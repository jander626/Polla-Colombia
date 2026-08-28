"""Estado persistente y seguimiento de las señales enviadas.

Sin esta pieza el bot no tendría forma de saber si funciona: solo dejaría la
sensación de que a veces acierta. Aquí se registra cada señal alertada y se
resuelve después contra el mercado, de modo que `/rendimiento` puede comparar
el acierto REAL con la confianza que se prometió.

Esa comparación es el único control externo que tiene la calibración. Si el
acierto real queda sistemáticamente por debajo de lo prometido, la tabla de
`calibration.json` está mal y hay que rehacer el backtest.

La resolución de las operaciones NO se reimplementa aquí: se delega en
`backtest.simulate_signal`. Si el seguimiento en vivo y el backtest aplicaran
reglas distintas, la comparación no significaría nada.

Pero el usuario opera a mano, y eso abre tres desenlaces donde el simulador
solo modela uno:

1. Toma el candidato y sale por objetivo o stop → el simulador lo resuelve.
2. Toma el candidato y **sale cuando quiere** → el simulador registraría una
   salida ficticia. Pasó con ABBV, cerrada a mano en 266 con el objetivo en
   268.27.
3. **No lo toma** → el simulador anotaría una operación que nadie tuvo. Pasó
   con F.

Los casos 2 y 3 se corregían editando este JSON a mano. `mark_taken`,
`close_manually` y `mark_skipped` son lo que hay detrás de `/tomada`,
`/cerrar` y `/paso` en Telegram, para que la única medición que no viene de
un backtest deje de depender de que alguien recuerde tocar un fichero.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from .backtest import Trade, simulate_signal
from .config import DEFAULT_BACKTEST, LIVE_DECISION_SAMPLE, STATE_FILE, BacktestParams
from .risk import Levels, mean_lower_bound
from .schedule import now_ny
from .strategy import Signal
from .universe import get as get_instrument


@dataclass
class ManualUpdate:
    """Resultado de una corrección manual del seguimiento.

    `ok` dice si el estado cambió; `reason` es un código estable que la capa
    de Telegram traduce a un mensaje. Se devuelve un código y no el texto
    porque decidir qué pasó y decidir cómo contarlo son dos trabajos
    distintos: mezclarlos obligaría a probar el formato para probar la regla.
    """

    ok: bool
    reason: str
    record: Optional[dict] = None
    warning: str = ""


@dataclass
class TradingState:
    chat_ids: list[str] = field(default_factory=list)
    tg_offset: int = 0
    last_scan_date: str = ""           # ISO, evita reenviar el mismo día
    # ISO, evita revisar stop/target más de una vez al día. Sin esto, cada
    # disparo del cron (hasta ~17 al día) repetía la consulta de mercado con
    # las mismas velas diarias: gastaba cupo de Twelve Data sin aprender nada
    # nuevo, porque una vela diaria no cambia entre un disparo y el siguiente
    # de la misma jornada.
    last_track_date: str = ""
    open_signals: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    # ── Persistencia ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str = STATE_FILE) -> "TradingState":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            # Un estado corrupto no puede impedir que el bot opere: se avisa y
            # se arranca limpio, que es preferible a quedarse mudo.
            print(f"[WARN] Estado ilegible ({exc}); se empieza de cero")
            return cls()

        return cls(
            chat_ids=[str(c) for c in raw.get("chat_ids", [])],
            tg_offset=int(raw.get("tg_offset", 0)),
            last_scan_date=str(raw.get("last_scan_date", "")),
            last_track_date=str(raw.get("last_track_date", "")),
            open_signals=list(raw.get("open_signals", [])),
            history=list(raw.get("history", [])),
        )

    def save(self, path: str = STATE_FILE) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)

    # ── Registro de chats ─────────────────────────────────────────────────────

    def register_chat(self, chat_id: str) -> bool:
        chat_id = str(chat_id)
        if chat_id and chat_id not in self.chat_ids:
            self.chat_ids.append(chat_id)
            print(f"[INFO] Chat registrado: {chat_id}")
            return True
        return False

    # ── Control del escaneo diario ────────────────────────────────────────────

    def already_scanned(self, day: date) -> bool:
        return self.last_scan_date == day.isoformat()

    def mark_scanned(self, day: date) -> None:
        self.last_scan_date = day.isoformat()

    def already_tracked(self, day: date) -> bool:
        return self.last_track_date == day.isoformat()

    def mark_tracked(self, day: date) -> None:
        self.last_track_date = day.isoformat()

    # ── Señales ───────────────────────────────────────────────────────────────

    def open_symbols(self) -> set[str]:
        return {record["symbol"] for record in self.open_signals}

    def record_signal(self, signal: Signal, confidence: float) -> dict:
        record = {
            "symbol": signal.symbol,
            "asset_class": signal.asset_class,
            # Sin esto, un corto guardado se releería como largo y su stop
            # quedaría del lado equivocado al resolverlo.
            "direction": signal.levels.direction,
            "signal_date": signal.bar_date.strftime("%Y-%m-%d"),
            "alerted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "score": round(signal.score, 2),
            "confidence": round(confidence, 1),
            "entry_max": signal.levels.entry_max,
            "stop": signal.levels.stop,
            "target": signal.levels.target,
            "risk_reward": round(signal.levels.risk_reward, 3),
        }
        self.open_signals.append(record)
        return record

    # ── Correcciones manuales (comandos de Telegram) ─────────────────────────

    def find_open(self, symbol: str) -> Optional[dict]:
        """Operación abierta por símbolo, sin distinguir mayúsculas."""
        wanted = symbol.strip().upper()
        for record in self.open_signals:
            if str(record.get("symbol", "")).upper() == wanted:
                return record
        return None

    def mark_taken(
        self, symbol: str, price: float, day: Optional[date] = None
    ) -> ManualUpdate:
        """`/tomada`: registra el precio al que el usuario entró de verdad.

        El simulador entra al techo de la zona (o a la apertura si hubo
        hueco), que casi nunca es el céntimo que dio Quantfury. Guardar el
        precio real es lo que permite que el R medido sea el del usuario y no
        el de una operación parecida.
        """
        record = self.find_open(symbol)
        if record is None:
            return ManualUpdate(False, "not_open")
        if not _is_price(price):
            return ManualUpdate(False, "bad_price")
        pasado_el_stop = (
            price >= record["stop"] if _is_short(record) else price <= record["stop"]
        )
        if pasado_el_stop:
            # Entrar ya pasado el stop no describe ninguna operación viva:
            # habría nacido cerrada. Es un dedazo con casi total seguridad, y
            # aceptarlo en silencio envenenaría la única medición real. En
            # corto el lado peligroso es el de arriba.
            return ManualUpdate(False, "below_stop", record)

        record["taken"] = True
        record["real_entry_price"] = float(price)
        record["real_entry_date"] = (day or _today()).isoformat()

        # Entrar fuera del límite de la zona no se rechaza —es un hecho, no una
        # propuesta—, pero cambia el seguimiento: el simulador nunca habría
        # ejecutado esa orden, así que no sabrá cerrarla solo. En largo eso es
        # comprar por encima del techo; en corto, vender por debajo del suelo.
        fuera = (
            price < record["entry_max"]
            if _is_short(record)
            else price > record["entry_max"]
        )
        warning = "outside_zone" if fuera else ""
        return ManualUpdate(True, "taken", record, warning)

    def mark_skipped(self, symbol: str) -> ManualUpdate:
        """`/paso`: el candidato se envió pero el usuario no lo tomó.

        Sale del seguimiento sin contar como operación. No se arriesgó
        dinero, así que sumarlo al acierto —en cualquiera de los dos lados—
        sería inventar historia.
        """
        record = self.find_open(symbol)
        if record is None:
            return ManualUpdate(False, "not_open")

        finished = dict(record)
        finished["status"] = "not_taken"
        finished["note"] = "Cribada y enviada, pero el usuario decidió no tomarla."
        self._drop_open(record)
        self.history.append(finished)
        return ManualUpdate(True, "skipped", finished)

    def close_manually(
        self,
        symbol: str,
        exit_price: float,
        entry_price: Optional[float] = None,
        day: Optional[date] = None,
        bt: BacktestParams = DEFAULT_BACKTEST,
    ) -> ManualUpdate:
        """`/cerrar`: el usuario salió cuando quiso, no donde decía el plan.

        La R se mide contra el riesgo PLANIFICADO (techo de la zona menos
        stop), igual que en el backtest: es lo que el usuario conocía al
        dimensionar la posición. Usar el riesgo realizado contaría como 1R
        completa una pérdida que fue menor.
        """
        record = self.find_open(symbol)
        if record is None:
            return ManualUpdate(False, "not_open")
        if not _is_price(exit_price):
            return ManualUpdate(False, "bad_price")
        if entry_price is not None and not _is_price(entry_price):
            return ManualUpdate(False, "bad_price")

        entry = entry_price if entry_price is not None else record.get("real_entry_price")
        if entry is None:
            # Sin entrada no hay resultado que medir. Inventarla con el techo
            # de la zona daría un número creíble y falso, que es peor que no
            # dar ninguno.
            return ManualUpdate(False, "unknown_entry", record)

        entry = float(entry)
        exit_price = float(exit_price)
        planned_risk = _planned_risk(record)
        move = _move(record, entry, exit_price)
        net_return = move / entry - bt.round_trip_cost

        finished = dict(record)
        finished.update(
            {
                "status": "closed",
                "outcome": "manual",
                "entry_price": entry,
                "exit_price": exit_price,
                "entry_date": record.get("real_entry_date"),
                "exit_date": (day or _today()).isoformat(),
                "r_multiple": _clean(move / planned_risk) if planned_risk > 0 else None,
                "return_pct": _clean(net_return),
                "is_win": net_return > 0,
                "note": "Cerrada a mano por el usuario con /cerrar.",
            }
        )
        self._drop_open(record)
        self.history.append(finished)
        return ManualUpdate(True, "closed", finished)

    def _drop_open(self, record: dict) -> None:
        # Por identidad y no por símbolo: si algún día hubiera dos registros
        # del mismo instrumento, borrar por símbolo se llevaría los dos.
        self.open_signals = [r for r in self.open_signals if r is not record]

    # ── Resolución contra el mercado ──────────────────────────────────────────

    def update_open_signals(
        self, bars: dict[str, pd.DataFrame], bt: BacktestParams
    ) -> list[dict]:
        """Cierra las operaciones resueltas. Devuelve los registros cerrados."""
        still_open: list[dict] = []
        closed: list[dict] = []

        for record in self.open_signals:
            df = bars.get(record["symbol"])
            if df is None or df.empty:
                still_open.append(record)  # sin datos hoy, se reintenta mañana
                continue

            status, trade = evaluate_record(record, df, bt)

            if status in ("pending", "open"):
                still_open.append(record)
                continue

            if record.get("taken") and not _simulation_describes(record, status, trade):
                # El usuario dice que está dentro y el simulador no lo sigue.
                # Manda el usuario: la operación existe. Se queda abierta hasta
                # que llegue `/cerrar`, que es quien conoce la salida real.
                print(
                    f"[INFO] {record['symbol']}: tomada a mano fuera de lo que "
                    "el simulador puede seguir; esperando /cerrar"
                )
                still_open.append(record)
                continue

            finished = dict(record)
            finished["status"] = status
            if trade is not None:
                finished.update(
                    {
                        "outcome": trade.outcome,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "entry_date": _iso(trade.entry_date),
                        "exit_date": _iso(trade.exit_date),
                        "r_multiple": _clean(trade.r_multiple),
                        "return_pct": _clean(trade.return_pct),
                        "is_win": trade.is_win,
                    }
                )
                _apply_real_entry(finished, bt)
            self.history.append(finished)
            closed.append(finished)

        self.open_signals = still_open
        return closed

    # ── Métricas en vivo ──────────────────────────────────────────────────────

    def performance(self) -> dict:
        # Las señales caducadas sin llegar a ejecutarse no son operaciones y no
        # deben contaminar el acierto: nunca se arriesgó dinero en ellas.
        executed = [h for h in self.history if h.get("status") == "closed"]
        if not executed:
            return {
                "closed": 0,
                "open": len(self.open_signals),
                "expired": _expired(self.history),
                "not_taken": _not_taken(self.history),
            }

        wins = [h for h in executed if h.get("is_win")]
        r_values = [h["r_multiple"] for h in executed if h.get("r_multiple") is not None]
        confidences = [h["confidence"] for h in executed if h.get("confidence")]

        return {
            "closed": len(executed),
            "wins": len(wins),
            "win_rate": len(wins) / len(executed),
            "avg_r": sum(r_values) / len(r_values) if r_values else 0.0,
            "total_r": sum(r_values),
            "open": len(self.open_signals),
            "expired": _expired(self.history),
            # Cuántos candidatos enviados el usuario decidió no tomar. Sin este
            # número, el rendimiento parece describir lo que hizo el usuario
            # cuando solo describe lo que el bot propuso.
            "not_taken": _not_taken(self.history),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
            **_live_decision(r_values),
        }


# ── Utilidades ────────────────────────────────────────────────────────────────

def _iso(value) -> Optional[str]:
    return value.strftime("%Y-%m-%d") if value is not None else None


def _clean(value: float) -> Optional[float]:
    """JSON no admite NaN ni infinitos."""
    if value is None or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return round(float(value), 4)


def _expired(history: list[dict]) -> int:
    return sum(1 for h in history if h.get("status") == "expired")


def _direction(record: dict) -> str:
    """Sentido del registro. Los guardados antes de existir cortos son largos."""
    return str(record.get("direction", "long"))


def _is_short(record: dict) -> bool:
    return _direction(record) == "short"


def _planned_risk(record: dict) -> float:
    """Riesgo planificado, siempre positivo: la distancia entrada-stop."""
    if _is_short(record):
        return record["stop"] - record["entry_max"]
    return record["entry_max"] - record["stop"]


def _move(record: dict, entry: float, exit_price: float) -> float:
    """Movimiento a favor, con el signo que le toque al sentido."""
    return (entry - exit_price) if _is_short(record) else (exit_price - entry)


def _not_taken(history: list[dict]) -> int:
    return sum(1 for h in history if h.get("status") == "not_taken")


def _live_decision(r_values: list[float]) -> dict:
    """Progreso hacia la regla de decisión fijada en `config.LIVE_DECISION_SAMPLE`.

    El veredicto ("sigue"/"se apaga") solo aparece con la muestra completa:
    calcularlo antes sería el mismo error que "mayor que cero" ya costó dos
    veces en este proyecto —una vez en las alertas, otra en la partición
    temporal—. Con menos muestra se publica el progreso y nada más.
    """
    objetivo = LIVE_DECISION_SAMPLE
    n = len(r_values)
    if n < objetivo:
        return {"decision_target": objetivo, "decision_ready": False}

    total = sum(r_values)
    total_sq = sum(r * r for r in r_values)
    r_lower = mean_lower_bound(total, total_sq, n)
    return {
        "decision_target": objetivo,
        "decision_ready": True,
        "r_lower": r_lower,
        "decision_sigue": r_lower > 0.0,
    }


def _today() -> date:
    """La fecha que manda es la de Nueva York: es la del mercado que se opera."""
    return now_ny().date()


def _is_price(value) -> bool:
    """Un precio válido es finito y positivo. `bool(nan)` es True; ojo con eso."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    value = float(value)
    return value == value and value not in (float("inf"), float("-inf")) and value > 0


def _simulation_describes(record: dict, status: str, trade: Optional[Trade]) -> bool:
    """¿Está el simulador siguiendo la operación que el usuario tiene de verdad?

    `/tomada` puede registrar una entrada que el simulador nunca planificó: el
    usuario compró por encima del techo de la zona, o entró tan tarde que la
    orden ya había caducado. Cerrar esos casos con los números del simulador
    inventaría un desenlace sobre dinero real.
    """
    if status == "expired" or trade is None or trade.exit_date is None:
        return False

    real_entry = record.get("real_entry_date")
    if real_entry and pd.Timestamp(real_entry) > pd.Timestamp(trade.exit_date):
        return False
    return True


def _apply_real_entry(record: dict, bt: BacktestParams) -> None:
    """Recalcula el resultado contra la entrada real cuando `/tomada` la dio.

    Lo que NO se toca es cómo se detecta la salida: eso lo sigue decidiendo
    `simulate_signal`, igual que en el backtest. Aquí solo cambia el precio
    contra el que se miden el retorno y la R, y se conserva el de la
    simulación para poder auditar la diferencia.
    """
    entry = record.get("real_entry_price")
    exit_price = record.get("exit_price")
    if not _is_price(entry) or not _is_price(exit_price):
        return

    entry, exit_price = float(entry), float(exit_price)
    planned_risk = _planned_risk(record)
    move = _move(record, entry, exit_price)
    net_return = move / entry - bt.round_trip_cost

    record["simulated_entry_price"] = record.get("entry_price")
    record["entry_price"] = entry
    if record.get("real_entry_date"):
        record["entry_date"] = record["real_entry_date"]
    record["r_multiple"] = _clean(move / planned_risk) if planned_risk > 0 else None
    record["return_pct"] = _clean(net_return)
    record["is_win"] = net_return > 0


def signal_from_record(record: dict) -> Signal:
    """Reconstruye la señal guardada para poder simularla."""
    entry, stop, target = record["entry_max"], record["stop"], record["target"]
    if _is_short(record):
        risk, reward = stop - entry, entry - target
    else:
        risk, reward = entry - stop, target - entry

    levels = Levels(
        entry_max=entry,
        stop=stop,
        target=target,
        risk_reward=reward / risk if risk else 0.0,
        risk_per_unit=risk,
        reward_per_unit=reward,
        direction=_direction(record),
    )
    instrument = get_instrument(record["symbol"])
    return Signal(
        symbol=record["symbol"],
        name=instrument.name,
        asset_class=record.get("asset_class", instrument.asset_class),
        bar_date=pd.Timestamp(record["signal_date"]),
        close=entry,
        atr=float("nan"),
        atr_pct=float("nan"),
        score=float(record.get("score", 0.0)),
        levels=levels,
    )


def evaluate_record(
    record: dict, df: pd.DataFrame, bt: BacktestParams
) -> tuple[str, Optional[Trade]]:
    """Estado actual de una señal: pending, open, expired o closed.

    Reutiliza `simulate_signal` para que la resolución en vivo sea idéntica a
    la del backtest. La única diferencia es distinguir "todavía no ha pasado
    nada" de "ya no va a pasar": el simulador, al quedarse sin velas, informa
    de `no_fill` o `timeout`, y aquí se comprueba si de verdad se agotó el
    plazo o simplemente aún no hay historia suficiente.
    """
    signal = signal_from_record(record)
    trade = simulate_signal(signal, df, bt)

    try:
        signal_pos = df.index.get_loc(signal.bar_date)
        bars_after = len(df) - 1 - signal_pos
    except KeyError:
        # La vela de la señal no está en los datos (símbolo sin cotizar ese
        # día, o histórico recortado): no se puede decidir, sigue abierta.
        return "pending", None

    if trade.outcome == "no_fill":
        if bars_after < bt.entry_valid_days:
            return "pending", None
        return "expired", trade

    if trade.outcome == "timeout" and trade.bars_held < bt.max_holding_days:
        return "open", None

    return "closed", trade
