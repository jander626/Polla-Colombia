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
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from .backtest import Trade, simulate_signal
from .config import STATE_FILE, BacktestParams
from .risk import Levels
from .strategy import Signal
from .universe import get as get_instrument


@dataclass
class TradingState:
    chat_ids: list[str] = field(default_factory=list)
    tg_offset: int = 0
    last_scan_date: str = ""           # ISO, evita reenviar el mismo día
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

    # ── Señales ───────────────────────────────────────────────────────────────

    def open_symbols(self) -> set[str]:
        return {record["symbol"] for record in self.open_signals}

    def record_signal(self, signal: Signal, confidence: float) -> dict:
        record = {
            "symbol": signal.symbol,
            "asset_class": signal.asset_class,
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
            return {"closed": 0, "open": len(self.open_signals), "expired": _expired(self.history)}

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
            "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
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


def signal_from_record(record: dict) -> Signal:
    """Reconstruye la señal guardada para poder simularla."""
    entry, stop, target = record["entry_max"], record["stop"], record["target"]
    risk, reward = entry - stop, target - entry

    levels = Levels(
        entry_max=entry,
        stop=stop,
        target=target,
        risk_reward=reward / risk if risk else 0.0,
        risk_per_unit=risk,
        reward_per_unit=reward,
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
