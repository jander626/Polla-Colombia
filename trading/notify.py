"""Entrega por Telegram y formato de los mensajes.

La capa de transporte viene del bot del Mundial que vivía en este repositorio
(`bot.py`, retirado). Se conserva porque resuelve dos problemas que solo se
descubren perdiendo mensajes en producción:

1. Telegram rechaza cualquier mensaje de más de 4096 caracteres, así que hay
   que trocearlo por saltos de línea y no a mitad de palabra.
2. Si el Markdown tiene un asterisco o un guion bajo descolocado, Telegram
   rechaza el mensaje ENTERO con un 400. El reintento en texto plano es la
   diferencia entre una alerta fea y ninguna alerta.

El formato usa el Markdown antiguo de Telegram: un solo asterisco para
negrita. Los dobles asteriscos y los guiones bajos rompen el análisis.
"""

from __future__ import annotations

from typing import Iterable, Optional

import requests

from .config import TELEGRAM_TOKEN
from .llm_filter import EventRisk
from .risk import Calibration, OutcomeStats
from .strategy import Signal
from .universe import Instrument, get as get_instrument

TELEGRAM_MAX_LEN = 4000  # el límite real es 4096; se deja margen
API_BASE = "https://api.telegram.org/bot"


class TelegramClient:
    def __init__(self, token: str = "") -> None:
        self.token = token or TELEGRAM_TOKEN

    @property
    def is_configured(self) -> bool:
        return bool(self.token)

    def _url(self, method: str) -> str:
        return f"{API_BASE}{self.token}/{method}"

    def send(self, chat_id: str, text: str) -> bool:
        if not self.is_configured:
            print("[WARN] Sin TELEGRAM_BOT_TOKEN: no se envía nada")
            return False

        ok = True
        for chunk in _split_message(text):
            payload = {"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"}
            try:
                resp = requests.post(self._url("sendMessage"), json=payload, timeout=15)
                if resp.status_code != 200:
                    # Un error de Markdown tumba el mensaje completo; se reenvía
                    # en texto plano antes que perder la alerta.
                    print(
                        f"[WARN] Telegram rechazó el Markdown ({resp.status_code}): "
                        f"{resp.text[:150]} — reintentando en texto plano"
                    )
                    resp = requests.post(
                        self._url("sendMessage"),
                        json={"chat_id": chat_id, "text": chunk},
                        timeout=15,
                    )
                if resp.status_code != 200:
                    print(f"[ERROR] Envío fallido ({resp.status_code}): {resp.text[:150]}")
                    ok = False
            except requests.RequestException as exc:
                print(f"[ERROR] Envío a Telegram: {exc}")
                ok = False
        return ok

    def get_updates(self, offset: int = 0) -> list[dict]:
        if not self.is_configured:
            return []
        try:
            resp = requests.get(
                self._url("getUpdates"),
                params={"offset": offset, "timeout": 5},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json().get("result", [])
            print(f"[WARN] getUpdates devolvió {resp.status_code}")
        except requests.RequestException as exc:
            print(f"[ERROR] getUpdates: {exc}")
        return []

    def broadcast(self, text: str, chat_ids: Iterable[str]) -> bool:
        """Envía a todos los chats. True si al menos uno recibió el mensaje."""
        delivered = False
        for chat_id in dict.fromkeys(str(c) for c in chat_ids if c):
            if self.send(chat_id, text):
                delivered = True
        return delivered


def _split_message(text: str) -> list[str]:
    """Trocea por saltos de línea para no cortar a mitad de una frase."""
    chunks: list[str] = []
    remaining = text
    while len(remaining) > TELEGRAM_MAX_LEN:
        cut = remaining.rfind("\n", 0, TELEGRAM_MAX_LEN)
        if cut <= 0:
            cut = TELEGRAM_MAX_LEN
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    chunks.append(remaining)
    return chunks


def sanitize(text: str) -> str:
    """Adapta texto libre al Markdown antiguo de Telegram."""
    return text.replace("**", "*").replace("__", "_").replace("`", "'").strip()


# ── Formato de las alertas ────────────────────────────────────────────────────

def _fmt(value: float, instrument: Instrument) -> str:
    return f"{value:.{instrument.price_decimals}f}"


def _confidence_block(
    stats: Optional[OutcomeStats],
    penalty: float = 0.0,
    stale: bool = False,
) -> str:
    """Bloque de confianza: acierto histórico Y beneficio esperado.

    Publicar solo el acierto engañaba. Este sistema acierta alrededor del 35% de
    las veces y aun así gana dinero, porque el objetivo está al doble de
    distancia que el stop; una alerta que dijera «confianza 35%» se leería como
    «esto va a fallar», que es justo lo contrario. Las dos cifras juntas, con la
    explicación, es lo único honesto.
    """
    if stats is None or stats.samples == 0:
        return (
            "📊 *Sin calibrar*\n"
            "     _No hay histórico para este tipo de instrumento._\n"
            "     _Ejecuta el backtest antes de fiarte de esta señal._"
        )

    win_rate = max(0.0, stats.win_rate_lower - penalty)
    expectancy = stats.expectancy_lower

    lines = [
        f"📊 *Acierto histórico: {win_rate:.0f}%*",
        # Tres decimales, no dos: con dos, una ventaja de +0.005R se imprimía
        # como "+0.00R" mientras el texto afirmaba que la señal ganaba.
        f"💰 *Beneficio esperado: {expectancy:+.3f}R por operación*",
    ]

    if stats.has_edge:
        lines.append(
            f"     _De cada 100 señales así, ~{win_rate:.0f} llegan al objetivo._\n"
            "     _Gana en el agregado porque el objetivo está más lejos que el stop._"
        )
    elif expectancy > 0:
        # Positiva pero demasiado pequeña para servir de algo: se la come el
        # primer slippage. Decirlo es más útil que redondearla a cero y callar.
        lines.append(
            "     ⚠️ _Ventaja demasiado pequeña para ser útil: la absorbería el\n"
            "     coste de operar. Trátala como informativa._"
        )
    else:
        lines.append(
            "     ⚠️ _Sin ventaja demostrada en el histórico para esta clase de\n"
            "     activo. Trátala como informativa, no como recomendación._"
        )

    if stale:
        lines.append(
            "     🔄 _Calibración de una versión anterior de la estrategia:\n"
            "     estas cifras no describen esta señal. Vuelve a calibrar._"
        )

    if not stats.is_reliable:
        lines.append(
            f"     _Solo {stats.samples} operaciones históricas: poco fiable._"
        )
    if penalty > 0:
        lines.append(f"     _Acierto reducido en {penalty:.0f} puntos por riesgo de evento._")

    return "\n".join(lines)


def format_signal(
    signal: Signal,
    calibration: Calibration,
    risk: Optional[EventRisk] = None,
    stale_calibration: bool = False,
) -> str:
    """Mensaje de alerta de compra.

    Incluye siempre stop y ratio riesgo/beneficio: sin stop, el porcentaje de
    confianza no significa nada, porque una sola operación sin limitar puede
    borrar diez ganadoras.
    """
    instrument = get_instrument(signal.symbol)
    levels = signal.levels
    kind = "PAR" if signal.is_forex else "ACCIÓN"

    lines = [
        f"🟢 *COMPRA — {signal.symbol}*",
        f"_{instrument.name} · {kind}_",
        "",
        f"📥 Entrada: hasta *{_fmt(levels.entry_max, instrument)}*",
        f"🎯 Objetivo: *{_fmt(levels.target, instrument)}*",
        f"🛑 Stop: *{_fmt(levels.stop, instrument)}*",
        f"⚖️ Riesgo/beneficio: *1:{levels.risk_reward:.2f}*",
        "",
        _confidence_block(
            calibration.for_asset_class(signal.asset_class),
            penalty=risk.penalty if risk is not None else 0.0,
            stale=stale_calibration,
        ),
    ]

    if risk is not None and risk.rationale:
        lines += ["", f"💡 {sanitize(risk.rationale)}"]
    if risk is not None and risk.risks:
        lines += [f"⚠️ {sanitize(risk.risks)}"]
    if risk is None:
        lines += ["", "_Sin verificación de noticias en esta señal._"]

    lines += [
        "",
        f"📈 Puntuación técnica: {signal.score:.0f}/100",
        f"🕯️ Calculado con el cierre del {signal.bar_date:%d/%m/%Y}",
    ]
    return "\n".join(lines)


def format_scan_header(count: int, shown: int) -> str:
    return (
        "📡 *ESCANEO DIARIO*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"Oportunidades detectadas: *{count}*"
        + (f" (se envían las {shown} mejores)" if shown < count else "")
    )


def format_no_signals() -> str:
    return (
        "📡 *ESCANEO DIARIO*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Hoy no hay ninguna oportunidad que cumpla los criterios.\n\n"
        "_Un bot que encuentra señales todos los días se las está inventando._"
    )


def format_outcome(record: dict) -> str:
    """Aviso de cierre de una operación previamente alertada."""
    instrument = get_instrument(record["symbol"])
    outcome = record.get("outcome")
    entry = record.get("entry_price")
    exit_price = record.get("exit_price")

    if outcome == "win":
        head, verdict = "✅", "Objetivo alcanzado"
    elif outcome == "loss":
        head, verdict = "❌", "Stop alcanzado"
    else:
        head, verdict = "⏳", "Cerrada por tiempo"

    lines = [
        f"{head} *{record['symbol']} — {verdict}*",
        f"_{instrument.name}_",
        "",
        f"Entrada: {entry:.{instrument.price_decimals}f}",
        f"Salida:  {exit_price:.{instrument.price_decimals}f}",
    ]
    if record.get("return_pct") is not None:
        lines.append(f"Resultado: *{record['return_pct'] * 100:+.2f}%*")
    if record.get("r_multiple") is not None:
        lines.append(f"En múltiplos de riesgo: *{record['r_multiple']:+.2f}R*")
    return "\n".join(lines)


def format_performance(stats: dict) -> str:
    """Rendimiento real en vivo, frente a la confianza que se prometió."""
    closed = stats.get("closed", 0)
    if not closed:
        return (
            "📊 *RENDIMIENTO*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Todavía no hay operaciones cerradas que medir."
        )

    lines = [
        "📊 *RENDIMIENTO EN VIVO*",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Operaciones cerradas: *{closed}*",
        f"Aciertos: *{stats['wins']}* ({stats['win_rate'] * 100:.1f}%)",
        f"R medio: *{stats['avg_r']:+.2f}*",
        f"Resultado acumulado: *{stats['total_r']:+.1f}R*",
        f"Abiertas ahora: {stats.get('open', 0)}",
    ]
    promised = stats.get("avg_confidence")
    if promised:
        lines += [
            "",
            f"Confianza media prometida: {promised:.0f}%",
            f"Acierto real: {stats['win_rate'] * 100:.0f}%",
            "_Si el acierto real queda muy por debajo de lo prometido de forma"
            " sostenida, la calibración está mal y hay que rehacer el backtest._",
        ]
    return "\n".join(lines)


def format_help() -> str:
    return (
        "🤖 *Bot de señales de trading*\n\n"
        "Analiza acciones y pares de forex operables en Quantfury una vez al "
        "día, antes de la apertura de EE.UU., y avisa solo de oportunidades de "
        "*compra*.\n\n"
        "Comandos:\n"
        "• /senales — Últimas señales enviadas\n"
        "• /abiertas — Operaciones en curso\n"
        "• /rendimiento — Aciertos reales frente a la confianza prometida\n"
        "• /instrumentos — Universo vigilado\n"
        "• /ayuda — Este mensaje\n\n"
        "_Esto no es asesoría financiera. Las órdenes las colocas tú en "
        "Quantfury; el bot solo criba el mercado._"
    )
