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


def _evidence_line(
    calibration: Calibration,
    asset_class: str,
    stale: bool = False,
) -> str:
    """Qué dice el histórico. Un hecho, no una promesa.

    Aquí vivía `_confidence_block`, que publicaba "acierto histórico 32%,
    beneficio esperado +0.053R". Se retiró cuando cuatro mediciones
    independientes coincidieron en que ese número no describe nada operable:

      - La partición temporal: ventaja en 2021-24, nada en 2024-26.
      - El barrido de 81 combinaciones: ninguna pasa la validación.
      - Contra comprar y mantener: el exceso es indistinguible de cero, y
        harían falta ~45 años de datos para demostrarlo al tamaño que tiene.
      - El diagnóstico: la puntuación no ordena.

    Un porcentaje que se lee como probabilidad de acierto, cuando detrás no
    hay ventaja demostrada, es la parte del bot que podía costar dinero de
    verdad. Lo que queda es una frase que dice lo que se sabe.
    """
    cierre = (
        "Criba, no recomendación: los niveles están calculados, "
        "la decisión es tuya."
    )

    if not calibration.is_calibrated or stale:
        return f"_Sin histórico aplicable a esta versión de los filtros. {cierre}_"

    recent = calibration.recent_for_asset_class(asset_class)
    period = f" ({calibration.recent_label})" if calibration.recent_label else ""

    if recent is None:
        return (
            "_No hay histórico reciente suficiente para esta clase de activo. "
            f"{cierre}_"
        )
    if not recent.has_edge:
        return (
            f"_En el histórico reciente{period} estos filtros NO muestran ventaja "
            f"demostrada ({recent.expectancy_lower:+.3f}R sobre {recent.samples} "
            f"operaciones). {cierre}_"
        )
    return (
        f"_En el histórico reciente{period} estos filtros sí muestran ventaja "
        f"({recent.expectancy_lower:+.3f}R sobre {recent.samples} operaciones). "
        f"{cierre}_"
    )


def _filters_passed(signal: Signal) -> list[str]:
    """Por qué apareció este instrumento. Es el producto del cribador.

    Sin esto, la tarjeta solo dice "aquí tienes un símbolo": el usuario no
    puede juzgar el candidato ni descartarlo con criterio propio, que es
    justo lo que un cribador tiene que permitirle hacer.
    """
    lines = ["✓ En tendencia alcista (cierre sobre la EMA200, EMA50 sobre EMA200)"]
    if signal.adx == signal.adx:                      # descarta NaN
        lines.append(f"✓ Con fuerza de tendencia (ADX {signal.adx:.0f})")
    if signal.pullback_rsi == signal.pullback_rsi:
        lines.append(
            f"✓ Retrocedió a la zona de la EMA20 (el RSI bajó a {signal.pullback_rsi:.0f})"
        )
    if signal.rsi == signal.rsi:
        lines.append(f"✓ Está reanudando (RSI {signal.rsi:.0f} y subiendo)")
    lines.append(f"✓ Volatilidad en rango (ATR {100 * signal.atr_pct:.1f}% del precio)")
    return lines


def format_signal(
    signal: Signal,
    calibration: Calibration,
    risk: Optional[EventRisk] = None,
    stale_calibration: bool = False,
) -> str:
    """Ficha de un instrumento que pasó la criba.

    Ya no es una señal de compra: es un candidato con sus niveles calculados
    y el motivo por el que aparece. Incluye siempre stop y ratio
    riesgo/beneficio, porque un nivel de entrada sin el riesgo al lado es la
    mitad de la información que hace falta para decidir.
    """
    instrument = get_instrument(signal.symbol)
    levels = signal.levels
    kind = "PAR" if signal.is_forex else "ACCIÓN"

    lines = [
        f"🔎 *{signal.symbol}* — {instrument.name}",
        f"_{kind}_",
        "",
        *_filters_passed(signal),
        "",
        f"📥 Zona de entrada: hasta *{_fmt(levels.entry_max, instrument)}*",
        f"🛑 Stop estructural: *{_fmt(levels.stop, instrument)}*",
        f"🎯 Objetivo (3 ATR): *{_fmt(levels.target, instrument)}*",
        f"⚖️ Riesgo/beneficio: *1:{levels.risk_reward:.2f}*",
    ]

    if risk is not None and risk.rationale:
        lines += ["", f"💡 {sanitize(risk.rationale)}"]
    if risk is not None and risk.risks:
        lines += [f"⚠️ {sanitize(risk.risks)}"]
    if risk is None:
        lines += ["", "_Sin verificación de noticias en este candidato._"]

    lines += [
        "",
        _evidence_line(calibration, signal.asset_class, stale_calibration),
        "",
        f"🕯️ Calculado con el cierre del {signal.bar_date:%d/%m/%Y}",
    ]
    return "\n".join(lines)


def _reasons(already_open: list[str], event_risk: list[str]) -> list[str]:
    lines = []
    if already_open:
        lines.append(
            f"• {', '.join(already_open)}: ya tienes la posición abierta"
        )
    if event_risk:
        lines.append(
            f"• {', '.join(event_risk)}: riesgo de evento alto (resultados o dato macro)"
        )
    return lines


def format_scan_header(
    count: int,
    shown: int,
    already_open: Optional[list[str]] = None,
    event_risk: Optional[list[str]] = None,
) -> str:
    """Cabecera del escaneo, con los descartes explicados.

    `shown` es lo que se va a enviar DE VERDAD, no lo que se encontró. La
    diferencia importa: el 11 de agosto la cabecera anunció una oportunidad,
    el único candidato quedó descartado por tener ya la posición abierta, y no
    llegó ninguna señal ni ninguna explicación.
    """
    lines = [
        "📡 *CRIBA DIARIA*",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Cumplen los filtros: *{count}*",
    ]
    if shown < count:
        lines.append(f"Se envían: *{shown}*")

    reasons = _reasons(already_open or [], event_risk or [])
    if reasons:
        lines += ["", "_Descartadas:_"] + reasons
    return "\n".join(lines)


def format_all_filtered(
    count: int, already_open: list[str], event_risk: list[str]
) -> str:
    """Se encontró algo pero no se envía nada. Hay que decir por qué.

    Un silencio sin explicación es indistinguible de un bot averiado, y eso
    fue exactamente lo que ocurrió el 11 de agosto.
    """
    lines = [
        "📡 *CRIBA DIARIA*",
        "━━━━━━━━━━━━━━━━━━━━",
        f"*{count}* instrumentos cumplen los filtros, pero ninguno se envía.",
        "",
        "_Motivo:_",
    ]
    reasons = _reasons(already_open, event_risk)
    lines += reasons or ["• Sin candidatos que superen los filtros de entrega."]

    if already_open:
        lines += [
            "",
            "_No se repite una señal sobre un instrumento que ya tienes "
            "abierto: duplicaría tu exposición sin que tú lo hayas decidido._",
        ]
    return "\n".join(lines)


def format_no_signals(funnel=None) -> str:
    """Día sin señales, con el embudo que explica dónde se quedaron.

    Sin el embudo, varios días seguidos de "hoy no hay nada" no permiten
    distinguir un mercado tranquilo de un filtro demasiado exigente o de un bot
    averiado. Con él, cada silencio viene firmado.
    """
    parts = [
        "📡 *CRIBA DIARIA*",
        "━━━━━━━━━━━━━━━━━━━━",
        "Hoy ningún instrumento cumple los filtros.",
    ]

    if funnel is not None and funnel.evaluated:
        parts.append("")
        parts.append(f"De *{funnel.evaluated}* instrumentos revisados:")
        for _, label, survivors, dropped in funnel.stages():
            if dropped <= 0 and survivors == funnel.evaluated:
                continue          # etapa que no descartó a nadie: no aporta
            parts.append(f"• {survivors} — {label}")
            if survivors == 0:
                break             # a partir de aquí ya no queda nadie

        bottleneck = funnel.bottleneck
        if bottleneck is not None:
            _, label, dropped = bottleneck
            parts.append("")
            parts.append(f"_El filtro que más descartó hoy: {label} ({dropped})._")

    parts.append("")
    parts.append("_Un cribador que encuentra algo todos los días no está cribando._")
    return "\n".join(parts)


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
    lines += [
        "",
        "_Estos son resultados en vivo sobre los candidatos que se enviaron._",
        "_Es la única medición que no viene de un backtest, y por eso la única"
        " que puede llegar a demostrar algo sobre estos filtros. Hacen falta"
        " cientos de operaciones antes de que signifique nada._",
    ]
    return "\n".join(lines)


def format_help() -> str:
    return (
        "🤖 *Cribador de Quantfury*\n\n"
        "Revisa cada día, antes de la apertura de EE.UU., las acciones y pares "
        "de forex operables en Quantfury, y te enseña los que cumplen seis "
        "filtros de retroceso en tendencia alcista — con su zona de entrada, "
        "su stop estructural y su ratio riesgo/beneficio ya calculados.\n\n"
        "*No emite señales con ventaja demostrada.* Cuatro mediciones "
        "independientes sobre cinco años coinciden en que estos filtros no "
        "baten al índice de forma distinguible del azar. Lo que hace es "
        "ahorrarte mirar 141 gráficos cada mañana; la decisión es tuya.\n\n"
        "Comandos:\n"
        "• /senales — Últimos candidatos enviados\n"
        "• /abiertas — Operaciones en curso\n"
        "• /rendimiento — Resultado real de lo que se cribó\n"
        "• /instrumentos — Universo vigilado\n"
        "• /ayuda — Este mensaje\n\n"
        "_Esto no es asesoría financiera. Las órdenes las colocas tú en "
        "Quantfury._"
    )
