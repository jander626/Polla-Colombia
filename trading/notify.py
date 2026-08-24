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

import numpy as np
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


def parse_price(text: str) -> Optional[float]:
    """Lee un precio tecleado en el móvil. None si no hay número utilizable.

    El usuario escribe desde Colombia, donde el separador decimal es la coma,
    y el teclado del móvil cuela espacios y símbolos de moneda sin esfuerzo.
    Rechazar "266,5" por eso le devolvería a editar `trading_state.json` a
    mano, que es exactamente lo que estos comandos vienen a quitar de en medio.
    """
    cleaned = text.strip().replace("$", "").replace(" ", "").replace("\u00a0", "")
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        # "1.234,56" o "1,234.56": manda el separador que está más a la
        # derecha, que en ambas convenciones es el decimal.
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        # Una coma sola es decimal: "1,0850" es un par de forex, no mil.
        cleaned = cleaned.replace(",", ".")

    try:
        value = float(cleaned)
    except ValueError:
        return None
    # Un precio no finito o negativo no es un dedazo recuperable: es basura.
    if value != value or value in (float("inf"), float("-inf")) or value <= 0:
        return None
    return value


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

    Con dos reglas activas (`retroceso` y `reversion`), esto tiene que
    describir la que disparó de verdad. Antes de que `Signal` llevara
    `entry_rule`, una señal de `reversion` se anunciaba con "tocó la EMA20"
    y "MACD girando" —cosas que esa regla no mira— porque el texto asumía
    siempre el retroceso de seis filtros.
    """
    short = signal.levels.is_short
    if signal.entry_rule == "rsi2":
        return _filters_passed_reversion(signal, short)
    return _filters_passed_retroceso(signal, short)


def _filters_passed_reversion(signal: Signal, short: bool) -> list[str]:
    lines = [
        "✓ En tendencia bajista (cierre bajo la EMA200)"
        if short else
        "✓ En tendencia alcista (cierre sobre la EMA200)"
    ]
    if signal.rsi_fast == signal.rsi_fast:            # descarta NaN
        extremo = "sobrecomprado" if short else "sobrevendido"
        lines.append(f"✓ Muy {extremo} (RSI de 2 sesiones en {signal.rsi_fast:.0f})")
    lines.append(f"✓ Volatilidad en rango (ATR {100 * signal.atr_pct:.1f}% del precio)")
    return lines


def _filters_passed_retroceso(signal: Signal, short: bool) -> list[str]:
    if short:
        lines = ["✓ En tendencia bajista (cierre bajo la EMA200, EMA50 bajo EMA200)"]
    else:
        lines = ["✓ En tendencia alcista (cierre sobre la EMA200, EMA50 sobre EMA200)"]
    if signal.adx == signal.adx:
        lines.append(f"✓ Con fuerza de tendencia (ADX {signal.adx:.0f})")
    if signal.pullback_rsi == signal.pullback_rsi:
        verbo = "Rebotó" if short else "Retrocedió"
        hacia = "subió" if short else "bajó"
        lines.append(
            f"✓ {verbo} a la zona de la EMA20 (el RSI {hacia} a "
            f"{signal.pullback_rsi:.0f})"
        )
    if signal.rsi == signal.rsi:
        rumbo = "y bajando" if short else "y subiendo"
        lines.append(f"✓ Está reanudando (RSI {signal.rsi:.0f} {rumbo})")
    lines.append(f"✓ Volatilidad en rango (ATR {100 * signal.atr_pct:.1f}% del precio)")
    return lines


def _target_label(signal: Signal) -> str:
    """El objetivo ya no es siempre 3 ATR: la regla `reversion` usa 1.

    Se calcula desde los niveles en vez de leer el parámetro para no tener
    que pasar `StrategyParams` hasta aquí solo por esta etiqueta.
    """
    atr = signal.atr
    if not np.isfinite(atr) or atr <= 0:
        return ""
    mult = signal.levels.reward_per_unit / atr
    return f" ({mult:.0f} ATR)"


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
    short = levels.is_short

    # El sentido va en la cabecera y no en una línea perdida abajo: confundir
    # una venta con una compra es el peor error que puede cometer quien lee
    # esto con el móvil en la mano y el mercado a punto de abrir.
    encabezado = f"🔻 *{signal.symbol}* — {instrument.name}" if short else (
        f"🔎 *{signal.symbol}* — {instrument.name}"
    )
    etiqueta = f"_{kind} · CORTO (vender)_" if short else f"_{kind}_"
    entrada = "desde" if short else "hasta"

    lines = [
        encabezado,
        etiqueta,
        "",
        *_filters_passed(signal),
        "",
        f"📥 Zona de entrada: {entrada} *{_fmt(levels.entry_max, instrument)}*",
        f"🛑 Stop estructural: *{_fmt(levels.stop, instrument)}*",
        f"🎯 Objetivo{_target_label(signal)}: *{_fmt(levels.target, instrument)}*",
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


def outcome_label(record: dict) -> tuple[str, str]:
    """Icono y nombre legible del desenlace de un registro del histórico.

    Vive aquí y no en cada sitio que lo necesita porque el histórico ya no
    contiene solo desenlaces del simulador: desde que existen `/cerrar` y
    `/paso` hay registros cerrados a mano y candidatos que nadie tomó. Antes,
    `/senales` los imprimía como "❌ F — None".
    """
    status = record.get("status")
    if status == "not_taken":
        return "🚫", "No tomada"
    if status == "expired":
        return "⌛", "Caducada sin ejecutarse"

    outcome = record.get("outcome")
    if outcome == "manual":
        # El icono lo decide el resultado, que es lo que le importa al usuario;
        # que la salida fuera manual lo dice el texto.
        return ("✅" if record.get("is_win") else "❌"), "Cerrada a mano"
    if outcome == "win":
        return "✅", "Objetivo alcanzado"
    if outcome == "loss":
        return "❌", "Stop alcanzado"
    return "⏳", "Cerrada por tiempo"


def format_outcome(record: dict) -> str:
    """Aviso de cierre de una operación previamente alertada."""
    instrument = get_instrument(record["symbol"])
    entry = record.get("entry_price")
    exit_price = record.get("exit_price")

    head, verdict = outcome_label(record)

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


# ── Correcciones manuales del seguimiento ────────────────────────────────────
#
# El usuario coloca las órdenes a mano, así que el bot solo sabe de su cuenta
# lo que él le cuente. Estos mensajes son la mitad visible de `/tomada`,
# `/cerrar` y `/paso`: confirman qué quedó anotado, porque una corrección que
# no se confirma es una corrección que el usuario repetirá o dará por perdida.

def format_taken(record: dict, warning: str = "") -> str:
    """`/tomada`: confirma el precio de entrada real que queda registrado."""
    instrument = get_instrument(record["symbol"])
    price = record["real_entry_price"]

    lines = [
        f"📈 *{record['symbol']} — anotada como tomada*",
        f"_{instrument.name}_",
        "",
        f"Entrada real: *{_fmt(price, instrument)}*",
        f"🛑 Stop: {_fmt(record['stop'], instrument)}  ·  "
        f"🎯 Objetivo: {_fmt(record['target'], instrument)}",
    ]

    if warning == "outside_zone":
        lines += [
            "",
            f"⚠️ Entraste por encima del techo de la zona "
            f"({_fmt(record['entry_max'], instrument)}). Queda registrado tal "
            "cual —es lo que pasó—, pero el simulador no puede seguir una "
            "orden que nunca habría ejecutado: cuando salgas, ciérrala tú con "
            f"/cerrar {record['symbol']} PRECIO.",
        ]
    else:
        lines += [
            "",
            "_A partir de ahora el resultado se medirá contra tu precio real, "
            "no contra el techo de la zona._",
        ]
    return "\n".join(lines)


def format_skipped(record: dict) -> str:
    """`/paso`: confirma que el candidato sale del seguimiento sin contar."""
    instrument = get_instrument(record["symbol"])
    return "\n".join(
        [
            f"🚫 *{record['symbol']} — marcada como no tomada*",
            f"_{instrument.name}_",
            "",
            "Sale del seguimiento. No contará ni como acierto ni como fallo: "
            "no se arriesgó dinero en ella.",
            "",
            "_Si vuelve a cumplir los filtros otro día, volverá a aparecer._",
        ]
    )


def format_manual_close(record: dict) -> str:
    """`/cerrar`: el desenlace de siempre, más de dónde salen los números."""
    return format_outcome(record) + "\n\n" + (
        "_Calculado con el precio que me diste, no con el del plan: es lo que "
        "de verdad pasó en tu cuenta._"
    )


def format_manual_usage(command: str) -> str:
    """Cómo se escribe el comando. Se responde en vez de callar."""
    usage = {
        "/tomada": (
            "📈 */tomada* — anota que entraste, y a qué precio.\n\n"
            "Escríbelo así:\n"
            "/tomada TXN 280.5"
        ),
        "/cerrar": (
            "🔚 */cerrar* — cierra una operación al precio al que saliste.\n\n"
            "Escríbelo así:\n"
            "/cerrar TXN 295.4\n\n"
            "Si nunca me dijiste a qué precio entraste, dímelo en el mismo "
            "mensaje (primero la salida, después la entrada):\n"
            "/cerrar TXN 295.4 280.5"
        ),
        "/paso": (
            "🚫 */paso* — marca un candidato que decidiste no tomar.\n\n"
            "Escríbelo así:\n"
            "/paso TXN"
        ),
    }
    return usage.get(command, "Comando desconocido. Usa /ayuda.")


def format_manual_error(
    reason: str,
    symbol: str,
    open_symbols: Optional[Iterable[str]] = None,
    record: Optional[dict] = None,
) -> str:
    """Por qué no se pudo anotar la corrección, y qué hacer en su lugar.

    Un "no se pudo" sin motivo deja al usuario sin saber si el bot le entendió
    mal o si el estado no es el que él cree. Cada error dice las dos cosas.
    """
    symbol = symbol.strip().upper()

    if reason == "not_open":
        abiertas = [s for s in (open_symbols or [])]
        cola = (
            "Abiertas ahora: " + ", ".join(sorted(abiertas))
            if abiertas
            else "Ahora mismo no hay ninguna operación abierta."
        )
        return (
            f"🤔 No tengo *{symbol}* entre las operaciones abiertas.\n\n"
            f"{cola}\n\n"
            "_Si ya se cerró, está en el histórico y no se puede volver a "
            "tocar desde aquí._"
        )

    if reason == "bad_price":
        return (
            f"🤔 No entendí el precio de *{symbol}*.\n\n"
            "Tiene que ser un número positivo. Valen tanto 280.5 como 280,5."
        )

    if reason == "below_stop" and record is not None:
        instrument = get_instrument(record["symbol"])
        return (
            f"🤔 Ese precio está por debajo del stop de *{symbol}* "
            f"({_fmt(record['stop'], instrument)}).\n\n"
            "Una entrada ahí habría nacido cerrada, así que casi seguro es un "
            "dedazo. No lo anoto: revisa el número y vuelve a mandarlo."
        )

    if reason == "unknown_entry":
        return (
            f"🤔 No sé a qué precio entraste en *{symbol}*, así que no puedo "
            "calcular el resultado.\n\n"
            "Dímelo en el mismo mensaje (primero la salida, después la "
            f"entrada):\n/cerrar {symbol} SALIDA ENTRADA"
        )

    return f"No pude anotar el cambio en *{symbol}*. Usa /ayuda."


def format_recent(history: list[dict], limit: int = 5) -> str:
    """Las últimas señales que salieron del seguimiento, con su desenlace."""
    recent = history[-limit:]
    if not recent:
        return "Todavía no hay señales cerradas."

    lines = ["🗒️ *ÚLTIMAS SEÑALES RESUELTAS*", ""]
    for record in reversed(recent):
        mark, verdict = outcome_label(record)
        ret = record.get("return_pct")
        tail = f" ({ret * 100:+.1f}%)" if ret is not None else ""
        lines.append(f"{mark} *{record['symbol']}* — {verdict}{tail}")
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
    # Sin esta línea el rendimiento parece describir lo que hizo el usuario,
    # cuando solo describe lo que el bot propuso y él aceptó.
    if stats.get("not_taken"):
        lines.append(f"Candidatos que dejaste pasar: {stats['not_taken']}")
    lines += [
        "",
        "_Estos son resultados en vivo sobre los candidatos que se enviaron._",
        "_Es la única medición que no viene de un backtest, y por eso la única"
        " que puede llegar a demostrar algo sobre estos filtros._",
        "",
        _decision_line(stats),
    ]
    return "\n".join(lines)


def _decision_line(stats: dict) -> str:
    """Progreso hacia la regla de decisión fijada de antemano (24/08/2026).

    Fijar el umbral y el criterio ANTES de ver el resultado es lo que impide
    que dentro de unos meses se reinterprete el número que haya salido. Con
    menos operaciones que el objetivo no hay veredicto, solo cuenta cuántas
    faltan; con el objetivo cumplido, el límite inferior de la R media decide.
    """
    objetivo = stats.get("decision_target")
    if objetivo is None:
        return ""

    if not stats.get("decision_ready"):
        return (
            f"_Regla de decisión: con {objetivo} operaciones cerradas se "
            f"decide si esto sigue o se apaga, según el límite inferior de la "
            f"R media. Van {stats['closed']}/{objetivo}._"
        )

    r_lower = stats["r_lower"]
    if stats["decision_sigue"]:
        return (
            f"_✅ Con {stats['closed']} operaciones, el límite inferior de la "
            f"R media es positivo ({r_lower:+.3f}R): la regla de decisión dice "
            f"que sigue._"
        )
    return (
        f"_🛑 Con {stats['closed']} operaciones, el límite inferior de la R "
        f"media NO es positivo ({r_lower:+.3f}R): la regla de decisión fijada "
        f"de antemano dice que se apague y el dinero vaya al índice._"
    )


def format_help() -> str:
    return (
        "🤖 *Cribador de Quantfury*\n\n"
        "Revisa cada día, antes de la apertura de EE.UU., las acciones y pares "
        "de forex operables en Quantfury, y te enseña los que están en "
        "tendencia y muy estirados en sentido contrario (EMA200 + RSI de 2 "
        "sesiones) — con su zona de entrada, su stop estructural y su ratio "
        "riesgo/beneficio ya calculados.\n\n"
        "*No emite señales con ventaja demostrada.* Con cinco años de datos, "
        "el límite inferior de la ventaja medida todavía no supera cero. Lo "
        "que hace es ahorrarte mirar 141 gráficos cada mañana; la decisión "
        "es tuya. Detalle en MEDICION_ESTRATEGIA.md del repositorio.\n\n"
        "Comandos:\n"
        "• /senales — Últimos candidatos resueltos\n"
        "• /abiertas — Operaciones en curso\n"
        "• /rendimiento — Resultado real de lo que se cribó\n"
        "• /instrumentos — Universo vigilado\n"
        "• /ayuda — Este mensaje\n\n"
        "Para contarme lo que hiciste de verdad:\n"
        "• /tomada SÍMBOLO PRECIO — Entraste, y a qué precio\n"
        "• /cerrar SÍMBOLO PRECIO — Saliste antes de tocar objetivo o stop\n"
        "• /paso SÍMBOLO — No la tomaste\n\n"
        "_Yo no veo tu cuenta: sin estos tres comandos doy por hecho que "
        "tomaste todo lo que te mandé y que saliste donde decía el plan. El "
        "seguimiento en vivo es la única medición que no viene de un "
        "backtest; con datos inventados no mide nada._\n\n"
        "_Esto no es asesoría financiera. Las órdenes las colocas tú en "
        "Quantfury._"
    )
