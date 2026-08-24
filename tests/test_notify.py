"""Tests del formato de las alertas.

Dos cosas que importan más de lo que parecen:

1. Telegram rechaza el mensaje ENTERO si el Markdown está mal, así que el
   formato tiene que salir limpio y los mensajes largos trocearse por saltos
   de línea.
2. Una señal sin calibrar debe decirlo. Enviar "confianza 0%" a secas
   invitaría a leerlo como "muy mala señal" en vez de "todavía no lo sé".
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading import universe
from trading.llm_filter import EventRisk
from trading.notify import (
    TELEGRAM_MAX_LEN,
    _split_message,
    format_all_filtered,
    format_help,
    format_no_signals,
    format_outcome,
    format_performance,
    format_scan_header,
    format_signal,
    sanitize,
)
from trading.risk import Calibration, Levels
from trading.strategy import Signal


def make_signal(symbol: str = "AAPL") -> Signal:
    instrument = universe.get(symbol)
    return Signal(
        symbol=symbol,
        name=instrument.name,
        asset_class=instrument.asset_class,
        bar_date=pd.Timestamp("2026-08-07"),
        close=99.0,
        atr=5.0,
        atr_pct=0.05,
        score=72.0,
        levels=Levels(231.40, 224.10, 248.90, 2.4, 7.3, 17.5),
        adx=23.7,
        rsi=52.0,
        pullback_rsi=41.0,
    )


def calibration_for(asset_class: str, wins: int, losses: int, win_r: float = 2.5):
    data = [(asset_class, 72.0, True, win_r)] * wins + [
        (asset_class, 72.0, False, -1.0)
    ] * losses
    return Calibration.from_outcomes(data)


def dated_outcomes(asset_class: str, wins: int, losses: int, win_r: float, start: str):
    """Operaciones fechadas con las ganadoras REPARTIDAS por todo el periodo.

    El reparto uniforme importa: agrupar las ganadoras al principio haría que
    el tramo reciente fuese el peor por construcción, y los tests pasarían
    midiendo el fixture en vez del código.
    """
    total = wins + losses
    rows = []
    for i in range(total):
        won = (i + 1) * wins // total > i * wins // total
        rows.append((won, win_r if won else -1.0))
    dates = pd.bdate_range(start, periods=total, freq="W")
    return [(asset_class, 72.0, won, r, d) for (won, r), d in zip(rows, dates)]


# Perfil realista: acierto bajo pero esperanza positiva por el pago asimétrico.
CALIBRATED = calibration_for("stock", 193, 356)
FOREX_CALIBRATED = calibration_for("forex", 31, 35)

# Con tramo reciente: el caso real (ventaja en el histórico largo, no en el
# reciente) y su contrario.
DECAYED = Calibration.from_outcomes(
    dated_outcomes("stock", 130, 170, 2.6, "2021-01-01")
    + dated_outcomes("stock", 55, 145, 1.4, "2024-07-01")
)
LIVE_EDGE = Calibration.from_outcomes(
    dated_outcomes("stock", 130, 170, 2.6, "2021-01-01")
    + dated_outcomes("stock", 90, 110, 2.6, "2024-07-01")
)


# ── Troceado ──────────────────────────────────────────────────────────────────

def test_short_messages_are_not_split():
    assert _split_message("hola") == ["hola"]


def test_long_messages_are_split_within_the_limit():
    text = "\n".join(f"línea {i}" for i in range(2000))
    chunks = _split_message(text)
    assert len(chunks) > 1
    assert all(len(c) <= TELEGRAM_MAX_LEN for c in chunks)


def test_splitting_prefers_line_breaks():
    text = "\n".join(["x" * 100] * 100)
    for chunk in _split_message(text):
        assert not chunk.startswith("x" * 100 + "x")  # no parte una línea a medias


def test_a_single_huge_line_is_still_split():
    """Sin saltos de línea hay que cortar igualmente, o Telegram lo rechaza."""
    chunks = _split_message("y" * 9000)
    assert len(chunks) >= 3
    assert all(len(c) <= TELEGRAM_MAX_LEN for c in chunks)


def test_sanitize_removes_markdown_telegram_rejects():
    assert "**" not in sanitize("esto es **negrita**")
    assert "__" not in sanitize("esto es __cursiva__")
    assert "`" not in sanitize("con `código`")


# ── Alertas ───────────────────────────────────────────────────────────────────

def test_the_card_contains_the_three_levels():
    message = format_signal(make_signal(), CALIBRATED)
    assert "231.40" in message   # zona de entrada
    assert "248.90" in message   # objetivo
    assert "224.10" in message   # stop
    assert "1:2.40" in message   # riesgo/beneficio
    assert "AAPL" in message


def test_the_card_never_publishes_a_confidence_percentage():
    """El cambio del 16 de agosto: no hay probabilidad de acierto que prometer.

    Cuatro mediciones coincidieron en que no hay ventaja demostrada. Un
    porcentaje que se lee como "esto acierta el 32% de las veces" sobre una
    ventaja inexistente era la única parte del bot capaz de costar dinero.
    """
    message = format_signal(make_signal(), CALIBRATED)

    assert "Acierto histórico" not in message
    assert "Beneficio esperado" not in message
    assert "Gana en el agregado" not in message
    assert "llegan al objetivo" not in message
    assert "COMPRA" not in message              # tampoco lo llama señal de compra


def test_the_card_says_why_the_instrument_appeared():
    """Es el producto del cribador: sin el porqué, solo es un símbolo suelto."""
    message = format_signal(make_signal(), CALIBRATED)

    assert "ADX 24" in message
    assert "RSI bajó a 41" in message
    assert "reanudando" in message
    assert "tendencia alcista" in message


def test_the_card_calls_itself_a_screen_not_a_recommendation():
    message = format_signal(make_signal(), CALIBRATED)
    assert "no recomendación" in message
    assert "la decisión es tuya" in message.lower()


def test_a_decayed_edge_is_stated_as_a_fact():
    """Sin ventaja reciente, la ficha lo dice con el número delante."""
    message = format_signal(make_signal(), DECAYED)
    assert "NO muestran ventaja demostrada" in message
    assert DECAYED.recent_label in message
    assert "operaciones" in message


def test_a_live_edge_is_reported_without_becoming_a_promise():
    """Aunque el histórico reciente sí aguante, sigue siendo una criba."""
    message = format_signal(make_signal(), LIVE_EDGE)
    assert "sí muestran ventaja" in message
    # Ni siquiera con ventaja viva se convierte en recomendación.
    assert "Criba, no recomendación" in message


def test_an_uncalibrated_screen_says_so():
    message = format_signal(make_signal(), Calibration.empty())
    assert "Sin histórico aplicable" in message


def test_a_stale_calibration_does_not_get_quoted():
    """Cifras de otra versión de los filtros no describen este candidato."""
    message = format_signal(make_signal(), DECAYED, stale_calibration=True)
    assert "Sin histórico aplicable" in message
    assert DECAYED.recent_label not in message


def test_forex_prices_use_more_decimals():
    """1.0842 con dos decimales sería inservible para operar."""
    signal = make_signal("EUR/USD")
    object.__setattr__(signal, "levels", Levels(1.08420, 1.07850, 1.09560, 2.0, 0.0057, 0.0114))
    message = format_signal(signal, FOREX_CALIBRATED)
    assert "1.08420" in message
    assert "PAR" in message


def test_an_unverified_candidate_says_so():
    """'Sin verificar' y 'sin riesgo' son cosas muy distintas."""
    message = format_signal(make_signal(), CALIBRATED, risk=None)
    assert "Sin verificación de noticias" in message


def test_event_risk_context_is_shown():
    risk = EventRisk("low", 10.0, "Tendencia intacta.", "Dato de empleo el viernes.")
    message = format_signal(make_signal(), CALIBRATED, risk=risk)

    assert "Tendencia intacta" in message
    assert "empleo" in message


def test_no_candidates_message_explains_why_thats_fine():
    message = format_no_signals()
    assert "no está cribando" in message


# ── Cierres y rendimiento ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "outcome,marker", [("win", "✅"), ("loss", "❌"), ("timeout", "⏳")]
)
def test_outcome_message_matches_the_result(outcome, marker):
    record = {
        "symbol": "AAPL", "outcome": outcome,
        "entry_price": 231.0, "exit_price": 248.0,
        "return_pct": 0.0735, "r_multiple": 2.3,
    }
    message = format_outcome(record)
    assert message.startswith(marker)
    assert "+7.35%" in message and "+2.30R" in message


def test_outcome_message_without_metrics_does_not_crash():
    record = {"symbol": "AAPL", "outcome": "loss", "entry_price": 231.0, "exit_price": 224.0}
    assert "AAPL" in format_outcome(record)


def test_performance_without_data():
    assert "Todavía no hay" in format_performance({"closed": 0})


def test_performance_reports_live_results_without_promising_anything():
    """Es la única medición que no viene de un backtest, y hay que decirlo.

    Antes comparaba el acierto real contra la confianza prometida. Ya no hay
    confianza prometida: lo que queda es el resultado en vivo, con el progreso
    hacia la regla de decisión fijada de antemano (config.LIVE_DECISION_SAMPLE).
    """
    message = format_performance(
        {
            "closed": 20, "wins": 11, "win_rate": 0.55, "avg_r": 0.42,
            "total_r": 8.4, "open": 3,
            "decision_target": 200, "decision_ready": False,
        }
    )
    assert "55.0%" in message
    assert "+0.42" in message
    assert "no viene de un backtest" in message
    assert "20/200" in message


def test_performance_gives_a_verdict_once_the_sample_is_complete():
    """Con la muestra completa, el límite inferior de la R media decide."""
    base = {
        "closed": 200, "wins": 90, "win_rate": 0.45, "avg_r": 0.10,
        "total_r": 20.0, "open": 0, "decision_target": 200, "decision_ready": True,
    }

    sigue = format_performance({**base, "r_lower": 0.02, "decision_sigue": True})
    assert "sigue" in sigue and "+0.020R" in sigue

    apaga = format_performance({**base, "r_lower": -0.05, "decision_sigue": False})
    assert "se apague" in apaga and "-0.050R" in apaga


def test_help_warns_it_is_not_financial_advice():
    assert "asesoría financiera" in format_help()


# ── El escaneo mudo del 11 de agosto ─────────────────────────────────────────

def test_the_header_reports_what_will_actually_be_sent():
    """Anunciaba lo encontrado, no lo que iba a enviar.

    Ese día detectó 1 oportunidad, la descartó por tener ya la posición
    abierta, y la cabecera igual dijo "1". Después, silencio.
    """
    header = format_scan_header(3, 1, already_open=["ABBV"], event_risk=["NVDA"])
    assert "Cumplen los filtros: *3*" in header
    assert "Se envían: *1*" in header
    assert "ABBV" in header and "posición abierta" in header
    assert "NVDA" in header and "riesgo de evento" in header


def test_the_header_stays_simple_when_nothing_is_discarded():
    header = format_scan_header(2, 2)
    assert "Descartadas" not in header
    assert "Se envían" not in header


def test_a_fully_filtered_scan_explains_itself():
    """El caso exacto que dejó al usuario sin saber si el bot estaba roto."""
    message = format_all_filtered(1, already_open=["ABBV"], event_risk=[])

    assert "ninguno se envía" in message
    assert "ABBV" in message
    assert "posición abierta" in message
    # Y explica el porqué de la regla, no solo el hecho.
    assert "duplicaría tu exposición" in message


def test_a_fully_filtered_scan_by_event_risk():
    message = format_all_filtered(2, already_open=[], event_risk=["NVDA", "AAPL"])
    assert "NVDA" in message and "AAPL" in message
    assert "riesgo de evento" in message
    assert "duplicaría" not in message      # ese motivo no aplica aquí


def test_a_fully_filtered_scan_without_a_known_reason():
    """Aun sin motivo identificable, hay que decir que no se envía nada."""
    message = format_all_filtered(1, already_open=[], event_risk=[])
    assert "ninguno se envía" in message
    assert "Sin candidatos" in message


# ── Cuatro días de silencio sin explicación ──────────────────────────────────

def _funnel(evaluated: int, counts: dict[str, int]):
    from trading.strategy import FUNNEL_STAGES, ScanFunnel

    full = {key: counts.get(key, 0) for key, _ in FUNNEL_STAGES}
    return ScanFunnel(evaluated=evaluated, no_data=0, counts=full)


def test_a_quiet_day_without_a_funnel_still_works():
    """La firma vieja se sigue usando en los tests y en cualquier llamada suelta."""
    assert "filtros" in format_no_signals()


def test_a_quiet_day_shows_where_the_candidates_died():
    """El bot llevaba cuatro días diciendo 'no hay nada' sin decir por qué.

    Desde fuera, eso es indistinguible de un filtro demasiado exigente o de un
    bot averiado. El embudo firma el silencio.
    """
    message = format_no_signals(
        _funnel(141, {"f_market": 141, "f_regime": 96, "f_trend_strength": 34,
                      "f_volatility": 34, "f_liquidity": 34, "f_pullback": 3,
                      "f_resume": 0})
    )
    assert "141" in message
    assert "96" in message and "34" in message and "3" in message


def test_a_quiet_day_names_the_tightest_filter():
    """Saber cuál es el cuello de botella es lo que permite decidir si aflojarlo."""
    message = format_no_signals(
        _funnel(141, {"f_market": 141, "f_regime": 130, "f_trend_strength": 120,
                      "f_volatility": 120, "f_liquidity": 120, "f_pullback": 4,
                      "f_resume": 2, "niveles": 2, "riesgo_beneficio": 0})
    )
    assert "retroceso" in message           # el que más descartó: 120 → 4
    assert "más descartó" in message


def test_a_quiet_day_hides_stages_that_rejected_nobody():
    """Listar diez etapas cuando ocho no descartaron a nadie es ruido."""
    message = format_no_signals(
        _funnel(141, {"f_market": 141, "f_regime": 141, "f_trend_strength": 141,
                      "f_volatility": 141, "f_liquidity": 141, "f_pullback": 0,
                      "f_resume": 0})
    )
    assert "tendencia alcista" not in message   # no descartó a nadie
    assert "retroceso" in message               # este sí


def test_a_quiet_day_stops_listing_once_nobody_is_left():
    """Repetir '0 — …' cinco veces seguidas no informa de nada."""
    message = format_no_signals(
        _funnel(141, {"f_market": 141, "f_regime": 0})
    )
    assert message.count("• 0") == 1


# ── La ventaja que se evaporó ────────────────────────────────────────────────

def test_the_recent_segment_survives_a_save_and_load(tmp_path):
    """Si no persiste, la ficha volvería a decir que no hay histórico reciente."""
    path = str(tmp_path / "calibration.json")
    DECAYED.save(path)

    restored = Calibration.load(path)
    assert restored.recent_for_asset_class("stock") is not None
    assert restored.edge_decayed("stock")
    assert "NO muestran ventaja demostrada" in format_signal(make_signal(), restored)
