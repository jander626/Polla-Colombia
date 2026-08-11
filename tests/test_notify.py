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
    )


def calibration_for(asset_class: str, wins: int, losses: int, win_r: float = 2.5):
    data = [(asset_class, 72.0, True, win_r)] * wins + [
        (asset_class, 72.0, False, -1.0)
    ] * losses
    return Calibration.from_outcomes(data)


# Perfil realista: acierto bajo pero esperanza positiva por el pago asimétrico.
CALIBRATED = calibration_for("stock", 193, 356)
FOREX_CALIBRATED = calibration_for("forex", 31, 35)


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

def test_signal_message_contains_the_three_prices():
    message = format_signal(make_signal(), CALIBRATED)
    assert "231.40" in message   # entrada
    assert "248.90" in message   # objetivo
    assert "224.10" in message   # stop
    assert "1:2.40" in message   # riesgo/beneficio
    assert "AAPL" in message and "COMPRA" in message


def test_an_uncalibrated_signal_says_so():
    """Sin explicación, un porcentaje bajo se leería como 'muy mala señal'."""
    message = format_signal(make_signal(), Calibration.empty())
    assert "sin calibrar" in message.lower()
    assert "backtest" in message.lower()


def test_the_message_shows_win_rate_and_expectancy_together():
    """Publicar solo el acierto engaña cuando los pagos son asimétricos.

    Un 35% de acierto con esperanza positiva es un buen sistema, pero leído a
    secas parece un mal sistema. Las dos cifras tienen que ir juntas.
    """
    message = format_signal(make_signal(), CALIBRATED)
    assert "Acierto histórico" in message
    assert "Beneficio esperado" in message
    assert "R por operación" in message
    assert "llegan al objetivo" in message


def test_a_segment_without_a_proven_edge_is_flagged():
    """Esperanza no demostrada: hay que decirlo, no esconderlo tras un %."""
    thin = calibration_for("stock", 6, 11)
    message = format_signal(make_signal(), thin)
    assert "Sin ventaja demostrada" in message


def test_a_low_sample_segment_is_flagged():
    message = format_signal(make_signal("EUR/USD"), calibration_for("forex", 8, 9))
    assert "poco fiable" in message.lower()


def test_a_reliable_segment_is_not_flagged_as_unreliable():
    message = format_signal(make_signal(), CALIBRATED)
    assert "poco fiable" not in message.lower()


def test_the_signal_uses_its_own_asset_class(monkeypatch):
    """Una señal de forex no puede calibrarse con el histórico de acciones."""
    both = Calibration.from_outcomes(
        [("stock", 72.0, True, 2.5)] * 193 + [("stock", 72.0, False, -1.0)] * 356
        + [("forex", 72.0, True, 2.5)] * 31 + [("forex", 72.0, False, -1.0)] * 35
    )
    stock_msg = format_signal(make_signal("AAPL"), both)
    forex_msg = format_signal(make_signal("EUR/USD"), both)
    assert stock_msg != forex_msg


def test_an_unverified_signal_says_so():
    """'Sin verificar' y 'sin riesgo' son cosas muy distintas."""
    message = format_signal(make_signal(), CALIBRATED, risk=None)
    assert "Sin verificación de noticias" in message


def test_event_risk_is_shown_and_lowers_the_win_rate():
    risk = EventRisk("low", 10.0, "Tendencia intacta.", "Dato de empleo el viernes.")
    with_risk = format_signal(make_signal(), CALIBRATED, risk=risk)
    without = format_signal(make_signal(), CALIBRATED)

    assert "Tendencia intacta" in with_risk
    assert "empleo" in with_risk
    assert "10 puntos" in with_risk
    assert with_risk != without


def test_forex_prices_use_more_decimals():
    """1.0842 con dos decimales sería inservible para operar."""
    signal = make_signal("EUR/USD")
    object.__setattr__(signal, "levels", Levels(1.08420, 1.07850, 1.09560, 2.0, 0.0057, 0.0114))
    message = format_signal(signal, FOREX_CALIBRATED)
    assert "1.08420" in message
    assert "PAR" in message


def test_no_signals_message_explains_why_thats_fine():
    message = format_no_signals()
    assert "inventando" in message or "inventa" in message


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


def test_performance_compares_promise_against_reality():
    """Es el control externo de la calibración: debe verse en el mensaje."""
    message = format_performance(
        {
            "closed": 20, "wins": 11, "win_rate": 0.55, "avg_r": 0.42,
            "total_r": 8.4, "open": 3, "avg_confidence": 58.0,
        }
    )
    assert "55.0%" in message or "55%" in message
    assert "58%" in message
    assert "calibración está mal" in message


def test_help_warns_it_is_not_financial_advice():
    assert "asesoría financiera" in format_help()


# ── Fallos encontrados en la primera alerta real ──────────────────────────────

def test_a_tiny_edge_is_not_announced_as_a_winner():
    """El fallo de la primera alerta real: '+0.00R' y 'gana en el agregado'.

    Una esperanza de +0.005R es positiva y económicamente nada. Con dos
    decimales se imprimía como +0.00R mientras el texto afirmaba que ganaba.
    Ahora debe reconocerse como insuficiente.
    """
    # Reproduce el caso real: media +0.160R, mínimo +0.028R. Positivo, pero
    # por debajo de lo que hace falta para que la operación valga la pena.
    tiny = calibration_for("stock", 193, 356, win_r=2.3)
    stats = tiny.for_asset_class("stock")
    assert stats is not None
    assert 0 < stats.expectancy_lower < 0.05     # positiva pero minúscula
    assert not stats.has_edge

    message = format_signal(make_signal(), tiny)
    assert "demasiado pequeña" in message
    assert "Gana en el agregado" not in message


def test_expectancy_is_shown_with_enough_precision():
    """Con dos decimales, una ventaja pequeña se imprimía como '+0.00R'."""
    message = format_signal(make_signal(), calibration_for("stock", 193, 356, win_r=2.3))
    assert "+0.00R" not in message      # el formato viejo, ambiguo
    assert "R por operación" in message


def test_a_real_edge_is_still_announced_as_such():
    """El umbral no puede silenciar una ventaja que sí es útil."""
    message = format_signal(make_signal(), CALIBRATED)
    assert "Gana en el agregado" in message
    assert "demasiado pequeña" not in message


def test_a_stale_calibration_is_announced():
    """Calibrar, cambiar la estrategia y seguir publicando las cifras viejas.

    Pasó de verdad: se calibró a las 04:07, la estrategia cambió a las 04:24, y
    el bot siguió mostrando la confianza de la versión anterior sin avisar.
    """
    fresh = format_signal(make_signal(), CALIBRATED, stale_calibration=False)
    stale = format_signal(make_signal(), CALIBRATED, stale_calibration=True)

    assert "versión anterior" in stale
    assert "versión anterior" not in fresh


# ── El escaneo mudo del 11 de agosto ─────────────────────────────────────────

def test_the_header_reports_what_will_actually_be_sent():
    """Anunciaba lo encontrado, no lo que iba a enviar.

    Ese día detectó 1 oportunidad, la descartó por tener ya la posición
    abierta, y la cabecera igual dijo "1". Después, silencio.
    """
    header = format_scan_header(3, 1, already_open=["ABBV"], event_risk=["NVDA"])
    assert "detectadas: *3*" in header
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

    assert "ninguna se envía" in message
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
    assert "ninguna se envía" in message
    assert "Sin candidatos" in message
