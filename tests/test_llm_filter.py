"""Tests del filtro de noticias.

La regla que estos tests defienden es una sola, y es la que sostiene la
honestidad del porcentaje de confianza: **el modelo solo puede restar**. Si
alguna vez consigue sumar, el número publicado deja de venir del backtest y
vuelve a ser una invención con aspecto de dato.
"""

from __future__ import annotations

import pytest

from trading.config import MAX_LLM_PENALTY
from trading.llm_filter import EventRisk, build_prompt, parse_response
from trading.strategy import Signal
from trading.risk import Levels
from trading import universe


def make_signal(symbol: str = "AAPL") -> Signal:
    instrument = universe.get(symbol)
    levels = Levels(100.0, 95.0, 115.0, 3.0, 5.0, 15.0)
    return Signal(
        symbol=symbol,
        name=instrument.name,
        asset_class=instrument.asset_class,
        bar_date=__import__("pandas").Timestamp("2026-08-07"),
        close=99.0,
        atr=5.0,
        atr_pct=0.05,
        score=72.0,
        levels=levels,
    )


# ── Parseo ────────────────────────────────────────────────────────────────────

def test_parses_a_clean_json_response():
    risk = parse_response(
        '{"event_risk":"low","confidence_penalty":8,'
        '"rationale_es":"Tendencia intacta.","risks_es":"Dato de empleo el viernes."}'
    )
    assert risk is not None
    assert risk.level == "low"
    assert risk.penalty == 8.0
    assert "Tendencia" in risk.rationale
    assert "empleo" in risk.risks


def test_parses_json_wrapped_in_a_code_fence():
    """Los modelos envuelven el JSON en ``` aunque se les pida lo contrario."""
    risk = parse_response(
        '```json\n{"event_risk":"none","confidence_penalty":0,'
        '"rationale_es":"Sin eventos.","risks_es":""}\n```'
    )
    assert risk is not None and risk.level == "none"


def test_parses_json_surrounded_by_prose():
    risk = parse_response(
        'Claro, aquí tienes el análisis:\n'
        '{"event_risk":"high","confidence_penalty":20,'
        '"rationale_es":"x","risks_es":"Resultados el martes."}\n'
        'Espero que sirva.'
    )
    assert risk is not None and risk.level == "high"


@pytest.mark.parametrize("text", ["", "lo siento, no puedo ayudar", "{roto", None])
def test_unparseable_responses_return_none(text):
    """Sin respuesta interpretable, la señal sale marcada como no verificada."""
    assert parse_response(text) is None


# ── La regla: solo restar ─────────────────────────────────────────────────────

def test_a_negative_penalty_is_clamped_to_zero():
    """Un modelo que intente subir la confianza no lo consigue."""
    risk = parse_response(
        '{"event_risk":"none","confidence_penalty":-30,"rationale_es":"","risks_es":""}'
    )
    assert risk is not None and risk.penalty == 0.0


def test_the_penalty_is_capped():
    risk = parse_response(
        '{"event_risk":"high","confidence_penalty":999,"rationale_es":"","risks_es":""}'
    )
    assert risk is not None and risk.penalty == MAX_LLM_PENALTY


def test_high_risk_always_carries_a_real_penalty():
    """Declarar riesgo alto y penalizar 0 sería incoherente."""
    risk = parse_response(
        '{"event_risk":"high","confidence_penalty":0,"rationale_es":"","risks_es":""}'
    )
    assert risk is not None and risk.penalty >= 15.0


def test_an_unknown_level_is_treated_as_some_risk():
    """Ante una respuesta rara se asume algo de riesgo, nunca ninguno."""
    risk = parse_response(
        '{"event_risk":"catastrófico","confidence_penalty":5,'
        '"rationale_es":"","risks_es":""}'
    )
    assert risk is not None and risk.level == "low"


def test_a_non_numeric_penalty_does_not_crash():
    risk = parse_response(
        '{"event_risk":"low","confidence_penalty":"mucho",'
        '"rationale_es":"","risks_es":""}'
    )
    assert risk is not None and risk.penalty == 0.0


# ── Bloqueo ───────────────────────────────────────────────────────────────────

def test_high_risk_blocks_the_signal():
    """Comprar la víspera de resultados es apostar a otra cosa distinta."""
    assert EventRisk("high", 20.0, "", "").is_blocking


@pytest.mark.parametrize("level", ["none", "low"])
def test_other_levels_do_not_block(level):
    assert not EventRisk(level, 5.0, "", "").is_blocking


# ── Prompts ───────────────────────────────────────────────────────────────────

def test_stock_prompt_asks_about_earnings():
    prompt = build_prompt(make_signal("AAPL"))
    assert "resultados trimestrales" in prompt
    assert "Apple" in prompt and "AAPL" in prompt


def test_forex_prompt_asks_about_central_banks():
    prompt = build_prompt(make_signal("EUR/USD"))
    assert "bancos centrales" in prompt
    assert "EUR" in prompt and "USD" in prompt
    assert "resultados trimestrales" not in prompt


def test_prompt_forbids_price_prediction():
    """El modelo aporta información, no pronóstico: eso ya lo hace el sistema."""
    prompt = build_prompt(make_signal())
    assert "NO vas a predecir" in prompt
