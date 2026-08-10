"""Tests de la estrategia.

El test que más importa aquí es `test_live_and_backtest_agree`: si el escaneo
en vivo y el backtest evaluasen las velas de forma distinta, el backtest
dejaría de decir nada sobre el comportamiento real y la calibración de
confianza sería ficción.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading import strategy, universe
from trading.config import DEFAULT_PARAMS, replace

STOCK = universe.get("AAPL")
FOREX = universe.get("EUR/USD")


# ── Detección ─────────────────────────────────────────────────────────────────

def test_detects_a_pullback_in_an_uptrend(uptrend_with_pullback):
    signal = strategy.latest_signal(STOCK, uptrend_with_pullback, DEFAULT_PARAMS)

    assert signal is not None, "la estrategia no reconoció su propio escenario"
    assert signal.symbol == "AAPL"
    assert signal.score >= DEFAULT_PARAMS.min_score
    assert signal.levels.is_valid
    assert signal.levels.risk_reward >= DEFAULT_PARAMS.min_risk_reward


def test_ignores_a_sideways_market(sideways):
    assert strategy.latest_signal(STOCK, sideways, DEFAULT_PARAMS) is None


def test_ignores_a_downtrend(downtrend):
    """Solo compras: una tendencia bajista nunca debe generar señal."""
    assert strategy.latest_signal(STOCK, downtrend, DEFAULT_PARAMS) is None


def test_requires_enough_history(uptrend_with_pullback):
    short = uptrend_with_pullback.tail(DEFAULT_PARAMS.min_bars - 1)
    assert strategy.latest_signal(STOCK, short, DEFAULT_PARAMS) is None


# ── Filtros ───────────────────────────────────────────────────────────────────

def test_every_hard_filter_can_veto_a_signal(uptrend_with_pullback):
    """Endurecer cualquiera de los filtros debe bastar para anular la señal."""
    assert strategy.latest_signal(STOCK, uptrend_with_pullback, DEFAULT_PARAMS)

    vetoes = {
        "tendencia": replace(DEFAULT_PARAMS, adx_min=99.0),
        "liquidez": replace(DEFAULT_PARAMS, min_dollar_volume=1e15),
        "volatilidad": replace(DEFAULT_PARAMS, max_atr_pct=0.0001),
        "reanudación": replace(DEFAULT_PARAMS, resume_rsi_min=99.0),
        "retroceso": replace(DEFAULT_PARAMS, pullback_rsi_max=1.0),
        "riesgo/beneficio": replace(DEFAULT_PARAMS, min_risk_reward=99.0),
        "puntuación": replace(DEFAULT_PARAMS, min_score=99.0),
    }
    for name, params in vetoes.items():
        assert strategy.latest_signal(STOCK, uptrend_with_pullback, params) is None, (
            f"el filtro de {name} no vetó la señal"
        )


def test_liquidity_filter_does_not_apply_to_forex(uptrend_with_pullback):
    """El forex no publica volumen centralizado; exigirlo lo excluiría entero."""
    params = replace(DEFAULT_PARAMS, min_dollar_volume=1e15)
    feats = strategy.compute_features(
        uptrend_with_pullback, params, benchmark_close=None, has_volume=False
    )
    assert bool(feats["f_liquidity"].iloc[-1]) is True


# ── Puntuación ────────────────────────────────────────────────────────────────

def test_score_stays_within_bounds(uptrend_with_pullback):
    feats = strategy.compute_features(uptrend_with_pullback, DEFAULT_PARAMS)
    score = feats["score"].dropna()
    assert len(score) > 200
    assert score.between(0.0, 100.0).all()


def test_forex_score_is_renormalized_not_penalized(uptrend_with_pullback):
    """Sin volumen ni benchmark el peso se reparte; el forex no sale castigado."""
    feats = strategy.compute_features(
        uptrend_with_pullback, DEFAULT_PARAMS, benchmark_close=None, has_volume=False
    )
    last = feats.iloc[-1]

    assert np.isnan(last["c_volume"])
    assert np.isnan(last["c_relative_strength"])
    assert np.isfinite(last["score"])
    assert 0.0 <= last["score"] <= 100.0


def test_beating_the_market_scores_higher_than_lagging_it(uptrend_with_pullback):
    """Mismo instrumento, distinto mercado: batirlo debe puntuar más que quedarse atrás."""
    df = uptrend_with_pullback
    falling_market = pd.Series(np.linspace(200.0, 100.0, len(df)), index=df.index)
    rising_market = pd.Series(np.linspace(100.0, 300.0, len(df)), index=df.index)

    outperforming = strategy.compute_features(
        df, DEFAULT_PARAMS, benchmark_close=falling_market
    ).iloc[-1]
    lagging = strategy.compute_features(
        df, DEFAULT_PARAMS, benchmark_close=rising_market
    ).iloc[-1]

    assert outperforming["c_relative_strength"] > lagging["c_relative_strength"]
    assert outperforming["score"] > lagging["score"]


def test_signal_components_are_reported(uptrend_with_pullback):
    signal = strategy.latest_signal(STOCK, uptrend_with_pullback, DEFAULT_PARAMS)
    assert signal is not None
    assert set(signal.components) <= set(strategy.WEIGHTS)
    assert all(0.0 <= v <= 1.0 for v in signal.components.values())


# ── Coherencia entre vivo y backtest ──────────────────────────────────────────

def test_live_and_backtest_agree(uptrend_with_pullback):
    """La última señal histórica debe ser idéntica a la del escaneo en vivo."""
    df = uptrend_with_pullback
    feats = strategy.compute_features(df, DEFAULT_PARAMS)
    historical = strategy.signals_from_features(STOCK, feats, DEFAULT_PARAMS)
    live = strategy.latest_signal(STOCK, df, DEFAULT_PARAMS)

    assert historical and live is not None
    last = historical[-1]
    assert last.bar_date == live.bar_date == df.index[-1]
    assert last.score == pytest.approx(live.score)
    assert last.levels.entry_max == pytest.approx(live.levels.entry_max)
    assert last.levels.stop == pytest.approx(live.levels.stop)
    assert last.levels.target == pytest.approx(live.levels.target)


def test_historical_signals_all_respect_the_risk_reward_floor(uptrend_with_pullback):
    feats = strategy.compute_features(uptrend_with_pullback, DEFAULT_PARAMS)
    signals = strategy.signals_from_features(STOCK, feats, DEFAULT_PARAMS)
    assert signals
    for signal in signals:
        assert signal.levels.risk_reward >= DEFAULT_PARAMS.min_risk_reward
        assert signal.levels.stop < signal.levels.entry_max < signal.levels.target


# ── Escaneo del universo ──────────────────────────────────────────────────────

def test_scan_ranks_by_score(uptrend_with_pullback, sideways):
    bars = {"AAPL": uptrend_with_pullback, "MSFT": uptrend_with_pullback, "KO": sideways}
    instruments = [universe.get(s) for s in ("AAPL", "MSFT", "KO")]

    found = strategy.scan(instruments, bars, DEFAULT_PARAMS)

    assert {s.symbol for s in found} == {"AAPL", "MSFT"}
    assert found == sorted(found, key=lambda s: s.score, reverse=True)


def test_scan_survives_a_broken_symbol(uptrend_with_pullback):
    broken = uptrend_with_pullback.copy()
    broken["close"] = np.nan

    found = strategy.scan(
        [universe.get("AAPL"), universe.get("MSFT")],
        {"AAPL": broken, "MSFT": uptrend_with_pullback},
        DEFAULT_PARAMS,
    )
    assert [s.symbol for s in found] == ["MSFT"]


def test_scan_skips_missing_data(uptrend_with_pullback):
    found = strategy.scan(
        [universe.get("AAPL"), universe.get("MSFT")],
        {"AAPL": uptrend_with_pullback},
        DEFAULT_PARAMS,
    )
    assert [s.symbol for s in found] == ["AAPL"]
