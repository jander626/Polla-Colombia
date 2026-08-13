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


def test_relative_strength_is_measured_but_no_longer_scores(uptrend_with_pullback):
    """La fuerza relativa se sigue midiendo, pero ya no mueve la puntuación.

    El backtest mostró que puntuaba AL REVÉS: el cuartil que más había batido
    al mercado rendía +0.002R y el que menos, +0.155R. Se le quitó el peso pero
    se conserva el cálculo para poder vigilarla en el diagnóstico. Este test
    fija ambas cosas: que se mide, y que no puntúa.
    """
    df = uptrend_with_pullback
    falling_market = pd.Series(np.linspace(200.0, 100.0, len(df)), index=df.index)
    rising_market = pd.Series(np.linspace(100.0, 300.0, len(df)), index=df.index)

    outperforming = strategy.compute_features(
        df, DEFAULT_PARAMS, benchmark_close=falling_market
    ).iloc[-1]
    lagging = strategy.compute_features(
        df, DEFAULT_PARAMS, benchmark_close=rising_market
    ).iloc[-1]

    # Se sigue midiendo, y distingue los dos escenarios.
    assert outperforming["c_relative_strength"] > lagging["c_relative_strength"]
    # Pero no tiene peso: la puntuación es idéntica.
    assert outperforming["score"] == pytest.approx(lagging["score"])
    assert "relative_strength" not in DEFAULT_PARAMS.weight_map


def test_signal_components_are_reported(uptrend_with_pullback):
    df = uptrend_with_pullback
    benchmark = pd.Series(np.linspace(100.0, 160.0, len(df)), index=df.index)
    signal = strategy.latest_signal(STOCK, df, DEFAULT_PARAMS, benchmark)

    assert signal is not None
    assert set(signal.components) <= set(strategy.ALL_COMPONENTS)
    assert all(0.0 <= v <= 1.0 for v in signal.components.values())
    # Los componentes SIN peso también se reportan: el diagnóstico por
    # cuartiles los necesita para poder vigilarlos.
    assert "relative_strength" in signal.components
    assert "relative_strength" not in DEFAULT_PARAMS.weight_map


def test_components_without_a_benchmark_omit_relative_strength(uptrend_with_pullback):
    """Sin benchmark no hay fuerza relativa que medir; no se inventa un cero."""
    signal = strategy.latest_signal(STOCK, uptrend_with_pullback, DEFAULT_PARAMS)
    assert signal is not None
    assert "relative_strength" not in signal.components
    assert "momentum_12_1" in signal.components  # este no necesita benchmark


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


# ── Filtro de régimen de mercado ──────────────────────────────────────────────

def test_market_regime_filter_vetoes_signals_in_a_bear_market(uptrend_with_pullback):
    """Con el índice bajo su media larga no se compra, por buena que sea la señal.

    Es la defensa contra los tramos donde la reversión deja de funcionar, que
    son los que producen la máxima caída del sistema.
    """
    df = uptrend_with_pullback
    bull = pd.Series(100.0 * np.cumprod(np.full(len(df), 1.0015)), index=df.index)
    bear = pd.Series(300.0 * np.cumprod(np.full(len(df), 0.9985)), index=df.index)

    params = replace(DEFAULT_PARAMS, use_market_regime_filter=True)
    assert strategy.latest_signal(STOCK, df, params, bull) is not None
    assert strategy.latest_signal(STOCK, df, params, bear) is None


def test_the_regime_filter_can_be_switched_off(uptrend_with_pullback):
    df = uptrend_with_pullback
    bear = pd.Series(300.0 * np.cumprod(np.full(len(df), 0.9985)), index=df.index)

    off = replace(DEFAULT_PARAMS, use_market_regime_filter=False)
    assert strategy.latest_signal(STOCK, df, off, bear) is not None


def test_the_regime_filter_does_not_exclude_forex(uptrend_with_pullback):
    """El S&P no es el régimen del euro-dólar; exigirlo excluiría todo el forex."""
    params = replace(DEFAULT_PARAMS, use_market_regime_filter=True)
    feats = strategy.compute_features(
        uptrend_with_pullback, params, benchmark_close=None, has_volume=False
    )
    assert bool(feats["f_market"].iloc[-1]) is True


# ── Variantes ─────────────────────────────────────────────────────────────────

def test_the_variants_differ_from_each_other():
    """Comparar variantes idénticas no probaría nada."""
    from trading.config import variants

    defined = variants()
    assert len(defined) == 3
    signatures = {
        (p.use_market_regime_filter, tuple(sorted(p.weight_map)))
        for p in defined.values()
    }
    assert len(signatures) == 3


def test_no_variant_gives_weight_to_the_inverted_component():
    """Ninguna variante puede reintroducir el componente que puntuaba al revés."""
    from trading.config import variants

    for name, params in variants().items():
        assert "relative_strength" not in params.weight_map, name


def test_variant_weights_are_normalised():
    from trading.config import variants

    for name, params in variants().items():
        assert sum(params.weight_map.values()) == pytest.approx(1.0), name


# ── Embudo del escaneo ────────────────────────────────────────────────────────
# Cuatro días seguidos de "hoy no hay nada" no dicen si el mercado está
# tranquilo, si un filtro es demasiado exigente o si el bot está roto. El
# embudo convierte ese silencio en un dato.

def test_the_funnel_finds_the_same_signals_as_the_plain_scan(uptrend_with_pullback):
    """El embudo no puede cambiar QUÉ se encuentra, solo explicar el resto."""
    bars = {"AAPL": uptrend_with_pullback}
    plain = strategy.scan([STOCK], bars, DEFAULT_PARAMS)
    with_funnel, _ = strategy.scan_with_funnel([STOCK], bars, DEFAULT_PARAMS)

    assert [s.symbol for s in plain] == [s.symbol for s in with_funnel]
    assert [s.score for s in plain] == [s.score for s in with_funnel]


def test_the_funnel_counts_a_signal_through_every_stage(uptrend_with_pullback):
    signals, funnel = strategy.scan_with_funnel(
        [STOCK], {"AAPL": uptrend_with_pullback}, DEFAULT_PARAMS
    )
    assert signals, "el fixture debe producir señal; si no, el test no mide nada"
    assert funnel.evaluated == 1
    assert funnel.no_data == 0
    # Sobrevive a todas las etapas, incluida la última.
    assert all(survivors == 1 for _, _, survivors, _ in funnel.stages())
    assert funnel.survivors == 1


def test_the_funnel_names_the_filter_that_rejected(downtrend):
    """Es la pregunta real del usuario: ¿cuál de los filtros manda hoy?"""
    signals, funnel = strategy.scan_with_funnel(
        [STOCK], {"AAPL": downtrend}, DEFAULT_PARAMS
    )
    assert signals == []
    assert funnel.evaluated == 1

    bottleneck = funnel.bottleneck
    assert bottleneck is not None
    key, _, dropped = bottleneck
    # Una tendencia bajista muere en el régimen, no en un filtro de detalle.
    assert key in {"f_market", "f_regime"}
    assert dropped == 1


def test_the_funnel_is_a_cascade_never_growing(sideways, downtrend, uptrend_with_pullback):
    """Cada etapa se aplica sobre lo que sobrevivió a la anterior."""
    bars = {"AAPL": uptrend_with_pullback, "MSFT": sideways, "NVDA": downtrend}
    instruments = [universe.get(s) for s in ("AAPL", "MSFT", "NVDA")]
    _, funnel = strategy.scan_with_funnel(instruments, bars, DEFAULT_PARAMS)

    assert funnel.evaluated == 3
    counts = [survivors for _, _, survivors, _ in funnel.stages()]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] <= funnel.evaluated


def test_an_instrument_without_enough_history_is_not_counted_as_rejected():
    """Sin datos no es lo mismo que rechazado, y mezclarlos falsearía el embudo."""
    short = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1e6]},
        index=pd.bdate_range("2026-01-01", periods=1),
    )
    _, funnel = strategy.scan_with_funnel([STOCK], {"AAPL": short}, DEFAULT_PARAMS)
    assert funnel.evaluated == 0
    assert funnel.no_data == 1


def test_a_nan_indicator_does_not_pass_a_stage():
    """`bool(np.nan)` es True en Python: sin cuidado, un NaN aprobaría el filtro."""
    row = pd.Series({"f_regime": np.nan, "f_market": True, "f_trend_strength": False})
    assert strategy._stage_ok(row, "f_regime") is False
    assert strategy._stage_ok(row, "f_market") is True
    assert strategy._stage_ok(row, "f_trend_strength") is False
    assert strategy._stage_ok(row, "no_existe") is False
