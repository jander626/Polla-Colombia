"""Tests de los indicadores.

Además de comprobar valores conocidos, aquí vive el test más importante del
paquete: `test_no_lookahead`. Si un indicador mirase al futuro, el backtest
mediría algo imposible de reproducir en vivo y toda la calibración de
confianza sería ficción.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from trading import indicators as ind


# ── Suavizado de Wilder ───────────────────────────────────────────────────────

def test_wilder_smooth_matches_hand_computation():
    # Semilla = media(1,2,3) = 2 en la posición 2; luego y = (y*2 + x)/3.
    series = pd.Series([1.0, 2, 3, 4, 5, 6])
    out = ind.wilder_smooth(series, 3)

    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx((2.0 * 2 + 4) / 3)
    assert out.iloc[4] == pytest.approx(((2.0 * 2 + 4) / 3 * 2 + 5) / 3)


def test_wilder_smooth_of_constant_is_constant():
    series = pd.Series([7.0] * 20)
    out = ind.wilder_smooth(series, 5)
    assert out.dropna().eq(7.0).all()


def test_wilder_smooth_without_enough_data_is_all_nan():
    out = ind.wilder_smooth(pd.Series([1.0, 2.0]), 14)
    assert out.isna().all()


# ── RSI ───────────────────────────────────────────────────────────────────────

def test_rsi_is_100_when_price_only_rises():
    closes = pd.Series(np.arange(1.0, 40.0))
    out = ind.rsi(closes, 14).dropna()
    assert out.eq(100.0).all()


def test_rsi_is_0_when_price_only_falls():
    closes = pd.Series(np.arange(40.0, 1.0, -1.0))
    out = ind.rsi(closes, 14).dropna()
    assert out.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_rsi_stays_within_bounds(uptrend_with_pullback):
    out = ind.rsi(uptrend_with_pullback["close"], 14).dropna()
    assert out.between(0.0, 100.0).all()
    assert len(out) > 250


# ── ATR ───────────────────────────────────────────────────────────────────────

def test_atr_of_constant_range_equals_that_range():
    n = 40
    index = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(np.full(n, 100.0), index=index)
    high = close + 1.0
    low = close - 1.0
    out = ind.atr(high, low, close, 14).dropna()
    # TR = max(2, |101-100|, |99-100|) = 2 en todas las velas.
    assert out.eq(2.0).all()


def test_atr_is_positive(uptrend_with_pullback):
    df = uptrend_with_pullback
    out = ind.atr(df["high"], df["low"], df["close"], 14).dropna()
    assert (out > 0).all()


# ── ADX ───────────────────────────────────────────────────────────────────────

def test_adx_is_bounded_and_high_in_a_strong_trend():
    closes = 100.0 * np.cumprod(np.full(200, 1.004))  # subida limpia y sostenida
    df = make_bars(closes, spread=0.003)
    out = ind.adx(df["high"], df["low"], df["close"], 14).dropna()

    assert out["adx"].between(0.0, 100.0).all()
    assert out["plus_di"].iloc[-1] > out["minus_di"].iloc[-1]
    assert out["adx"].iloc[-1] > 25.0


def test_adx_is_low_in_a_sideways_market(sideways):
    out = ind.adx(sideways["high"], sideways["low"], sideways["close"], 14).dropna()
    assert out["adx"].iloc[-1] < 30.0


# ── MACD y EMA ────────────────────────────────────────────────────────────────

def test_macd_hist_is_difference_between_macd_and_signal(uptrend_with_pullback):
    out = ind.macd(uptrend_with_pullback["close"]).dropna()
    assert np.allclose(out["hist"], out["macd"] - out["signal"])


def test_ema_of_constant_is_constant():
    out = ind.ema(pd.Series([50.0] * 30), 10).dropna()
    assert out.eq(50.0).all()


# ── Liquidez y estructura ─────────────────────────────────────────────────────

def test_dollar_volume_multiplies_price_by_volume():
    close = pd.Series([10.0] * 25)
    volume = pd.Series([1_000.0] * 25)
    out = ind.dollar_volume(close, volume, 20).dropna()
    assert out.eq(10_000.0).all()


def test_relative_strength_is_zero_against_itself(uptrend_with_pullback):
    close = uptrend_with_pullback["close"]
    out = ind.relative_strength(close, close, 60).dropna()
    assert np.allclose(out, 0.0, atol=1e-12)


# ── Invariante de causalidad ──────────────────────────────────────────────────

@pytest.mark.parametrize("cut", [280, 295, 305])
def test_no_lookahead(uptrend_with_pullback, cut):
    """Truncar el futuro no puede cambiar el valor presente de un indicador.

    Se calcula cada indicador sobre la serie completa y sobre la serie cortada
    en `cut`, y se exige que el último valor de la versión cortada coincida con
    el valor en esa misma fecha de la versión completa.
    """
    full = uptrend_with_pullback
    truncated = full.iloc[:cut]

    checks = {
        "ema20": lambda d: ind.ema(d["close"], 20),
        "ema200": lambda d: ind.ema(d["close"], 200),
        "rsi": lambda d: ind.rsi(d["close"], 14),
        "atr": lambda d: ind.atr(d["high"], d["low"], d["close"], 14),
        "adx": lambda d: ind.adx(d["high"], d["low"], d["close"], 14)["adx"],
        "macd_hist": lambda d: ind.macd(d["close"])["hist"],
        "swing_low": lambda d: ind.rolling_low(d["low"], 10),
    }

    for name, fn in checks.items():
        full_value = fn(full).iloc[cut - 1]
        truncated_value = fn(truncated).iloc[-1]
        assert truncated_value == pytest.approx(full_value, rel=1e-9), (
            f"{name} mira al futuro: {truncated_value} != {full_value}"
        )
