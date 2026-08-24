"""Fixtures compartidas: generadores de velas sintéticas.

Se usan datos sintéticos y no un CSV descargado para que los tests sean
deterministas y no dependan de la red ni de la cuota de la API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_bars(closes: list[float] | np.ndarray, spread: float = 0.01) -> pd.DataFrame:
    """Convierte una serie de cierres en velas OHLCV plausibles.

    El máximo y el mínimo se construyen alrededor del cierre con una amplitud
    proporcional, y la apertura es el cierre anterior. Suficiente para ejercitar
    los indicadores sin inventar microestructura.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    index = pd.bdate_range("2020-01-01", periods=n)

    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * (1.0 + spread)
    lows = np.minimum(opens, closes) * (1.0 - spread)
    volume = np.full(n, 5_000_000.0)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=index,
    )


@pytest.fixture
def uptrend_with_pullback() -> pd.DataFrame:
    """Tendencia alcista larga, retroceso hacia la media rápida y reanudación.

    Es el escenario que la estrategia debe detectar. Se construye a mano para
    que el test falle de forma clara si algún filtro deja de reconocerlo.
    """
    rng = np.random.default_rng(7)

    # Tramo 1: subida sostenida con ruido pequeño. Son 400 velas y no 300
    # porque el momentum a 12 meses exige 273 sesiones de historia antes
    # de dar su primer valor.
    trend = 100.0 * np.cumprod(1.0 + rng.normal(0.0018, 0.006, 400))

    # Tramo 2: retroceso de ~4% en 5 velas — enfría el RSI y acerca el precio
    # a la EMA20 sin romper la tendencia de fondo.
    last = trend[-1]
    pullback = last * np.cumprod(np.full(5, 1.0 - 0.008))

    # Tramo 3: dos velas de reanudación contenidas. La magnitud importa y no
    # es arbitraria: cuanto más fuerte rebota el precio antes de la señal, más
    # arriba queda la entrada y peor sale el ratio riesgo/beneficio, porque se
    # compra más lejos del stop. Con este rebote el R:B queda en ~1.68, con
    # holgura sobre el mínimo de 1.5 para que el test no sea frágil.
    resume = pullback[-1] * np.cumprod([1.010, 1.018])

    closes = np.concatenate([trend, pullback, resume])
    # El volumen de la vela de reanudación destaca sobre la media.
    bars = make_bars(closes)
    bars.iloc[-1, bars.columns.get_loc("volume")] = 12_000_000.0
    return bars


@pytest.fixture
def downtrend_with_rally() -> pd.DataFrame:
    """El espejo de `uptrend_with_pullback`: lo que debe detectar el lado corto.

    Se construye reflejando la misma serie (`p' = K - p`) en vez de generarla
    aparte, para que cualquier diferencia entre lo que detecta el largo y lo
    que detecta el corto sea del código y no del escenario.
    """
    rng = np.random.default_rng(7)
    trend = 100.0 * np.cumprod(1.0 + rng.normal(0.0018, 0.006, 400))
    pullback = trend[-1] * np.cumprod(np.full(5, 1.0 - 0.008))
    resume = pullback[-1] * np.cumprod([1.010, 1.018])
    bars = make_bars(np.concatenate([trend, pullback, resume]))
    bars.iloc[-1, bars.columns.get_loc("volume")] = 12_000_000.0

    level = float(bars["high"].max()) + 50.0
    mirrored = bars.copy()
    mirrored["open"] = level - bars["open"]
    mirrored["close"] = level - bars["close"]
    mirrored["high"] = level - bars["low"]
    mirrored["low"] = level - bars["high"]
    return mirrored


@pytest.fixture
def sideways() -> pd.DataFrame:
    """Mercado lateral: la estrategia NO debe generar señales aquí.

    Se genera con reversión a la media en lugar de un paseo aleatorio: un
    paseo aleatorio largo deriva y acaba pareciendo una tendencia, que es
    justo lo contrario de lo que este escenario debe representar.
    """
    rng = np.random.default_rng(11)
    level, pull = 100.0, 0.08
    closes = np.empty(420)
    price = level
    for i in range(420):
        price += pull * (level - price) + rng.normal(0.0, 1.0)
        closes[i] = price
    return make_bars(closes)


@pytest.fixture
def downtrend() -> pd.DataFrame:
    """Tendencia bajista sostenida: tampoco debe generar compras."""
    rng = np.random.default_rng(13)
    closes = 200.0 * np.cumprod(1.0 + rng.normal(-0.002, 0.006, 420))
    return make_bars(closes)
