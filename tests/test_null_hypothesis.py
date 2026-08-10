"""Prueba nula: sobre ruido puro, la estrategia no puede ganar dinero.

Este es el test que protege la credibilidad de todo el proyecto. Un backtest
con look-ahead, con ejecución optimista o con cualquier fuga de información del
futuro produce beneficios incluso sobre datos sin ninguna estructura. Si la
estrategia gana dinero sobre paseos aleatorios sin deriva, el problema está en
el motor y no en el mercado.

Se comprueba con un contraste estadístico, no con una comparación exacta:
sobre unos cientos de operaciones simuladas el retorno medio tiene ruido
propio, así que lo que se exige es que no sea *significativamente* distinto de
cero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading import universe
from trading.backtest import run_backtest
from trading.config import DEFAULT_BACKTEST, DEFAULT_PARAMS, replace

# Sin costes: aquí se aísla el sesgo del motor, no la rentabilidad neta.
NO_COST = replace(DEFAULT_BACKTEST, round_trip_cost=0.0)

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "JPM", "XOM", "KO", "WMT", "PFE", "BA", "GE",
    "CVX", "MRK", "ORCL", "CSCO", "INTC", "TGT", "LOW", "UPS", "CAT", "DE",
]


def random_walk(seed: int, n: int = 1600, drift: float = 0.0) -> pd.DataFrame:
    """Paseo aleatorio geométrico sin ninguna estructura explotable."""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.009, 0.018)
    closes = rng.uniform(30.0, 300.0) * np.cumprod(1.0 + rng.normal(drift, vol, n))

    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * (1.0 + np.abs(rng.normal(0, 0.004, n)))
    lows = np.minimum(opens, closes) * (1.0 - np.abs(rng.normal(0, 0.004, n)))

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.uniform(5e6, 4e7, n),
        },
        index=pd.bdate_range("2019-01-01", periods=n),
    )


@pytest.fixture(scope="module")
def noise_backtest():
    bars = {sym: random_walk(seed=100 + i) for i, sym in enumerate(SYMBOLS)}
    instruments = [universe.get(sym) for sym in SYMBOLS]
    report = run_backtest(instruments, bars, DEFAULT_PARAMS, NO_COST)
    return report


def test_the_null_test_actually_generates_trades(noise_backtest):
    """Sin operaciones el contraste no probaría nada."""
    assert len(noise_backtest.filled) >= 50


def _t_statistic(values: np.ndarray) -> float:
    return float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values))))


# El contraste es de UNA cola, y esto es deliberado. Lo que delata un motor
# defectuoso es *ganar* sobre ruido: eso solo puede venir de mirar al futuro o
# de suponer ejecuciones imposibles. Una esperanza ligeramente negativa, en
# cambio, es la consecuencia correcta de las reglas conservadoras del motor
# (vela ambigua contada como pérdida, huecos ejecutados peor que el stop), así
# que exigirle simetría al contraste castigaría precisamente lo que queremos.
MAX_T = 3.0


def test_no_edge_on_pure_noise(noise_backtest):
    """El retorno medio no puede ser significativamente positivo sobre ruido."""
    returns = np.array([t.return_pct for t in noise_backtest.filled])
    t_stat = _t_statistic(returns)

    assert t_stat < MAX_T, (
        f"la estrategia gana sobre ruido puro (t={t_stat:+.2f}): "
        "sospecha de look-ahead o de ejecución optimista en el motor"
    )


def test_r_expectancy_is_not_positive_on_noise(noise_backtest):
    r = np.array([t.r_multiple for t in noise_backtest.filled])
    assert _t_statistic(r) < MAX_T


def test_a_losing_stop_never_exceeds_one_r(noise_backtest):
    """Con R medido sobre el riesgo planificado, un stop limpio pierde ≤ 1R.

    Solo los huecos por debajo del stop pueden superarlo, y esos se identifican
    porque la salida ocurre por debajo del nivel de stop.
    """
    for trade in noise_backtest.filled:
        if trade.outcome == "loss" and trade.exit_price >= trade.stop:
            assert trade.r_multiple >= -1.0000001, (
                f"{trade.symbol}: pérdida de {trade.r_multiple:.2f}R sin hueco"
            )


def test_upward_drift_is_detected(noise_backtest):
    """Contraprueba: con deriva alcista real, la estrategia sí debe capturarla.

    Sin esto, `test_no_edge_on_pure_noise` pasaría también con una estrategia
    rota que no genera ninguna operación rentable jamás.
    """
    bars = {
        sym: random_walk(seed=300 + i, drift=0.0012)
        for i, sym in enumerate(SYMBOLS)
    }
    instruments = [universe.get(sym) for sym in SYMBOLS]
    report = run_backtest(instruments, bars, DEFAULT_PARAMS, NO_COST)

    assert len(report.filled) >= 20
    drift_return = np.mean([t.return_pct for t in report.filled])
    noise_return = np.mean([t.return_pct for t in noise_backtest.filled])
    assert drift_return > noise_return
