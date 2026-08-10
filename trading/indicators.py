"""Indicadores técnicos sobre pandas.

Escritos a mano en lugar de usar pandas-ta / TA-Lib: son pocos, la
implementación cabe en un archivo, evita conflictos de versiones en CI y —
sobre todo — se puede testear contra valores conocidos.

Todas las funciones devuelven una Series alineada con el índice de entrada,
con NaN en las posiciones donde el indicador aún no tiene historia suficiente.
Ninguna mira hacia el futuro: el valor en la posición i solo usa datos de
las posiciones <= i.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Suavizados ────────────────────────────────────────────────────────────────

def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Suavizado de Wilder: y_t = (y_{t-1} * (n-1) + x_t) / n.

    Se siembra con la media simple de las primeras `period` observaciones
    válidas, que es la convención original de Wilder y la que usan las
    plataformas de gráficos. Equivale a un EWM con alpha = 1/period una vez
    fijada esa semilla.
    """
    s = series.astype(float)
    first = s.first_valid_index()
    if first is None:
        return pd.Series(np.nan, index=s.index, dtype=float)

    pos = s.index.get_loc(first)
    if len(s) - pos < period:
        return pd.Series(np.nan, index=s.index, dtype=float)

    seeded = pd.Series(np.nan, index=s.index, dtype=float)
    seed_end = pos + period
    seeded.iloc[seed_end - 1] = s.iloc[pos:seed_end].mean()
    seeded.iloc[seed_end:] = s.iloc[seed_end:]
    return seeded.ewm(alpha=1.0 / period, adjust=False).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Media móvil exponencial estándar (alpha = 2/(span+1))."""
    return series.astype(float).ewm(span=span, adjust=False, min_periods=span).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.astype(float).rolling(window, min_periods=window).mean()


# ── Momentum ──────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder. Devuelve valores en [0, 100]."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = wilder_smooth(gain.iloc[1:], period).reindex(close.index)
    avg_loss = wilder_smooth(loss.iloc[1:], period).reindex(close.index)

    # avg_loss == 0 significa que no hubo ninguna bajada en la ventana: RSI = 100.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    return out.where(avg_gain.notna() & avg_loss.notna())


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD clásico. Columnas: macd, signal, hist."""
    c = close.astype(float)
    macd_line = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    # Antes de `slow` barras la línea está contaminada por el arranque del EWM.
    macd_line.iloc[: slow - 1] = np.nan
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line}
    )


# ── Volatilidad y tendencia ───────────────────────────────────────────────────

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range de Wilder, en unidades de precio."""
    return wilder_smooth(true_range(high, low, close).iloc[1:], period).reindex(close.index)


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """ADX de Wilder. Columnas: plus_di, minus_di, adx.

    ADX mide la *fuerza* de la tendencia, no su dirección; la dirección la dan
    plus_di y minus_di. Por eso la estrategia lo combina con el filtro de medias.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    tr = true_range(high, low, close)
    atr_ = wilder_smooth(tr.iloc[1:], period).reindex(close.index)
    plus_smoothed = wilder_smooth(plus_dm.iloc[1:], period).reindex(close.index)
    minus_smoothed = wilder_smooth(minus_dm.iloc[1:], period).reindex(close.index)

    safe_atr = atr_.replace(0.0, np.nan)
    plus_di = 100.0 * plus_smoothed / safe_atr
    minus_di = 100.0 * minus_smoothed / safe_atr

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_line = wilder_smooth(dx, period)

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_line})


# ── Volumen y estructura de precio ────────────────────────────────────────────

def dollar_volume(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Volumen medio en dólares de las últimas `window` sesiones.

    Es la medida de liquidez que usa el bot: el volumen en acciones no es
    comparable entre un valor de 5 USD y uno de 500 USD.
    """
    return (close.astype(float) * volume.astype(float)).rolling(
        window, min_periods=window
    ).mean()


def rolling_low(low: pd.Series, window: int) -> pd.Series:
    """Mínimo de las últimas `window` velas, incluida la actual."""
    return low.astype(float).rolling(window, min_periods=1).min()


def rolling_high(high: pd.Series, window: int) -> pd.Series:
    """Máximo de las últimas `window` velas, incluida la actual."""
    return high.astype(float).rolling(window, min_periods=1).max()


def pct_from(series: pd.Series, reference: pd.Series) -> pd.Series:
    """Distancia relativa de `series` respecto a `reference`, en tanto por uno."""
    ref = reference.replace(0.0, np.nan)
    return (series - ref) / ref


def relative_strength(close: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
    """Rendimiento del instrumento menos el del benchmark en `window` sesiones.

    Positivo = lo está haciendo mejor que el mercado. El benchmark se realinea
    al índice del instrumento, así que los días sin dato del benchmark
    (festivos distintos, forex en fin de semana) quedan como NaN en vez de
    producir una comparación falsa.
    """
    c = close.astype(float)
    b = benchmark.astype(float).reindex(c.index)
    own = c.pct_change(window)
    bench = b.pct_change(window)
    return own - bench
