"""Reglas de entrada simples, para probar hipótesis con POCOS indicadores.

Cada regla es una hipótesis económica con nombre, no una combinación de
parámetros sacada de una rejilla. Eso importa: una rejilla de 162 celdas
produce una ganadora aunque no haya nada que ganar, y este proyecto ya lo
midió dos veces. Con seis hipótesis declaradas de antemano, el umbral de
credibilidad es mucho más alto.

Todas comparten la MISMA salida (stop y objetivo por ATR, los del proyecto),
así que lo único que varía entre ellas es *cuándo entrar*. Es un experimento
con una sola variable, no seis estrategias distintas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ── Indicadores. Solo estos seis, y cada regla usa uno o dos. ────────────────

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0.0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def indicadores(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    out = df.copy()
    out["sma200"] = sma(c, 200)
    out["sma50"] = sma(c, 50)
    out["rsi2"] = rsi(c, 2)
    out["rsi14"] = rsi(c, 14)
    out["atr"] = atr(h, l, c, 14)
    out["atr_pct"] = out["atr"] / c
    out["max20"] = h.rolling(20, min_periods=20).max()
    out["min20"] = l.rolling(20, min_periods=20).min()
    # Momentum 12-1: 12 meses excluyendo el último (Jegadeesh y Titman).
    out["mom12_1"] = c.shift(21) / c.shift(252) - 1.0
    return out


# ── Las hipótesis ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Regla:
    nombre: str
    indicadores_usados: str
    tesis: str
    entrada: Callable[[pd.DataFrame], pd.Series]
    direccion: str = "long"


def _prev_max(f: pd.DataFrame) -> pd.Series:
    """Máximo de 20 sesiones SIN incluir la de hoy: si no, mira al futuro."""
    return f["max20"].shift(1)


def _prev_min(f: pd.DataFrame) -> pd.Series:
    return f["min20"].shift(1)


REGLAS: tuple[Regla, ...] = (
    Regla(
        "L1 tendencia+RSI2",
        "SMA200, RSI(2)",
        "Reversión corta dentro de tendencia alcista (Connors). El clásico "
        "con más respaldo publicado de los que caben en dos indicadores.",
        lambda f: (f["close"] > f["sma200"]) & (f["rsi2"] < 10.0),
    ),
    Regla(
        "L2 ruptura 20d",
        "máximo de 20 sesiones",
        "Seguimiento de tendencia puro (Donchian). Un solo indicador.",
        lambda f: f["close"] > _prev_max(f),
    ),
    Regla(
        "L3 ruptura en tendencia",
        "SMA200 + máximo de 20 sesiones",
        "La ruptura solo cuenta si el fondo es alcista: filtra las rupturas "
        "de mercado bajista, que son las que devuelven el precio.",
        lambda f: (f["close"] > f["sma200"]) & (f["close"] > _prev_max(f)),
    ),
    Regla(
        "L4 momentum 12-1",
        "SMA200 + momentum 12-1",
        "El efecto con más respaldo académico que existe en acciones, y el "
        "único que su revisión de 2023 confirma que no ha decaído.",
        lambda f: (f["close"] > f["sma200"]) & (f["mom12_1"] > 0.15),
    ),
    Regla(
        "L5 retroceso a la SMA50",
        "SMA200 + SMA50",
        "Retroceso simple: tendencia intacta y el precio vuelve a tocar la "
        "media intermedia. Es la estrategia actual reducida a dos medias.",
        lambda f: (f["close"] > f["sma200"])
        & (f["low"] <= f["sma50"])
        & (f["low"].shift(1) > f["sma50"].shift(1)),
    ),
    Regla(
        "L6 sobreventa profunda",
        "RSI(14)",
        "Sin filtro de tendencia, a propósito: sirve de control. Si rinde "
        "como las demás, el filtro de tendencia no aporta nada.",
        lambda f: (f["rsi14"] < 30.0) & (f["rsi14"].shift(1) >= 30.0),
    ),
    # ── Los espejos ──────────────────────────────────────────────────────────
    Regla(
        "S1 tendencia+RSI2",
        "SMA200, RSI(2)",
        "El espejo de L1.",
        lambda f: (f["close"] < f["sma200"]) & (f["rsi2"] > 90.0),
        direccion="short",
    ),
    Regla(
        "S3 ruptura en tendencia",
        "SMA200 + mínimo de 20 sesiones",
        "El espejo de L3.",
        lambda f: (f["close"] < f["sma200"]) & (f["close"] < _prev_min(f)),
        direccion="short",
    ),
)
