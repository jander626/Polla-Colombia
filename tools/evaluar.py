"""Evalúa las hipótesis con instrumentos Y fechas reservados.

Por qué así y no como antes: el proyecto siempre validó reservando el tramo
final del periodo. Eso protege del sobreajuste temporal, pero no del otro: una
regla puede estar ajustada a los instrumentos concretos con los que se probó.
Aquí se reservan las dos cosas, lo que da cuatro cuadrantes:

                 fechas de descubrimiento   fechas reservadas
  instr. desc.        DESCUBRIMIENTO            fuera de tiempo
  instr. reserv.      fuera de muestra          NADA COMPARTIDO

El cuadrante que decide es el último: instrumentos que la regla no vio, en
fechas que no vio. Es el más parecido a operar mañana.

Dos decisiones que evitan inflar la muestra:

1. **Operaciones no solapadas.** Si una regla dispara cinco días seguidos en
   el mismo valor, eso no son cinco observaciones: es una. Se toma la primera
   y no se vuelve a entrar en ese símbolo hasta que la anterior cierra. Es
   además lo que puede hacer alguien que opera a mano.
2. **El nulo lleva el sentido de la operación.** Para un corto es vender el
   índice, no comprarlo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tools.reglas import REGLAS, Regla, indicadores
from trading.backtest import Trade, simulate_signal
from trading.config import DEFAULT_BACKTEST, DEFAULT_PARAMS, replace
from trading.risk import compute_levels
from trading.strategy import Signal

CALENTAMIENTO = 273        # el momentum 12-1 es el que más historia exige
BT = DEFAULT_BACKTEST


def _levels(fila, direccion):
    params = replace(DEFAULT_PARAMS, direction=direccion)
    swing = fila["max20"] if direccion == "short" else fila["min20"]
    return compute_levels(
        close=float(fila["close"]), atr=float(fila["atr"]),
        swing_low=float(swing), params=params,
    )


def operaciones(regla: Regla, bars: dict[str, pd.DataFrame]) -> list[Trade]:
    """Todas las operaciones de una regla, sin solapar dentro de un símbolo."""
    salida: list[Trade] = []
    for simbolo, df in bars.items():
        f = indicadores(df)
        dispara = regla.entrada(f).fillna(False)
        libre_desde = pd.Timestamp.min

        for pos in range(CALENTAMIENTO, len(f)):
            fecha = f.index[pos]
            if fecha < libre_desde or not bool(dispara.iloc[pos]):
                continue
            fila = f.iloc[pos]
            if not np.isfinite(fila["atr"]) or fila["atr"] <= 0:
                continue
            niveles = _levels(fila, regla.direccion)
            if niveles is None or not niveles.is_valid:
                continue

            señal = Signal(
                symbol=simbolo, name=simbolo, asset_class="stock",
                bar_date=fecha, close=float(fila["close"]),
                atr=float(fila["atr"]), atr_pct=float(fila["atr_pct"]),
                score=50.0, levels=niveles,
            )
            t = simulate_signal(señal, f, BT)
            salida.append(t)
            # Bloquea el símbolo hasta que esta operación termine.
            libre_desde = t.exit_date if t.exit_date is not None else fecha
    return salida


@dataclass
class Celda:
    ops: int
    acierto: float
    r_medio: float
    exceso: float
    exceso_min: float

    @property
    def demuestra(self) -> bool:
        return self.ops >= 30 and self.exceso_min > 0.0


def evaluar(trades: list[Trade], indice: pd.Series) -> Celda:
    """Estadística de un grupo de operaciones frente al nulo del mismo sentido."""
    utiles = [
        t for t in trades
        if t.was_filled and t.entry_date is not None and t.exit_date is not None
        and np.isfinite(t.return_pct)
    ]
    if not utiles:
        return Celda(0, float("nan"), float("nan"), float("nan"), float("nan"))

    idx = indice.sort_index()

    def precio(cuando):
        p = idx.index.searchsorted(cuando, side="right") - 1
        return float(idx.iloc[p]) if p >= 0 else float("nan")

    est, nulo = [], []
    for t in utiles:
        e, s = precio(t.entry_date), precio(t.exit_date)
        if not (np.isfinite(e) and np.isfinite(s)) or e <= 0:
            continue
        mov = s / e - 1.0
        est.append(t.return_pct)
        nulo.append(-mov if t.is_short else mov)

    if not est:
        return Celda(0, float("nan"), float("nan"), float("nan"), float("nan"))

    exceso = np.array(est) - np.array(nulo)
    ee = exceso.std(ddof=1) / np.sqrt(len(exceso)) if len(exceso) > 1 else float("nan")
    return Celda(
        ops=len(utiles),
        acierto=float(np.mean([t.is_win for t in utiles])),
        r_medio=float(np.mean([t.r_multiple for t in utiles])),
        exceso=float(exceso.mean()),
        exceso_min=float(exceso.mean() - 1.96 * ee),
    )
