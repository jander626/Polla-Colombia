"""¿Puede un backtest demostrar esto alguna vez, y con cuántas operaciones?"""
from __future__ import annotations

import numpy as np

from tools.evaluar import evaluar, operaciones
from tools.ibkr_data import load_all
from tools.reglas import REGLAS

bars = load_all(); indice = bars["SPY"]["close"]

print(f"{'regla':24} {'ops':>5} {'acierto':>8} {'R med':>7} {'exceso':>9} "
      f"{'mín':>9} {'ops para demostrar':>20}")
print("─" * 88)

for r in REGLAS:
    ts = operaciones(r, bars)
    c = evaluar(ts, indice)
    if c.ops < 30:
        print(f"{r.nombre:24} {c.ops:5}  muestra corta"); continue

    # Desviación típica del exceso, despejada del error estándar observado.
    ee = (c.exceso - c.exceso_min) / 1.96
    sigma = ee * np.sqrt(c.ops)
    # n para que el límite inferior toque cero manteniendo el exceso medio.
    if c.exceso > 0:
        n = (1.96 * sigma / c.exceso) ** 2
        años = n / (c.ops / 5.0)
        falta = f"{n:,.0f}  (~{años:.0f} años)"
    else:
        falta = "nunca (exceso negativo)"
    print(f"{r.nombre:24} {c.ops:5} {100*c.acierto:7.1f}% {c.r_medio:+7.3f} "
          f"{100*c.exceso:+8.3f}% {100*c.exceso_min:+8.3f}% {falta:>20}")

print("\nMuestra COMPLETA (descubrimiento + reservado). Ya no es fuera de")
print("muestra: es la mejor estimación puntual que permiten estos datos, y")
print("sirve para ver de qué tamaño tendría que ser la muestra.")
