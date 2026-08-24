"""¿L1 es un efecto o un filo de cuchillo? Se publican TODOS los umbrales."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from tools.evaluar import evaluar, operaciones
from tools.ibkr_data import load_all
from tools.reglas import Regla

bars = load_all(); indice = bars["SPY"]["close"]

print("Sensibilidad al umbral del RSI(2) — se publican todos, no el mejor")
print(f"{'umbral':>7} {'ops':>6} {'acierto':>8} {'R med':>7} {'exceso':>9} {'mín':>9}")
print("─" * 52)
for u in (5, 10, 15, 20, 25):
    r = Regla(f"rsi2<{u}", "SMA200, RSI(2)", "",
              lambda f, u=u: (f["close"] > f["sma200"]) & (f["rsi2"] < float(u)))
    c = evaluar(operaciones(r, bars), indice)
    print(f"{u:>7} {c.ops:6} {100*c.acierto:7.1f}% {c.r_medio:+7.3f} "
          f"{100*c.exceso:+8.3f}% {100*c.exceso_min:+8.3f}%")

print("\nSensibilidad a la media larga")
print(f"{'media':>7} {'ops':>6} {'acierto':>8} {'R med':>7} {'exceso':>9} {'mín':>9}")
print("─" * 52)
for n in (100, 150, 200, 250):
    r = Regla(f"sma{n}", "", "",
              lambda f, n=n: (f["close"] > f["close"].rolling(n, min_periods=n).mean())
              & (f["rsi2"] < 10.0))
    c = evaluar(operaciones(r, bars), indice)
    print(f"{n:>7} {c.ops:6} {100*c.acierto:7.1f}% {c.r_medio:+7.3f} "
          f"{100*c.exceso:+8.3f}% {100*c.exceso_min:+8.3f}%")

print("\nL1 año a año (¿aguanta o vive de un tramo?)")
base = Regla("L1", "", "", lambda f: (f["close"] > f["sma200"]) & (f["rsi2"] < 10.0))
ts = [t for t in operaciones(base, bars) if t.entry_date is not None and t.was_filled]
print(f"{'año':>6} {'ops':>6} {'acierto':>8} {'R med':>8}")
print("─" * 32)
for a in sorted({t.entry_date.year for t in ts}):
    sel = [t for t in ts if t.entry_date.year == a]
    ac = 100 * sum(t.is_win for t in sel) / len(sel)
    rm = sum(t.r_multiple for t in sel) / len(sel)
    print(f"{a:>6} {len(sel):6} {ac:7.1f}% {rm:+8.3f}")

print(f"\nseñales por año y por instrumento: {len(ts)/5/29:.1f}")
print(f"señales por día en un universo de 141: {len(ts)/5/29*141/252:.1f}")
