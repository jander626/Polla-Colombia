"""Corre las hipótesis declaradas y publica los cuatro cuadrantes."""
from __future__ import annotations

import hashlib

import pandas as pd

from tools.evaluar import Celda, evaluar, operaciones
from tools.ibkr_data import load_all
from tools.reglas import REGLAS

bars = load_all()
indice = bars["SPY"]["close"]
fechas = bars["SPY"].index
corte = fechas[int(len(fechas) * 0.6)]

# El reparto de instrumentos va por hash del símbolo, no a mano: elegirlos yo
# sería el primer sitio por donde se cuela el sobreajuste.
def reservado(s: str) -> bool:
    return hashlib.sha256(s.encode()).hexdigest()[0] in "0123"

simbolos = [s for s in bars if s != "SPY"]
reserva = sorted(s for s in simbolos if reservado(s))
descubre = sorted(s for s in simbolos if not reservado(s))

print(f"Instrumentos de descubrimiento ({len(descubre)}): {' '.join(descubre)}")
print(f"Instrumentos RESERVADOS       ({len(reserva)}): {' '.join(reserva)}")
print(f"Corte temporal: {corte.date()}  "
      f"(descubrimiento {fechas[0].date()}→{corte.date()}, "
      f"reservado {corte.date()}→{fechas[-1].date()})\n")

def cuadrante(ts, simbolos_ok, desde=None, hasta=None):
    # Sin fecha de entrada la orden nunca se ejecutó: no cae en ningún
    # cuadrante y no debe contar como operación en ninguno.
    sel = [t for t in ts
           if t.symbol in simbolos_ok and t.entry_date is not None
           and (desde is None or t.entry_date >= desde)
           and (hasta is None or t.entry_date < hasta)]
    return evaluar(sel, indice)

def linea(nombre: str, c: Celda) -> str:
    if c.ops == 0:
        return f"    {nombre:22} sin operaciones"
    marca = "✓" if c.demuestra else ("·" if c.ops >= 30 else "?")
    return (f"    {nombre:22} {c.ops:4} ops  acierto {100*c.acierto:4.1f}%  "
            f"R {c.r_medio:+.3f}  exceso {100*c.exceso:+.3f}%  "
            f"mín {100*c.exceso_min:+.3f}% {marca}")

resumen = []
for r in REGLAS:
    ts = operaciones(r, bars)
    d_d = cuadrante(ts, descubre, hasta=corte)
    d_t = cuadrante(ts, descubre, desde=corte)
    r_d = cuadrante(ts, reserva, hasta=corte)
    r_t = cuadrante(ts, reserva, desde=corte)

    print(f"── {r.nombre}  [{r.indicadores_usados}]")
    print(linea("descubrimiento", d_d))
    print(linea("otras fechas", d_t))
    print(linea("otros instrumentos", r_d))
    print(linea("NADA COMPARTIDO", r_t))
    print()
    resumen.append((r.nombre, d_d, r_t))

print("═" * 78)
print("VEREDICTO — solo cuenta el cuadrante sin nada compartido")
print("═" * 78)
for nombre, d, r in resumen:
    est = ("DEMUESTRA" if r.demuestra
           else "muestra corta" if r.ops < 30
           else "no demuestra")
    print(f"  {nombre:24} descubrimiento {100*d.exceso:+7.3f}%  →  "
           f"reservado {100*r.exceso:+7.3f}% (mín {100*r.exceso_min:+7.3f}%)  {est}")
