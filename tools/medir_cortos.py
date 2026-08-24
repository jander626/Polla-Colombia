"""Mide largos y cortos sobre las velas reales de IBKR."""
from __future__ import annotations

import numpy as np

from tools.ibkr_data import load_all
from trading import universe
from trading.backtest import benchmark_comparison, run_backtest
from trading.config import DEFAULT_BACKTEST, DEFAULT_PARAMS, replace

bars = load_all()
bench = bars["SPY"]["close"]
insts = [universe.get(s) for s in bars if s in universe.all_symbols()]

# Régimen: cuántos días el S&P estuvo bajo su media de 200 (el único terreno
# donde el cribador corto puede disparar).
sma200 = bench.rolling(200).mean()
bajo = (bench < sma200)
print(f"Universo: {len(insts)} instrumentos · {len(bars['SPY'])} sesiones "
      f"({bars['SPY'].index[0].date()} → {bars['SPY'].index[-1].date()})")
print(f"Sesiones con el S&P bajo su media de 200: {int(bajo.sum())} "
      f"({100*bajo.mean():.1f}%)\n")

for d in ("long", "short"):
    p = replace(DEFAULT_PARAMS, direction=d)
    rep = run_backtest(insts, bars, p, DEFAULT_BACKTEST, bench)
    f = rep.filled
    print("═" * 64)
    print(f"  {d.upper()}")
    print("═" * 64)
    print(f"señales generadas   : {rep.signals_generated}")
    print(f"órdenes ejecutadas  : {len(f)}")
    if not f:
        print("sin operaciones\n"); continue
    r = np.array([t.r_multiple for t in f])
    print(f"acierto             : {100*sum(t.is_win for t in f)/len(f):.1f}%")
    print(f"R media             : {r.mean():+.3f}   (total {r.sum():+.1f}R)")
    print(f"retorno medio       : {100*np.mean([t.return_pct for t in f]):+.3f}%")
    print(f"días en posición    : {np.mean([t.bars_held for t in f]):.1f}")

    c = benchmark_comparison(rep, bench)
    print("\n── ¿bate al nulo del MISMO sentido? ──")
    if c is None:
        print(f"  MUESTRA INSUFICIENTE (hacen falta 30 operaciones, hay {len(f)})")
    else:
        print(f"  estrategia          : {100*c.strategy_avg:+.3f}% por operación")
        print(f"  nulo (índice)       : {100*c.benchmark_avg:+.3f}%")
        print(f"  exceso              : {100*c.excess_avg:+.3f}%")
        print(f"  LÍMITE INFERIOR     : {100*c.excess_lower:+.3f}%   <-- el que decide")
        print(f"  veces que gana      : {100*c.beat_rate:.1f}%")
    print()
    print("── partición temporal ──")
    print(rep.out_of_sample_split())
    print()
