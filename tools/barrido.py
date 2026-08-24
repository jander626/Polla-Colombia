"""Barre la rejilla en los dos sentidos y ordena por VALIDACIÓN."""
from __future__ import annotations

from tools.ibkr_data import load_all
from trading import universe
from trading.backtest import format_search, run_search
from trading.config import DEFAULT_BACKTEST, SEARCH_TRAILS, replace, search_grid

bars = load_all()
bench = bars["SPY"]["close"]
insts = [universe.get(s) for s in bars if s in universe.all_symbols()]

grid = search_grid(("long", "short"))
print(f"{len(grid)} combinaciones × {len(SEARCH_TRAILS)} salidas = "
      f"{len(grid)*len(SEARCH_TRAILS)} mediciones\n")

res = run_search(insts, bars, grid, SEARCH_TRAILS,
                 replace(DEFAULT_BACKTEST, years=5), bench)
print(format_search(res))
