"""¿Y si la salida comparte tesis con la entrada?

No es afinar parámetros: es corregir un desajuste de diseño. Una entrada por
reversión dice "el precio se pasó y volverá a su sitio", y eso tiene un
horizonte de días y un objetivo cercano. Emparejarla con "aguanta hasta un
movimiento de 3 ATR durante 30 días" es una tesis distinta pegada detrás.

Se declaran TRES salidas y se publican las tres en los dos cuadrantes. No se
elige la mejor del descubrimiento y se presenta el reservado como comprobación.
"""
from __future__ import annotations

import hashlib

from tools.evaluar import evaluar
from tools.ibkr_data import load_all
from trading import universe
from trading.backtest import run_backtest
from trading.config import DEFAULT_BACKTEST, DEFAULT_PARAMS, replace

bars = load_all(); indice = bars["SPY"]["close"]
fechas = bars["SPY"].index
corte = fechas[int(len(fechas) * 0.6)]
reservado = lambda s: hashlib.sha256(s.encode()).hexdigest()[0] in "0123"

SALIDAS = {
    "tendencia (3 ATR, 30 d)": dict(target_atr_mult=3.00, max_holding_days=30),
    "media       (1 ATR, 5 d)": dict(target_atr_mult=1.00, max_holding_days=5),
    "intermedia  (2 ATR, 10 d)": dict(target_atr_mult=2.00, max_holding_days=10),
}

for entrada in ("retroceso", "rsi2"):
    print(f"\n{'═'*76}\n  ENTRADA: {entrada}\n{'═'*76}")
    for nombre, ajuste in SALIDAS.items():
        p = replace(DEFAULT_PARAMS, entry_rule=entrada,
                    target_atr_mult=ajuste["target_atr_mult"],
                    min_risk_reward=0.0)   # el R:B mínimo vetaría 1 ATR entero
        bt = replace(DEFAULT_BACKTEST, max_holding_days=ajuste["max_holding_days"])
        insts = [universe.get(s) for s in bars if s in universe.all_symbols()]
        rep = run_backtest(insts, bars, p, bt, indice)

        ts = [t for t in rep.trades if t.entry_date is not None]
        desc = evaluar([t for t in ts if not reservado(t.symbol) and t.entry_date < corte], indice)
        res  = evaluar([t for t in ts if reservado(t.symbol) and t.entry_date >= corte], indice)

        print(f"\n  {nombre}")
        for etiqueta, c in (("descubrimiento", desc), ("NADA COMPARTIDO", res)):
            if c.ops < 5:
                print(f"    {etiqueta:18} sin muestra"); continue
            marca = "✓" if c.demuestra else " "
            print(f"    {etiqueta:18} {c.ops:5} ops  acierto {100*c.acierto:5.1f}%  "
                  f"R {c.r_medio:+.3f}  exceso {100*c.exceso:+7.3f}%  "
                  f"mín {100*c.exceso_min:+7.3f}% {marca}")
