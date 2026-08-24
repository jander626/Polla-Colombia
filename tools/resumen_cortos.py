from tools.ibkr_data import load_all
from trading import universe
from trading.backtest import run_search
from trading.config import MIN_MEANINGFUL_EXPECTANCY as UMBRAL
from trading.config import DEFAULT_BACKTEST, SEARCH_TRAILS, replace, search_grid

bars = load_all(); bench = bars["SPY"]["close"]
insts = [universe.get(s) for s in bars if s in universe.all_symbols()]
res = run_search(insts, bars, search_grid(("long","short")), SEARCH_TRAILS,
                 replace(DEFAULT_BACKTEST, years=5), bench)

largos = [r for r in res if r.label.startswith("long")]
cortos = [r for r in res if r.label.startswith("short")]
print(f"umbral de ventaja demostrada: +{UMBRAL:.3f}R en validación\n")

def bloque(nombre, rs):
    pasan = [r for r in rs if r.test_lower >= UMBRAL]
    print(f"{nombre}: {len(rs)} mediciones · pasan validación: {len(pasan)}")
    ops = sorted(r.ops for r in rs)
    print(f"  operaciones por medición: mediana {ops[len(ops)//2]}, máx {ops[-1]}")
    mejor = max(rs, key=lambda r: r.test_lower)
    print(f"  mejor: {mejor.label}·trail{mejor.trail}")
    print(f"    ops={mejor.ops}  R med={mejor.avg_r:+.3f}  "
          f"val.ops={mejor.test_ops}  val.mín={mejor.test_lower:+.3f}\n")

bloque("LARGOS", largos); bloque("CORTOS", cortos)

print("── Los 8 mejores cortos por validación ──")
for r in sorted(cortos, key=lambda r: -r.test_lower)[:8]:
    print(f"  {r.name:40} ops={r.ops:4}  R med={r.avg_r:+.3f}  "
          f"val.ops={r.test_ops:3}  val.mín={r.test_lower:+.3f}")
