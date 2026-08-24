"""Asocia los ficheros que el harness guarda con el símbolo que se pidió.

Los resultados grandes de IBKR se escriben en disco con un nombre por marca de
tiempo, así que el emparejamiento va por orden de descarga. Se verifica con el
último cierre: si el número no es plausible, el símbolo está cruzado.
"""
import glob, json, os, shutil, sys

TOOLDIR = "/root/.claude/projects/-home-user-Quantfury-Alerts/0d113b71-831b-533a-97a0-833f7a6e3ad7/tool-results"

def pending():
    fs = sorted(glob.glob(f"{TOOLDIR}/*get_price_history*.txt"), key=os.path.getmtime)
    return [f for f in fs if os.path.getsize(f) > 50_000]

def claim(symbols):
    fs = pending()
    if len(fs) < len(symbols):
        sys.exit(f"faltan ficheros: {len(fs)} para {len(symbols)} símbolos")
    for sym, f in zip(symbols, fs[-len(symbols):]):
        d = json.load(open(f))
        shutil.move(f, f"data/ibkr/{sym}.json")
        print(f"{sym:6} {len(d['time']):5} barras  {d['time'][0][:10]} → {d['time'][-1][:10]}"
              f"  último cierre {d['close'][-1]}")

if __name__ == "__main__":
    claim(sys.argv[1:])
