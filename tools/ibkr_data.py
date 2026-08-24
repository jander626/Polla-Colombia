"""Carga las velas descargadas de IBKR para medir sin depender de la red.

El bot en producción NO usa esto: corre en GitHub Actions, donde no hay
conector de IBKR, y se queda con Twelve Data (`trading/data.py`). Esto existe
solo para poder medir desde una sesión, que es la diferencia entre comprobar
una hipótesis en minutos o por ronda de clic en el workflow.

Los ficheros vienen tal cual los deja el conector: arrays columnares con las
marcas de tiempo en UTC.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import date

import pandas as pd

DEFAULT_DIR = "data/ibkr"


def load_one(path: str, today: date | None = None) -> pd.DataFrame:
    raw = json.load(open(path, encoding="utf-8"))
    df = pd.DataFrame(
        {
            "open": raw["open"],
            "high": raw["high"],
            "low": raw["low"],
            "close": raw["close"],
            "volume": raw["volume"],
        },
        index=pd.to_datetime(raw["time"], utc=True).tz_localize(None).normalize(),
    ).astype(float)

    # La última vela es la sesión en curso y viene a medias: su volumen es una
    # fracción del habitual y su máximo/mínimo todavía pueden moverse. Usarla
    # metería en el backtest una barra que en vivo no existía todavía.
    hoy = pd.Timestamp(today or date.today())
    df = df[df.index < hoy]

    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Una vela sin precio o con precio cero no es un dato, es un hueco.
    return df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]


def load_all(directory: str = DEFAULT_DIR, today: date | None = None) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        symbol = os.path.splitext(os.path.basename(path))[0]
        bars[symbol] = load_one(path, today)
    return bars
