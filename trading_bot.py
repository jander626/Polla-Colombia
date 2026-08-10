#!/usr/bin/env python3
"""Punto de entrada del bot de señales de trading.

Modos disponibles en esta fase:

    python trading_bot.py backtest        Mide la ventaja de la estrategia y
                                          genera calibration.json
    python trading_bot.py scan            Escaneo de hoy, impreso en consola

El envío por Telegram y el filtro de noticias con Gemini llegan en la fase
siguiente, deliberadamente: no tiene sentido montar el reparto de alertas
antes de saber si las señales valen algo. `backtest` es la puerta de decisión.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from trading import universe
from trading.backtest import run_backtest
from trading.config import (
    CALIBRATION_FILE,
    DEFAULT_BACKTEST,
    DEFAULT_PARAMS,
    StrategyParams,
)
from trading.data import MarketData
from trading.risk import Calibration
from trading.strategy import Signal, scan
from trading.universe import BENCHMARK_SYMBOL, Instrument

TRADING_DAYS_PER_YEAR = 252


def _load_bars(
    instruments: list[Instrument], bars_needed: int, offline: bool
) -> tuple[dict[str, pd.DataFrame], pd.Series | None]:
    """Descarga (o lee de caché) las velas del universo y del benchmark."""
    market = MarketData()
    if offline:
        # Caché sin caducidad: permite iterar sobre el backtest sin gastar cuota.
        market.cache_max_age_hours = float("inf")

    benchmark_instrument = universe.get(BENCHMARK_SYMBOL)
    to_fetch = list(instruments)
    if benchmark_instrument not in to_fetch:
        to_fetch.append(benchmark_instrument)

    bars = market.get_many(to_fetch, bars=bars_needed)
    benchmark_df = bars.get(BENCHMARK_SYMBOL)
    benchmark_close = benchmark_df["close"] if benchmark_df is not None else None

    if benchmark_close is None:
        print(
            "[WARN] Sin datos de SPY: la fuerza relativa quedará fuera del "
            "cálculo y las puntuaciones no serán comparables con las de un "
            "escaneo normal."
        )
    return bars, benchmark_close


def _format_signal(signal: Signal, calibration: Calibration) -> str:
    instrument = universe.get(signal.symbol)
    decimals = instrument.price_decimals
    levels = signal.levels
    confidence, reliable = calibration.confidence_for(signal.score)

    if not reliable:
        confidence_text = (
            f"{confidence:.0f}% (sin calibrar — ejecuta el backtest)"
            if confidence == 0
            else f"{confidence:.0f}% (muestra escasa, poco fiable)"
        )
    else:
        confidence_text = f"{confidence:.0f}%"

    return (
        f"  {signal.symbol:<9} {instrument.name[:26]:<26} "
        f"puntuación {signal.score:>5.1f}\n"
        f"      entrada hasta {levels.entry_max:.{decimals}f}  |  "
        f"objetivo {levels.target:.{decimals}f}  |  "
        f"stop {levels.stop:.{decimals}f}\n"
        f"      riesgo/beneficio 1:{levels.risk_reward:.2f}  |  "
        f"confianza {confidence_text}"
    )


# ── Modos ─────────────────────────────────────────────────────────────────────

def cmd_backtest(args: argparse.Namespace) -> int:
    params: StrategyParams = DEFAULT_PARAMS
    bt = DEFAULT_BACKTEST
    bars_needed = args.years * TRADING_DAYS_PER_YEAR + params.min_bars

    instruments = list(universe.ALL_INSTRUMENTS)
    print(
        f"[INFO] Backtest sobre {len(instruments)} instrumentos, "
        f"{args.years} años ({bars_needed} velas por símbolo)"
    )

    bars, benchmark_close = _load_bars(instruments, bars_needed, args.offline)
    if not bars:
        print(
            "[ERROR] No se descargó ningún dato. Configura TWELVEDATA_API_KEY "
            "o ejecuta con --offline si ya tienes caché."
        )
        return 1

    tradable = [i for i in instruments if i.symbol in bars]
    print(f"[INFO] Con datos utilizables: {len(tradable)}/{len(instruments)}")

    report = run_backtest(tradable, bars, params, bt, benchmark_close)
    print()
    print(report.summary())

    calibration = report.to_calibration()
    print()
    print("── Calibración de confianza ───────────────────────────")
    print(calibration.summary_table())

    if args.write:
        calibration.save(CALIBRATION_FILE)
        print(f"\n[INFO] Calibración guardada en {CALIBRATION_FILE}")
    else:
        print("\n[INFO] Ejecuta con --write para guardar la calibración.")

    # La decisión de la fase 2, dicha sin adornos.
    if report.filled and report.avg_r <= 0:
        print(
            "\n⚠️  R medio negativo: con estos parámetros la estrategia NO tiene "
            "ventaja. Hay que ajustarlos antes de construir las alertas."
        )
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    params = DEFAULT_PARAMS
    instruments = list(universe.ALL_INSTRUMENTS)

    bars, benchmark_close = _load_bars(instruments, params.min_bars + 60, args.offline)
    if not bars:
        print("[ERROR] Sin datos de mercado.")
        return 1

    liquid = universe.filter_by_liquidity(
        [i for i in instruments if i.symbol in bars], bars, params
    )
    print(f"[INFO] Instrumentos que superan el filtro de liquidez: {len(liquid)}")

    signals = scan(liquid, bars, params, benchmark_close)
    calibration = Calibration.load(CALIBRATION_FILE)

    print()
    if not signals:
        print("Hoy no hay ninguna oportunidad que cumpla los criterios.")
        print("Eso es correcto: un bot que encuentra señales todos los días se las inventa.")
        return 0

    shown = signals[: params.max_signals_per_scan]
    print(f"Señales detectadas: {len(signals)} (se muestran las {len(shown)} mejores)\n")
    for signal in shown:
        print(_format_signal(signal, calibration))
        print()

    if not calibration.buckets:
        print(
            "⚠️  Sin calibration.json los porcentajes de confianza no significan "
            "nada. Ejecuta primero: python trading_bot.py backtest --write"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bot de señales de trading")
    sub = parser.add_subparsers(dest="mode", required=True)

    bt = sub.add_parser("backtest", help="Mide la ventaja y calibra la confianza")
    bt.add_argument("--years", type=int, default=DEFAULT_BACKTEST.years)
    bt.add_argument("--write", action="store_true", help="Guarda calibration.json")
    bt.add_argument("--offline", action="store_true", help="Usa solo la caché local")
    bt.set_defaults(func=cmd_backtest)

    sc = sub.add_parser("scan", help="Escaneo de hoy, impreso en consola")
    sc.add_argument("--offline", action="store_true", help="Usa solo la caché local")
    sc.set_defaults(func=cmd_scan)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
