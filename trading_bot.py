#!/usr/bin/env python3
"""Punto de entrada del bot de señales de trading.

    python trading_bot.py auto        Lo que ejecuta el cron: comandos,
                                      seguimiento y escaneo si toca
    python trading_bot.py scan        Fuerza el escaneo ahora
    python trading_bot.py track       Solo revisa las operaciones abiertas
    python trading_bot.py poll        Solo atiende comandos de Telegram
    python trading_bot.py backtest    Mide la ventaja y calibra la confianza
    python trading_bot.py test        Envía un mensaje de prueba

El bot no se conecta a Quantfury —no existe API pública de trading—, así que
alerta y las órdenes las coloca el usuario a mano. Quantfury interviene como
restricción de universo: solo se recomienda lo que se puede comprar allí.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from trading import llm_filter, notify, universe
from trading.backtest import run_backtest
from trading.config import (
    CALIBRATION_FILE,
    DEFAULT_BACKTEST,
    DEFAULT_PARAMS,
    MAX_LLM_PENALTY,
    STATE_FILE,
    StrategyParams,
)
from trading.data import MarketData
from trading.risk import Calibration, apply_llm_penalty
from trading.schedule import should_scan
from trading.state import TradingState
from trading.strategy import Signal, scan as scan_universe
from trading.universe import BENCHMARK_SYMBOL, Instrument

TRADING_DAYS_PER_YEAR = 252


# ── Datos ─────────────────────────────────────────────────────────────────────

def _load_bars(
    instruments: list[Instrument], bars_needed: int, offline: bool = False
) -> tuple[dict[str, pd.DataFrame], pd.Series | None]:
    market = MarketData()
    if offline:
        market.cache_max_age_hours = float("inf")

    to_fetch = list(instruments)
    benchmark = universe.get(BENCHMARK_SYMBOL)
    if benchmark not in to_fetch:
        to_fetch.append(benchmark)

    bars = market.get_many(to_fetch, bars=bars_needed)
    benchmark_df = bars.get(BENCHMARK_SYMBOL)
    if benchmark_df is None:
        print("[WARN] Sin datos de SPY: la fuerza relativa queda fuera del cálculo")
        return bars, None
    return bars, benchmark_df["close"]


# ── Escaneo ───────────────────────────────────────────────────────────────────

def _deliver_signals(
    signals: list[Signal],
    state: TradingState,
    calibration: Calibration,
    client: notify.TelegramClient,
    params: StrategyParams,
    dry_run: bool,
    use_llm: bool,
) -> int:
    """Verifica, formatea y envía. Devuelve cuántas señales se entregaron."""
    # No se repite una señal sobre un instrumento que ya tiene operación
    # abierta: duplicaría la exposición sin que el usuario lo haya decidido.
    busy = state.open_symbols()
    candidates = [s for s in signals if s.symbol not in busy][
        : params.max_signals_per_scan
    ]
    if not candidates:
        return 0

    delivered = 0
    for signal in candidates:
        confidence, reliable = calibration.confidence_for(signal.score)

        risk = llm_filter.check(signal) if use_llm else None
        if risk is not None:
            if risk.is_blocking:
                print(f"[INFO] {signal.symbol} descartada: riesgo de evento alto")
                continue
            confidence = apply_llm_penalty(confidence, risk.penalty, MAX_LLM_PENALTY)

        message = notify.format_signal(signal, calibration, confidence, reliable, risk)

        if dry_run:
            print(message)
            print("-" * 55)
            delivered += 1
            continue

        if client.broadcast(message, state.chat_ids):
            state.record_signal(signal, confidence)
            delivered += 1
            time.sleep(1)  # respeta el límite de tasa de Telegram
        else:
            # No se registra lo que no se entregó: si el usuario nunca vio la
            # alerta, contarla en el rendimiento sería falsear el historial.
            print(f"[WARN] No se pudo entregar {signal.symbol}; no se registra")

    return delivered


def cmd_scan(args: argparse.Namespace) -> int:
    params = DEFAULT_PARAMS
    state = TradingState.load(STATE_FILE)
    client = notify.TelegramClient()

    if not args.force:
        allowed, reason = should_scan(last_scan_date=state.last_scan_date)
        if not allowed:
            print(f"[INFO] No toca escanear: {reason}")
            return 0
        print(f"[INFO] Escaneando: {reason}")

    instruments = list(universe.ALL_INSTRUMENTS)
    bars, benchmark = _load_bars(instruments, params.min_bars + 60, args.offline)
    if not bars:
        print("[ERROR] Sin datos de mercado")
        return 1

    liquid = universe.filter_by_liquidity(
        [i for i in instruments if i.symbol in bars], bars, params
    )
    print(f"[INFO] Superan el filtro de liquidez: {len(liquid)}/{len(bars)}")

    signals = scan_universe(liquid, bars, params, benchmark)
    calibration = Calibration.load(CALIBRATION_FILE)
    if not calibration.buckets:
        print(
            "[WARN] Sin calibration.json: las alertas saldrán marcadas como "
            "'sin calibrar'. Ejecuta el backtest."
        )

    if not signals:
        print("[INFO] Ninguna oportunidad cumple los criterios hoy")
        if not args.dry_run:
            client.broadcast(notify.format_no_signals(), state.chat_ids)
            state.mark_scanned(datetime.now(timezone.utc).astimezone().date())
            state.save(STATE_FILE)
        return 0

    print(f"[INFO] Señales detectadas: {len(signals)}")
    if not args.dry_run:
        shown = min(len(signals), params.max_signals_per_scan)
        client.broadcast(
            notify.format_scan_header(len(signals), shown), state.chat_ids
        )

    delivered = _deliver_signals(
        signals, state, calibration, client, params,
        dry_run=args.dry_run, use_llm=not args.no_llm,
    )
    print(f"[INFO] Señales entregadas: {delivered}")

    if not args.dry_run:
        from trading.schedule import now_ny

        state.mark_scanned(now_ny().date())
        state.save(STATE_FILE)
    return 0


# ── Seguimiento ───────────────────────────────────────────────────────────────

def cmd_track(args: argparse.Namespace) -> int:
    state = TradingState.load(STATE_FILE)
    if not state.open_signals:
        print("[INFO] No hay operaciones abiertas que seguir")
        return 0

    instruments = [universe.get(s) for s in state.open_symbols()]
    print(f"[INFO] Revisando {len(instruments)} operaciones abiertas")

    market = MarketData()
    if args.offline:
        market.cache_max_age_hours = float("inf")
    bars = market.get_many(instruments, bars=90)

    closed = state.update_open_signals(bars, DEFAULT_BACKTEST)
    if not closed:
        print(f"[INFO] Ninguna se resolvió; siguen abiertas {len(state.open_signals)}")
        state.save(STATE_FILE)
        return 0

    client = notify.TelegramClient()
    for record in closed:
        if record.get("status") == "expired":
            # La orden condicional caducó sin ejecutarse. No es una pérdida ni
            # merece una alerta: nunca se arriesgó dinero.
            print(f"[INFO] {record['symbol']}: orden caducada sin ejecutarse")
            continue
        print(f"[INFO] {record['symbol']} cerrada: {record.get('outcome')}")
        if not args.dry_run:
            client.broadcast(notify.format_outcome(record), state.chat_ids)

    if not args.dry_run:
        state.save(STATE_FILE)
    return 0


# ── Comandos de Telegram ──────────────────────────────────────────────────────

def _handle_command(command: str, chat_id: str, state: TradingState,
                    client: notify.TelegramClient) -> None:
    if command in ("/ayuda", "/start", "/help"):
        client.send(chat_id, notify.format_help())

    elif command in ("/rendimiento", "/stats"):
        client.send(chat_id, notify.format_performance(state.performance()))

    elif command in ("/abiertas", "/open"):
        if not state.open_signals:
            client.send(chat_id, "No hay operaciones abiertas.")
        else:
            lines = ["📌 *OPERACIONES ABIERTAS*", ""]
            for record in state.open_signals:
                inst = universe.get(record["symbol"])
                d = inst.price_decimals
                lines.append(
                    f"• *{record['symbol']}* — entrada ≤{record['entry_max']:.{d}f} · "
                    f"objetivo {record['target']:.{d}f} · stop {record['stop']:.{d}f}"
                )
            client.send(chat_id, "\n".join(lines))

    elif command in ("/senales", "/señales"):
        recent = state.history[-5:]
        if not recent:
            client.send(chat_id, "Todavía no hay señales cerradas.")
        else:
            lines = ["🗒️ *ÚLTIMAS SEÑALES CERRADAS*", ""]
            for record in reversed(recent):
                mark = "✅" if record.get("is_win") else "❌"
                ret = record.get("return_pct")
                tail = f" ({ret * 100:+.1f}%)" if ret is not None else ""
                lines.append(f"{mark} {record['symbol']} — {record.get('outcome')}{tail}")
            client.send(chat_id, "\n".join(lines))

    elif command in ("/instrumentos", "/universo"):
        client.send(
            chat_id,
            "🌐 *UNIVERSO VIGILADO*\n\n"
            f"• {len(universe.STOCKS)} acciones de NYSE/NASDAQ\n"
            f"• {len(universe.ETFS)} ETFs\n"
            f"• {len(universe.FOREX)} pares de divisas de Quantfury\n\n"
            "_Las acciones se filtran además por volumen real en dólares antes "
            "de cada escaneo._",
        )
    else:
        client.send(chat_id, "Comando desconocido. Usa /ayuda.")


def cmd_poll(args: argparse.Namespace) -> int:
    state = TradingState.load(STATE_FILE)
    client = notify.TelegramClient()
    if not client.is_configured:
        print("[ERROR] Falta TELEGRAM_BOT_TOKEN")
        return 1

    updates = client.get_updates(state.tg_offset)
    changed = False

    for update in updates:
        state.tg_offset = max(state.tg_offset, update["update_id"] + 1)
        changed = True

        message = update.get("message") or update.get("channel_post")
        if not message:
            continue

        chat_id = str(message.get("chat", {}).get("id", ""))
        text = (message.get("text") or "").strip()
        if state.register_chat(chat_id):
            changed = True
        if not text.startswith("/"):
            continue

        command = text.split()[0].lower().split("@")[0]
        print(f"[INFO] Comando '{command}' de {chat_id}")
        _handle_command(command, chat_id, state, client)

    if changed:
        state.save(STATE_FILE)
    return 0


# ── Backtest ──────────────────────────────────────────────────────────────────

def cmd_backtest(args: argparse.Namespace) -> int:
    params = DEFAULT_PARAMS
    bars_needed = args.years * TRADING_DAYS_PER_YEAR + params.min_bars
    instruments = list(universe.ALL_INSTRUMENTS)

    print(
        f"[INFO] Backtest sobre {len(instruments)} instrumentos, "
        f"{args.years} años ({bars_needed} velas por símbolo)"
    )
    bars, benchmark = _load_bars(instruments, bars_needed, args.offline)
    if not bars:
        print("[ERROR] No se descargó ningún dato. Configura TWELVEDATA_API_KEY.")
        return 1

    tradable = [i for i in instruments if i.symbol in bars]
    print(f"[INFO] Con datos utilizables: {len(tradable)}/{len(instruments)}")

    report = run_backtest(tradable, bars, params, DEFAULT_BACKTEST, benchmark)
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

    if report.filled and report.avg_r <= 0:
        print(
            "\n⚠️  R medio negativo: con estos parámetros la estrategia NO tiene "
            "ventaja. Hay que ajustarlos antes de fiarse de las alertas."
        )
    return 0


# ── Modos auxiliares ──────────────────────────────────────────────────────────

def cmd_test(args: argparse.Namespace) -> int:
    state = TradingState.load(STATE_FILE)
    client = notify.TelegramClient()
    if not client.is_configured:
        print("[ERROR] Falta TELEGRAM_BOT_TOKEN")
        return 1
    if not state.chat_ids:
        print("[ERROR] No hay chats registrados. Escribe /start al bot en Telegram.")
        return 1

    ok = client.broadcast(
        "✅ Bot de señales de trading activo.\nEscribe /ayuda para ver los comandos.",
        state.chat_ids,
    )
    print("[INFO] Mensaje entregado" if ok else "[ERROR] No se pudo entregar")
    return 0 if ok else 1


def cmd_auto(args: argparse.Namespace) -> int:
    """Lo que ejecuta el cron: atiende comandos, sigue las abiertas y escanea."""
    cmd_poll(args)
    cmd_track(args)
    return cmd_scan(args)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Bot de señales de trading")
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p, offline=True, dry=True):
        if offline:
            p.add_argument("--offline", action="store_true", help="Usa solo la caché")
        if dry:
            p.add_argument("--dry-run", action="store_true", help="No envía nada")
        return p

    p = common(sub.add_parser("auto", help="Modo del cron"))
    p.add_argument("--force", action="store_true", help="Ignora la ventana horaria")
    p.add_argument("--no-llm", action="store_true", help="Sin filtro de noticias")
    p.set_defaults(func=cmd_auto)

    p = common(sub.add_parser("scan", help="Escaneo del día"))
    p.add_argument("--force", action="store_true", help="Ignora la ventana horaria")
    p.add_argument("--no-llm", action="store_true", help="Sin filtro de noticias")
    p.set_defaults(func=cmd_scan)

    common(sub.add_parser("track", help="Revisa las operaciones abiertas")).set_defaults(
        func=cmd_track
    )
    common(sub.add_parser("poll", help="Atiende comandos"), dry=False).set_defaults(
        func=cmd_poll
    )
    sub.add_parser("test", help="Mensaje de prueba").set_defaults(func=cmd_test)

    p = sub.add_parser("backtest", help="Mide la ventaja y calibra")
    p.add_argument("--years", type=int, default=DEFAULT_BACKTEST.years)
    p.add_argument("--write", action="store_true", help="Guarda calibration.json")
    p.add_argument("--offline", action="store_true", help="Usa solo la caché")
    p.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    # Los modos comparten helpers que consultan estos flags aunque no todos los
    # definan; se rellenan para no tener que comprobar su existencia en cada uso.
    for flag, default in (
        ("offline", False), ("dry_run", False), ("force", False), ("no_llm", False),
    ):
        if not hasattr(args, flag):
            setattr(args, flag, default)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
