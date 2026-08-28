"""Tests del estado y del seguimiento de señales.

El seguimiento es el único control externo que tiene la calibración: compara
lo que el bot prometió con lo que de verdad ocurrió. Estos tests protegen que
esa comparación sea limpia — en particular, que las órdenes que caducan sin
ejecutarse no cuenten como pérdidas, porque nunca se arriesgó dinero en ellas.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading import universe
from trading.config import DEFAULT_BACKTEST, replace
from trading.risk import Levels
from trading.state import TradingState, evaluate_record, signal_from_record
from trading.strategy import Signal

BT = replace(DEFAULT_BACKTEST, round_trip_cost=0.0)


def make_record(symbol: str = "AAPL") -> dict:
    return {
        "symbol": symbol,
        "asset_class": "stock",
        "signal_date": "2024-01-01",
        "score": 72.0,
        "confidence": 61.0,
        "entry_max": 100.0,
        "stop": 95.0,
        "target": 115.0,
        "risk_reward": 3.0,
    }


def make_df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1e6] * len(rows),
        },
        index=pd.bdate_range("2024-01-01", periods=len(rows)),
    )


SIGNAL_BAR = (99.0, 100.0, 98.0, 99.0)
FILL_BAR = (99.0, 101.0, 98.0, 100.0)


# ── Persistencia ──────────────────────────────────────────────────────────────

def test_state_survives_a_round_trip(tmp_path):
    path = str(tmp_path / "estado.json")
    state = TradingState(chat_ids=["123"], tg_offset=42)
    state.open_signals.append(make_record())
    state.save(path)

    restored = TradingState.load(path)
    assert restored.chat_ids == ["123"]
    assert restored.tg_offset == 42
    assert restored.open_signals[0]["symbol"] == "AAPL"


def test_a_missing_state_file_starts_empty(tmp_path):
    state = TradingState.load(str(tmp_path / "no-existe.json"))
    assert state.chat_ids == [] and state.open_signals == []


def test_a_corrupt_state_does_not_block_the_bot(tmp_path):
    """Un estado ilegible no puede dejar al bot mudo."""
    path = tmp_path / "estado.json"
    path.write_text("{roto", encoding="utf-8")
    assert TradingState.load(str(path)).chat_ids == []


def test_chats_are_registered_once():
    state = TradingState()
    assert state.register_chat("555") is True
    assert state.register_chat("555") is False
    assert state.chat_ids == ["555"]


# ── Deduplicación del escaneo ─────────────────────────────────────────────────

def test_marking_a_scan_prevents_repeating_it():
    from datetime import date

    state = TradingState()
    day = date(2026, 8, 12)
    assert not state.already_scanned(day)
    state.mark_scanned(day)
    assert state.already_scanned(day)
    assert not state.already_scanned(date(2026, 8, 13))


def test_marking_a_track_prevents_repeating_it():
    """El 27 de agosto de 2026 el cron disparó `auto` (y con él, track) hasta
    diecisiete veces contra las mismas velas diarias: gastaba cupo de Twelve
    Data sin aprender nada nuevo entre un disparo y el siguiente del mismo
    día. Este es el mismo corte que ya protegía al escaneo, aplicado a track.
    """
    from datetime import date

    state = TradingState()
    day = date(2026, 8, 27)
    assert not state.already_tracked(day)
    state.mark_tracked(day)
    assert state.already_tracked(day)
    assert not state.already_tracked(date(2026, 8, 28))


def test_scan_and_track_dedup_are_independent():
    """Marcar uno no puede marcar el otro: son verificaciones distintas."""
    from datetime import date

    state = TradingState()
    day = date(2026, 8, 27)
    state.mark_scanned(day)
    assert state.already_scanned(day) and not state.already_tracked(day)


# ── Resolución ────────────────────────────────────────────────────────────────

def test_a_signal_with_no_history_yet_is_pending():
    df = make_df([SIGNAL_BAR])
    status, trade = evaluate_record(make_record(), df, BT)
    assert status == "pending" and trade is None


def test_a_filled_but_unresolved_signal_stays_open():
    df = make_df([SIGNAL_BAR, FILL_BAR, (100.0, 101.0, 99.0, 100.0)])
    status, trade = evaluate_record(make_record(), df, BT)
    assert status == "open" and trade is None


def test_a_target_hit_closes_as_a_win():
    df = make_df([SIGNAL_BAR, FILL_BAR, (100.0, 116.0, 99.0, 115.5)])
    status, trade = evaluate_record(make_record(), df, BT)
    assert status == "closed"
    assert trade is not None and trade.outcome == "win" and trade.is_win


def test_a_stop_hit_closes_as_a_loss():
    df = make_df([SIGNAL_BAR, FILL_BAR, (100.0, 101.0, 94.0, 94.5)])
    status, trade = evaluate_record(make_record(), df, BT)
    assert status == "closed"
    assert trade is not None and trade.outcome == "loss" and not trade.is_win


def test_an_untouched_order_expires_instead_of_losing():
    """Caducar sin ejecutarse no es perder: nunca se arriesgó dinero."""
    above = (108.0, 112.0, 107.0, 110.0)
    df = make_df([SIGNAL_BAR, above, above, above])
    status, trade = evaluate_record(make_record(), df, replace(BT, entry_valid_days=2))
    assert status == "expired"


def test_a_record_whose_bar_is_missing_stays_pending():
    df = make_df([SIGNAL_BAR, FILL_BAR])
    record = make_record()
    record["signal_date"] = "1999-01-01"
    status, _ = evaluate_record(record, df, BT)
    assert status == "pending"


def test_signal_is_rebuilt_faithfully_from_its_record():
    signal = signal_from_record(make_record())
    assert isinstance(signal, Signal)
    assert signal.levels.entry_max == 100.0
    assert signal.levels.stop == 95.0
    assert signal.levels.target == 115.0
    assert signal.levels.risk_reward == pytest.approx(3.0)


# ── Actualización en bloque ───────────────────────────────────────────────────

def test_update_moves_closed_signals_to_history():
    state = TradingState()
    state.open_signals.append(make_record("AAPL"))
    bars = {"AAPL": make_df([SIGNAL_BAR, FILL_BAR, (100.0, 116.0, 99.0, 115.5)])}

    closed = state.update_open_signals(bars, BT)

    assert len(closed) == 1
    assert state.open_signals == []
    assert state.history[0]["outcome"] == "win"


def test_a_symbol_without_data_keeps_its_signal_open():
    """Sin datos hoy se reintenta mañana; no se cierra a ciegas."""
    state = TradingState()
    state.open_signals.append(make_record("AAPL"))
    closed = state.update_open_signals({}, BT)
    assert closed == [] and len(state.open_signals) == 1


def test_open_symbols_prevents_duplicate_exposure():
    state = TradingState()
    state.open_signals.append(make_record("AAPL"))
    assert state.open_symbols() == {"AAPL"}


# ── Rendimiento ───────────────────────────────────────────────────────────────

def test_performance_without_closed_trades():
    stats = TradingState().performance()
    assert stats["closed"] == 0


def test_performance_counts_only_executed_trades():
    """Las órdenes caducadas no deben ensuciar el porcentaje de acierto."""
    state = TradingState()
    state.history = [
        {"status": "closed", "is_win": True, "r_multiple": 3.0, "confidence": 60.0},
        {"status": "closed", "is_win": False, "r_multiple": -1.0, "confidence": 60.0},
        {"status": "expired"},
        {"status": "expired"},
    ]
    stats = state.performance()

    assert stats["closed"] == 2
    assert stats["wins"] == 1
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["avg_r"] == pytest.approx(1.0)
    assert stats["total_r"] == pytest.approx(2.0)
    assert stats["expired"] == 2
    assert stats["avg_confidence"] == pytest.approx(60.0)


def test_recording_a_signal_stores_what_was_promised():
    """Sin guardar la confianza prometida no se podría auditar la calibración."""
    state = TradingState()
    instrument = universe.get("AAPL")
    signal = Signal(
        symbol="AAPL",
        name=instrument.name,
        asset_class="stock",
        bar_date=pd.Timestamp("2026-08-07"),
        close=99.0,
        atr=5.0,
        atr_pct=0.05,
        score=72.0,
        levels=Levels(100.0, 95.0, 115.0, 3.0, 5.0, 15.0),
    )
    record = state.record_signal(signal, confidence=64.0)

    assert record["confidence"] == 64.0
    assert record["score"] == 72.0
    assert record["signal_date"] == "2026-08-07"
    assert state.open_signals == [record]


def test_state_json_never_contains_nan(tmp_path):
    """NaN no es JSON válido y rompería la lectura del estado en la siguiente corrida."""
    import json

    state = TradingState()
    state.open_signals.append(make_record("AAPL"))
    # Una vela sin objetivo ni stop tocados durante todo el plazo -> timeout.
    quiet = [(100.0, 100.5, 99.5, 100.0)] * 40
    state.update_open_signals({"AAPL": make_df([SIGNAL_BAR, FILL_BAR, *quiet])}, BT)

    path = str(tmp_path / "estado.json")
    state.save(path)
    raw = path and open(path, encoding="utf-8").read()
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)  # debe poder releerse


# ── Preparación de la entrega (trading_bot) ──────────────────────────────────

def test_an_open_symbol_is_not_alerted_again():
    """Repetir una señal sobre lo que ya tienes abierto duplicaría exposición."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import trading_bot
    from trading.risk import Calibration
    from trading.strategy import Signal
    from trading.config import DEFAULT_PARAMS

    instrument = universe.get("AAPL")
    signal = Signal(
        symbol="AAPL", name=instrument.name, asset_class="stock",
        bar_date=pd.Timestamp("2026-08-10"), close=99.0, atr=5.0, atr_pct=0.05,
        score=72.0, levels=Levels(100.0, 95.0, 115.0, 3.0, 5.0, 15.0),
    )

    state = TradingState()
    state.open_signals.append({**make_record("AAPL")})

    delivery = trading_bot._prepare_delivery(
        [signal], state, Calibration.empty(), DEFAULT_PARAMS,
        use_llm=False, stale_calibration=False,
    )

    assert delivery.total_found == 1
    assert not delivery.has_messages          # no se envía
    assert delivery.already_open == ["AAPL"]  # y se sabe por qué


def test_a_free_symbol_is_prepared_for_delivery():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import trading_bot
    from trading.risk import Calibration
    from trading.strategy import Signal
    from trading.config import DEFAULT_PARAMS

    instrument = universe.get("MSFT")
    signal = Signal(
        symbol="MSFT", name=instrument.name, asset_class="stock",
        bar_date=pd.Timestamp("2026-08-10"), close=99.0, atr=5.0, atr_pct=0.05,
        score=72.0, levels=Levels(100.0, 95.0, 115.0, 3.0, 5.0, 15.0),
    )

    state = TradingState()
    state.open_signals.append({**make_record("AAPL")})   # otra posición abierta

    delivery = trading_bot._prepare_delivery(
        [signal], state, Calibration.empty(), DEFAULT_PARAMS,
        use_llm=False, stale_calibration=False,
    )

    assert delivery.has_messages
    assert delivery.already_open == []
    assert delivery.messages[0][0].symbol == "MSFT"


# ── cmd_track no repite la consulta de mercado el mismo día ─────────────────

def test_cmd_track_only_hits_the_market_once_per_day(tmp_path, monkeypatch):
    """Reproduce el 27 de agosto: el cron dispara `auto` varias veces en el
    mismo día y cada una ejecutaba `track` contra las mismas velas diarias.
    Con el corte, la segunda llamada del mismo día no debe tocar el mercado.
    """
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import trading_bot
    from trading.data import MarketData

    state_path = str(tmp_path / "estado.json")
    state = TradingState(open_signals=[make_record("AAPL")])
    state.save(state_path)
    monkeypatch.setattr(trading_bot, "STATE_FILE", state_path)

    llamadas = []
    monkeypatch.setattr(
        MarketData, "get_many",
        lambda self, instruments, bars=90: (llamadas.append(1) or {})
    )

    args = argparse.Namespace(offline=True, dry_run=False, force=False)

    trading_bot.cmd_track(args)   # primer disparo del día: sí consulta
    assert len(llamadas) == 1

    trading_bot.cmd_track(args)   # segundo disparo, mismo día: no debería
    assert len(llamadas) == 1, "no debe volver a tocar el mercado el mismo día"

    trading_bot.cmd_track(argparse.Namespace(offline=True, dry_run=False, force=True))
    assert len(llamadas) == 2, "--force sí debe saltarse el corte"
