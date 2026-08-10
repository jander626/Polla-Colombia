"""Tests del motor de backtest.

El backtest solo sirve si sus decisiones ambiguas caen en contra de la
estrategia. Estos tests fijan ese comportamiento por escrito: vela ambigua
contada como pérdida, huecos ejecutados al precio de apertura y no al nivel
teórico, y órdenes condicionales que caducan si el precio no acude.

Si alguien relaja alguna de estas reglas, el backtest empezará a mentir y la
calibración de confianza dejará de significar nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading import universe
from trading.backtest import BacktestReport, run_backtest, simulate_signal
from trading.config import DEFAULT_BACKTEST, DEFAULT_PARAMS, replace
from trading.risk import Levels
from trading.strategy import Signal

STOCK = universe.get("AAPL")

ENTRY_MAX, STOP, TARGET = 100.0, 95.0, 115.0
NO_COST = replace(DEFAULT_BACKTEST, round_trip_cost=0.0)


def make_signal(score: float = 70.0) -> Signal:
    levels = Levels(
        entry_max=ENTRY_MAX,
        stop=STOP,
        target=TARGET,
        risk_reward=(TARGET - ENTRY_MAX) / (ENTRY_MAX - STOP),
        risk_per_unit=ENTRY_MAX - STOP,
        reward_per_unit=TARGET - ENTRY_MAX,
    )
    return Signal(
        symbol=STOCK.symbol,
        name=STOCK.name,
        asset_class=STOCK.asset_class,
        bar_date=pd.Timestamp("2024-01-01"),
        close=99.0,
        atr=5.0,
        atr_pct=0.05,
        score=score,
        levels=levels,
    )


def make_df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Construye velas a partir de tuplas (open, high, low, close).

    La primera fila es la vela que generó la señal; la simulación empieza en
    la siguiente, igual que en vivo.
    """
    index = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1_000_000.0] * len(rows),
        },
        index=index,
    )


SIGNAL_BAR = (99.0, 100.0, 98.0, 99.0)


# ── Ejecución de la orden condicional ─────────────────────────────────────────

def test_fills_at_the_open_when_it_gaps_below_the_entry():
    df = make_df([SIGNAL_BAR, (97.0, 101.0, 96.5, 100.0), (100.0, 116.0, 99.0, 116.0)])
    trade = simulate_signal(make_signal(), df, NO_COST)

    assert trade.was_filled
    assert trade.entry_price == pytest.approx(97.0)
    assert trade.entry_date == df.index[1]


def test_fills_at_the_entry_ceiling_when_price_dips_intraday():
    """Abre por encima del techo pero baja a tocarlo: se compra en el techo."""
    df = make_df([SIGNAL_BAR, (105.0, 106.0, 99.0, 104.0), (104.0, 116.0, 103.0, 116.0)])
    trade = simulate_signal(make_signal(), df, NO_COST)

    assert trade.entry_price == pytest.approx(ENTRY_MAX)


def test_a_gap_above_the_entry_means_no_trade():
    """Si abre muy por encima, el precio ya no justifica la señal. No se persigue."""
    df = make_df(
        [SIGNAL_BAR, (108.0, 112.0, 107.0, 110.0), (110.0, 120.0, 109.0, 119.0)]
    )
    trade = simulate_signal(make_signal(), df, NO_COST)

    assert trade.outcome == "no_fill"
    assert not trade.was_filled
    assert not trade.is_win


def test_the_conditional_order_expires():
    """La orden solo vive `entry_valid_days`; después la señal caduca."""
    late = (99.0, 100.0, 90.0, 95.0)  # tocaría la entrada, pero llega tarde
    df = make_df(
        [SIGNAL_BAR, (105.0, 106.0, 104.0, 105.0), (105.0, 106.0, 104.0, 105.0), late]
    )
    params = replace(NO_COST, entry_valid_days=2)
    assert simulate_signal(make_signal(), df, params).outcome == "no_fill"


# ── Resolución de la operación ────────────────────────────────────────────────

def test_target_hit_is_a_win():
    df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (100.0, 116.0, 99.0, 115.5)])
    trade = simulate_signal(make_signal(), df, NO_COST)

    assert trade.outcome == "win" and trade.is_win
    assert trade.exit_price == pytest.approx(TARGET)
    # entrada 99, riesgo PLANIFICADO 5 (techo 100 - stop 95) -> (115-99)/5 = 3.2R
    assert trade.r_multiple == pytest.approx(3.2)


def test_stop_hit_is_a_loss():
    df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (100.0, 101.0, 94.0, 94.5)])
    trade = simulate_signal(make_signal(), df, NO_COST)

    assert trade.outcome == "loss" and not trade.is_win
    assert trade.exit_price == pytest.approx(STOP)
    # Entró en 99, por debajo del techo de 100: pierde 4 de los 5 planificados.
    assert trade.r_multiple == pytest.approx(-0.8)


def test_an_ambiguous_bar_counts_as_a_loss():
    """Stop y objetivo en la misma vela: sin datos intradía se asume lo peor."""
    df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (100.0, 116.0, 94.0, 100.0)])
    trade = simulate_signal(make_signal(), df, NO_COST)

    assert trade.outcome == "loss"
    assert trade.exit_price == pytest.approx(STOP)


def test_the_optimistic_reading_of_an_ambiguous_bar_is_opt_in():
    df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (100.0, 116.0, 94.0, 100.0)])
    optimistic = replace(NO_COST, ambiguous_bar_is_loss=False)
    assert simulate_signal(make_signal(), df, optimistic).outcome == "win"


def test_a_gap_down_exits_worse_than_the_stop():
    """Un hueco bajista no respeta el stop: se sale en la apertura, y duele más."""
    df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (90.0, 92.0, 89.0, 91.0)])
    trade = simulate_signal(make_signal(), df, NO_COST)

    assert trade.outcome == "loss"
    assert trade.exit_price == pytest.approx(90.0)
    assert trade.r_multiple < -1.0  # el hueco hace perder más de lo planificado


def test_a_gap_up_exits_better_than_the_target():
    df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (120.0, 122.0, 119.0, 121.0)])
    trade = simulate_signal(make_signal(), df, NO_COST)

    assert trade.outcome == "win"
    assert trade.exit_price == pytest.approx(120.0)
    assert trade.r_multiple > 3.2  # el hueco al alza da más de lo planificado


def test_a_position_that_goes_nowhere_times_out():
    quiet = [(99.0, 100.5, 98.5, 99.5)] * 10
    df = make_df([SIGNAL_BAR, (99.0, 100.0, 98.0, 99.0), *quiet])
    params = replace(NO_COST, max_holding_days=5)
    trade = simulate_signal(make_signal(), df, params)

    assert trade.outcome == "timeout"
    assert trade.bars_held == 5


def test_stop_is_checked_on_the_entry_bar_itself():
    """Comprar en la apertura y desplomarse el mismo día es una pérdida real."""
    df = make_df([SIGNAL_BAR, (99.0, 100.0, 93.0, 94.0), (94.0, 95.0, 93.0, 94.0)])
    trade = simulate_signal(make_signal(), df, NO_COST)

    assert trade.outcome == "loss"
    assert trade.entry_date == trade.exit_date


def test_running_out_of_bars_does_not_crash():
    df = make_df([SIGNAL_BAR, (99.0, 100.0, 98.0, 99.5)])
    trade = simulate_signal(make_signal(), df, NO_COST)
    assert trade.outcome == "timeout"


def test_a_signal_whose_bar_is_missing_is_skipped():
    df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0)])
    orphan = make_signal()
    object.__setattr__(orphan, "bar_date", pd.Timestamp("1999-01-01"))
    assert simulate_signal(orphan, df, NO_COST).outcome == "no_fill"


# ── Costes ────────────────────────────────────────────────────────────────────

def test_trading_costs_are_deducted_from_the_return():
    df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (100.0, 116.0, 99.0, 115.5)])
    free = simulate_signal(make_signal(), df, NO_COST)
    costly = simulate_signal(
        make_signal(), df, replace(DEFAULT_BACKTEST, round_trip_cost=0.01)
    )

    assert costly.return_pct == pytest.approx(free.return_pct - 0.01)


def test_costs_can_turn_a_marginal_win_into_a_loss():
    """Una ganancia por debajo del spread no es una ganancia."""
    tiny = Levels(
        entry_max=100.0, stop=95.0, target=100.2,
        risk_reward=0.04, risk_per_unit=5.0, reward_per_unit=0.2,
    )
    signal = make_signal()
    object.__setattr__(signal, "levels", tiny)

    df = make_df([SIGNAL_BAR, (100.0, 100.5, 99.0, 100.3), (100.3, 101.0, 100.0, 100.5)])
    trade = simulate_signal(signal, df, replace(DEFAULT_BACKTEST, round_trip_cost=0.01))

    assert trade.outcome == "win"       # tocó el objetivo
    assert not trade.is_win             # pero perdió dinero tras costes


# ── Informe y calibración ─────────────────────────────────────────────────────

def test_report_metrics_are_consistent():
    win_df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (100.0, 116.0, 99.0, 115.5)])
    loss_df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (100.0, 101.0, 94.0, 94.5)])

    report = BacktestReport(
        trades=[
            simulate_signal(make_signal(85.0), win_df, NO_COST),
            simulate_signal(make_signal(85.0), win_df, NO_COST),
            simulate_signal(make_signal(55.0), loss_df, NO_COST),
        ],
        signals_generated=3,
    )

    assert len(report.filled) == 3
    assert len(report.wins) == 2
    assert report.win_rate == pytest.approx(2 / 3)
    assert report.avg_r == pytest.approx((3.2 + 3.2 - 0.8) / 3)
    assert report.profit_factor > 1.0
    assert report.max_drawdown_r <= 0.0
    assert "Sesgo de supervivencia" not in report.summary()  # las notas las pone run_backtest


def test_report_with_no_fills_reports_it_plainly():
    df = make_df([SIGNAL_BAR, (108.0, 112.0, 107.0, 110.0), (110.0, 120.0, 109.0, 119.0)])
    report = BacktestReport(
        trades=[simulate_signal(make_signal(), df, NO_COST)], signals_generated=1
    )
    assert report.filled == []
    assert "Ninguna llegó a ejecutarse" in report.summary()


def test_calibration_is_built_from_actual_outcomes():
    win_df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (100.0, 116.0, 99.0, 115.5)])
    loss_df = make_df([SIGNAL_BAR, (99.0, 101.0, 98.0, 100.0), (100.0, 101.0, 94.0, 94.5)])

    report = BacktestReport(
        trades=(
            [simulate_signal(make_signal(85.0), win_df, NO_COST) for _ in range(3)]
            + [simulate_signal(make_signal(55.0), loss_df, NO_COST) for _ in range(2)]
        ),
        signals_generated=5,
    )
    calibration = report.to_calibration()

    top = calibration.bucket_for(85.0)
    bottom = calibration.bucket_for(55.0)
    assert top is not None and top.samples == 3 and top.wins == 3
    assert bottom is not None and bottom.samples == 2 and bottom.wins == 0
    # Aun con 3 de 3, la confianza publicada no llega al 100%.
    assert 0.0 < top.confidence < 100.0


# ── Extremo a extremo ─────────────────────────────────────────────────────────

def test_end_to_end_backtest_produces_trades_and_warnings(uptrend_with_pullback):
    report = run_backtest(
        [STOCK], {"AAPL": uptrend_with_pullback}, DEFAULT_PARAMS, DEFAULT_BACKTEST
    )

    assert report.signals_generated > 0
    assert len(report.trades) == report.signals_generated
    assert any("Sesgo de supervivencia" in note for note in report.notes)
    assert "Resultado del backtest" in report.summary() or report.filled == []


def test_backtest_skips_instruments_without_enough_history(uptrend_with_pullback):
    short = uptrend_with_pullback.tail(50)
    report = run_backtest([STOCK], {"AAPL": short}, DEFAULT_PARAMS, DEFAULT_BACKTEST)
    assert report.signals_generated == 0


def test_backtest_survives_corrupt_data(uptrend_with_pullback):
    broken = uptrend_with_pullback.copy()
    broken["high"] = np.nan
    report = run_backtest(
        [STOCK, universe.get("MSFT")],
        {"AAPL": broken, "MSFT": uptrend_with_pullback},
        DEFAULT_PARAMS,
        DEFAULT_BACKTEST,
    )
    assert report.signals_generated > 0
