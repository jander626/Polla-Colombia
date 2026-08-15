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
from trading.backtest import BacktestReport, Trade, run_backtest, simulate_signal
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

    # La calibración que alimenta las alertas se segmenta por clase de activo.
    stats = calibration.for_asset_class("stock")
    assert stats is not None
    assert stats.samples == 5 and stats.wins == 3

    # Aun con 3 de 5, el acierto publicado queda por debajo del crudo.
    assert 0.0 < stats.win_rate_lower < 100.0 * stats.win_rate

    # Los tramos de puntuación se conservan solo como diagnóstico.
    by_label = {b.label: b for b in calibration.by_score}
    assert by_label["80-100"].samples == 3 and by_label["80-100"].wins == 3
    assert by_label["50-60"].samples == 2 and by_label["50-60"].wins == 0


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


# ── Trailing: la palanca que mejora operaciones en vez de multiplicarlas ──────

def _trail_signal(entry_max=100.0, stop=95.0, target=115.0, atr=5.0):
    from trading.risk import Levels

    return Signal(
        symbol="AAPL", name="Apple", asset_class="stock",
        bar_date=pd.Timestamp("2024-01-01"), close=99.0, atr=atr, atr_pct=atr / 99.0,
        score=72.0, levels=Levels(entry_max, stop, target, 3.0, entry_max - stop,
                                  target - entry_max),
    )


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1e6] * len(rows)},
        index=pd.bdate_range("2024-01-01", periods=len(rows)),
    )


NO_COST = replace(DEFAULT_BACKTEST, round_trip_cost=0.0)
TRAIL = replace(NO_COST, trail_atr_mult=2.0)     # 2 ATR = 10 puntos


def test_trailing_off_changes_nothing():
    """El comportamiento en vivo es el de siempre; el trailing es opt-in."""
    bars = _bars([(99.0, 100.0, 98.0, 99.0), (99.0, 101.0, 98.0, 100.0),
                  (100.0, 110.0, 99.0, 109.0), (109.0, 110.0, 94.0, 95.0)])
    plain = simulate_signal(_trail_signal(), bars, NO_COST)
    assert plain.outcome == "loss"
    assert plain.exit_price == pytest.approx(95.0)


def test_trailing_turns_a_full_loss_into_a_small_one():
    """Es el efecto que se quiere medir: la que se da la vuelta cuesta menos.

    Sube a 110 (el trail pasa a 110 − 2 ATR = 100) y luego se desploma. Sin
    trailing sale en 95 (−1R); con trailing sale en 100, por encima de la
    entrada.
    """
    bars = _bars([(99.0, 100.0, 98.0, 99.0), (99.0, 101.0, 98.0, 100.0),
                  (100.0, 110.0, 99.0, 109.0), (109.0, 110.0, 94.0, 95.0)])
    trailed = simulate_signal(_trail_signal(), bars, TRAIL)

    assert trailed.exit_price == pytest.approx(100.0)
    assert trailed.r_multiple > simulate_signal(_trail_signal(), bars, NO_COST).r_multiple


def test_a_trailing_stop_above_entry_is_not_counted_as_a_loss():
    """Salir por el stop no es perder cuando el stop ya está sobre la entrada."""
    bars = _bars([(99.0, 100.0, 98.0, 99.0), (99.0, 101.0, 98.0, 100.0),
                  (100.0, 114.0, 99.0, 113.0), (113.0, 113.0, 100.0, 101.0)])
    trailed = simulate_signal(_trail_signal(), bars, TRAIL)

    assert trailed.exit_price > trailed.entry_price
    assert trailed.outcome == "win"
    assert trailed.is_win


def test_the_trailing_stop_never_goes_down():
    """Un stop que baja no es un stop: convertiría cada retroceso en más riesgo."""
    bars = _bars([(99.0, 100.0, 98.0, 99.0), (99.0, 101.0, 98.0, 100.0),
                  (100.0, 112.0, 99.0, 111.0),   # trail sube a 102
                  (111.0, 111.5, 103.0, 104.0),  # máximo menor: no debe bajar
                  (104.0, 105.0, 101.0, 101.5)]) # toca 102 -> sale ahí
    trailed = simulate_signal(_trail_signal(), bars, TRAIL)
    assert trailed.exit_price == pytest.approx(102.0)


def test_the_trail_does_not_use_todays_high_against_todays_low():
    """El fallo más fácil de cometer aquí, y el que inventaría rentabilidad.

    Dentro de una vela diaria no se sabe si el máximo llegó antes o después del
    mínimo. Si el stop de hoy se calculase con el máximo de hoy, la operación
    saldría en un nivel que todavía no existía cuando el precio pasó por ahí.

    La tercera vela sube a 120 y baja a 99 el MISMO día. Con mirada al futuro,
    su trail (120 − 2 ATR = 110) se compararía con su propio mínimo de 99 y
    cerraría la operación ahí. Honestamente, esa vela no cierra nada: el stop
    que rige durante ella se fijó con máximos anteriores.
    """
    bars = _bars([(99.0, 100.0, 98.0, 99.0),      # 0 señal
                  (99.0, 101.0, 98.0, 100.0),     # 1 entrada en 99
                  (100.0, 120.0, 99.0, 100.0),    # 2 máximo 120, mínimo 99
                  (115.0, 118.0, 112.0, 116.0)])  # 3 tranquila, sobre el trail
    trailed = simulate_signal(_trail_signal(target=200.0), bars, TRAIL)

    assert trailed.exit_date != bars.index[2], "el trail miró al futuro"
    assert trailed.exit_price != pytest.approx(110.0)


def test_letting_winners_run_ignores_the_fixed_target():
    """Cortar en 3 ATR renuncia a la cola derecha; esta variante la persigue."""
    rows = [(99.0, 100.0, 98.0, 99.0), (99.0, 101.0, 98.0, 100.0)]
    rows += [(100.0 + 5 * i, 106.0 + 5 * i, 99.0 + 5 * i, 105.0 + 5 * i) for i in range(8)]
    rows.append((145.0, 145.0, 120.0, 121.0))
    bars = _bars(rows)

    capped = simulate_signal(_trail_signal(), bars, TRAIL)
    running = simulate_signal(_trail_signal(), bars, replace(TRAIL, let_winners_run=True))

    assert capped.exit_price == pytest.approx(115.0)   # se corta en el objetivo
    assert running.exit_price > capped.exit_price      # el trail deja correr
    assert running.r_multiple > capped.r_multiple


def test_letting_winners_run_needs_a_trail_to_exit():
    """Sin trailing no habría con qué salir: la bandera no puede quedar suelta."""
    bars = _bars([(99.0, 100.0, 98.0, 99.0), (99.0, 101.0, 98.0, 100.0),
                  (100.0, 116.0, 99.0, 115.5)])
    trade = simulate_signal(
        _trail_signal(), bars, replace(NO_COST, let_winners_run=True)
    )
    assert trade.outcome == "win"
    assert trade.exit_price == pytest.approx(115.0)   # el objetivo sigue cerrando


# ── El veredicto de la partición temporal ────────────────────────────────────
# El informe del 14 de agosto firmó "✓ la ventaja aparece en los dos tramos"
# con el tramo reciente en límite inferior +0.000. El veredicto miraba la
# media. Es el mismo error que ya se corrigió en las alertas, en el sitio donde
# más caro sale: el que dice si conviene operar.

def _report_from_r(first_r: list[float], second_r: list[float]) -> BacktestReport:
    report = BacktestReport()
    dates = pd.bdate_range("2021-01-01", periods=len(first_r) + len(second_r))
    for i, r in enumerate([*first_r, *second_r]):
        report.trades.append(
            Trade(
                symbol="AAPL", asset_class="stock", signal_date=dates[i],
                score=60.0, outcome="win" if r > 0 else "loss",
                entry_date=dates[i], entry_price=100.0, exit_price=100.0 + r,
                r_multiple=r, return_pct=r / 100.0,
            )
        )
    return report


def test_a_positive_but_undemonstrated_second_half_is_not_a_pass():
    """Media positiva y ventaja demostrada no son lo mismo.

    Reproduce el caso real: segundo tramo con media claramente positiva pero
    tan disperso que su límite inferior se queda en cero.
    """
    rng = np.random.default_rng(7)
    solid = list(rng.normal(0.60, 0.8, 240))     # media alta, poca dispersión
    noisy = list(rng.normal(0.17, 3.0, 160))     # media positiva, límite ~0

    text = _report_from_r(solid, noisy).out_of_sample_split()

    assert "✗" in text
    assert "positiva y demostrada no son lo mismo" in text


def test_two_solid_halves_still_pass():
    """El umbral no puede rechazar una ventaja que sí está demostrada."""
    rng = np.random.default_rng(11)
    first = list(rng.normal(0.55, 0.8, 240))
    second = list(rng.normal(0.50, 0.8, 160))

    text = _report_from_r(first, second).out_of_sample_split()
    assert "✓ La ventaja aparece en los dos tramos" in text


def test_neither_half_demonstrating_says_so_with_both_numbers():
    rng = np.random.default_rng(13)
    flat = list(rng.normal(0.0, 1.5, 240))
    also_flat = list(rng.normal(0.0, 1.5, 160))

    text = _report_from_r(flat, also_flat).out_of_sample_split()
    assert "Ningún tramo demuestra ventaja" in text


# ── Búsqueda sistemática ──────────────────────────────────────────────────────
# Barrer 81 combinaciones sobre una muestra produce una ganadora aunque no haya
# nada que ganar. Estos tests protegen lo único que hace útil el barrido: que
# el ranking use el tramo que las combinaciones NO vieron, y que cuando nada
# funciona lo diga en vez de coronar a la menos mala.

def _tiny_grid():
    from trading.config import PREDICTIVE_WEIGHTS

    base = DEFAULT_PARAMS
    return {
        "rsi45·actual·adx20": base,
        "rsi50·predictivo·adx20": replace(
            base, pullback_rsi_max=50.0, resume_rsi_min=50.0,
            weights=PREDICTIVE_WEIGHTS,
        ),
    }


def test_the_search_measures_every_combination(uptrend_with_pullback):
    from trading.backtest import run_search

    bars = {"AAPL": uptrend_with_pullback}
    results = run_search(
        [STOCK], bars, _tiny_grid(), (0.0, 2.0), NO_COST, train_fraction=0.6
    )

    assert len(results) == 4                       # 2 estrategias × 2 salidas
    assert {r.trail for r in results} == {0.0, 2.0}
    assert len({r.name for r in results}) == 4


def test_generating_signals_once_per_strategy_gives_the_same_signals(
    uptrend_with_pullback,
):
    """La optimización no puede cambiar el resultado, solo el tiempo.

    El barrido genera las señales una vez por estrategia y las simula con cada
    salida. Si eso difiriese de medir cada combinación por separado, todo el
    ranking estaría midiendo otra cosa.
    """
    from trading.backtest import run_search

    bars = {"AAPL": uptrend_with_pullback}
    searched = run_search([STOCK], bars, _tiny_grid(), (0.0,), NO_COST)
    direct = run_backtest([STOCK], bars, DEFAULT_PARAMS, NO_COST)

    mine = next(r for r in searched if r.label == "rsi45·actual·adx20")
    assert mine.signals == direct.signals_generated
    assert mine.ops == direct.headline()["ops"]
    assert mine.avg_r == pytest.approx(direct.headline()["avg_r"], abs=1e-9)


def test_the_ranking_uses_the_validation_half_not_the_total():
    """Ordenar por el total sería ordenar por cuánto memorizó cada combinación."""
    from trading.backtest import SearchResult, format_search

    def result(label: str, total: float, validation: float) -> SearchResult:
        return SearchResult(
            label=label, trail=0.0, params=DEFAULT_PARAMS, bt=NO_COST,
            signals=500, years=5, ops=400, win_rate=0.4, avg_r=total,
            lower_r=total, max_dd=-20.0, train_ops=240, train_lower=total,
            test_ops=160, test_avg=validation, test_lower=validation,
        )

    text = format_search(
        [result("memorizo", 0.90, 0.01), result("aguanto", 0.30, 0.25)]
    )
    lines = [ln for ln in text.splitlines() if "·trail" in ln]
    assert lines[0].startswith("aguanto"), "el ranking premió al que memorizó"


def test_a_search_where_nothing_survives_says_so():
    """Con 81 intentos, que la mejor no pase el listón ES la respuesta."""
    from trading.backtest import SearchResult, format_search

    losers = [
        SearchResult(
            label=f"c{i}", trail=0.0, params=DEFAULT_PARAMS, bt=NO_COST,
            signals=500, years=5, ops=400, win_rate=0.35, avg_r=0.30,
            lower_r=0.10, max_dd=-30.0, train_ops=240, train_lower=0.20,
            test_ops=160, test_avg=0.05, test_lower=0.01,
        )
        for i in range(5)
    ]
    text = format_search(losers)

    assert "NINGUNA combinación demuestra ventaja" in text
    assert "afinarle" in text
    assert "Mejor por validación" not in text


def test_a_credible_winner_is_still_labelled_as_a_candidate():
    """Pasar el listón la hace candidata, no demostrada. Hay que decirlo."""
    from trading.backtest import SearchResult, format_search

    winner = SearchResult(
        label="ganadora", trail=2.5, params=DEFAULT_PARAMS, bt=NO_COST,
        signals=600, years=5, ops=500, win_rate=0.42, avg_r=0.40,
        lower_r=0.20, max_dd=-18.0, train_ops=300, train_lower=0.25,
        test_ops=200, test_avg=0.35, test_lower=0.18,
    )
    text = format_search([winner])

    assert "Mejor por validación" in text
    assert "candidata, no demostrada" in text
