"""Tests de niveles de operación y de la calibración de confianza.

La calibración es la pieza que hace que el porcentaje publicado signifique
algo. Estos tests defienden esa propiedad: que la confianza nunca se invente,
nunca suba por obra del LLM, y que castigue la falta de muestra.
"""

from __future__ import annotations

import json

import pytest

from trading.config import DEFAULT_PARAMS, replace
from trading.risk import (
    Calibration,
    OutcomeStats,
    compute_levels,
    mean_lower_bound,
    wilson_lower_bound,
)


# ── Intervalo de Wilson ───────────────────────────────────────────────────────

def test_wilson_with_no_samples_is_zero():
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_punishes_small_samples():
    """2 de 3 aciertos no puede publicarse como 67% de confianza."""
    small = wilson_lower_bound(2, 3)
    assert small < 0.35


def test_wilson_converges_with_more_evidence():
    """Con la misma proporción, más muestra estrecha el intervalo."""
    few = wilson_lower_bound(6, 10)
    many = wilson_lower_bound(600, 1000)
    assert few < many < 0.60


def test_wilson_never_reaches_certainty():
    assert wilson_lower_bound(50, 50) < 1.0


def test_wilson_is_always_a_lower_bound():
    for wins, total in [(1, 10), (5, 10), (9, 10), (30, 40), (99, 100)]:
        assert wilson_lower_bound(wins, total) <= wins / total


def test_wilson_rejects_impossible_input():
    with pytest.raises(ValueError):
        wilson_lower_bound(11, 10)


# ── Niveles de la operación ───────────────────────────────────────────────────

def test_levels_are_ordered_and_consistent():
    levels = compute_levels(close=100.0, atr=2.0, swing_low=96.0, params=DEFAULT_PARAMS)
    assert levels is not None and levels.is_valid
    assert levels.stop < levels.entry_max < levels.target
    # entry = 100 + 0.30*2 = 100.6
    # stop  = min(96 - 0.10*2, 100.6 - 0.80*2) = min(95.8, 99.0) = 95.8
    assert levels.entry_max == pytest.approx(100.6)
    assert levels.stop == pytest.approx(95.8)
    assert levels.target == pytest.approx(100.6 + 3.0 * 2.0)
    assert levels.risk_reward == pytest.approx(
        levels.reward_per_unit / levels.risk_per_unit
    )


def test_shallow_pullback_keeps_a_minimum_stop_distance():
    """Un mínimo pegado al precio no puede producir un stop de un céntimo.

    Sin distancia mínima, el stop saldría por puro ruido intradía y el ratio
    riesgo/beneficio parecería magnífico sobre el papel.
    """
    levels = compute_levels(close=100.0, atr=2.0, swing_low=100.5, params=DEFAULT_PARAMS)
    assert levels is not None
    # Manda la distancia mínima (0.80 ATR), no el mínimo estructural.
    assert levels.stop == pytest.approx(100.6 - 0.8 * 2.0)
    assert levels.risk_per_unit == pytest.approx(1.6)


def test_a_deep_pullback_ruins_the_risk_reward():
    """Un retroceso profundo aleja el stop y la señal acaba descartada por R:B.

    Es deliberado: a esa profundidad ya no es un descanso dentro de la
    tendencia, sino un posible cambio de tendencia.
    """
    shallow = compute_levels(100.0, 2.0, 99.0, DEFAULT_PARAMS)
    deep = compute_levels(100.0, 2.0, 90.0, DEFAULT_PARAMS)
    assert shallow is not None and deep is not None

    assert deep.stop < shallow.stop
    assert deep.risk_reward < shallow.risk_reward
    assert shallow.risk_reward >= DEFAULT_PARAMS.min_risk_reward
    assert deep.risk_reward < DEFAULT_PARAMS.min_risk_reward


def test_target_does_not_depend_on_nearby_highs():
    """El objetivo es puro ATR: la tesis es que el precio hace nuevos máximos."""
    levels = compute_levels(100.0, 2.0, 96.0, DEFAULT_PARAMS)
    assert levels is not None
    assert levels.reward_per_unit == pytest.approx(3.0 * 2.0)


@pytest.mark.parametrize(
    "close,atr,swing",
    [(0.0, 2.0, 96.0), (100.0, 0.0, 96.0), (float("nan"), 2.0, 96.0), (100.0, 2.0, float("nan"))],
)
def test_incoherent_inputs_produce_no_levels(close, atr, swing):
    assert compute_levels(close, atr, swing, DEFAULT_PARAMS) is None


def test_target_below_entry_produces_no_levels():
    params = replace(DEFAULT_PARAMS, target_atr_mult=0.0)
    assert compute_levels(100.0, 2.0, 96.0, params) is None


# ── Límite inferior de la media ───────────────────────────────────────────────

def test_mean_lower_bound_punishes_small_samples():
    """+0.50R sacado de 12 operaciones no puede publicarse como +0.50R."""
    values = [3.0, -1.0] * 6                       # media +1.0, n=12
    total, squares = sum(values), sum(v * v for v in values)
    assert mean_lower_bound(total, squares, len(values)) < 1.0


def test_mean_lower_bound_converges_with_evidence():
    few = mean_lower_bound(12.0, 60.0, 12)
    many = mean_lower_bound(1200.0, 6000.0, 1200)   # misma media, más muestra
    assert few < many


def test_mean_lower_bound_needs_two_samples():
    assert mean_lower_bound(5.0, 25.0, 1) == 0.0


def test_mean_lower_bound_of_constant_values_is_the_value():
    values = [2.0] * 50
    total, squares = sum(values), sum(v * v for v in values)
    assert mean_lower_bound(total, squares, 50) == pytest.approx(2.0)


# ── Calibración ───────────────────────────────────────────────────────────────

def outcomes(
    asset_class: str, score: float, wins: int, losses: int, win_r: float = 2.5
):
    """Genera (clase, puntuación, ¿ganó?, R) con pagos asimétricos realistas.

    Las ganadoras rinden `win_r` y las perdedoras -1R, que es la forma del
    sistema real: pocas aciertos, pero cada uno vale por varias pérdidas.
    """
    return [(asset_class, score, True, win_r)] * wins + [
        (asset_class, score, False, -1.0)
    ] * losses


def test_calibration_segments_by_asset_class():
    """Es lo que sí discrimina: el forex rinde muy distinto de las acciones."""
    data = outcomes("forex", 55.0, 30, 34) + outcomes("stock", 55.0, 160, 320)
    calibration = Calibration.from_outcomes(data)

    forex = calibration.for_asset_class("forex")
    stock = calibration.for_asset_class("stock")
    assert forex is not None and stock is not None
    assert forex.win_rate > stock.win_rate
    assert forex.expectancy_r > stock.expectancy_r


def test_an_unknown_asset_class_has_no_stats():
    calibration = Calibration.from_outcomes(outcomes("stock", 55.0, 10, 10))
    assert calibration.for_asset_class("crypto") is None


def test_uncalibrated_is_flagged():
    """Un bot recién instalado no ha demostrado nada y debe decirlo."""
    empty = Calibration.empty()
    assert not empty.is_calibrated
    assert empty.for_asset_class("stock") is None


def test_published_win_rate_is_below_the_raw_one():
    stats = Calibration.from_outcomes(
        outcomes("stock", 55.0, 8, 2)
    ).for_asset_class("stock")
    assert stats is not None
    assert stats.win_rate == pytest.approx(0.8)
    assert stats.win_rate_lower < 80.0
    assert not stats.is_reliable          # 10 operaciones no bastan


def test_enough_samples_makes_a_segment_reliable():
    stats = Calibration.from_outcomes(
        outcomes("stock", 55.0, 30, 20)
    ).for_asset_class("stock")
    assert stats is not None and stats.is_reliable


def test_a_low_win_rate_can_still_carry_an_edge():
    """El hallazgo central del primer backtest: 35% de acierto y aun así gana.

    Publicar solo el acierto haría leer esta señal como perdedora, cuando su
    esperanza es claramente positiva porque el objetivo está el doble de lejos
    que el stop.
    """
    # Las cifras del primer backtest real: 193 aciertos de 549 operaciones.
    stats = Calibration.from_outcomes(
        outcomes("stock", 55.0, 193, 356)
    ).for_asset_class("stock")
    assert stats is not None
    assert stats.win_rate < 0.40
    assert stats.expectancy_r > 0
    assert stats.has_edge


def test_a_thin_edge_does_not_survive_its_own_uncertainty():
    """Con pago de 2R exactos, un 35% de acierto ya no demuestra ventaja.

    La esperanza sale en +0.05R, positiva pero indistinguible de cero. Es
    justo el caso que `has_edge` existe para atrapar: una media favorable que
    no aguanta su propio intervalo de confianza.
    """
    stats = Calibration.from_outcomes(
        outcomes("stock", 55.0, 193, 356, win_r=2.0)
    ).for_asset_class("stock")
    assert stats is not None
    assert stats.expectancy_r > 0
    assert not stats.has_edge


def test_an_edge_that_does_not_survive_its_uncertainty_is_not_an_edge():
    stats = Calibration.from_outcomes(
        outcomes("forex", 55.0, 4, 7)
    ).for_asset_class("forex")
    assert stats is not None
    assert stats.expectancy_r > 0        # media positiva por casualidad
    assert not stats.has_edge            # pero no sobrevive al intervalo


# ── Diagnóstico: ¿ordena la puntuación? ───────────────────────────────────────

def test_a_monotonic_score_is_detected_as_ranking():
    data = (
        outcomes("stock", 55.0, 30, 70)
        + outcomes("stock", 65.0, 40, 60)
        + outcomes("stock", 75.0, 50, 50)
        + outcomes("stock", 85.0, 60, 40)
    )
    assert Calibration.from_outcomes(data).score_ranks_correctly()


def test_a_non_monotonic_score_is_rejected():
    """Lo que ocurrió de verdad: el tramo 60-70 acertaba menos que el 50-60."""
    data = (
        outcomes("stock", 55.0, 122, 223)   # 35.4%
        + outcomes("stock", 65.0, 57, 118)  # 32.6%, peor
    )
    assert not Calibration.from_outcomes(data).score_ranks_correctly()


def test_score_ranking_needs_two_reliable_buckets():
    assert not Calibration.from_outcomes(outcomes("stock", 55.0, 2, 1)).score_ranks_correctly()


# ── Persistencia ──────────────────────────────────────────────────────────────

def test_calibration_survives_a_save_load_round_trip(tmp_path):
    data = outcomes("stock", 55.0, 40, 60) + outcomes("forex", 75.0, 20, 15)
    original = Calibration.from_outcomes(data, notes="prueba")
    path = tmp_path / "calibration.json"
    original.save(str(path))

    restored = Calibration.load(str(path))
    assert restored.total_signals == original.total_signals
    assert restored.notes == "prueba"
    for name, before in original.by_asset_class.items():
        after = restored.for_asset_class(name)
        assert after is not None
        assert (after.samples, after.wins) == (before.samples, before.wins)
        assert after.expectancy_r == pytest.approx(before.expectancy_r)
        assert after.win_rate_lower == pytest.approx(before.win_rate_lower)


def test_corrupt_calibration_file_degrades_instead_of_crashing(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{ esto no es json", encoding="utf-8")
    restored = Calibration.load(str(path))
    assert not restored.is_calibrated
    assert restored.for_asset_class("stock") is None


def test_missing_calibration_file_is_empty(tmp_path):
    assert not Calibration.load(str(tmp_path / "no-existe.json")).is_calibrated


def test_saved_file_is_readable_json(tmp_path):
    path = tmp_path / "calibration.json"
    Calibration.from_outcomes(outcomes("stock", 65.0, 25, 15)).save(str(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["total_signals"] == 40
    assert raw["by_asset_class"]["stock"]["reliable"] is True
    assert "score_ranks_correctly" in raw


def test_summary_table_reports_both_views():
    calibration = Calibration.from_outcomes(
        outcomes("stock", 55.0, 40, 60) + outcomes("forex", 75.0, 20, 15)
    )
    table = calibration.summary_table()
    assert "stock" in table and "forex" in table
    assert "50-60" in table and "70-80" in table
    assert "NO ordena" in table or "ordena" in table


def test_summary_table_without_calibration():
    assert "Sin calibración" in Calibration.empty().summary_table()


def test_outcome_stats_with_zero_samples_is_safe():
    stats = OutcomeStats(label="vacío")
    assert stats.win_rate == 0.0
    assert stats.win_rate_lower == 0.0
    assert stats.expectancy_r == 0.0
    assert not stats.is_reliable
    assert not stats.has_edge


# La penalización del filtro de noticias se probaba aquí. Desapareció con la
# confianza que penalizaba: sin porcentaje que publicar no hay nada que restar.
# Lo que el filtro sí sigue decidiendo —retirar un candidato con un evento
# binario cerca— se prueba en test_llm_filter.py.


# ── Firma de la estrategia ────────────────────────────────────────────────────

def test_a_calibration_knows_which_strategy_produced_it():
    calibration = Calibration.from_outcomes(
        outcomes("stock", 55.0, 40, 60), signature=DEFAULT_PARAMS.signature
    )
    assert calibration.matches(DEFAULT_PARAMS)


def test_a_calibration_from_another_strategy_is_rejected():
    """Su acierto describe operaciones que esta estrategia ya no genera."""
    calibration = Calibration.from_outcomes(
        outcomes("stock", 55.0, 40, 60), signature=DEFAULT_PARAMS.signature
    )
    other = replace(DEFAULT_PARAMS, adx_min=30.0)
    assert not calibration.matches(other)


def test_a_calibration_without_signature_never_matches():
    """Las calibraciones anteriores a esta comprobación no pueden validarse."""
    assert not Calibration.from_outcomes(outcomes("stock", 55.0, 40, 60)).matches(
        DEFAULT_PARAMS
    )


def test_the_signature_survives_a_round_trip(tmp_path):
    path = tmp_path / "calibration.json"
    Calibration.from_outcomes(
        outcomes("stock", 55.0, 40, 60), signature=DEFAULT_PARAMS.signature
    ).save(str(path))
    assert Calibration.load(str(path)).matches(DEFAULT_PARAMS)


def test_cosmetic_changes_do_not_change_the_signature():
    """Solo debe cambiar con lo que altera qué señales se generan."""
    same = replace(DEFAULT_PARAMS, max_signals_per_scan=99)
    assert same.signature == DEFAULT_PARAMS.signature


@pytest.mark.parametrize(
    "change",
    [{"adx_min": 25.0}, {"target_atr_mult": 4.0}, {"use_market_regime_filter": False},
     {"min_risk_reward": 2.0}, {"resume_rsi_min": 55.0}],
)
def test_meaningful_changes_change_the_signature(change):
    assert replace(DEFAULT_PARAMS, **change).signature != DEFAULT_PARAMS.signature


def test_a_thin_edge_is_not_counted_as_an_edge():
    """Una ventaja positiva pero minúscula no es una ventaja utilizable.

    El caso real medido: media +0.130R con mínimo +0.005R en acciones. Aquí se
    reproduce con mínimo +0.028R — positivo, y aun así insuficiente.
    """
    stats = Calibration.from_outcomes(
        outcomes("stock", 55.0, 193, 356, win_r=2.3)
    ).for_asset_class("stock")
    assert stats is not None
    assert stats.expectancy_lower > 0
    assert not stats.has_edge
