"""Tests de niveles de operación y de la calibración de confianza.

La calibración es la pieza que hace que el porcentaje publicado signifique
algo. Estos tests defienden esa propiedad: que la confianza nunca se invente,
nunca suba por obra del LLM, y que castigue la falta de muestra.
"""

from __future__ import annotations

import json

import pytest

from trading.config import DEFAULT_PARAMS, MAX_LLM_PENALTY, replace
from trading.risk import (
    BucketStats,
    Calibration,
    apply_llm_penalty,
    compute_levels,
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


# ── Calibración ───────────────────────────────────────────────────────────────

def test_outcomes_land_in_the_right_buckets():
    outcomes = [(55.0, True), (55.0, False), (75.0, True), (85.0, True)]
    calibration = Calibration.from_results(outcomes)

    low = calibration.bucket_for(55.0)
    assert low is not None and low.samples == 2 and low.wins == 1
    assert low.win_rate == pytest.approx(0.5)

    high = calibration.bucket_for(85.0)
    assert high is not None and high.samples == 1 and high.wins == 1


def test_uncalibrated_confidence_is_zero_and_flagged():
    """Un bot recién instalado no ha demostrado nada y debe decirlo."""
    confidence, reliable = Calibration.empty().confidence_for(75.0)
    assert confidence == 0.0
    assert reliable is False


def test_confidence_is_below_raw_win_rate():
    outcomes = [(75.0, True)] * 8 + [(75.0, False)] * 2  # 80% crudo, n=10
    bucket = Calibration.from_results(outcomes).bucket_for(75.0)
    assert bucket is not None
    assert bucket.win_rate == pytest.approx(0.8)
    assert bucket.confidence < 80.0
    assert bucket.is_reliable is False  # 10 muestras no bastan


def test_bucket_becomes_reliable_with_enough_samples():
    outcomes = [(65.0, True)] * 30 + [(65.0, False)] * 20
    bucket = Calibration.from_results(outcomes).bucket_for(65.0)
    assert bucket is not None and bucket.is_reliable


def test_calibration_survives_a_save_load_round_trip(tmp_path):
    outcomes = [(55.0, True), (65.0, False), (75.0, True), (85.0, True)]
    original = Calibration.from_results(outcomes, notes="prueba")
    path = tmp_path / "calibration.json"
    original.save(str(path))

    restored = Calibration.load(str(path))
    assert restored.total_signals == original.total_signals
    assert restored.notes == "prueba"
    for before, after in zip(original.buckets, restored.buckets):
        assert (before.samples, before.wins) == (after.samples, after.wins)
        assert before.confidence == pytest.approx(after.confidence)


def test_corrupt_calibration_file_degrades_instead_of_crashing(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{ esto no es json", encoding="utf-8")
    restored = Calibration.load(str(path))
    assert restored.buckets == []
    assert restored.confidence_for(75.0) == (0.0, False)


def test_missing_calibration_file_is_empty(tmp_path):
    assert Calibration.load(str(tmp_path / "no-existe.json")).buckets == []


def test_summary_table_reports_every_bucket():
    calibration = Calibration.from_results([(55.0, True), (85.0, False)])
    table = calibration.summary_table()
    assert "50-60" in table and "80-100" in table


def test_saved_file_is_readable_json(tmp_path):
    path = tmp_path / "calibration.json"
    Calibration.from_results([(65.0, True)] * 40).save(str(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["total_signals"] == 40
    assert any(b["reliable"] for b in raw["buckets"])


# ── Penalización del filtro de noticias ───────────────────────────────────────

def test_llm_can_only_subtract_confidence():
    """Dejar que el LLM sume reintroduciría el número inventado."""
    assert apply_llm_penalty(70.0, -30.0, MAX_LLM_PENALTY) == 70.0


def test_llm_penalty_is_capped():
    assert apply_llm_penalty(70.0, 999.0, MAX_LLM_PENALTY) == 70.0 - MAX_LLM_PENALTY


def test_confidence_never_goes_negative():
    assert apply_llm_penalty(5.0, 25.0, MAX_LLM_PENALTY) == 0.0


def test_bucket_stats_with_zero_samples_is_safe():
    bucket = BucketStats(low=50.0, high=60.0, samples=0, wins=0)
    assert bucket.win_rate == 0.0
    assert bucket.confidence == 0.0
    assert bucket.is_reliable is False
