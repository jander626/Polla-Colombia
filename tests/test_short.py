"""Tests del lado corto: que sea el espejo exacto del largo, y nada más.

El riesgo de añadir cortos no es que no funcionen —eso lo dirá la medición—,
sino que un signo mal puesto haga que el backtest mienta sin que se note. Un
corto con el stop del lado equivocado produce números plausibles y falsos, que
es la peor clase de error que puede tener este proyecto.

La defensa principal es una propiedad, no una lista de casos: **reflejar la
serie de precios y la señal tiene que dar exactamente el mismo resultado**. Si
`precio' = K - precio`, entonces un largo sobre la serie original y un corto
sobre la reflejada son la misma operación vista al revés, y su R debe coincidir
hasta el último decimal. Un solo `<` girado rompe esa igualdad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.backtest import simulate_signal
from trading.config import DEFAULT_BACKTEST, DEFAULT_PARAMS, replace
from trading.risk import Levels, compute_levels
from trading.strategy import Signal

BT = replace(DEFAULT_BACKTEST, round_trip_cost=0.0)
LARGO = DEFAULT_PARAMS
CORTO = replace(DEFAULT_PARAMS, direction="short")

# Nivel de reflexión. Cualquiera vale mientras deje todos los precios
# positivos; 300 sobra para las series de estos tests.
K = 300.0


def bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
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


def reflect(df: pd.DataFrame) -> pd.DataFrame:
    """Refleja las velas: `p' = K - p`. Máximos y mínimos se intercambian."""
    out = df.copy()
    out["open"] = K - df["open"]
    out["close"] = K - df["close"]
    out["high"] = K - df["low"]
    out["low"] = K - df["high"]
    return out


def signal_with(levels: Levels, atr: float = 5.0) -> Signal:
    return Signal(
        symbol="AAPL",
        name="Apple",
        asset_class="stock",
        bar_date=pd.Timestamp("2024-01-01"),
        close=levels.entry_max,
        atr=atr,
        atr_pct=0.05,
        score=72.0,
        levels=levels,
    )


def mirror_levels(levels: Levels) -> Levels:
    """El reflejo exacto de unos niveles largos: los mismos, en corto."""
    return Levels(
        entry_max=K - levels.entry_max,
        stop=K - levels.stop,
        target=K - levels.target,
        risk_reward=levels.risk_reward,
        risk_per_unit=levels.risk_per_unit,
        reward_per_unit=levels.reward_per_unit,
        direction="short",
    )


LARGOS = Levels(100.0, 95.0, 115.0, 3.0, 5.0, 15.0)
SIGNAL_BAR = (99.0, 100.0, 98.0, 99.0)
FILL_BAR = (99.0, 101.0, 98.0, 100.0)


# ── La propiedad que lo fija todo ─────────────────────────────────────────────

ESCENARIOS = {
    "objetivo": [SIGNAL_BAR, FILL_BAR, (100.0, 116.0, 99.0, 115.5)],
    "stop": [SIGNAL_BAR, FILL_BAR, (100.0, 101.0, 94.0, 94.5)],
    "hueco_contra_el_stop": [SIGNAL_BAR, FILL_BAR, (90.0, 91.0, 88.0, 89.0)],
    "hueco_a_favor": [SIGNAL_BAR, FILL_BAR, (120.0, 121.0, 118.0, 119.0)],
    "vela_ambigua": [SIGNAL_BAR, FILL_BAR, (100.0, 116.0, 94.0, 100.0)],
    "sin_ejecutar": [SIGNAL_BAR, (108.0, 112.0, 107.0, 110.0), (108.0, 112.0, 107.0, 110.0)],
    "por_tiempo": [SIGNAL_BAR, FILL_BAR] + [(100.0, 100.5, 99.5, 100.0)] * 40,
}


@pytest.mark.parametrize("nombre", sorted(ESCENARIOS))
def test_el_corto_es_el_espejo_exacto_del_largo(nombre):
    """Misma operación reflejada: mismo desenlace y misma R, al decimal.

    Cubre las decisiones delicadas del simulador —huecos, vela ambigua que toca
    stop y objetivo, caducidad, salida por tiempo— en los dos sentidos a la vez.
    """
    rows = ESCENARIOS[nombre]
    largo = simulate_signal(signal_with(LARGOS), bars(rows), BT)
    corto = simulate_signal(
        signal_with(mirror_levels(LARGOS)), reflect(bars(rows)), BT
    )

    assert corto.outcome == largo.outcome
    assert corto.was_filled == largo.was_filled
    assert corto.bars_held == largo.bars_held
    if largo.was_filled:
        assert corto.r_multiple == pytest.approx(largo.r_multiple)
        assert corto.entry_price == pytest.approx(K - largo.entry_price)
        assert corto.exit_price == pytest.approx(K - largo.exit_price)
        assert corto.is_win == largo.is_win


@pytest.mark.parametrize("nombre", sorted(ESCENARIOS))
def test_el_espejo_tambien_aguanta_con_trailing(nombre):
    """El trailing es lo único que mueve el stop; en corto solo puede bajar."""
    bt = replace(BT, trail_atr_mult=2.5)
    rows = ESCENARIOS[nombre]

    largo = simulate_signal(signal_with(LARGOS), bars(rows), bt)
    corto = simulate_signal(
        signal_with(mirror_levels(LARGOS)), reflect(bars(rows)), bt
    )

    assert corto.outcome == largo.outcome
    if largo.was_filled:
        assert corto.r_multiple == pytest.approx(largo.r_multiple)


def test_los_niveles_son_el_espejo_exacto_del_largo():
    """`compute_levels` reflejado da los mismos tres precios, invertidos."""
    close, atr, swing = 100.0, 4.0, 96.0

    largo = compute_levels(close, atr, swing, LARGO)
    corto = compute_levels(K - close, atr, K - swing, CORTO)

    assert largo is not None and corto is not None
    assert corto.entry_max == pytest.approx(K - largo.entry_max)
    assert corto.stop == pytest.approx(K - largo.stop)
    assert corto.target == pytest.approx(K - largo.target)
    assert corto.risk_reward == pytest.approx(largo.risk_reward)
    assert corto.risk_per_unit == pytest.approx(largo.risk_per_unit)


# ── Coherencia de los niveles cortos ──────────────────────────────────────────

def test_en_corto_el_stop_queda_arriba_y_el_objetivo_abajo():
    corto = compute_levels(100.0, 4.0, 104.0, CORTO)
    assert corto is not None
    assert corto.target < corto.entry_max < corto.stop
    assert corto.is_valid and corto.is_short


def test_unos_niveles_largos_no_se_validan_como_cortos():
    """`is_valid` tiene que distinguir el sentido, o cualquier orden colaría."""
    invertidos = Levels(100.0, 95.0, 115.0, 3.0, 5.0, 15.0, direction="short")
    assert not invertidos.is_valid


def test_un_objetivo_bajo_cero_se_descarta():
    """Un ATR grande sobre un precio bajo puede pedir un objetivo negativo.

    El precio no puede caer por debajo de cero, así que esa operación no
    existe: dejarla pasar metería una ganancia imposible en el histórico.
    """
    assert compute_levels(close=5.0, atr=3.0, swing_low=6.0, params=CORTO) is None


def test_la_r_del_corto_se_mide_contra_el_riesgo_planificado():
    """Misma regla que en largo: stop menos límite de entrada."""
    niveles = Levels(100.0, 105.0, 85.0, 3.0, 5.0, 15.0, direction="short")
    # Entra en 100, sale en el objetivo 85 -> (100 - 85) / (105 - 100) = 3.0
    df = bars([(101.0, 102.0, 100.5, 101.0), (100.0, 100.5, 99.0, 99.5),
               (99.0, 99.5, 84.0, 85.0)])
    trade = simulate_signal(signal_with(niveles), df, BT)

    assert trade.outcome == "win"
    assert trade.r_multiple == pytest.approx(3.0)


def test_un_hueco_por_encima_del_stop_no_se_toma():
    """En corto el hueco peligroso es al alza: abriría ya pasado el stop."""
    niveles = Levels(100.0, 105.0, 85.0, 3.0, 5.0, 15.0, direction="short")
    df = bars([(99.0, 99.5, 98.0, 99.0), (110.0, 112.0, 109.0, 111.0)])
    trade = simulate_signal(signal_with(niveles), df, BT)

    assert trade.outcome == "no_fill"


# ── El nulo de la comparación contra el índice ───────────────────────────────

def test_el_nulo_de_un_corto_es_vender_el_indice():
    """Comparar un corto contra COMPRAR el índice mediría el signo del mercado.

    Es la comparación que decidió el rumbo del proyecto: si se rompe al añadir
    cortos, los cortos parecerían pésimos en un mercado alcista y geniales en
    uno bajista, sin que la estrategia tuviera nada que ver.
    """
    from trading.backtest import BacktestReport, benchmark_comparison

    fechas = pd.bdate_range("2024-01-01", periods=200)
    # El índice sube un 20% en el periodo: viento a favor del largo.
    indice = pd.Series(np.linspace(100.0, 120.0, len(fechas)), index=fechas)

    def trade(i: int, direction: str):
        from trading.backtest import Trade

        return Trade(
            symbol="AAPL", asset_class="stock", signal_date=fechas[i],
            score=70.0, outcome="win", entry_date=fechas[i],
            exit_date=fechas[i + 5], entry_price=100.0, exit_price=101.0,
            r_multiple=0.2, return_pct=0.01, bars_held=5, direction=direction,
        )

    largos = BacktestReport(trades=[trade(i, "long") for i in range(0, 150)])
    cortos = BacktestReport(trades=[trade(i, "short") for i in range(0, 150)])

    c_largo = benchmark_comparison(largos, indice)
    c_corto = benchmark_comparison(cortos, indice)

    assert c_largo is not None and c_corto is not None
    # Mismo retorno declarado en los dos, pero el nulo cambia de signo.
    assert c_largo.benchmark_avg > 0 and c_corto.benchmark_avg < 0
    assert c_corto.benchmark_avg == pytest.approx(-c_largo.benchmark_avg)
    # Y por tanto el exceso del corto es MAYOR: no tuvo la deriva a favor.
    assert c_corto.excess_avg > c_largo.excess_avg


# ── El reflejo completo, filtro a filtro ─────────────────────────────────────

def test_los_filtros_del_corto_son_los_del_largo_reflejados(
    uptrend_with_pullback, downtrend_with_rally
):
    """Sobre una serie reflejada, cada filtro tiene que decidir lo mismo.

    Es la prueba de que el lado corto no es una estrategia nueva sino la misma
    con el signo cambiado. Si un filtro se desincroniza —un `<` que se queda
    sin girar—, los recuentos dejan de coincidir aquí antes de que nadie
    arriesgue dinero.
    """
    from trading.strategy import compute_features

    largo = compute_features(uptrend_with_pullback, LARGO, None, has_volume=True)
    corto = compute_features(downtrend_with_rally, CORTO, None, has_volume=True)

    for filtro in ("f_regime", "f_trend_strength", "f_pullback", "f_resume"):
        a = largo[filtro].fillna(False).astype(bool)
        b = corto[filtro].fillna(False).astype(bool)
        assert (a.values == b.values).all(), f"{filtro} no está reflejado"

    # Y los indicadores de los que dependen, en la relación que les toca.
    assert np.allclose(largo["rsi"].dropna(), 100.0 - corto["rsi"].dropna())
    assert np.allclose(largo["atr"].dropna(), corto["atr"].dropna())
    assert np.allclose(largo["adx"].dropna(), corto["adx"].dropna())


def test_el_cribador_corto_encuentra_la_tendencia_bajista(downtrend_with_rally):
    """De punta a punta: rebote en tendencia bajista con niveles coherentes."""
    from trading import universe
    from trading.strategy import latest_signal

    señal = latest_signal(universe.get("AAPL"), downtrend_with_rally, CORTO)

    assert señal is not None
    niveles = señal.levels
    assert niveles.is_short
    assert niveles.target < niveles.entry_max < niveles.stop
    assert niveles.risk_reward >= CORTO.min_risk_reward


def test_el_cribador_corto_no_vende_en_tendencia_alcista(uptrend_with_pullback):
    """El filtro de régimen tiene que excluir justo lo que el largo persigue."""
    from trading import universe
    from trading.strategy import latest_signal

    assert latest_signal(universe.get("AAPL"), uptrend_with_pullback, CORTO) is None


def test_el_cribador_largo_no_compra_en_tendencia_bajista(downtrend_with_rally):
    """Y el reflejo: añadir cortos no puede haber aflojado el lado largo."""
    from trading import universe
    from trading.strategy import latest_signal

    assert latest_signal(universe.get("AAPL"), downtrend_with_rally, LARGO) is None


def test_el_regimen_de_mercado_se_invierte_en_corto():
    """Solo se vende con el índice bajo su media larga, y al revés en largo."""
    from trading.strategy import compute_features

    fechas = pd.bdate_range("2020-01-01", periods=400)
    subiendo = pd.Series(np.linspace(100.0, 200.0, 400), index=fechas)
    precios = pd.DataFrame(
        {
            "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0,
            "volume": 5e6,
        },
        index=fechas,
    )

    largo = compute_features(precios, LARGO, subiendo, has_volume=True)
    corto = compute_features(precios, CORTO, subiendo, has_volume=True)

    # Con el índice en tendencia alcista clara: mercado bueno para comprar,
    # malo para vender.
    assert bool(largo["f_market"].iloc[-1]) is True
    assert bool(corto["f_market"].iloc[-1]) is False


# ── Las correcciones manuales, reflejadas ────────────────────────────────────

def record_corto() -> dict:
    """Un corto abierto: entrada 100, stop 105 arriba, objetivo 85 abajo."""
    return {
        "symbol": "AAPL",
        "asset_class": "stock",
        "direction": "short",
        "signal_date": "2024-01-01",
        "score": 72.0,
        "confidence": 0.0,
        "entry_max": 100.0,
        "stop": 105.0,
        "target": 85.0,
        "risk_reward": 3.0,
    }


def estado_corto():
    from trading.state import TradingState

    state = TradingState()
    state.open_signals.append(record_corto())
    return state


def test_en_corto_el_dedazo_peligroso_es_por_encima_del_stop():
    """El espejo de la regla: en corto la entrada imposible es la de arriba."""
    from datetime import date

    state = estado_corto()
    malo = state.mark_taken("AAPL", 106.0, day=date(2026, 8, 21))   # pasado el stop
    assert not malo.ok and malo.reason == "below_stop"

    # Y el precio que en largo sería un dedazo, en corto es perfectamente sano.
    bueno = state.mark_taken("AAPL", 99.0, day=date(2026, 8, 21))
    assert bueno.ok


def test_en_corto_entrar_por_debajo_del_suelo_avisa():
    """Fuera de la zona es, en corto, vender más barato que el suelo."""
    from datetime import date

    state = estado_corto()
    update = state.mark_taken("AAPL", 96.0, day=date(2026, 8, 21))
    assert update.ok and update.warning == "outside_zone"


def test_un_corto_cerrado_a_mano_gana_cuando_el_precio_baja():
    """El signo del resultado se invierte; la R sigue midiéndose igual."""
    from datetime import date

    state = estado_corto()
    state.close_manually("AAPL", 90.0, entry_price=100.0, day=date(2026, 8, 21), bt=BT)

    cerrada = state.history[0]
    assert cerrada["is_win"] is True
    assert cerrada["return_pct"] == pytest.approx(0.10)
    # (100 - 90) / (105 - 100) = 2.0
    assert cerrada["r_multiple"] == pytest.approx(2.0)


def test_un_corto_cerrado_mas_arriba_es_una_perdida():
    from datetime import date

    state = estado_corto()
    state.close_manually("AAPL", 104.0, entry_price=100.0, day=date(2026, 8, 21), bt=BT)

    cerrada = state.history[0]
    assert cerrada["is_win"] is False
    assert cerrada["r_multiple"] == pytest.approx(-0.8)


def test_una_senal_corta_guardada_se_reconstruye_en_el_mismo_sentido():
    """Sin el sentido en el registro, el stop volvería del lado equivocado."""
    from trading.state import signal_from_record

    señal = signal_from_record(record_corto())
    assert señal.levels.is_short and señal.levels.is_valid
    assert señal.levels.risk_per_unit == pytest.approx(5.0)
    assert señal.levels.reward_per_unit == pytest.approx(15.0)


def test_un_registro_antiguo_sin_sentido_sigue_siendo_largo():
    """Los registros guardados antes de existir cortos no pueden cambiar."""
    from trading.state import signal_from_record

    antiguo = {**record_corto(), "entry_max": 100.0, "stop": 95.0, "target": 115.0}
    del antiguo["direction"]

    señal = signal_from_record(antiguo)
    assert not señal.levels.is_short and señal.levels.is_valid


# ── La regla de dos indicadores ──────────────────────────────────────────────

def test_la_regla_de_dos_indicadores_entra_en_sobreventa(uptrend_with_pullback):
    """EMA200 y RSI(2): tendencia de fondo alcista y precio muy estirado."""
    from trading.config import reversion_params
    from trading.strategy import compute_features

    p = reversion_params()
    f = compute_features(uptrend_with_pullback, p, None, has_volume=True)

    assert "rsi_fast" in f
    dispara = f["f_regime"] & f["f_pullback"]
    # Cuando dispara, siempre por encima de la EMA200 y con el RSI(2) bajo.
    for fecha in f.index[dispara.fillna(False)]:
        fila = f.loc[fecha]
        assert fila["close"] > fila["ema_slow"]
        assert fila["rsi_fast"] < p.rsi2_entry_max


def test_la_regla_de_dos_indicadores_se_refleja_en_corto(downtrend_with_rally):
    from trading.config import replace, reversion_params
    from trading.strategy import compute_features

    p = replace(reversion_params(), direction="short")
    f = compute_features(downtrend_with_rally, p, None, has_volume=True)

    dispara = (f["f_regime"] & f["f_pullback"]).fillna(False)
    for fecha in f.index[dispara]:
        fila = f.loc[fecha]
        assert fila["close"] < fila["ema_slow"]
        assert fila["rsi_fast"] > 100.0 - p.rsi2_entry_max


def test_la_salida_va_atada_a_la_entrada():
    """El preset trae su plazo: separarlos costó 19 puntos de acierto medidos."""
    from trading.config import reversion_backtest, reversion_params

    p, bt = reversion_params(), reversion_backtest()
    assert p.entry_rule == "rsi2"
    assert p.target_atr_mult == 1.00      # objetivo cercano, no 3 ATR
    assert bt.max_holding_days == 5       # días, no un mes
    assert p.min_risk_reward == 0.0       # el suelo de 1.5 vetaría la regla


def test_las_dos_reglas_tienen_calibraciones_distintas():
    """Una calibración de la regla vieja no describe a la nueva."""
    from trading.config import DEFAULT_PARAMS, reversion_params

    assert DEFAULT_PARAMS.signature != reversion_params().signature
