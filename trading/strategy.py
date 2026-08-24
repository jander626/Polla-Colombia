"""Estrategia: retroceso en tendencia, en los dos sentidos.

La idea, en una frase: no perseguir rupturas ni intentar adivinar extremos,
sino esperar a que un instrumento que ya se mueve en una dirección se tome un
descanso y entrar cuando reanuda.

En largo eso es una tendencia alcista, un retroceso y una reanudación al alza.
En corto (`StrategyParams.direction == "short"`) es exactamente lo mismo
reflejado: tendencia bajista, rebote, y vuelta a caer. No son dos estrategias
sino una con el signo cambiado, y así está escrito a propósito — dos cuerpos de
reglas independientes se habrían desincronizado a la primera.

Se eligió esta y no otra por tres motivos concretos:

1. El retroceso da un punto de entrada con estructura a ambos lados.
2. Funciona sobre velas diarias, lo que encaja con un solo escaneo al día y
   mantiene el consumo de la API dentro del plan gratuito.
3. El retroceso deja un punto de stop natural justo debajo, así que el ratio
   riesgo/beneficio sale bueno en lugar de forzado.

**Invariante crítico**: el backtest y el escaneo en vivo llaman a las mismas
funciones. `compute_features` produce las columnas y `signals_from_features`
las evalúa; el modo en vivo solo mira la última fila. Si estas dos rutas se
separasen, el backtest dejaría de decir nada sobre el comportamiento real y
toda la calibración de confianza sería mentira.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import StrategyParams
from .risk import Levels, compute_levels
from .universe import Instrument


# Componentes que se calculan siempre. Los que puntúan y con qué peso lo
# decide `StrategyParams.weights`, para que el backtest pueda comparar
# variantes sin tocar esta lógica.
#
# Para forex no hay volumen centralizado ni un benchmark comparable, así que
# esos componentes se marcan como no disponibles y su peso se reparte entre el
# resto, en lugar de puntuar cero (lo que penalizaría a todo el forex sin
# ninguna razón).
ALL_COMPONENTS = (
    "trend",
    "pullback",
    "momentum",
    "headroom",
    "volume",
    "momentum_12_1",
    "reversal",
    # Se sigue calculando pero ya no puntúa: la primera medición mostró que
    # puntuaba al revés. Se conserva para poder vigilarlo en el diagnóstico.
    "relative_strength",
)


@dataclass(frozen=True)
class Signal:
    symbol: str
    name: str
    asset_class: str
    bar_date: pd.Timestamp
    close: float
    atr: float
    atr_pct: float
    score: float
    levels: Levels
    components: dict[str, float] = field(default_factory=dict)
    # Valores de los indicadores en la vela de la señal. Un cribador tiene que
    # poder decir POR QUÉ apareció un instrumento —"ADX 24, el RSI bajó a 41 y
    # está reanudando"— y no solo que apareció. Con defaults para que los
    # registros guardados antes de esto sigan cargando.
    adx: float = float("nan")
    rsi: float = float("nan")
    pullback_rsi: float = float("nan")

    @property
    def is_forex(self) -> bool:
        return self.asset_class == "forex"


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


# ── Cálculo de indicadores y filtros ──────────────────────────────────────────

def compute_features(
    df: pd.DataFrame,
    params: StrategyParams,
    benchmark_close: Optional[pd.Series] = None,
    has_volume: bool = True,
) -> pd.DataFrame:
    """Añade indicadores, filtros booleanos y puntuación a las velas.

    Ninguna columna mira al futuro: el valor de la fila i solo depende de las
    filas <= i. Es lo que permite que el backtest sea honesto.
    """
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]

    out["ema_fast"] = ind.ema(close, params.ema_fast)
    out["ema_mid"] = ind.ema(close, params.ema_mid)
    out["ema_slow"] = ind.ema(close, params.ema_slow)
    out["rsi"] = ind.rsi(close, params.rsi_period)
    out["atr"] = ind.atr(high, low, close, params.atr_period)
    out["atr_pct"] = out["atr"] / close.replace(0.0, np.nan)

    macd_df = ind.macd(close)
    out["macd_hist"] = macd_df["hist"]

    adx_df = ind.adx(high, low, close, params.adx_period)
    out["adx"] = adx_df["adx"]

    if has_volume:
        out["dollar_volume"] = ind.dollar_volume(close, out["volume"], params.volume_window)
        out["avg_volume"] = out["volume"].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean()
    else:
        out["dollar_volume"] = np.nan
        out["avg_volume"] = np.nan

    out["momentum_12_1"] = ind.momentum_12_1(
        close, params.momentum_window, params.momentum_skip
    )
    out["reversal"] = ind.short_term_reversal(close, params.reversal_window)

    if benchmark_close is not None:
        out["rel_strength"] = ind.relative_strength(
            close, benchmark_close, params.relative_strength_window
        )
        # Régimen de mercado: solo se compra con el índice sobre su media
        # larga. Es la forma más barata de evitar los tramos donde la reversión
        # deja de funcionar, que son los que producen la máxima caída.
        benchmark = benchmark_close.reindex(out.index)
        above = ind.above_moving_average(benchmark, params.market_regime_ma)
        # En corto el régimen favorable es el contrario: solo se vende con el
        # índice POR DEBAJO de su media larga. Esto reduce muchísimo el número
        # de días operables —el S&P pasa la mayor parte del tiempo por
        # encima—, y esa escasez es un resultado de la medición, no un fallo:
        # las operaciones cortas salen agrupadas en unos pocos episodios.
        out["market_ok"] = (~above if params.is_short else above).fillna(False)
    else:
        out["rel_strength"] = np.nan
        # Sin benchmark (forex) no hay régimen que filtrar: el S&P no es su
        # régimen, y exigirlo excluiría todo el forex sin fundamento.
        out["market_ok"] = True

    out["swing_low"] = ind.rolling_low(low, params.stop_swing_lookback)
    out["resistance"] = ind.rolling_high(high, params.resistance_lookback)
    # Los espejos de las dos anteriores: en corto el stop se apoya en el máximo
    # del rebote y el "espacio libre" se mide hacia el soporte, no hacia arriba.
    out["swing_high"] = ind.rolling_high(high, params.stop_swing_lookback)
    out["support"] = ind.rolling_low(low, params.resistance_lookback)

    _add_filters(out, params, has_volume)
    _add_score(out, params, has_volume, benchmark_close is not None)
    return out


def _add_filters(out: pd.DataFrame, params: StrategyParams, has_volume: bool) -> None:
    close, low, high = out["close"], out["low"], out["high"]
    window = params.pullback_lookback
    short = params.is_short

    # Los umbrales del RSI se reflejan sobre 50: si en largo el retroceso exige
    # caer por debajo de 45, en corto el rebote exige subir por encima de 55.
    # Reflejarlos —en vez de darles parámetros propios— evita duplicar la
    # rejilla del barrido y con ella la superficie de sobreajuste.
    pullback_rsi = 100.0 - params.pullback_rsi_max if short else params.pullback_rsi_max
    resume_rsi = 100.0 - params.resume_rsi_min if short else params.resume_rsi_min

    # 1 — Régimen de fondo, en el sentido que toque.
    if short:
        out["f_regime"] = (close < out["ema_slow"]) & (out["ema_mid"] < out["ema_slow"])
    else:
        out["f_regime"] = (close > out["ema_slow"]) & (out["ema_mid"] > out["ema_slow"])

    # 2 — La tendencia tiene fuerza. Sin esto la estrategia se desangra en
    #     mercados laterales, donde los retrocesos no reanudan nada. El ADX no
    #     tiene signo: mide fuerza, no dirección, así que vale igual en corto.
    out["f_trend_strength"] = out["adx"] > params.adx_min

    # 3 — Hubo un retroceso real: el RSI se descargó y el precio visitó la zona
    #     de la media rápida, sin llegar a romper la tendencia de fondo.
    out["rsi_min"] = out["rsi"].rolling(window, min_periods=1).min()
    out["rsi_max"] = out["rsi"].rolling(window, min_periods=1).max()
    if short:
        rsi_dipped = out["rsi_max"] > pullback_rsi
        touched = high >= out["ema_fast"] - params.pullback_ema_touch_atr * out["atr"]
        structure_intact = high.rolling(window, min_periods=1).max() < out["ema_slow"]
    else:
        rsi_dipped = out["rsi_min"] < pullback_rsi
        touched = low <= out["ema_fast"] + params.pullback_ema_touch_atr * out["atr"]
        structure_intact = low.rolling(window, min_periods=1).min() > out["ema_slow"]
    touched_recently = touched.rolling(window, min_periods=1).max().astype(bool)
    out["f_pullback"] = rsi_dipped & touched_recently & structure_intact

    # 4 — Está reanudando. Este es el filtro que evita comprar un cuchillo
    #     cayendo —o venderle a un cohete—: exige una señal activa de giro, no
    #     solo un precio que parezca bueno.
    if short:
        rsi_crossed = (out["rsi"] < resume_rsi) & (out["rsi"].shift(1) >= resume_rsi)
        macd_turned = (out["macd_hist"] < 0) & (out["macd_hist"].shift(1) >= 0)
        broke_prev = close < out["low"].shift(1)
        out["f_resume"] = (out["rsi"] < resume_rsi) & (
            rsi_crossed | macd_turned | broke_prev
        )
    else:
        rsi_crossed = (out["rsi"] > resume_rsi) & (out["rsi"].shift(1) <= resume_rsi)
        macd_turned = (out["macd_hist"] > 0) & (out["macd_hist"].shift(1) <= 0)
        broke_prev = close > out["high"].shift(1)
        out["f_resume"] = (out["rsi"] > resume_rsi) & (
            rsi_crossed | macd_turned | broke_prev
        )

    # 5 — Liquidez. El forex no publica volumen centralizado, pero los 14 pares
    #     de Quantfury son de por sí los más líquidos del mundo.
    if has_volume:
        out["f_liquidity"] = out["dollar_volume"] >= params.min_dollar_volume
    else:
        out["f_liquidity"] = True

    # 6 — Volatilidad sana: ni muerta (no llega al objetivo) ni caótica (el
    #     stop salta por ruido).
    out["f_volatility"] = (out["atr_pct"] >= params.min_atr_pct) & (
        out["atr_pct"] <= params.max_atr_pct
    )

    if params.use_market_regime_filter:
        out["f_market"] = out["market_ok"].astype(bool)
    else:
        out["f_market"] = True

    out["passes"] = (
        out["f_market"]
        & out["f_regime"].fillna(False)
        & out["f_trend_strength"].fillna(False)
        & out["f_pullback"].fillna(False)
        & out["f_resume"].fillna(False)
        & out["f_liquidity"].fillna(False)
        & out["f_volatility"].fillna(False)
    )


def _add_score(
    out: pd.DataFrame, params: StrategyParams, has_volume: bool, has_benchmark: bool
) -> None:
    """Puntuación 0-100 combinando cinco componentes normalizados a [0, 1].

    Cada componente se refleja para el lado corto con la misma fórmula y el
    signo cambiado. Se mantiene la forma exacta —y no una puntuación nueva
    "pensada para cortos"— por una razón de método: el diagnóstico ya mostró
    que esta puntuación no ordena resultados en largo, así que inventarle
    componentes propios al lado corto sería añadir grados de libertad a algo
    que todavía no ha demostrado predecir nada. Si el reflejo tampoco ordena,
    esa es la respuesta.
    """
    window = params.pullback_lookback
    short = params.is_short

    # Fuerza de tendencia: ADX por encima del umbral y separación de medias.
    adx_part = _clip01((out["adx"] - params.adx_min) / (40.0 - params.adx_min))
    gap = out["ema_slow"] - out["ema_mid"] if short else out["ema_mid"] - out["ema_slow"]
    spread = gap / out["close"].replace(0.0, np.nan)
    spread_part = _clip01(spread / 0.10)
    out["c_trend"] = 0.6 * adx_part + 0.4 * spread_part

    # Calidad del retroceso: profundo pero no roto. Un RSI que cae a 40 es un
    # descanso; uno que cae a 20 suele ser un cambio de tendencia.
    # El punto dulce del retroceso se refleja sobre 50: RSI ~40 en largo,
    # RSI ~60 en corto.
    if short:
        rsi_extreme = out["rsi"].rolling(window, min_periods=1).max()
        depth_part = _clip01(1.0 - (rsi_extreme - 60.0).abs() / 25.0)
        touch_dist = (
            (out["high"] - out["ema_fast"]) / out["atr"].replace(0.0, np.nan)
        ).rolling(window, min_periods=1).max().abs()
    else:
        rsi_extreme = out["rsi"].rolling(window, min_periods=1).min()
        depth_part = _clip01(1.0 - (rsi_extreme - 40.0).abs() / 25.0)
        touch_dist = (
            (out["low"] - out["ema_fast"]) / out["atr"].replace(0.0, np.nan)
        ).rolling(window, min_periods=1).min().abs()
    touch_part = _clip01(1.0 - touch_dist / 1.5)
    out["c_pullback"] = 0.6 * depth_part + 0.4 * touch_part

    # Confirmación de momentum en la vela de reanudación.
    bar_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    if short:
        hist_rising = (out["macd_hist"] < out["macd_hist"].shift(1)).astype(float)
        rsi_slope = _clip01((out["rsi"].shift(2) - out["rsi"]) / 15.0)
        close_strength = _clip01((out["high"] - out["close"]) / bar_range)
    else:
        hist_rising = (out["macd_hist"] > out["macd_hist"].shift(1)).astype(float)
        rsi_slope = _clip01((out["rsi"] - out["rsi"].shift(2)) / 15.0)
        close_strength = _clip01((out["close"] - out["low"]) / bar_range)
    out["c_momentum"] = (hist_rising + rsi_slope + close_strength) / 3.0

    # Espacio libre hasta el máximo reciente, en múltiplos de ATR. Sustituye al
    # antiguo recorte del objetivo contra resistencia: en vez de vetar la señal
    # (lo que eliminaba casi todos los retrocesos, porque siempre vienen de un
    # máximo cercano), penaliza su puntuación y deja que el backtest mida
    # cuánto cuesta realmente tener un techo encima.
    # En corto el espacio libre se mide hacia abajo, hasta el soporte.
    room = out["close"] - out["support"] if short else out["resistance"] - out["close"]
    headroom_atr = room / out["atr"].replace(0.0, np.nan)
    out["c_headroom"] = _clip01(headroom_atr / params.headroom_target_atr)

    # Momentum 12-1 (Jegadeesh y Titman). Se normaliza sobre un rango de
    # -20% a +60%, que cubre lo que hace una acción líquida en un año.
    # En corto se premia el momentum 12-1 NEGATIVO: el mismo efecto con el
    # signo cambiado, sobre el mismo rango.
    out["c_momentum_12_1"] = _clip01(
        (0.60 - out["momentum_12_1"]) / 0.80
        if short
        else (out["momentum_12_1"] + 0.20) / 0.80
    )

    # Reversión a corto plazo: ya viene invertida, así que un valor alto
    # significa "se quedó atrás el último mes", que es lo que se premia.
    out["c_reversal"] = _clip01(
        (0.10 - out["reversal"]) / 0.20 if short else (out["reversal"] + 0.10) / 0.20
    )

    # Fuerza relativa frente al mercado. YA NO PUNTÚA —su peso es cero— pero se
    # sigue calculando para vigilarla en el diagnóstico por cuartiles.
    if not has_benchmark:
        out["c_relative_strength"] = np.nan
    else:
        out["c_relative_strength"] = _clip01(
            (0.15 - out["rel_strength"]) / 0.20
            if short
            else (out["rel_strength"] + 0.05) / 0.20
        )

    # Volumen de la vela de reanudación frente a su media.
    if has_volume:
        ratio = out["volume"] / out["avg_volume"].replace(0.0, np.nan)
        out["c_volume"] = _clip01((ratio - 0.8) / 1.2)
    else:
        out["c_volume"] = np.nan

    # Suma ponderada renormalizada sobre los componentes disponibles, para que
    # el forex no quede penalizado por carecer de volumen y benchmark.
    total = pd.Series(0.0, index=out.index)
    weight_sum = pd.Series(0.0, index=out.index)
    for key, weight in params.weight_map.items():
        column = out[f"c_{key}"]
        available = column.notna()
        total += column.fillna(0.0) * weight * available
        weight_sum += weight * available

    out["score"] = 100.0 * (total / weight_sum.replace(0.0, np.nan))


# ── Construcción de señales ───────────────────────────────────────────────────

def _evaluate(
    instrument: Instrument, row: pd.Series, bar_date: pd.Timestamp, params: StrategyParams
) -> tuple[Signal | None, str | None]:
    """Construye la señal y, si la descarta, dice en qué etapa fue.

    El motivo del descarte importa tanto como el descarte: sin él, un día sin
    alertas es indistinguible de un bot averiado. Devolver las dos cosas juntas
    —en vez de tener una función que decide y otra que explica— es lo que evita
    que la explicación se desincronice de la decisión real.
    """
    # En corto el stop se apoya en el máximo del rebote, no en el mínimo.
    swing = row["swing_high"] if params.is_short else row["swing_low"]
    levels = compute_levels(
        close=float(row["close"]),
        atr=float(row["atr"]),
        swing_low=float(swing),
        params=params,
    )
    if levels is None or not levels.is_valid:
        return None, "niveles"
    if levels.risk_reward < params.min_risk_reward:
        return None, "riesgo_beneficio"

    score = float(row["score"])
    if not np.isfinite(score) or score < params.min_score:
        return None, "score"

    components = {
        key: float(row[f"c_{key}"])
        for key in ALL_COMPONENTS
        if pd.notna(row.get(f"c_{key}", np.nan))
    }

    return (
        Signal(
            symbol=instrument.symbol,
            name=instrument.name,
            asset_class=instrument.asset_class,
            bar_date=bar_date,
            close=float(row["close"]),
            atr=float(row["atr"]),
            atr_pct=float(row["atr_pct"]),
            score=score,
            levels=levels,
            components=components,
            adx=float(row.get("adx", float("nan"))),
            rsi=float(row.get("rsi", float("nan"))),
            pullback_rsi=float(
                row.get("rsi_max" if params.is_short else "rsi_min", float("nan"))
            ),
        ),
        None,
    )


def _build_signal(
    instrument: Instrument, row: pd.Series, bar_date: pd.Timestamp, params: StrategyParams
) -> Signal | None:
    return _evaluate(instrument, row, bar_date, params)[0]


def signals_from_features(
    instrument: Instrument, feats: pd.DataFrame, params: StrategyParams
) -> list[Signal]:
    """Todas las señales históricas del instrumento. Lo que consume el backtest."""
    passing = feats[feats["passes"]]
    signals: list[Signal] = []
    for bar_date, row in passing.iterrows():
        signal = _build_signal(instrument, row, bar_date, params)
        if signal is not None:
            signals.append(signal)
    return signals


def latest_signal(
    instrument: Instrument,
    df: pd.DataFrame,
    params: StrategyParams,
    benchmark_close: Optional[pd.Series] = None,
) -> Signal | None:
    """Señal de la última vela cerrada. Lo que consume el escaneo en vivo."""
    has_volume = not instrument.is_forex
    if len(df) < params.min_bars:
        return None

    feats = compute_features(df, params, benchmark_close, has_volume=has_volume)
    last = feats.iloc[-1]
    if not bool(last.get("passes", False)):
        return None
    return _build_signal(instrument, last, feats.index[-1], params)


# ── Embudo del escaneo ────────────────────────────────────────────────────────
# Las etapas se aplican en cascada y en este orden, de lo más estructural a lo
# más puntual. El orden importa para leer el resultado: cada número es "cuántos
# candidatos siguen vivos tras aplicar esta etapa y todas las anteriores", así
# que la caída más grande señala al filtro que de verdad manda ese día.

FUNNEL_STAGES: tuple[tuple[str, str], ...] = (
    ("f_market", "el mercado está en régimen alcista"),
    ("f_regime", "el instrumento está en tendencia alcista"),
    ("f_trend_strength", "la tendencia tiene fuerza (ADX)"),
    ("f_volatility", "la volatilidad es sana"),
    ("f_liquidity", "hay liquidez suficiente"),
    ("f_pullback", "hubo un retroceso"),
    ("f_resume", "está reanudando"),
    ("niveles", "el stop cabe donde debe"),
    ("riesgo_beneficio", "el ratio riesgo/beneficio llega a 1.5"),
    ("score", "la puntuación llega al mínimo"),
)

_STAGE_LABELS = dict(FUNNEL_STAGES)
_POST_STAGES = ("niveles", "riesgo_beneficio", "score")


@dataclass(frozen=True)
class ScanFunnel:
    """Cuántos candidatos sobreviven a cada etapa del escaneo.

    Existe porque "hoy no hay nada" es una respuesta inútil por sí sola: no
    distingue un mercado tranquilo de un filtro mal calibrado ni de un bot
    roto. Con el embudo, un día sin señales viene con su propia explicación.
    """

    evaluated: int = 0
    no_data: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def stages(self) -> list[tuple[str, str, int, int]]:
        """(clave, etiqueta, supervivientes, descartados en esta etapa)."""
        rows: list[tuple[str, str, int, int]] = []
        alive = self.evaluated
        for key, label in FUNNEL_STAGES:
            survivors = self.counts.get(key, 0)
            rows.append((key, label, survivors, alive - survivors))
            alive = survivors
        return rows

    @property
    def survivors(self) -> int:
        return self.counts.get(FUNNEL_STAGES[-1][0], 0)

    @property
    def bottleneck(self) -> tuple[str, str, int] | None:
        """La etapa que más candidatos elimina: (clave, etiqueta, descartados)."""
        rows = [row for row in self.stages() if row[3] > 0]
        if not rows:
            return None
        key, label, _, dropped = max(rows, key=lambda row: row[3])
        return key, label, dropped

    def summary(self) -> str:
        lines = [f"Instrumentos evaluados: {self.evaluated} (sin datos: {self.no_data})"]
        for _, label, survivors, dropped in self.stages():
            lines.append(f"  {survivors:>4} tras «{label}»  (−{dropped})")
        return "\n".join(lines)


def _stage_ok(row: pd.Series, key: str) -> bool:
    """Un NaN es un 'no sé', y un 'no sé' no puede contar como aprobado.

    `bool(np.nan)` es True en Python, así que sin este envoltorio un indicador
    todavía sin calentar dejaría pasar al instrumento.
    """
    value = row.get(key, False)
    return bool(value) if pd.notna(value) else False


def scan_with_funnel(
    instruments: list[Instrument],
    bars: dict[str, pd.DataFrame],
    params: StrategyParams,
    benchmark_close: Optional[pd.Series] = None,
) -> tuple[list[Signal], ScanFunnel]:
    """Escaneo completo con el recuento de por qué cae cada candidato."""
    found: list[Signal] = []
    counts = {key: 0 for key, _ in FUNNEL_STAGES}
    evaluated = 0
    no_data = 0

    for instrument in instruments:
        df = bars.get(instrument.symbol)
        if df is None or df.empty or len(df) < params.min_bars:
            no_data += 1
            continue

        benchmark = None if instrument.is_forex else benchmark_close
        try:
            feats = compute_features(
                df, params, benchmark, has_volume=not instrument.is_forex
            )
        except Exception as exc:  # un símbolo con datos raros no tumba el escaneo
            print(f"[WARN] Fallo evaluando {instrument.symbol}: {exc}")
            no_data += 1
            continue

        evaluated += 1
        last = feats.iloc[-1]

        alive = True
        for key, _ in FUNNEL_STAGES:
            if key in _POST_STAGES:
                break
            alive = alive and _stage_ok(last, key)
            if not alive:
                break
            counts[key] += 1
        if not alive:
            continue

        signal, failed = _evaluate(instrument, last, feats.index[-1], params)
        for key in _POST_STAGES:
            if failed == key:
                break
            counts[key] += 1
        if signal is not None:
            found.append(signal)

    found.sort(key=lambda s: s.score, reverse=True)
    return found, ScanFunnel(evaluated=evaluated, no_data=no_data, counts=counts)


def scan(
    instruments: list[Instrument],
    bars: dict[str, pd.DataFrame],
    params: StrategyParams,
    benchmark_close: Optional[pd.Series] = None,
) -> list[Signal]:
    """Escaneo completo del universo, ordenado de mejor a peor puntuación."""
    return scan_with_funnel(instruments, bars, params, benchmark_close)[0]
