"""Configuración central del bot de trading.

Todos los umbrales de la estrategia viven aquí para que el backtest pueda
barrerlos sin tocar la lógica. Los valores por defecto son una hipótesis
razonable de partida, NO parámetros validados: la fase de backtest existe
precisamente para ajustarlos con datos.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from datetime import date
from zoneinfo import ZoneInfo


# ── Entorno ───────────────────────────────────────────────────────────────────
# Se leen con .get() y no con [] para que importar el módulo nunca falle: los
# modos offline (backtest sobre caché, tests) no necesitan credenciales.

TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

STATE_FILE = os.environ.get("TRADING_STATE_FILE", "trading_state.json")
CALIBRATION_FILE = os.environ.get("TRADING_CALIBRATION_FILE", "calibration.json")
CACHE_DIR = os.environ.get("TRADING_CACHE_DIR", ".cache/market_data")


# ── Husos horarios y calendario ───────────────────────────────────────────────

NY_TZ = ZoneInfo("America/New_York")
LONDON_TZ = ZoneInfo("Europe/London")
BOGOTA_TZ = ZoneInfo("America/Bogota")  # zona del usuario, solo para mostrar horas

# Ventana de escaneo, en hora de Nueva York.
#
# El cron de GitHub Actions es en UTC y —esto es lo importante— NO se respeta:
# está programado cada 15 minutos y en la práctica dispara dos o tres veces al
# día, a horas impredecibles. Por eso la decisión de escanear se toma aquí, con
# zonas horarias reales, y no en el cron.
#
# La ventana empezaba a las 08:30 y duraba 90 minutos. El 11 de agosto de 2026
# el bot no alertó en todo el día: sus dos únicas ejecuciones cayeron a las
# 07:38 y a las 08:27 de Nueva York — la segunda por DOS MINUTOS fuera.
#
# El fallo era de criterio. La señal se calcula con el CIERRE DE AYER, que no
# cambia entre las 06:00 y las 09:00 de la mañana: escanear a las 07:38 habría
# dado exactamente el mismo resultado. Aquella hora de inicio era una
# preferencia, no una necesidad, y estaba rechazando escaneos perfectamente
# válidos.
#
# El único límite real es el otro extremo: no alertar con el mercado ya
# abierto, porque entonces el precio de entrada calculado ya no sirve. De ahí
# que la ventana termine a las 09:25, cinco minutos antes de la apertura.
SCAN_HOUR_NY = 6
SCAN_MINUTE_NY = 0
SCAN_WINDOW_MINUTES = 205  # 06:00 → 09:25, cinco minutos antes de la apertura

# Festivos de NYSE/NASDAQ. Lista estática porque son ~10 al año y la
# alternativa (pandas_market_calendars) arrastra dependencias pesadas para
# resolver un problema trivial. Revisar y extender cada diciembre.
MARKET_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2026
        date(2026, 1, 1),    # Año Nuevo
        date(2026, 1, 19),   # Martin Luther King Jr.
        date(2026, 2, 16),   # Presidents' Day
        date(2026, 4, 3),    # Viernes Santo
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Día de la Independencia (observado)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Acción de Gracias
        date(2026, 12, 25),  # Navidad
        # 2027
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),   # Juneteenth (observado)
        date(2027, 7, 5),    # Independencia (observado)
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24),  # Navidad (observado)
    }
)


def is_trading_day(day: date) -> bool:
    """Días hábiles de bolsa estadounidense: ni fin de semana ni festivo."""
    return day.weekday() < 5 and day not in MARKET_HOLIDAYS


# ── Parámetros de la estrategia ───────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyParams:
    """Umbrales de los filtros y del cálculo de niveles.

    Congelado (frozen) a propósito: el backtest genera variantes con
    `replace(params, adx_min=25)` en vez de mutar un objeto compartido, lo que
    evita que un barrido de parámetros contamine al siguiente.
    """

    # Filtro 1 — régimen alcista
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200

    # Filtro 2 — fuerza de tendencia
    adx_period: int = 14
    adx_min: float = 20.0

    # Filtro 3 — retroceso
    pullback_lookback: int = 5          # velas en las que buscar el retroceso
    pullback_rsi_max: float = 45.0      # el RSI tuvo que caer por debajo de esto
    pullback_ema_touch_atr: float = 0.5  # "tocar la EMA20" = estar a < 0.5 ATR de ella

    # Filtro 4 — reanudación
    resume_rsi_min: float = 45.0        # RSI recuperando por encima de este nivel

    # Filtro 5 — liquidez
    min_dollar_volume: float = 50_000_000.0  # media de 20 sesiones, en USD
    volume_window: int = 20

    # Filtro 6 — volatilidad sana (ATR como fracción del precio)
    atr_period: int = 14
    min_atr_pct: float = 0.005   # por debajo: no se mueve, no llega al objetivo
    max_atr_pct: float = 0.080   # por encima: el stop salta solo por ruido

    # Niveles de la operación
    entry_buffer_atr: float = 0.30   # techo de entrada = cierre + 0.30 ATR
    stop_swing_lookback: int = 10
    stop_buffer_atr: float = 0.10    # colchón bajo el mínimo, evita mechas
    min_stop_atr: float = 0.80       # distancia mínima al stop, evita salir por ruido
    # Objetivo de 3 ATR: junto al suelo de R:B de 1.5 admite retrocesos de hasta
    # ~1.6 ATR de profundidad, que es el rango de un descanso sano hacia la media
    # rápida. Con 2.5 ATR el sistema solo aceptaba retrocesos de menos de 1.3 ATR
    # y descartaba casi todos los escenarios que la estrategia busca.
    target_atr_mult: float = 3.00
    min_risk_reward: float = 1.50    # descarte duro por debajo de este ratio

    # El máximo reciente NO recorta el objetivo. En un retroceso dentro de una
    # tendencia alcista siempre hay un máximo cercano por encima —el retroceso
    # viene de ahí—, así que recortar contra él contradice la tesis de la
    # estrategia y hunde el ratio riesgo/beneficio de forma sistemática.
    # En su lugar, el espacio libre hasta ese techo es un componente de la
    # puntuación: menos espacio, menos puntuación, y que sea el backtest quien
    # decida cuánto vale esa penalización.
    resistance_lookback: int = 60
    headroom_target_atr: float = 3.0  # espacio libre "de sobra", en múltiplos de ATR

    # ── Componentes con respaldo académico ───────────────────────────────────
    # La primera medición mostró que la fuerza relativa a 60 sesiones puntuaba
    # AL REVÉS: el cuartil que más había batido al mercado rendía +0.002R y el
    # que menos, +0.155R. No fue mala suerte: 60 sesiones cae en la zona de la
    # *reversión a corto plazo* (Jegadeesh, 1990), donde los ganadores
    # recientes se quedan atrás.
    #
    # Son dos efectos distintos y opuestos según el horizonte, y mezclarlos en
    # una sola ventana los cancelaba:
    #   - Momentum 12-1 (Jegadeesh y Titman, 1993): 12 meses excluyendo el
    #     último. Su revisión de 2023 confirma que no ha decaído.
    #   - Reversión corta: el último mes, con el signo invertido.
    momentum_window: int = 252       # ~12 meses
    momentum_skip: int = 21          # se excluye el último mes
    reversal_window: int = 21        # ~1 mes, se puntúa invertido
    relative_strength_window: int = 60   # solo diagnóstico; ya no puntúa

    # Filtro de régimen de mercado: solo comprar acciones con el índice de
    # referencia por encima de su media de 200 sesiones. Documentado como la
    # forma más barata de recortar la máxima caída en estrategias de reversión,
    # que es justo el peor número del sistema (-28.9R). No se aplica al forex:
    # el S&P no es su régimen.
    use_market_regime_filter: bool = True
    market_regime_ma: int = 200

    # Pesos del scoring. Viven en los parámetros —y no como constante del
    # módulo— para que el backtest pueda comparar variantes sin tocar la lógica.
    weights: tuple[tuple[str, float], ...] = (
        ("pullback", 0.30),
        ("momentum", 0.18),
        ("trend", 0.15),
        ("momentum_12_1", 0.15),
        ("reversal", 0.12),
        ("headroom", 0.05),
        ("volume", 0.05),
    )

    # Otros
    rsi_period: int = 14
    max_signals_per_scan: int = 5    # cuántas pasan al filtro de noticias (Gemini)
    min_score: float = 50.0          # puntuación mínima para considerar la señal

    # Barras mínimas de historia para que los indicadores sean fiables. El
    # momentum a 12 meses es ahora el indicador más exigente en historia.
    @property
    def min_bars(self) -> int:
        return max(self.ema_slow, self.momentum_window + self.momentum_skip) + 60

    @property
    def weight_map(self) -> dict[str, float]:
        return dict(self.weights)

    @property
    def signature(self) -> str:
        """Huella de los parámetros que cambian qué señales se generan.

        Sirve para detectar una calibración obsoleta. Ocurrió de verdad: se
        calibró a las 04:07, se cambió la estrategia a las 04:24, y el bot
        siguió publicando la confianza de la versión anterior sin que nada
        avisara. Un número que dice venir de un backtest tiene que venir del
        backtest de ESTA estrategia, o no significa nada.
        """
        parts = [
            f"w={sorted(self.weights)}",
            f"regime={self.use_market_regime_filter}:{self.market_regime_ma}",
            f"adx={self.adx_min}",
            f"pull={self.pullback_rsi_max}:{self.pullback_lookback}",
            f"resume={self.resume_rsi_min}",
            f"target={self.target_atr_mult}",
            f"stop={self.min_stop_atr}:{self.stop_swing_lookback}",
            f"rr={self.min_risk_reward}",
            f"score={self.min_score}",
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ── Variantes a comparar ──────────────────────────────────────────────────────
# Se definen A PRIORI desde la literatura, no buscando en los datos hasta que
# algo salga bien. Probar diez variantes sobre la misma muestra garantiza que
# una parezca buena por azar; por eso son tres, están fijadas de antemano, y el
# informe publica las tres — no solo la que gane.

BASELINE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("pullback", 0.42),
    ("momentum", 0.22),
    ("trend", 0.18),
    ("headroom", 0.09),
    ("volume", 0.09),
)


# Pesos que conservan solo los componentes con Δ R positiva en el diagnóstico
# del 14 de agosto, repartidos aproximadamente según cuánto predice cada uno:
#
#   momentum       Δ +0.172   predice   (era 0.18)
#   volume         Δ +0.108   ruido     (era 0.05)
#   pullback       Δ +0.068   ruido     (era 0.30)
#   headroom       Δ +0.059   ruido     (era 0.05)
#   ─ fuera ────────────────────────────────────────
#   momentum_12_1  Δ -0.247   INVERTIDO (era 0.15)
#   trend          Δ -0.164   INVERTIDO (era 0.15)
#   reversal       Δ -0.035   ruido     (era 0.12)
#
# Las dos invertidas se llevaban el 30% del peso puntuando al revés. La
# explicación que encaja con las tres: dentro de una muestra ya filtrada a
# "tendencia alcista + ADX fuerte + retroceso reciente", todo lo que mide
# CUÁNTO se ha extendido el movimiento puntúa al revés, porque los valores más
# extendidos son los que peor reanudan.
PREDICTIVE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("momentum", 0.40),
    ("volume", 0.25),
    ("pullback", 0.20),
    ("headroom", 0.15),
)


# La puntuación más simple posible: el único componente que el diagnóstico
# midió como predictivo, y nada más. Está aquí porque "más simple" es una
# hipótesis con derecho a competir, no un consuelo: siete componentes de los
# que uno predice, tres puntúan al revés y tres son ruido tienen menos
# información que ese uno solo.
MOMENTUM_ONLY_WEIGHTS: tuple[tuple[str, float], ...] = (("momentum", 1.0),)


# ── Búsqueda sistemática ──────────────────────────────────────────────────────

# Cuatro perillas, que es el máximo que admite una búsqueda honesta sobre 609
# operaciones. Cada una sale de una medición previa, no de una corazonada:
#
#   pullback_rsi_max  el embudo lo señaló como el cuello de botella (48 → 4)
#   weights           el diagnóstico mostró que el 30% del peso va al revés
#   adx_min           segundo mayor descarte del embudo (93 → 49)
#   trail_atr_mult    la única palanca que mejora operaciones en vez de contarlas
#
# 27 combinaciones de estrategia × 3 de salida = 81 mediciones. Las tres de
# salida son gratis: el trailing no cambia qué señales se generan, solo cómo se
# cierran, así que las señales se generan UNA vez y se simulan tres.
SEARCH_PULLBACK_RSI: tuple[float, ...] = (45.0, 50.0, 55.0)
SEARCH_ADX_MIN: tuple[float, ...] = (15.0, 20.0, 25.0)
SEARCH_TRAILS: tuple[float, ...] = (0.0, 1.5, 2.5)
SEARCH_WEIGHTS: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = (
    ("actual", StrategyParams().weights),
    ("predictivo", PREDICTIVE_WEIGHTS),
    ("momentum", MOMENTUM_ONLY_WEIGHTS),
)


def search_grid() -> dict[str, "StrategyParams"]:
    """Las 27 combinaciones de estrategia que barre `backtest --search`.

    El umbral de reanudación va SIEMPRE atado al de retroceso. Si el RSI "cae
    por debajo de 55" y basta con "estar por encima de 45" para darlo por
    reanudado, un RSI plano en 50 cumpliría las dos condiciones a la vez y la
    búsqueda encontraría "ventaja" en un filtro que no filtra nada.
    """
    base = StrategyParams()
    grid: dict[str, StrategyParams] = {}
    for rsi in SEARCH_PULLBACK_RSI:
        for weight_name, weights in SEARCH_WEIGHTS:
            for adx in SEARCH_ADX_MIN:
                label = f"rsi{rsi:.0f}·{weight_name}·adx{adx:.0f}"
                grid[label] = replace(
                    base,
                    pullback_rsi_max=rsi,
                    resume_rsi_min=rsi,
                    adx_min=adx,
                    weights=weights,
                )
    return grid


def variants() -> dict[str, tuple["StrategyParams", "BacktestParams"]]:
    """Las variantes a comparar, cada una respondiendo UNA pregunta.

    Salen del embudo del escaneo del 14 de agosto, no de buscar en los datos
    hasta que algo saliera bien. Ese día, de 141 instrumentos:

        141 → mercado en régimen alcista        (−0)
         93 → instrumento en tendencia alcista  (−48)
         49 → tendencia con fuerza (ADX)        (−44)
         48 → volatilidad y liquidez            (−1)
          4 → hubo un retroceso                 (−44)  ← el 92% muere aquí
          0 → está reanudando                   (−4)

    El diagnóstico es inequívoco: el cuello de botella es el filtro de
    retroceso. Exigir que el RSI(14) baje de 45 es pedir una corrección
    profunda en un valor que, por los filtros anteriores, ya está en tendencia
    alcista con fuerza. El descanso típico de un valor así lleva el RSI a
    45-55, no por debajo de 45.

    Hay dos palancas distintas y conviene no confundirlas:

    - Aflojar filtros da MÁS operaciones, no mejores. Sube la frecuencia y
      diluye la ventaja por operación.
    - El trailing da MEJORES operaciones sin cambiar cuántas hay: recorta a las
      perdedoras que se dan la vuelta antes de tocar el stop planificado.

    Por eso se miden por separado, y luego juntas. La columna que decide sigue
    siendo el límite inferior de la esperanza: si es negativo, la variante no
    ha demostrado nada por muchas señales que produzca.
    """
    base = StrategyParams()
    bt = BacktestParams()

    # Aflojar el retroceso obliga a mover también la reanudación: si el RSI
    # "cae por debajo de 50" y basta con "estar por encima de 45" para
    # considerarlo reanudado, un RSI plano en 47 cumpliría las dos cosas a la
    # vez y no habría giro ninguno que confirmar.
    suave = replace(base, pullback_rsi_max=50.0, resume_rsi_min=50.0)

    return {
        "0_actual": (base, bt),
        # 1 — ¿Es el umbral de RSI lo que deja al bot mudo una semana entera?
        "1_retroceso_a_50": (suave, bt),
        # 2 — 1 + más margen entre el hundimiento y el giro. Con 5 velas, un
        #     valor que se hundió y rebotó hace 6 días ya no es alcanzable.
        "2_ventana_10": (replace(suave, pullback_lookback=10), bt),
        # 3 — Sin tocar un solo filtro: ¿sube la esperanza solo con salir mejor?
        "3_trailing": (base, replace(bt, trail_atr_mult=2.5)),
        # 4 — Las dos palancas juntas. Más señales y mejores salidas.
        "4_ambos": (replace(suave, pullback_lookback=10), replace(bt, trail_atr_mult=2.5)),
        # 5 — 4 pero dejando correr la cola derecha en vez de cortar en 3 ATR.
        "5_dejar_correr": (
            replace(suave, pullback_lookback=10),
            replace(bt, trail_atr_mult=2.5, let_winners_run=True),
        ),
        # 6 — Solo los componentes que el diagnóstico mide como predictivos.
        #
        # ATENCIÓN al leer esta: las cinco anteriores se definieron ANTES de
        # ver ningún resultado; esta se eligió DESPUÉS de ver el diagnóstico
        # del 14 de agosto, así que tiene ventaja injusta sobre la misma
        # muestra. Su media no vale nada por sí sola: hay que creerle solo si
        # la partición temporal aguanta, y aun así con reservas.
        "6_solo_lo_que_predice": (replace(base, weights=PREDICTIVE_WEIGHTS), bt),
    }


DEFAULT_PARAMS = StrategyParams()


# ── Calibración de confianza ──────────────────────────────────────────────────

# Tramos de puntuación sobre los que se mide el acierto histórico. La confianza
# de una señal nueva es el acierto del tramo al que pertenece, corregido a la
# baja por incertidumbre (ver risk.wilson_lower_bound).
SCORE_BUCKETS: tuple[tuple[float, float], ...] = (
    (50.0, 60.0),
    (60.0, 70.0),
    (70.0, 80.0),
    (80.0, 100.01),
)

# Muestras mínimas para publicar la confianza de un tramo. Por debajo, la
# corrección de Wilson ya castiga el número, pero además lo marcamos como
# poco fiable en el mensaje.
MIN_BUCKET_SAMPLES = 30

# Confianza usada cuando todavía no existe calibration.json. Deliberadamente
# baja y marcada como no calibrada: un bot recién instalado no ha demostrado nada.
UNCALIBRATED_CONFIDENCE = 0.0

# Penalización máxima que el filtro de noticias puede aplicar. Gemini solo resta.
MAX_LLM_PENALTY = 25.0

# Ventaja mínima para considerarla útil, en múltiplos de R.
#
# Una esperanza de +0.005R es estadísticamente positiva y económicamente nada:
# al imprimirla con dos decimales sale "+0.00R" mientras el mensaje afirma que
# la señal gana. Esa contradicción apareció en la primera alerta real, y no era
# un fallo de formato sino de criterio: "mayor que cero" no es lo mismo que
# "suficiente para operar". Por debajo de este umbral la alerta lo dice.
MIN_MEANINGFUL_EXPECTANCY = 0.05


@dataclass(frozen=True)
class BacktestParams:
    """Ajustes del motor de backtest."""

    years: int = 5
    # Días hábiles que la orden condicional sigue viva. Si el precio no entra
    # en la zona de compra en este plazo, la señal se descarta sin operar.
    entry_valid_days: int = 2
    # Días máximos en posición. Evita que una operación zombi distorsione
    # las estadísticas durante meses.
    max_holding_days: int = 30
    # Coste de ida y vuelta como fracción del precio de entrada. Quantfury no
    # cobra comisión pero opera sobre el spread; esto lo aproxima.
    round_trip_cost: float = 0.0010
    # Si una misma vela toca stop y objetivo, no sabemos cuál llegó primero.
    # Contarlo como pérdida sesga el backtest en contra, que es el lado
    # correcto en el que equivocarse.
    ambiguous_bar_is_loss: bool = True

    # ── Gestión de la salida ─────────────────────────────────────────────────
    # Stop dinámico tipo "chandelier": el stop sube a `máximo alcanzado −
    # N × ATR` y nunca baja. En 0 queda desactivado y la salida es la de
    # siempre (stop fijo u objetivo), que es el comportamiento en vivo.
    #
    # Es el único cambio que puede subir la esperanza sin tocar los filtros:
    # aflojar filtros da MÁS operaciones, no mejores. Recortar a las
    # perdedoras da mejores. Pero se mide antes de adoptarlo.
    trail_atr_mult: float = 0.0
    # Con el objetivo fijo, una ganadora se corta en 3 ATR aunque siguiera
    # subiendo. Activando esto el objetivo deja de cerrar la operación y solo
    # sale el trailing: es la hipótesis de "dejar correr la cola derecha".
    # Sin `trail_atr_mult` no tiene efecto (no habría con qué salir).
    let_winners_run: bool = False


DEFAULT_BACKTEST = BacktestParams()


__all__ = [
    "StrategyParams",
    "BacktestParams",
    "DEFAULT_PARAMS",
    "DEFAULT_BACKTEST",
    "SCORE_BUCKETS",
    "MIN_BUCKET_SAMPLES",
    "MAX_LLM_PENALTY",
    "UNCALIBRATED_CONFIDENCE",
    "is_trading_day",
    "replace",
]
