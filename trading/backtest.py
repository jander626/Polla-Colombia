"""Motor de backtest y generación de la tabla de calibración.

Este módulo es la puerta de decisión del proyecto. Su trabajo no es producir
un número bonito, sino responder con honestidad a una pregunta: *¿esta
estrategia tiene ventaja real, y cuánta?*

Por eso todas las decisiones ambiguas se resuelven en contra de la estrategia:

- Si una misma vela toca el stop y el objetivo, no sabemos cuál llegó primero.
  Se cuenta como pérdida.
- La entrada es condicional y caduca: si el precio no entra en la zona de
  compra en pocos días, la señal se descarta sin operar, igual que en vivo.
- Se descuenta un coste de ida y vuelta que aproxima el spread de Quantfury.
- Los huecos se ejecutan al precio de apertura, no al nivel teórico: un hueco
  bajista por debajo del stop sale peor que el stop, como en la realidad.

Aun así, **el resultado sigue siendo una cota superior**. No captura el
slippage real, ni que el usuario pueda no estar disponible para colocar la
orden, ni el sesgo de supervivencia del universo (ver `notes` del informe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    MIN_MEANINGFUL_EXPECTANCY,
    BacktestParams,
    StrategyParams,
    replace,
)
from .risk import Calibration
from .strategy import Signal, compute_features, signals_from_features
from .universe import Instrument

Outcome = str  # "win" | "loss" | "timeout" | "no_fill"


@dataclass(frozen=True)
class Trade:
    symbol: str
    asset_class: str
    signal_date: pd.Timestamp
    score: float
    outcome: Outcome
    # Componentes de la puntuación en el momento de la señal. Se arrastran
    # hasta aquí para poder medir cuáles predicen de verdad el resultado: la
    # primera medición reveló que la puntuación agregada no ordenaba, y sin
    # este desglose no hay forma de saber qué parte la estaba estropeando.
    components: dict[str, float] = field(default_factory=dict)
    entry_date: Optional[pd.Timestamp] = None
    entry_price: float = float("nan")
    exit_date: Optional[pd.Timestamp] = None
    exit_price: float = float("nan")
    stop: float = float("nan")
    target: float = float("nan")
    r_multiple: float = float("nan")
    return_pct: float = float("nan")
    bars_held: int = 0

    @property
    def was_filled(self) -> bool:
        return self.outcome != "no_fill"

    @property
    def is_win(self) -> bool:
        """Ganadora = terminó con retorno positivo tras costes.

        Se define por el resultado económico y no por "tocó el objetivo", para
        que las salidas por tiempo cuenten en el lado que les corresponde.
        """
        return self.was_filled and np.isfinite(self.return_pct) and self.return_pct > 0


def _stop_outcome(exit_price: float, entry_price: float) -> Outcome:
    """Salir por el stop no implica perder cuando el stop ha subido.

    Con trailing, un stop alcanzado por encima de la entrada es una ganadora
    recortada, no una pérdida. Etiquetarla como "loss" haría que el porcentaje
    de acierto contase al revés justo en las operaciones que el trailing salva.
    """
    return "win" if exit_price > entry_price else "loss"


def simulate_signal(
    signal: Signal,
    df: pd.DataFrame,
    bt: BacktestParams,
) -> Trade:
    """Simula una señal barra a barra desde el día siguiente a su generación."""
    levels = signal.levels
    index = df.index

    try:
        signal_pos = index.get_loc(signal.bar_date)
    except KeyError:
        return Trade(
            symbol=signal.symbol,
            asset_class=signal.asset_class,
            signal_date=signal.bar_date,
            score=signal.score,
            outcome="no_fill",
            components=dict(signal.components),
        )

    base = Trade(
        symbol=signal.symbol,
        asset_class=signal.asset_class,
        signal_date=signal.bar_date,
        score=signal.score,
        outcome="no_fill",
        stop=levels.stop,
        target=levels.target,
        components=dict(signal.components),
    )

    # ── Fase 1: ¿se llega a ejecutar la orden condicional? ────────────────────
    entry_pos: int | None = None
    entry_price = float("nan")

    first = signal_pos + 1
    last = min(signal_pos + bt.entry_valid_days, len(df) - 1)
    for pos in range(first, last + 1):
        bar = df.iloc[pos]
        if bar["open"] <= levels.entry_max:
            entry_pos, entry_price = pos, float(bar["open"])
            break
        if bar["low"] <= levels.entry_max:
            entry_pos, entry_price = pos, float(levels.entry_max)
            break

    if entry_pos is None or not math.isfinite(entry_price) or entry_price <= 0:
        return base

    if entry_price - levels.stop <= 0:
        # Un hueco bajista puede dejar la entrada por debajo del stop; la
        # operación deja de tener sentido y no se toma.
        return base

    # R se mide contra el riesgo PLANIFICADO (techo de entrada - stop), no
    # contra el riesgo realizado. Es lo económicamente fiel: el usuario
    # dimensiona la posición al colocar la orden, cuando solo conoce esos dos
    # números. Si luego entra más barato por un hueco, su pérdida en el stop es
    # menor de 1R — y usar el riesgo realizado lo contaría como 1R completo,
    # exagerando las pérdidas y amplificando artificialmente las ganancias de
    # las entradas con hueco.
    planned_risk = levels.entry_max - levels.stop

    # ── Fase 2: resolución por primer toque ──────────────────────────────────
    exit_pos = min(entry_pos + bt.max_holding_days, len(df) - 1)
    outcome: Outcome = "timeout"
    exit_price = float(df.iloc[exit_pos]["close"])
    exit_at = exit_pos

    # El stop puede moverse (trailing), el objetivo no. `stop` arranca en el
    # nivel planificado y solo sube.
    stop = float(levels.stop)
    trailing = bt.trail_atr_mult > 0 and math.isfinite(signal.atr) and signal.atr > 0
    # Sin objetivo que cierre, el trailing es lo único que saca de la posición.
    target_exits = not (trailing and bt.let_winners_run)

    for pos in range(entry_pos, exit_pos + 1):
        bar = df.iloc[pos]
        open_, high, low = float(bar["open"]), float(bar["high"]), float(bar["low"])

        # Huecos: la apertura manda sobre el nivel teórico.
        if open_ <= stop:
            outcome, exit_price, exit_at = _stop_outcome(open_, entry_price), open_, pos
            break
        if target_exits and open_ >= levels.target:
            outcome, exit_price, exit_at = "win", open_, pos
            break

        hit_stop = low <= stop
        hit_target = target_exits and high >= levels.target

        if hit_stop and hit_target:
            # Vela ambigua: sin datos intradía no sabemos el orden. Contarla
            # como pérdida sesga el backtest en contra, que es el lado
            # correcto en el que equivocarse.
            if bt.ambiguous_bar_is_loss:
                outcome, exit_price = _stop_outcome(stop, entry_price), stop
            else:
                outcome, exit_price = "win", float(levels.target)
            exit_at = pos
            break
        if hit_stop:
            outcome, exit_price, exit_at = _stop_outcome(stop, entry_price), float(stop), pos
            break
        if hit_target:
            outcome, exit_price, exit_at = "win", float(levels.target), pos
            break

        # La barra no cerró la operación: ahora —y solo ahora— se sube el
        # trailing con el máximo de HOY.
        #
        # El orden importa y es la diferencia entre un backtest honesto y uno
        # que mira al futuro: dentro de una vela diaria no se sabe si el máximo
        # llegó antes o después del mínimo. Si se subiera el stop con el máximo
        # de hoy y luego se comparase contra el mínimo de hoy, se estaría
        # cerrando la operación en un nivel que aún no existía. Por eso el stop
        # que se aplica en la barra N solo usa máximos de barras anteriores.
        if trailing:
            stop = max(stop, high - bt.trail_atr_mult * signal.atr)

    gross_return = (exit_price - entry_price) / entry_price
    net_return = gross_return - bt.round_trip_cost
    r_multiple = (exit_price - entry_price) / planned_risk

    return Trade(
        symbol=signal.symbol,
        asset_class=signal.asset_class,
        signal_date=signal.bar_date,
        score=signal.score,
        outcome=outcome,
        components=dict(signal.components),
        entry_date=index[entry_pos],
        entry_price=entry_price,
        exit_date=index[exit_at],
        exit_price=exit_price,
        stop=levels.stop,
        target=levels.target,
        r_multiple=r_multiple,
        return_pct=net_return,
        bars_held=exit_at - entry_pos,
    )


# ── Métricas ──────────────────────────────────────────────────────────────────

@dataclass
class BacktestReport:
    trades: list[Trade] = field(default_factory=list)
    signals_generated: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def filled(self) -> list[Trade]:
        return [t for t in self.trades if t.was_filled]

    @property
    def wins(self) -> list[Trade]:
        return [t for t in self.filled if t.is_win]

    @property
    def win_rate(self) -> float:
        return len(self.wins) / len(self.filled) if self.filled else 0.0

    @property
    def avg_r(self) -> float:
        values = [t.r_multiple for t in self.filled if np.isfinite(t.r_multiple)]
        return float(np.mean(values)) if values else 0.0

    @property
    def profit_factor(self) -> float:
        gains = sum(t.return_pct for t in self.filled if t.return_pct > 0)
        losses = -sum(t.return_pct for t in self.filled if t.return_pct < 0)
        if losses <= 0:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    @property
    def expectancy_r(self) -> float:
        """R esperado por operación. Es la métrica que decide si hay ventaja."""
        return self.avg_r

    @property
    def max_drawdown_r(self) -> float:
        """Máxima caída de la curva de capital medida en múltiplos de R.

        Asume riesgo constante de 1R por señal, sin límite de posiciones
        simultáneas: mide la calidad de la señal, no la de una cartera concreta.
        """
        ordered = sorted(
            (t for t in self.filled if np.isfinite(t.r_multiple)),
            key=lambda t: t.exit_date or t.signal_date,
        )
        equity, peak, worst = 0.0, 0.0, 0.0
        for trade in ordered:
            equity += trade.r_multiple
            peak = max(peak, equity)
            worst = min(worst, equity - peak)
        return worst

    def outcomes(self) -> list[tuple[str, float, bool, float]]:
        """(clase de activo, puntuación, ¿ganó?, R) de cada operación ejecutada."""
        return [
            (t.asset_class, t.score, t.is_win, t.r_multiple)
            for t in self.filled
            if np.isfinite(t.r_multiple)
        ]

    def to_calibration(self, params: StrategyParams | None = None) -> Calibration:
        return Calibration.from_outcomes(
            self.outcomes(),
            notes=" | ".join(self.notes),
            signature=params.signature if params is not None else "",
        )

    def component_diagnostics(self, quantiles: int = 4) -> str:
        """Mide si cada componente de la puntuación predice el resultado.

        Divide las operaciones en cuartiles según el valor de cada componente y
        compara el R medio del cuartil más bajo con el más alto. Un componente
        útil produce una diferencia positiva y clara; uno inútil da ruido, y uno
        con la diferencia invertida está *empeorando* la puntuación agregada y
        conviene quitarlo o darle la vuelta.

        Es el diagnóstico que faltaba cuando la primera medición mostró que la
        puntuación no ordenaba: sin él solo se sabía que algo iba mal, no qué.
        """
        filled = [t for t in self.filled if t.components and np.isfinite(t.r_multiple)]
        if len(filled) < quantiles * 10:
            return "Muestra insuficiente para diagnosticar los componentes."

        keys = sorted({k for t in filled for k in t.components})
        lines = [
            f"{'Componente':<20} {'Q1 (bajo)':>12} {'Q4 (alto)':>12} "
            f"{'Δ R':>9} {'Veredicto':>12}",
        ]

        for key in keys:
            usable = [t for t in filled if key in t.components]
            if len(usable) < quantiles * 10:
                continue

            values = np.array([t.components[key] for t in usable])
            r_values = np.array([t.r_multiple for t in usable])
            edges = np.quantile(values, np.linspace(0, 1, quantiles + 1))

            # Un componente constante no tiene cuartiles que comparar.
            if edges[0] == edges[-1]:
                lines.append(f"{key:<20} {'—':>12} {'—':>12} {'—':>9} {'constante':>12}")
                continue

            low_mask = values <= edges[1]
            high_mask = values >= edges[quantiles - 1]
            if low_mask.sum() < 5 or high_mask.sum() < 5:
                continue

            low_r = float(r_values[low_mask].mean())
            high_r = float(r_values[high_mask].mean())
            delta = high_r - low_r

            if delta > 0.15:
                verdict = "predice"
            elif delta < -0.15:
                verdict = "INVERTIDO"
            else:
                verdict = "ruido"

            lines.append(
                f"{key:<20} {low_r:>+12.3f} {high_r:>+12.3f} "
                f"{delta:>+9.3f} {verdict:>12}"
            )

        lines.append(
            "\n  'predice' = el cuartil alto rinde más que el bajo, como debe ser."
            "\n  'INVERTIDO' = el componente está restando: penaliza justo lo que"
            "\n                debería premiar, y arrastra a la puntuación agregada."
            "\n  'ruido' = no aporta información; su peso podría repartirse mejor."
        )
        return "\n".join(lines)

    def out_of_sample_split(self, train_fraction: float = 0.6) -> str:
        """Divide las operaciones por fecha y compara las dos mitades.

        Es la defensa contra el autoengaño. Estamos ajustando la estrategia
        sobre la misma muestra con la que la medimos, así que cualquier mejora
        está en parte inflada. Una ventaja que aparece en el primer tramo y
        desaparece en el segundo no es una ventaja: es un ajuste a los datos.

        No es una validación fuera de muestra estricta —los parámetros se
        eligieron viendo el conjunto entero—, pero detecta el caso más común y
        más caro: que todo el resultado venga de un periodo concreto.
        """
        filled = sorted(
            (t for t in self.filled if np.isfinite(t.r_multiple)),
            key=lambda t: t.signal_date,
        )
        if len(filled) < 60:
            return "Muestra insuficiente para partir en dos."

        cut = int(len(filled) * train_fraction)
        first, second = filled[:cut], filled[cut:]

        def describe(name: str, trades: list[Trade]) -> str:
            r = np.array([t.r_multiple for t in trades])
            wins = sum(1 for t in trades if t.is_win)
            se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else float("nan")
            lower = r.mean() - 1.96 * se
            return (
                f"  {name:<12} {trades[0].signal_date:%Y-%m}→{trades[-1].signal_date:%Y-%m}  "
                f"{len(trades):>4} ops  {100 * wins / len(trades):>5.1f}% aciertos  "
                f"R medio {r.mean():>+7.3f}  R mínimo {lower:>+7.3f}"
            )

        lines = [describe("Primer tramo", first), describe("Segundo tramo", second)]

        # El veredicto se decide con el LÍMITE INFERIOR, no con la media.
        #
        # Antes usaba la media, y con eso el informe del 14 de agosto firmó un
        # "✓ la ventaja aparece en los dos tramos" cuando el tramo reciente
        # tenía media +0.174 y límite inferior +0.000: es decir, ninguna
        # ventaja demostrada. Es el mismo error que ya se corrigió en las
        # alertas —confundir "mayor que cero" con "demostrado"— repetido en el
        # sitio donde más caro sale, que es el que dice si conviene operar.
        def lower_bound(trades: list[Trade]) -> float:
            r = np.array([t.r_multiple for t in trades])
            if len(r) < 2:
                return float("-inf")
            return float(r.mean() - 1.96 * r.std(ddof=1) / np.sqrt(len(r)))

        l1, l2 = lower_bound(first), lower_bound(second)
        floor = MIN_MEANINGFUL_EXPECTANCY

        if l1 >= floor and l2 >= floor:
            lines.append("\n  ✓ La ventaja aparece en los dos tramos.")
        elif l1 >= floor > l2:
            lines.append(
                f"\n  ✗ La ventaja está demostrada en el primer tramo y NO en el\n"
                f"    segundo (límite inferior {l2:+.3f}, hace falta {floor:+.3f}).\n"
                "    Eso apunta a ajuste a los datos, o a que el efecto dejó de\n"
                "    funcionar. La media del tramo reciente puede seguir siendo\n"
                "    positiva: positiva y demostrada no son lo mismo."
            )
        elif l2 >= floor > l1:
            lines.append(
                "\n  ~ Solo hay ventaja demostrada en el tramo reciente. Puede ser\n"
                "    una mejora real del régimen, o casualidad; hace falta más muestra."
            )
        else:
            lines.append(
                f"\n  ✗ Ningún tramo demuestra ventaja (límites inferiores {l1:+.3f}\n"
                f"    y {l2:+.3f}, hace falta {floor:+.3f} en ambos)."
            )
        return "\n".join(lines)

    def headline(self) -> dict:
        """Métricas comparables entre variantes."""
        filled = [t for t in self.filled if np.isfinite(t.r_multiple)]
        if not filled:
            return {"ops": 0, "win_rate": 0.0, "avg_r": 0.0, "lower_r": 0.0, "max_dd": 0.0}
        r = np.array([t.r_multiple for t in filled])
        se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else float("nan")
        return {
            "ops": len(filled),
            "win_rate": len(self.wins) / len(filled),
            "avg_r": float(r.mean()),
            "lower_r": float(r.mean() - 1.96 * se),
            "max_dd": self.max_drawdown_r,
        }

    def score_distribution(self) -> str:
        """Cuántas señales caen en cada tramo de puntuación.

        La primera medición mostró un 95% de las señales entre 50 y 70 y ninguna
        por encima de 80: con la puntuación tan comprimida, elegir «las mejores
        del día» era elegir casi al azar.
        """
        scores = np.array([t.score for t in self.filled])
        if not len(scores):
            return "Sin operaciones."
        return (
            f"Puntuación — mín {scores.min():.1f} · p25 {np.percentile(scores, 25):.1f} · "
            f"mediana {np.median(scores):.1f} · p75 {np.percentile(scores, 75):.1f} · "
            f"máx {scores.max():.1f}"
        )

    def summary(self) -> str:
        filled = self.filled
        if not filled:
            return (
                f"Señales generadas: {self.signals_generated}\n"
                "Ninguna llegó a ejecutarse. Revisa los umbrales de entrada."
            )

        by_class: dict[str, list[Trade]] = {}
        for trade in filled:
            by_class.setdefault(trade.asset_class, []).append(trade)

        lines = [
            "── Resultado del backtest ─────────────────────────────",
            f"Señales generadas   : {self.signals_generated}",
            f"Operaciones abiertas: {len(filled)} "
            f"({100 * len(filled) / max(self.signals_generated, 1):.0f}% de las señales)",
            f"Aciertos            : {len(self.wins)} ({100 * self.win_rate:.1f}%)",
            f"R medio por operación: {self.avg_r:+.3f}",
            f"Factor de beneficio : {self.profit_factor:.2f}",
            f"Máxima caída        : {self.max_drawdown_r:.1f}R",
            "",
            "Por clase de activo:",
        ]
        for asset_class, trades in sorted(by_class.items()):
            wins = sum(1 for t in trades if t.is_win)
            avg = np.mean([t.r_multiple for t in trades if np.isfinite(t.r_multiple)])
            lines.append(
                f"  {asset_class:<6} {len(trades):>5} ops  "
                f"{100 * wins / len(trades):>5.1f}% aciertos  R medio {avg:+.3f}"
            )

        if self.notes:
            lines += ["", "Advertencias:"] + [f"  • {n}" for n in self.notes]
        return "\n".join(lines)


# ── Ejecución ─────────────────────────────────────────────────────────────────

def run_backtest(
    instruments: list[Instrument],
    bars: dict[str, pd.DataFrame],
    params: StrategyParams,
    bt: BacktestParams,
    benchmark_close: Optional[pd.Series] = None,
) -> BacktestReport:
    """Recorre el universo generando y simulando señales."""
    report = BacktestReport()

    for instrument in instruments:
        df = bars.get(instrument.symbol)
        if df is None or len(df) < params.min_bars:
            continue

        has_volume = not instrument.is_forex
        benchmark = None if instrument.is_forex else benchmark_close

        try:
            feats = compute_features(df, params, benchmark, has_volume=has_volume)
            signals = signals_from_features(instrument, feats, params)
        except Exception as exc:
            print(f"[WARN] {instrument.symbol}: fallo al generar señales ({exc})")
            continue

        report.signals_generated += len(signals)
        for signal in signals:
            report.trades.append(simulate_signal(signal, df, bt))

    report.notes.append(
        "Sesgo de supervivencia: el universo son los instrumentos líquidos de hoy, "
        "así que el resultado histórico sale mejor de lo que habría sido en vivo."
    )
    report.notes.append(
        "No incluye slippage real ni el riesgo de no colocar la orden a tiempo; "
        "tómalo como cota superior."
    )
    return report


# ── Comparación de variantes ──────────────────────────────────────────────────

def compare_variants(
    named_params: dict[str, tuple[StrategyParams, BacktestParams]],
    instruments: list[Instrument],
    bars: dict[str, pd.DataFrame],
    default_bt: BacktestParams,
    benchmark_close: Optional[pd.Series] = None,
) -> tuple[str, dict[str, BacktestReport]]:
    """Mide varias variantes sobre los mismos datos y las publica TODAS.

    Reportar solo la ganadora sería hacer trampa: con suficientes intentos,
    alguna variante siempre parece buena por azar. Verlas todas, con su límite
    inferior y su partición temporal, permite distinguir una mejora real de una
    casualidad afortunada.

    Cada variante trae sus propios ajustes de backtest además de los de
    estrategia, porque la gestión de la salida (trailing) es una palanca tan
    legítima como los filtros y hay que poder medirla por separado.
    """
    reports: dict[str, BacktestReport] = {}
    used_bt: dict[str, BacktestParams] = {}
    for name, spec in named_params.items():
        params, bt = spec if isinstance(spec, tuple) else (spec, default_bt)
        print(f"[INFO] Midiendo variante {name}…")
        reports[name] = run_backtest(instruments, bars, params, bt, benchmark_close)
        used_bt[name] = bt

    lines = [
        f"{'Variante':<20} {'Señales':>8} {'/año':>7} {'Ops':>6} {'Acierto':>9} "
        f"{'R medio':>9} {'R mínimo':>9} {'Caída máx':>10}",
    ]
    for name, report in reports.items():
        h = report.headline()
        years = max(used_bt[name].years, 1)
        lines.append(
            f"{name:<20} {report.signals_generated:>8} "
            f"{report.signals_generated / years:>7.0f} {h['ops']:>6} "
            f"{100 * h['win_rate']:>8.1f}% {h['avg_r']:>+9.3f} "
            f"{h['lower_r']:>+9.3f} {h['max_dd']:>9.1f}R"
        )

    lines.append(
        "\n  '/año' es la frecuencia: cuántas alertas al año produciría. Dividido\n"
        "  entre 252 sesiones da las que verías al día. Es la columna que explica\n"
        "  una semana en silencio, pero NO es la que decide si vale la pena.\n"
        "\n  'R mínimo' es la ventaja que sobrevive a su propia incertidumbre.\n"
        "  Si es negativo, la variante NO ha demostrado nada, por buena que\n"
        "  parezca su media, ni por muchas señales que produzca. Es la única\n"
        "  columna que decide."
    )
    return "\n".join(lines), reports


# ── Búsqueda sistemática ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class SearchResult:
    """Una combinación medida, con sus números dentro y fuera de muestra.

    Los dos juegos de números van juntos a propósito. Buscar sobre 81
    combinaciones garantiza que alguna parezca excelente por azar: la que gana
    lo hace, en parte, por haber tenido suerte en esta muestra concreta. La
    única defensa es mirar cómo se comporta en el tramo que no se usó para
    elegirla, y para poder mirarlo hay que publicarlo al lado.
    """

    label: str
    trail: float
    params: StrategyParams
    bt: BacktestParams
    signals: int
    years: int
    ops: int
    win_rate: float
    avg_r: float
    lower_r: float
    max_dd: float
    train_ops: int
    train_lower: float
    test_ops: int
    test_avg: float
    test_lower: float

    @property
    def name(self) -> str:
        return f"{self.label}·trail{self.trail:g}"

    @property
    def signals_per_year(self) -> float:
        return self.signals / max(self.years, 1)

    @property
    def is_credible(self) -> bool:
        """Ventaja demostrada en el tramo que NO se usó para elegirla."""
        return self.test_lower >= MIN_MEANINGFUL_EXPECTANCY


def _half_metrics(trades: list[Trade], train_fraction: float):
    """(ops_train, lower_train, ops_test, media_test, lower_test)."""
    usable = sorted(
        (t for t in trades if t.was_filled and np.isfinite(t.r_multiple)),
        key=lambda t: t.signal_date,
    )
    if len(usable) < 40:
        return 0, float("nan"), 0, float("nan"), float("nan")

    cut = int(len(usable) * train_fraction)
    train, test = usable[:cut], usable[cut:]

    def lower(chunk: list[Trade]) -> float:
        r = np.array([t.r_multiple for t in chunk])
        if len(r) < 2:
            return float("nan")
        return float(r.mean() - 1.96 * r.std(ddof=1) / np.sqrt(len(r)))

    test_r = np.array([t.r_multiple for t in test])
    return (
        len(train),
        lower(train),
        len(test),
        float(test_r.mean()) if len(test_r) else float("nan"),
        lower(test),
    )


def run_search(
    instruments: list[Instrument],
    bars: dict[str, pd.DataFrame],
    grid: dict[str, StrategyParams],
    trails: tuple[float, ...],
    base_bt: BacktestParams,
    benchmark_close: Optional[pd.Series] = None,
    train_fraction: float = 0.6,
) -> list[SearchResult]:
    """Mide toda la rejilla en una sola pasada sobre los datos.

    La parte cara es generar las señales (indicadores sobre 141 instrumentos);
    simularlas es casi gratis. Y el trailing no cambia QUÉ señales existen,
    solo cómo se cierran. Así que por cada combinación de estrategia se generan
    las señales una vez y se simulan con todas las salidas: 27 cálculos de
    indicadores en vez de 81.
    """
    results: list[SearchResult] = []

    for position, (label, params) in enumerate(grid.items(), start=1):
        print(f"[INFO] ({position}/{len(grid)}) {label}…")

        generated = 0
        pending: list[tuple[Signal, pd.DataFrame]] = []
        for instrument in instruments:
            df = bars.get(instrument.symbol)
            if df is None or len(df) < params.min_bars:
                continue
            benchmark = None if instrument.is_forex else benchmark_close
            try:
                feats = compute_features(
                    df, params, benchmark, has_volume=not instrument.is_forex
                )
                signals = signals_from_features(instrument, feats, params)
            except Exception as exc:
                print(f"[WARN] {instrument.symbol}: {exc}")
                continue
            generated += len(signals)
            pending.extend((signal, df) for signal in signals)

        for trail in trails:
            bt = replace(base_bt, trail_atr_mult=trail)
            report = BacktestReport()
            report.signals_generated = generated
            report.trades = [simulate_signal(s, df, bt) for s, df in pending]

            head = report.headline()
            train_ops, train_lower, test_ops, test_avg, test_lower = _half_metrics(
                report.trades, train_fraction
            )
            results.append(
                SearchResult(
                    label=label, trail=trail, params=params, bt=bt,
                    signals=generated, years=bt.years, ops=head["ops"],
                    win_rate=head["win_rate"], avg_r=head["avg_r"],
                    lower_r=head["lower_r"], max_dd=head["max_dd"],
                    train_ops=train_ops, train_lower=train_lower,
                    test_ops=test_ops, test_avg=test_avg, test_lower=test_lower,
                )
            )

    return results


def format_search(results: list[SearchResult], top: int = 15) -> str:
    """Ranking por el tramo que NO se usó para elegir, no por el total.

    Ordenar por el resultado completo sería ordenar por cuánto se ajustó cada
    combinación a esta muestra concreta: la primera de la lista sería, por
    construcción, la que mejor memorizó estos cinco años. Ordenar por el tramo
    reciente —que ninguna de las 81 vio al ser definida— no elimina el
    problema, pero sí lo reduce a lo que de verdad importa: qué seguía
    funcionando al final del periodo.
    """
    if not results:
        return "Sin resultados: ninguna combinación produjo operaciones."

    usable = [r for r in results if r.test_ops > 0 and np.isfinite(r.test_lower)]
    if not usable:
        return (
            "Ninguna combinación dejó operaciones suficientes en el tramo de\n"
            "validación. La rejilla es demasiado estricta para esta muestra."
        )

    ranked = sorted(usable, key=lambda r: r.test_lower, reverse=True)
    credible = [r for r in ranked if r.is_credible]

    lines = [
        f"{'Combinación':<34} {'/año':>6} {'Ops':>5} {'Acier':>7} "
        f"{'R med':>8} {'R mín':>8} {'Val.ops':>8} {'Val.R':>8} {'Val.mín':>8} {'Caída':>8}",
    ]
    for r in ranked[:top]:
        lines.append(
            f"{r.name:<34} {r.signals_per_year:>6.0f} {r.ops:>5} "
            f"{100 * r.win_rate:>6.1f}% {r.avg_r:>+8.3f} {r.lower_r:>+8.3f} "
            f"{r.test_ops:>8} {r.test_avg:>+8.3f} {r.test_lower:>+8.3f} "
            f"{r.max_dd:>7.1f}R"
        )

    baseline = next(
        (r for r in results if r.label.startswith("rsi45·actual·adx20") and r.trail == 0),
        None,
    )
    if baseline is not None and baseline not in ranked[:top]:
        lines.append("  " + "─" * 106)
        lines.append(
            f"{'(actual) ' + baseline.name:<34} {baseline.signals_per_year:>6.0f} "
            f"{baseline.ops:>5} {100 * baseline.win_rate:>6.1f}% "
            f"{baseline.avg_r:>+8.3f} {baseline.lower_r:>+8.3f} "
            f"{baseline.test_ops:>8} {baseline.test_avg:>+8.3f} "
            f"{baseline.test_lower:>+8.3f} {baseline.max_dd:>7.1f}R"
        )

    lines.append(
        f"\n  Se midieron {len(results)} combinaciones. 'Val.' es el 40% final del\n"
        "  periodo, que ninguna combinación vio al ser definida; el ranking va por\n"
        f"  'Val.mín'. Con ventaja demostrada ahí (≥ {MIN_MEANINGFUL_EXPECTANCY:+.3f}): "
        f"{len(credible)} de {len(results)}."
    )

    if not credible:
        lines.append(
            "\n  ✗ NINGUNA combinación demuestra ventaja en el tramo de validación.\n"
            "    Con 81 intentos sobre la misma muestra, que la mejor no pase el\n"
            "    listón no es mala suerte: es la respuesta. Esta familia de\n"
            "    estrategias dejó de funcionar en el periodo reciente, y afinarle\n"
            "    los umbrales no lo va a arreglar."
        )
    else:
        best = ranked[0]
        lines.append(
            f"\n  Mejor por validación: {best.name}\n"
            f"    {best.signals_per_year:.0f} señales/año · {best.test_ops} ops de validación · "
            f"límite inferior {best.test_lower:+.3f}R\n"
            "\n  Antes de adoptarla: 81 intentos sobre una muestra producen una\n"
            "  ganadora aunque no haya nada que ganar. Que esta pase el listón la\n"
            "  hace candidata, no demostrada. Lo que la demostraría es funcionar\n"
            "  en los meses que vienen, sobre datos que todavía no existen."
        )

    return "\n".join(lines)
