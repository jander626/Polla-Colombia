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

from .config import BacktestParams, StrategyParams
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
        )

    base = Trade(
        symbol=signal.symbol,
        asset_class=signal.asset_class,
        signal_date=signal.bar_date,
        score=signal.score,
        outcome="no_fill",
        stop=levels.stop,
        target=levels.target,
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

    for pos in range(entry_pos, exit_pos + 1):
        bar = df.iloc[pos]
        open_, high, low = float(bar["open"]), float(bar["high"]), float(bar["low"])

        # Huecos: la apertura manda sobre el nivel teórico.
        if open_ <= levels.stop:
            outcome, exit_price, exit_at = "loss", open_, pos
            break
        if open_ >= levels.target:
            outcome, exit_price, exit_at = "win", open_, pos
            break

        hit_stop = low <= levels.stop
        hit_target = high >= levels.target

        if hit_stop and hit_target:
            # Vela ambigua: sin datos intradía no sabemos el orden. Contarla
            # como pérdida sesga el backtest en contra, que es el lado
            # correcto en el que equivocarse.
            outcome = "loss" if bt.ambiguous_bar_is_loss else "win"
            exit_price = levels.stop if bt.ambiguous_bar_is_loss else levels.target
            exit_at = pos
            break
        if hit_stop:
            outcome, exit_price, exit_at = "loss", float(levels.stop), pos
            break
        if hit_target:
            outcome, exit_price, exit_at = "win", float(levels.target), pos
            break

    gross_return = (exit_price - entry_price) / entry_price
    net_return = gross_return - bt.round_trip_cost
    r_multiple = (exit_price - entry_price) / planned_risk

    return Trade(
        symbol=signal.symbol,
        asset_class=signal.asset_class,
        signal_date=signal.bar_date,
        score=signal.score,
        outcome=outcome,
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

    def scored_outcomes(self) -> list[tuple[float, bool]]:
        return [(t.score, t.is_win) for t in self.filled]

    def to_calibration(self) -> Calibration:
        return Calibration.from_results(
            self.scored_outcomes(), notes=" | ".join(self.notes)
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
