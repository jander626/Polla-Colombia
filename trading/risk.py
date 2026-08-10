"""Niveles de la operación y calibración honesta de la confianza.

Aquí vive la decisión de diseño más importante del bot: **el porcentaje de
confianza no lo inventa un modelo de lenguaje**. Se mide sobre el histórico.

El primer backtest (549 operaciones, 5 años) obligó a rehacer este módulo por
dos motivos que solo se vieron con datos delante:

1. **La puntuación técnica no ordenaba.** El tramo 60-70 acertaba menos que el
   50-60, y ninguna señal superaba 80. Calibrar por puntuación era publicar
   ruido con aspecto de orden, así que la calibración pasa a segmentarse por
   clase de activo —que sí discrimina— y los tramos de puntuación se conservan
   solo como diagnóstico.

2. **Publicar el acierto a secas engaña** cuando los pagos son asimétricos.
   Este sistema acierta el 35% de las veces y aun así gana dinero, porque el
   objetivo está al doble de distancia que el stop. Una alerta que dijera
   «confianza 35%» se leería como «esto va a fallar», que es justo lo
   contrario de lo que significa. Por eso ahora se publican las dos cifras:
   acierto histórico y beneficio esperado.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import MIN_BUCKET_SAMPLES, SCORE_BUCKETS, StrategyParams


# ── Estadística ───────────────────────────────────────────────────────────────

def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Límite inferior del intervalo de Wilson para una proporción.

    Se usa en lugar del acierto crudo para que un tramo con 3 operaciones y 2
    aciertos no publique «67% de confianza». Wilson penaliza la falta de
    muestra: con n pequeño el límite inferior se hunde, y solo converge al
    valor real cuando hay evidencia suficiente.
    """
    if total <= 0:
        return 0.0
    if successes < 0 or successes > total:
        raise ValueError("successes debe estar entre 0 y total")

    p = successes / total
    denominator = 1.0 + (z * z) / total
    center = p + (z * z) / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + (z * z) / (4.0 * total * total))
    return max(0.0, (center - margin) / denominator)


def mean_lower_bound(
    total: float, total_squares: float, samples: int, z: float = 1.96
) -> float:
    """Límite inferior de la media, por aproximación normal.

    El equivalente de Wilson para el beneficio esperado: R no es una
    proporción, así que su incertidumbre se acota con el error estándar. Sirve
    para no publicar «+0.50R esperado» cuando eso sale de doce operaciones.
    """
    if samples < 2:
        return 0.0
    mean = total / samples
    variance = (total_squares - samples * mean * mean) / (samples - 1)
    if variance <= 0:
        return mean
    return mean - z * math.sqrt(variance / samples)


# ── Niveles de entrada, stop y objetivo ───────────────────────────────────────

@dataclass(frozen=True)
class Levels:
    entry_max: float      # techo de la orden condicional de compra
    stop: float
    target: float
    risk_reward: float
    risk_per_unit: float
    reward_per_unit: float

    @property
    def is_valid(self) -> bool:
        return (
            self.risk_per_unit > 0
            and self.reward_per_unit > 0
            and self.stop < self.entry_max < self.target
        )


def compute_levels(
    close: float,
    atr: float,
    swing_low: float,
    params: StrategyParams,
) -> Levels | None:
    """Calcula los tres precios de la operación. Devuelve None si no son coherentes.

    La entrada es un *techo*, no un precio fijo: el escaneo corre antes de la
    apertura y no sabemos a qué precio abrirá. Si el instrumento abre con un
    hueco alcista fuerte, el precio ya no es el que justificaba la señal y la
    operación simplemente no se ejecuta. Esto evita perseguir.

    El stop se apoya en estructura real de precio —el mínimo del retroceso—,
    con una distancia mínima para no salir por puro ruido intradía. Un
    retroceso muy profundo produce un stop lejano y un ratio riesgo/beneficio
    pobre, y la señal acaba descartada por el filtro de R:B; eso es deliberado,
    porque un retroceso así ya no es un descanso sino un cambio de tendencia.

    El objetivo es puro ATR. El máximo reciente no lo recorta: ver la nota en
    `StrategyParams.resistance_lookback`.
    """
    if not all(math.isfinite(x) for x in (close, atr, swing_low)):
        return None
    if close <= 0 or atr <= 0:
        return None

    entry_max = close + params.entry_buffer_atr * atr

    stop_by_structure = swing_low - params.stop_buffer_atr * atr
    stop_min_distance = entry_max - params.min_stop_atr * atr
    stop = min(stop_by_structure, stop_min_distance)

    target = entry_max + params.target_atr_mult * atr

    risk = entry_max - stop
    reward = target - entry_max
    if risk <= 0 or reward <= 0:
        return None

    return Levels(
        entry_max=entry_max,
        stop=stop,
        target=target,
        risk_reward=reward / risk,
        risk_per_unit=risk,
        reward_per_unit=reward,
    )


# ── Calibración ───────────────────────────────────────────────────────────────

@dataclass
class OutcomeStats:
    """Resultado histórico de un grupo de operaciones.

    Se usa para dos cosas distintas: los segmentos por clase de activo (que
    alimentan la confianza publicada) y los tramos de puntuación (que solo
    sirven para comprobar si la puntuación discrimina).
    """

    label: str
    samples: int = 0
    wins: int = 0
    sum_r: float = 0.0
    sum_r_squared: float = 0.0

    def add(self, won: bool, r_multiple: float) -> None:
        self.samples += 1
        self.wins += int(won)
        if math.isfinite(r_multiple):
            self.sum_r += r_multiple
            self.sum_r_squared += r_multiple * r_multiple

    @property
    def win_rate(self) -> float:
        return self.wins / self.samples if self.samples else 0.0

    @property
    def win_rate_lower(self) -> float:
        """Acierto publicable (0-100), corregido a la baja por incertidumbre."""
        return 100.0 * wilson_lower_bound(self.wins, self.samples)

    @property
    def expectancy_r(self) -> float:
        return self.sum_r / self.samples if self.samples else 0.0

    @property
    def expectancy_lower(self) -> float:
        return mean_lower_bound(self.sum_r, self.sum_r_squared, self.samples)

    @property
    def is_reliable(self) -> bool:
        return self.samples >= MIN_BUCKET_SAMPLES

    @property
    def has_edge(self) -> bool:
        """Ventaja que sobrevive a su propia incertidumbre.

        Una esperanza positiva cuyo límite inferior es negativo no es una
        ventaja demostrada: es ruido que casualmente salió a favor.
        """
        return self.expectancy_lower > 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "samples": self.samples,
            "wins": self.wins,
            "sum_r": round(self.sum_r, 6),
            "sum_r_squared": round(self.sum_r_squared, 6),
            "win_rate": round(self.win_rate, 4),
            "win_rate_lower": round(self.win_rate_lower, 2),
            "expectancy_r": round(self.expectancy_r, 4),
            "expectancy_lower": round(self.expectancy_lower, 4),
            "reliable": self.is_reliable,
            "has_edge": self.has_edge,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "OutcomeStats":
        return cls(
            label=str(raw.get("label", "")),
            samples=int(raw.get("samples", 0)),
            wins=int(raw.get("wins", 0)),
            sum_r=float(raw.get("sum_r", 0.0)),
            sum_r_squared=float(raw.get("sum_r_squared", 0.0)),
        )


@dataclass
class Calibration:
    """Tabla de resultados históricos, producida por el backtest.

    `by_asset_class` es lo que alimenta las alertas. `by_score` se conserva
    únicamente como diagnóstico: mientras la puntuación no ordene de forma
    monótona, calibrar con ella sería publicar ruido.
    """

    by_asset_class: dict[str, OutcomeStats] = field(default_factory=dict)
    by_score: list[OutcomeStats] = field(default_factory=list)
    generated_at: str = ""
    total_signals: int = 0
    notes: str = ""

    @classmethod
    def empty(cls) -> "Calibration":
        return cls()

    @property
    def is_calibrated(self) -> bool:
        return any(stats.samples > 0 for stats in self.by_asset_class.values())

    @classmethod
    def from_outcomes(
        cls, outcomes: list[tuple[str, float, bool, float]], notes: str = ""
    ) -> "Calibration":
        """Construye la tabla desde (clase de activo, puntuación, ¿ganó?, R)."""
        by_class: dict[str, OutcomeStats] = {}
        by_score = [
            OutcomeStats(label=f"{low:.0f}-{min(high, 100):.0f}")
            for low, high in SCORE_BUCKETS
        ]

        for asset_class, score, won, r_multiple in outcomes:
            stats = by_class.setdefault(asset_class, OutcomeStats(label=asset_class))
            stats.add(won, r_multiple)

            for bucket, (low, high) in zip(by_score, SCORE_BUCKETS):
                if low <= score < high:
                    bucket.add(won, r_multiple)
                    break

        return cls(
            by_asset_class=by_class,
            by_score=by_score,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            total_signals=len(outcomes),
            notes=notes,
        )

    def for_asset_class(self, asset_class: str) -> OutcomeStats | None:
        stats = self.by_asset_class.get(asset_class)
        return stats if stats and stats.samples > 0 else None

    def score_ranks_correctly(self) -> bool:
        """¿La puntuación ordena? Solo cierto si el acierto crece de forma monótona.

        Es la comprobación que descalificó a la puntuación como base de la
        calibración en la primera medición. Se mantiene para detectar cuándo
        vuelve a ser utilizable.
        """
        usable = [b for b in self.by_score if b.is_reliable]
        if len(usable) < 2:
            return False
        rates = [b.win_rate for b in usable]
        return all(earlier <= later for earlier, later in zip(rates, rates[1:]))

    # ── Persistencia ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total_signals": self.total_signals,
            "notes": self.notes,
            "score_ranks_correctly": self.score_ranks_correctly(),
            "by_asset_class": {k: v.to_dict() for k, v in self.by_asset_class.items()},
            "by_score": [b.to_dict() for b in self.by_score],
        }

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        if not os.path.exists(path):
            return cls.empty()
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] calibration.json ilegible ({exc}); se usará sin calibrar")
            return cls.empty()

        return cls(
            by_asset_class={
                k: OutcomeStats.from_dict(v)
                for k, v in (raw.get("by_asset_class") or {}).items()
            },
            by_score=[OutcomeStats.from_dict(b) for b in raw.get("by_score", [])],
            generated_at=raw.get("generated_at", ""),
            total_signals=int(raw.get("total_signals", 0)),
            notes=raw.get("notes", ""),
        )

    # ── Informes ──────────────────────────────────────────────────────────────

    def summary_table(self) -> str:
        if not self.is_calibrated:
            return "Sin calibración: ejecuta `python trading_bot.py backtest --write`."

        lines = [
            "Por clase de activo (es lo que alimenta las alertas):",
            f"  {'Clase':<8} {'Ops':>6} {'Acierto':>9} {'Publicado':>10} "
            f"{'R medio':>9} {'R mínimo':>9} {'Ventaja':>8}",
        ]
        for stats in sorted(
            self.by_asset_class.values(), key=lambda s: s.samples, reverse=True
        ):
            lines.append(
                f"  {stats.label:<8} {stats.samples:>6} "
                f"{100 * stats.win_rate:>8.1f}% {stats.win_rate_lower:>9.1f}% "
                f"{stats.expectancy_r:>+9.3f} {stats.expectancy_lower:>+9.3f} "
                f"{'sí' if stats.has_edge else 'no':>8}"
            )

        lines += [
            "",
            "Por tramo de puntuación (solo diagnóstico):",
            f"  {'Tramo':<8} {'Ops':>6} {'Acierto':>9} {'R medio':>9} {'Fiable':>8}",
        ]
        for bucket in self.by_score:
            lines.append(
                f"  {bucket.label:<8} {bucket.samples:>6} "
                f"{100 * bucket.win_rate:>8.1f}% {bucket.expectancy_r:>+9.3f} "
                f"{'sí' if bucket.is_reliable else 'no':>8}"
            )

        if self.score_ranks_correctly():
            lines.append("\n  ✓ La puntuación ordena: el acierto crece con ella.")
        else:
            lines.append(
                "\n  ✗ La puntuación NO ordena: el acierto no crece con ella, así que\n"
                "    no se usa para calibrar. Sirve solo para elegir qué señales enviar."
            )
        return "\n".join(lines)


def apply_llm_penalty(confidence: float, penalty: float, max_penalty: float) -> float:
    """Aplica la penalización por riesgo de evento.

    El filtro de noticias solo puede *restar*. Dejarle sumar confianza
    reintroduciría por la puerta de atrás el número inventado que toda esta
    calibración existe para eliminar.
    """
    bounded = max(0.0, min(float(penalty), max_penalty))
    return max(0.0, confidence - bounded)
