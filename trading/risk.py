"""Niveles de la operación y calibración honesta de la confianza.

Aquí vive la decisión de diseño más importante del bot: **el porcentaje de
confianza no lo inventa un modelo de lenguaje**. Se mide.

Pedirle a un LLM "dame la probabilidad de que esta operación funcione"
devuelve un número plausible y completamente ficticio, que es peor que no dar
ninguno, porque invita a confiar en él. En su lugar, el backtest mide el
acierto real de cada tramo de puntuación sobre años de historia, y ese número
—corregido a la baja por incertidumbre muestral— es la confianza que se
publica.

Cuando la alerta dice 64%, significa algo comprobable: de las señales
históricas con este perfil, el 64% alcanzó el objetivo antes que el stop.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .config import (
    MIN_BUCKET_SAMPLES,
    SCORE_BUCKETS,
    UNCALIBRATED_CONFIDENCE,
    StrategyParams,
)


# ── Estadística ───────────────────────────────────────────────────────────────

def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Límite inferior del intervalo de Wilson para una proporción.

    Se usa en lugar del acierto crudo para que un tramo con 3 operaciones y 2
    aciertos no publique "67% de confianza". Wilson penaliza la falta de
    muestra: con n pequeño el límite inferior se hunde, y solo converge al
    valor real cuando hay evidencia suficiente. Es la diferencia entre un
    número honesto y uno que solo parece informado.
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
class BucketStats:
    """Resultado histórico de un tramo de puntuación."""

    low: float
    high: float
    samples: int
    wins: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.samples if self.samples else 0.0

    @property
    def confidence(self) -> float:
        """Confianza publicable, en porcentaje (0-100)."""
        return 100.0 * wilson_lower_bound(self.wins, self.samples)

    @property
    def is_reliable(self) -> bool:
        return self.samples >= MIN_BUCKET_SAMPLES


@dataclass
class Calibration:
    """Tabla puntuación → acierto histórico, producida por el backtest."""

    buckets: list[BucketStats]
    generated_at: str = ""
    total_signals: int = 0
    notes: str = ""

    @classmethod
    def empty(cls) -> "Calibration":
        return cls(buckets=[], generated_at="", total_signals=0)

    @classmethod
    def from_results(
        cls, scored_outcomes: list[tuple[float, bool]], notes: str = ""
    ) -> "Calibration":
        """Construye la tabla a partir de pares (puntuación, ¿ganó?)."""
        buckets = [
            BucketStats(low=low, high=high, samples=0, wins=0)
            for low, high in SCORE_BUCKETS
        ]
        for score, won in scored_outcomes:
            for bucket in buckets:
                if bucket.low <= score < bucket.high:
                    bucket.samples += 1
                    bucket.wins += int(won)
                    break

        return cls(
            buckets=buckets,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            total_signals=len(scored_outcomes),
            notes=notes,
        )

    def bucket_for(self, score: float) -> BucketStats | None:
        for bucket in self.buckets:
            if bucket.low <= score < bucket.high:
                return bucket
        return None

    def confidence_for(self, score: float) -> tuple[float, bool]:
        """Devuelve (confianza en %, ¿es fiable?).

        Sin calibración la confianza es 0 y se marca como no fiable: un bot
        recién instalado no ha demostrado nada, y fingir lo contrario sería
        exactamente el problema que este módulo existe para evitar.
        """
        bucket = self.bucket_for(score)
        if bucket is None or bucket.samples == 0:
            return UNCALIBRATED_CONFIDENCE, False
        return bucket.confidence, bucket.is_reliable

    # ── Persistencia ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total_signals": self.total_signals,
            "notes": self.notes,
            "buckets": [
                {
                    **asdict(b),
                    "win_rate": round(b.win_rate, 4),
                    "confidence": round(b.confidence, 2),
                    "reliable": b.is_reliable,
                }
                for b in self.buckets
            ],
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

        buckets = [
            BucketStats(
                low=float(b["low"]),
                high=float(b["high"]),
                samples=int(b["samples"]),
                wins=int(b["wins"]),
            )
            for b in raw.get("buckets", [])
        ]
        return cls(
            buckets=buckets,
            generated_at=raw.get("generated_at", ""),
            total_signals=int(raw.get("total_signals", 0)),
            notes=raw.get("notes", ""),
        )

    def summary_table(self) -> str:
        """Tabla legible para la consola y para el informe de la fase 2."""
        if not self.buckets:
            return "Sin calibración: ejecuta primero `python trading_bot.py backtest`."

        lines = [
            f"{'Tramo':>12} {'Señales':>9} {'Aciertos':>9} {'% real':>8} "
            f"{'Confianza':>10} {'Fiable':>7}"
        ]
        for b in self.buckets:
            tramo = f"{b.low:.0f}-{min(b.high, 100):.0f}"
            lines.append(
                f"{tramo:>12} {b.samples:>9} {b.wins:>9} "
                f"{100 * b.win_rate:>7.1f}% {b.confidence:>9.1f}% "
                f"{'sí' if b.is_reliable else 'no':>7}"
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
