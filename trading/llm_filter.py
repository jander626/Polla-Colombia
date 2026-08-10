"""Filtro de riesgo de evento con Gemini y búsqueda en Google.

Este módulo tiene un poder deliberadamente limitado: **solo puede restar
confianza**. El porcentaje que se publica sale del backtest (ver `risk.py`);
si se dejara que el modelo lo subiera, volvería por la puerta de atrás el
número inventado que toda la calibración existe para eliminar.

Lo que sí aporta un modelo con búsqueda es lo que el análisis técnico no puede
ver: que la empresa presenta resultados en tres días, que hay reunión de la
Reserva Federal, o que acaba de salir una noticia que rompe la tesis. Eso es
información, no predicción.

Se aplica solo a los mejores candidatos del día, no a todo el universo: acota
el coste y la latencia sin perder nada, porque las señales que no se van a
enviar no necesitan verificación.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from .config import GEMINI_API_KEY, MAX_LLM_PENALTY
from .strategy import Signal
from .universe import get as get_instrument

MODELS = ("gemini-2.5-pro", "gemini-2.5-flash")
_TRANSIENT = ("503", "429", "high demand", "overloaded", "timeout", "deadline")


@dataclass(frozen=True)
class EventRisk:
    level: str            # "none" | "low" | "high"
    penalty: float        # puntos de confianza a restar, 0..MAX_LLM_PENALTY
    rationale: str        # por qué entra la señal, en español
    risks: str            # qué evento puede romperla

    @property
    def is_blocking(self) -> bool:
        """Riesgo alto: mejor no enviar la señal que enviarla penalizada.

        Comprar el día antes de unos resultados trimestrales no es operar con
        menos confianza, es operar otra cosa distinta: un evento binario que la
        estrategia técnica no modela en absoluto.
        """
        return self.level == "high"


def _client():
    """Crea el cliente de Gemini. None si no hay clave o falta la librería."""
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
    except ImportError:
        print("[WARN] google-genai no está instalado: sin filtro de noticias")
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


def build_prompt(signal: Signal) -> str:
    instrument = get_instrument(signal.symbol)
    if signal.is_forex:
        base, quote = signal.symbol.split("/")
        focus = (
            f"Busca eventos macroeconómicos de las próximas 72 horas que afecten a "
            f"{base} o a {quote}: decisiones de tipos de interés, actas de bancos "
            f"centrales, datos de empleo (NFP), IPC, PIB o discursos relevantes."
        )
        # Los criterios se adaptan al activo: hablarle de resultados
        # trimestrales a un par de divisas solo introduce ruido.
        high_criteria = (
            "decisión de tipos o comparecencia de un banco central que afecte "
            "directamente al par en menos de 72 horas, o un dato macro de primer "
            "orden (NFP, IPC) inminente"
        )
    else:
        focus = (
            f"Busca: (a) si {instrument.name} ({signal.symbol}) presenta resultados "
            f"trimestrales en los próximos 5 días hábiles; (b) noticias de las "
            f"últimas 48 horas sobre la empresa: profit warnings, investigaciones "
            f"regulatorias, cambios en la cúpula, fusiones; (c) eventos "
            f"macroeconómicos inminentes que afecten a su sector."
        )
        high_criteria = (
            "resultados trimestrales en menos de 5 días hábiles, o una noticia "
            "grave y reciente sobre la empresa"
        )

    return f"""Eres un analista de riesgo de mercado. NO vas a predecir el precio ni
valorar si la operación es buena: eso ya está decidido por un sistema técnico
calibrado. Tu único trabajo es detectar EVENTOS PROGRAMADOS O RECIENTES que
puedan invalidar una operación alcista de swing (2 a 10 días) sobre
{signal.symbol}.

{focus}

Responde ÚNICAMENTE con un objeto JSON válido, sin texto alrededor ni bloques
de código, con exactamente estas claves:

{{
  "event_risk": "none" | "low" | "high",
  "confidence_penalty": <entero de 0 a {int(MAX_LLM_PENALTY)}>,
  "rationale_es": "<máximo 2 frases en español: contexto favorable o neutro que hayas encontrado>",
  "risks_es": "<máximo 1 frase en español: el evento concreto que puede romper la operación, o cadena vacía si no hay ninguno>"
}}

Criterios para event_risk:
- "high": {high_criteria}.
- "low": hay algún evento en el horizonte pero de impacto moderado o indirecto.
- "none": nada relevante encontrado.

La penalización debe ser coherente: 0 para "none", entre 5 y 12 para "low",
entre 15 y {int(MAX_LLM_PENALTY)} para "high"."""


def _extract_json(text: str) -> Optional[dict]:
    """Rescata el objeto JSON de la respuesta.

    Los modelos envuelven el JSON en ```json ... ``` con frecuencia pese a que
    se les pida lo contrario, así que se limpia antes de intentar el parseo.
    """
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Último intento: quedarse con el primer objeto {...} que aparezca.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_response(text: str) -> Optional[EventRisk]:
    """Convierte la respuesta en EventRisk, saneando lo que haga falta."""
    data = _extract_json(text)
    if not isinstance(data, dict):
        return None

    level = str(data.get("event_risk", "none")).lower().strip()
    if level not in ("none", "low", "high"):
        level = "low"  # respuesta rara: se asume algo de riesgo, nunca ninguno

    try:
        penalty = float(data.get("confidence_penalty", 0))
    except (TypeError, ValueError):
        penalty = 0.0
    # El modelo no decide el techo: se recorta aquí. Y una penalización
    # negativa (que subiría la confianza) se descarta sin contemplaciones.
    penalty = max(0.0, min(penalty, MAX_LLM_PENALTY))

    if level == "high":
        penalty = max(penalty, 15.0)

    return EventRisk(
        level=level,
        penalty=penalty,
        rationale=str(data.get("rationale_es", "") or "").strip(),
        risks=str(data.get("risks_es", "") or "").strip(),
    )


def _call(client, model: str, prompt: str, with_search: bool) -> Optional[str]:
    """Una llamada con hasta 3 reintentos ante errores transitorios."""
    from google.genai import types as genai_types

    config = genai_types.GenerateContentConfig(
        tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
        if with_search
        else [],
        max_output_tokens=2048,
        temperature=0.2,  # se quiere consistencia, no creatividad
    )

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
            return resp.text
        except Exception as exc:
            error = str(exc)
            transient = any(token in error.lower() for token in _TRANSIENT)
            if transient and attempt < 2:
                wait = 15 * (attempt + 1)
                print(f"[WARN] {model} error transitorio, reintento en {wait}s")
                time.sleep(wait)
            else:
                print(f"[ERROR] {model}: {error[:200]}")
                return None
    return None


def check(signal: Signal) -> Optional[EventRisk]:
    """Evalúa el riesgo de evento de una señal.

    Devuelve None si no se pudo consultar. El llamante debe interpretarlo como
    "sin verificación" y avisarlo en el mensaje, nunca como "sin riesgo": son
    cosas muy distintas y confundirlas sería justo el tipo de silencio que hace
    perder dinero.
    """
    client = _client()
    if client is None:
        return None

    prompt = build_prompt(signal)
    for model in MODELS:
        for with_search in (True, False):
            raw = _call(client, model, prompt, with_search)
            if not raw:
                continue
            risk = parse_response(raw)
            if risk is not None:
                suffix = "con búsqueda" if with_search else "sin búsqueda"
                print(f"[INFO] {signal.symbol}: riesgo '{risk.level}' ({model}, {suffix})")
                return risk
            print(f"[WARN] {signal.symbol}: respuesta no interpretable de {model}")

    print(f"[WARN] {signal.symbol}: sin verificación de noticias")
    return None
