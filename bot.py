#!/usr/bin/env python3
"""
World Cup 2026 Forecast Bot
- Sends forecasts 24h before each match via Telegram
- Responds to manual Telegram commands
- Uses Google Gemini with Google Search grounding for analysis
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from google import genai
from google.genai import types as genai_types

# ── Configuration ────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
FOOTBALL_API_KEY = os.environ["FOOTBALL_DATA_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

_gemini = genai.Client(api_key=GEMINI_API_KEY)

FOOTBALL_BASE = "https://api.football-data.org/v4"
WORLD_CUP_ID = "WC"  # football-data.org competition code for World Cup
WC2026_SEASON = 2026

STATE_FILE = "state.json"
FORECAST_WINDOW_HOURS = 24   # send forecast this many hours before kickoff
FORECAST_REPEAT_HOURS = 2    # re-send if match is within this window and no forecast yet

HEADERS = {"X-Auth-Token": FOOTBALL_API_KEY}

# ── State helpers ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"sent_forecasts": [], "chat_ids": []}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ── Football-data.org helpers ─────────────────────────────────────────────────

def get_matches(status: Optional[str] = None) -> list:
    """Fetch WC 2026 matches, optionally filtered by status."""
    params = {"season": WC2026_SEASON}
    if status:
        params["status"] = status
    url = f"{FOOTBALL_BASE}/competitions/{WORLD_CUP_ID}/matches"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    if resp.status_code == 200:
        return resp.json().get("matches", [])
    # Fallback: try without season param (some free-tier restrictions)
    resp2 = requests.get(url, headers=HEADERS, timeout=15)
    if resp2.status_code == 200:
        return resp2.json().get("matches", [])
    print(f"[WARN] Football API error {resp.status_code}: {resp.text[:200]}")
    return []


def get_standings() -> list:
    """Fetch current WC 2026 group standings."""
    url = f"{FOOTBALL_BASE}/competitions/{WORLD_CUP_ID}/standings"
    params = {"season": WC2026_SEASON}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    if resp.status_code == 200:
        return resp.json().get("standings", [])
    return []


STAGE_NAMES = {
    "GROUP_STAGE": "Fase de grupos",
    "LAST_32": "Dieciseisavos de final",
    "LAST_16": "Octavos de final",
    "QUARTER_FINALS": "Cuartos de final",
    "SEMI_FINALS": "Semifinales",
    "THIRD_PLACE": "Tercer puesto",
    "FINAL": "Final",
}


def stage_label(match: dict) -> str:
    """Human-readable stage/group label, safe for Telegram Markdown
    (raw API values like GROUP_A contain underscores that break parsing)."""
    stage = STAGE_NAMES.get(match.get("stage", ""), match.get("stage", "").replace("_", " "))
    group = (match.get("group") or "").replace("GROUP_", "Grupo ").replace("_", " ")
    return f"{stage} — {group}" if group else stage


def teams_known(match: dict) -> bool:
    """False for knockout-stage placeholder matches where football-data.org
    hasn't resolved the qualifying teams yet (homeTeam/awayTeam keys exist
    but their "name" value is null)."""
    home = (match.get("homeTeam") or {}).get("name")
    away = (match.get("awayTeam") or {}).get("name")
    return bool(home) and bool(away)


def match_display_name(match: dict) -> str:
    home = (match.get("homeTeam") or {}).get("name") or "Por definir"
    away = (match.get("awayTeam") or {}).get("name") or "Por definir"
    return f"{home} vs {away}"


def match_kickoff_utc(match: dict) -> Optional[datetime]:
    dt_str = match.get("utcDate")
    if not dt_str:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def upcoming_matches(hours_ahead: int = 36) -> list:
    """Return scheduled matches kicking off within the next `hours_ahead` hours."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)
    matches = get_matches(status="SCHEDULED") + get_matches(status="TIMED")
    result = []
    for m in matches:
        ko = match_kickoff_utc(m)
        if ko and now <= ko <= cutoff:
            result.append(m)
    return sorted(result, key=lambda m: m.get("utcDate", ""))


def all_upcoming_matches(limit: int = 10) -> list:
    """Return next `limit` scheduled matches regardless of window."""
    now = datetime.now(timezone.utc)
    matches = get_matches(status="SCHEDULED") + get_matches(status="TIMED")
    future = [m for m in matches if (match_kickoff_utc(m) or now) > now]
    return sorted(future, key=lambda m: m.get("utcDate", ""))[:limit]


# ── Claude analysis ───────────────────────────────────────────────────────────

def build_analysis_prompt(match: dict, standings: list) -> str:
    home = (match.get("homeTeam") or {}).get("name") or "Por definir"
    away = (match.get("awayTeam") or {}).get("name") or "Por definir"
    stage = match.get("stage", "")
    group = match.get("group", "")
    ko = match_kickoff_utc(match)
    ko_str = ko.strftime("%Y-%m-%d %H:%M UTC") if ko else "?"

    standings_text = ""
    if standings:
        for table in standings:
            grp = table.get("group", "")
            if group and group != grp:
                continue
            rows = table.get("table", [])[:6]
            lines = [f"  {r['position']}. {r['team']['name']} — PJ:{r['playedGames']} P:{r['points']} GF:{r['goalsFor']} GC:{r['goalsAgainst']}" for r in rows]
            if lines:
                standings_text += f"\nGrupo {grp}:\n" + "\n".join(lines)

    return f"""Eres un experto analista de fútbol. Analiza el siguiente partido del Mundial 2026 y proporciona un pronóstico detallado.

PARTIDO: {home} vs {away}
FASE: {stage} {group}
FECHA: {ko_str}

CLASIFICACIÓN ACTUAL:{standings_text if standings_text else ' No disponible'}

Por favor:
1. Busca noticias recientes (últimas 72 horas) sobre ambos equipos: lesiones, sanciones, forma reciente, declaraciones del técnico.
2. Analiza estadísticas de los últimos 5 partidos de cada equipo.
3. Revisa el historial de enfrentamientos directos recientes.
4. Considera el contexto (qué necesita cada equipo según su posición en el grupo / fase del torneo).

Responde EXACTAMENTE con este formato, sin preámbulo, sin introducción, sin disclaimers — empieza directamente con la primera línea:

🎯 *RESULTADO MÁS PROBABLE*: [1 (gana {home}) / X (empate) / 2 (gana {away})]
⚽ *MARCADOR PROBABLE*: X-X
📊 *CONFIANZA*: XX%
🔑 *FACTORES CLAVE*:
• [factor 1]
• [factor 2]
• [factor 3]
📝 *RESUMEN*: [2-3 oraciones explicando el pronóstico]

Máximo 200 palabras en total. Usa un solo asterisco para negritas (formato Telegram), nunca dobles asteriscos."""


def analyze_match(match: dict, standings: list) -> Optional[str]:
    """Returns the analysis text, or None if the Gemini API call failed
    (so callers can retry on the next run instead of marking it as sent)."""
    prompt = build_analysis_prompt(match, standings)

    try:
        response = _gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                max_output_tokens=8192,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=1024),
            ),
        )
        return _clean_analysis(response.text)
    except Exception as e:
        print(f"[ERROR] Gemini API (with search): {e}")
        # Fallback without search
        try:
            response = _gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=8192,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=1024),
                ),
            )
            return _clean_analysis(response.text)
        except Exception as e2:
            print(f"[ERROR] Gemini API fallback: {e2}")
            return None


def _clean_analysis(text: Optional[str]) -> Optional[str]:
    """Adapt model output to Telegram's legacy Markdown: ** (bold) and __
    are not supported and make sendMessage reject the whole message."""
    if not text:
        return None
    return text.replace("**", "*").replace("__", "_").strip()


# ── Telegram helpers ──────────────────────────────────────────────────────────

TELEGRAM_MAX_LEN = 4000  # actual limit is 4096; leave margin


def tg_send(chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Split long messages (Telegram rejects anything over 4096 chars)
    chunks = []
    while len(text) > TELEGRAM_MAX_LEN:
        cut = text.rfind("\n", 0, TELEGRAM_MAX_LEN)
        if cut <= 0:
            cut = TELEGRAM_MAX_LEN
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    chunks.append(text)

    ok = True
    for chunk in chunks:
        try:
            resp = requests.post(url, json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"}, timeout=15)
            if resp.status_code != 200:
                # Markdown parse errors reject the whole message — resend as plain text
                print(f"[WARN] Telegram rejected Markdown ({resp.status_code}): {resp.text[:150]} — retrying plain")
                resp = requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=15)
            if resp.status_code != 200:
                print(f"[ERROR] Telegram send failed ({resp.status_code}): {resp.text[:150]}")
                ok = False
        except Exception as e:
            print(f"[ERROR] Telegram send: {e}")
            ok = False
    return ok


def tg_get_updates(offset: int = 0) -> list:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"offset": offset, "timeout": 5}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("result", [])
    except Exception as e:
        print(f"[ERROR] Telegram getUpdates: {e}")
    return []


def broadcast(text: str, state: dict) -> bool:
    """Send a message to all known chat IDs. Returns True if at least one
    delivery succeeded."""
    chat_ids = set(state.get("chat_ids", []))
    if CHAT_ID:
        chat_ids.add(CHAT_ID)
    delivered = False
    for cid in chat_ids:
        if tg_send(str(cid), text):
            delivered = True
    return delivered


def format_forecast_message(match: dict, analysis: str) -> str:
    home = (match.get("homeTeam") or {}).get("name") or "Por definir"
    away = (match.get("awayTeam") or {}).get("name") or "Por definir"
    ko = match_kickoff_utc(match)
    ko_str = ko.strftime("%d/%m/%Y %H:%M UTC") if ko else "?"

    header = (
        f"🌍 *PRONÓSTICO MUNDIAL 2026*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚽ *{home} vs {away}*\n"
        f"📅 {ko_str}\n"
        f"🏆 {stage_label(match)}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
    )
    return header + analysis


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_proximos(chat_id: str) -> None:
    matches = all_upcoming_matches(10)
    if not matches:
        tg_send(chat_id, "📅 No hay partidos programados próximamente.")
        return

    lines = ["🌍 *PRÓXIMOS PARTIDOS — MUNDIAL 2026*\n"]
    for i, m in enumerate(matches, 1):
        ko = match_kickoff_utc(m)
        ko_str = ko.strftime("%d/%m %H:%M UTC") if ko else "?"
        name = match_display_name(m)
        lines.append(f"{i}. *{name}*\n   📅 {ko_str} — {stage_label(m)}")

    tg_send(chat_id, "\n".join(lines))


def cmd_pronostico(chat_id: str, args: str, state: dict) -> None:
    matches = all_upcoming_matches(20)
    if not matches:
        tg_send(chat_id, "No hay partidos próximos disponibles.")
        return

    # Try to match by index or team name substring
    target = None
    args_lower = args.lower().strip()

    if args_lower.isdigit():
        idx = int(args_lower) - 1
        if 0 <= idx < len(matches):
            target = matches[idx]
    else:
        for m in matches:
            name = match_display_name(m).lower()
            if args_lower in name:
                target = m
                break

    if target is None:
        tg_send(chat_id, f"⚠️ No encontré el partido '{args}'. Usa /proximos para ver la lista y escribe el número.")
        return

    tg_send(chat_id, f"🔍 Analizando *{match_display_name(target)}*… (puede tardar 20-30 segundos)")
    standings = get_standings()
    analysis = analyze_match(target, standings)
    if analysis is None:
        tg_send(chat_id, "⚠️ No pude generar el análisis (problema con la API de Claude, posiblemente sin créditos). Intenta más tarde.")
        return
    msg = format_forecast_message(target, analysis)
    tg_send(chat_id, msg)


def cmd_ayuda(chat_id: str) -> None:
    text = (
        "🤖 *Bot de Pronósticos — Mundial 2026*\n\n"
        "Comandos disponibles:\n"
        "• /proximos — Lista los próximos 10 partidos\n"
        "• /pronostico <número o equipo> — Pide el pronóstico de un partido\n"
        "  Ejemplo: `/pronostico 1` o `/pronostico Colombia`\n"
        "• /ayuda — Muestra este mensaje\n\n"
        "También recibirás pronósticos automáticos 24h antes de cada partido."
    )
    tg_send(chat_id, text)


# ── Automatic forecast loop ───────────────────────────────────────────────────

def run_auto_forecasts(state: dict) -> bool:
    """Check for matches in the 24h window and send forecasts. Returns True if state changed."""
    now = datetime.now(timezone.utc)
    matches = upcoming_matches(hours_ahead=FORECAST_WINDOW_HOURS + 2)
    standings = get_standings() if matches else []
    changed = False

    for match in matches:
        match_id = str(match.get("id"))
        ko = match_kickoff_utc(match)
        if ko is None:
            continue

        hours_to_ko = (ko - now).total_seconds() / 3600
        if hours_to_ko < 0 or hours_to_ko > FORECAST_WINDOW_HOURS:
            continue

        if match_id in state.get("sent_forecasts", []):
            continue  # already sent

        if not teams_known(match):
            print(f"[INFO] Skipping {match_id}: qualifying teams not yet confirmed")
            continue  # don't mark as sent — retry once football-data.org resolves the bracket

        name = match_display_name(match)
        print(f"[INFO] Sending forecast for {name} (kickoff in {hours_to_ko:.1f}h)")

        analysis = analyze_match(match, standings)
        if analysis is None:
            # Claude API failed (e.g. no credits) — don't mark as sent so the
            # next run retries this match
            print(f"[WARN] Analysis failed for {name}; will retry next run")
            continue

        msg = format_forecast_message(match, analysis)
        if not broadcast(msg, state):
            print(f"[WARN] Delivery failed for {name}; will retry next run")
            continue

        state.setdefault("sent_forecasts", []).append(match_id)
        changed = True
        time.sleep(2)  # avoid Telegram rate limiting between matches

    return changed


# ── Telegram polling for commands ─────────────────────────────────────────────

def poll_commands(state: dict) -> bool:
    """Process pending Telegram commands. Returns True if state changed."""
    offset = state.get("tg_offset", 0)
    updates = tg_get_updates(offset)
    changed = False

    for upd in updates:
        offset = max(offset, upd["update_id"] + 1)
        msg = upd.get("message") or upd.get("channel_post")
        if not msg:
            continue

        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = msg.get("text", "").strip()

        # Register chat for broadcasts
        if chat_id and chat_id not in state.get("chat_ids", []):
            state.setdefault("chat_ids", []).append(chat_id)
            changed = True
            print(f"[INFO] Registered new chat_id: {chat_id}")

        if not text.startswith("/"):
            continue

        parts = text.split(None, 1)
        command = parts[0].lower().split("@")[0]
        args = parts[1] if len(parts) > 1 else ""

        print(f"[INFO] Command '{command}' from {chat_id}")

        if command == "/proximos":
            cmd_proximos(chat_id)
        elif command == "/pronostico":
            if not args:
                tg_send(chat_id, "Uso: /pronostico <número o nombre del equipo>\nEjemplo: /pronostico 1")
            else:
                cmd_pronostico(chat_id, args, state)
        elif command in ("/ayuda", "/start", "/help"):
            cmd_ayuda(chat_id)
        else:
            tg_send(chat_id, "Comando desconocido. Usa /ayuda para ver los comandos disponibles.")

    if offset != state.get("tg_offset", 0):
        state["tg_offset"] = offset
        changed = True

    return changed


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    state = load_state()
    changed = False

    if mode == "auto":
        # Run by GitHub Actions cron: process commands + send scheduled forecasts
        changed |= poll_commands(state)
        changed |= run_auto_forecasts(state)
    elif mode == "poll":
        # Only process Telegram commands (used for interactive testing)
        changed |= poll_commands(state)
    elif mode == "test":
        # Send a quick health-check message to configured chats
        broadcast("✅ Bot de pronósticos activo. Escribe /ayuda para ver los comandos.", state)
    else:
        print(f"Unknown mode: {mode}. Use: auto | poll | test")
        sys.exit(1)

    if changed:
        save_state(state)
        print("[INFO] State saved.")
    else:
        print("[INFO] No changes.")


if __name__ == "__main__":
    main()
