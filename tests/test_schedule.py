"""Tests de la ventana horaria.

Dos formas silenciosas de que el bot falle, y las dos han ocurrido:

1. El horario de verano mueve la hora de Nueva York respecto al UTC del cron.
2. GitHub no respeta el cron. Pide cada 15 minutos y concede dos o tres
   disparos al día, a horas impredecibles — de ahí que la ventana tenga que
   ser ancha, y que su anchura se verifique contra el cron real del workflow.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from trading.config import NY_TZ
from trading.schedule import previous_trading_day, scan_window, should_scan


def ny(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY_TZ)


# ── Ventana ───────────────────────────────────────────────────────────────────

def test_scans_inside_the_window():
    allowed, reason = should_scan(ny(2026, 8, 12, 8, 35))
    assert allowed, reason


def test_does_not_scan_before_the_window():
    allowed, reason = should_scan(ny(2026, 8, 12, 5, 0))
    assert not allowed and "pronto" in reason


def test_does_not_scan_after_the_window():
    """Una alerta con el mercado ya abierto lleva precios obsoletos."""
    allowed, reason = should_scan(ny(2026, 8, 12, 11, 0))
    assert not allowed and "ventana perdida" in reason


def test_the_window_ends_before_the_opening_bell():
    """El único límite real: no alertar con el mercado ya abierto.

    Con el mercado abierto el precio de entrada calculado sobre el cierre de
    ayer deja de servir. La hora de INICIO, en cambio, es holgada a propósito:
    la señal sale del cierre anterior y no cambia durante toda la mañana.
    """
    start, end = scan_window(date(2026, 8, 12))
    assert (end.hour, end.minute) == (9, 25)      # 5 min antes de la apertura
    assert end > start
    assert (end - start).total_seconds() / 3600 > 3   # banda ancha, no 90 min


# ── Horario de verano ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "day,expected_utc_hour",
    [
        (date(2026, 8, 12), 10),  # verano: EDT = UTC-4 -> 06:00 NY = 10:00 UTC
        (date(2026, 1, 14), 11),  # invierno: EST = UTC-5 -> 06:00 NY = 11:00 UTC
    ],
)
def test_the_window_follows_daylight_saving(day, expected_utc_hour):
    """El cron es UTC; si la ventana no siguiera al DST, se desfasaría una hora."""
    start, _ = scan_window(day)
    assert start.astimezone(tz=None).utcoffset() is not None
    assert start.utctimetuple().tm_hour == expected_utc_hour


def _workflow_cron_hours() -> set[int]:
    """Horas UTC en las que dispara el cron real del workflow.

    Se lee del YAML en lugar de codificarlo aquí: si alguien cambia el cron y
    deja de cubrir la ventana, el bot no alertaría nunca y no habría ningún
    error visible. Este test es lo único que lo detectaría.
    """
    import yaml

    path = Path(__file__).resolve().parents[1] / ".github/workflows/trading.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    # 'on' es la palabra reservada YAML para True, así que puede venir como bool.
    triggers = workflow.get("on") or workflow.get(True)
    expression = triggers["schedule"][0]["cron"]

    hour_field = expression.split()[1]  # p.ej. "11-14"
    start, end = (int(x) for x in hour_field.split("-"))
    return set(range(start, end + 1))


@pytest.mark.parametrize("day", [date(2026, 8, 12), date(2026, 1, 14)])
def test_the_cron_fires_inside_the_window_all_year(day):
    """Verano e invierno: el cron debe dispararse dentro de la ventana.

    En verano la ventana cae en 10:00–13:25 UTC y en invierno en 11:00–14:25.
    Un cron que solo cubriera una de las dos dejaría al bot mudo medio año.
    """
    start, end = scan_window(day)
    cron_hours = _workflow_cron_hours()

    # El cron dispara cada 15 minutos dentro de sus horas.
    firings = [
        datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)
        for hour in sorted(cron_hours)
        for minute in (0, 15, 30, 45)
    ]
    inside = [f for f in firings if start <= f <= end]

    assert inside, (
        f"el cron ({sorted(cron_hours)} UTC) no dispara nunca dentro de la "
        f"ventana {start:%H:%M}–{end:%H:%M} UTC del {day}"
    )
    # Margen amplio a propósito. GitHub concede dos o tres disparos al día de
    # los que se le piden, así que la ventana debe ser lo bastante ancha para
    # que cualquiera de ellos caiga dentro.
    assert len(inside) >= 12


# ── Días no hábiles ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("day", [(2026, 8, 15), (2026, 8, 16)])  # sábado y domingo
def test_does_not_scan_on_weekends(day):
    allowed, reason = should_scan(ny(*day, 8, 35))
    assert not allowed and "hábil" in reason


def test_does_not_scan_on_a_market_holiday():
    # 2026-07-03: Día de la Independencia (observado)
    allowed, reason = should_scan(ny(2026, 7, 3, 8, 35))
    assert not allowed and "hábil" in reason


# ── Deduplicación ─────────────────────────────────────────────────────────────

def test_does_not_scan_twice_the_same_day():
    """El cron dispara cada 15 min dentro de la ventana; solo una alerta."""
    now = ny(2026, 8, 12, 8, 45)
    allowed, reason = should_scan(now, last_scan_date="2026-08-12")
    assert not allowed and "ya se escaneó" in reason


def test_a_previous_scan_does_not_block_the_next_day():
    allowed, _ = should_scan(ny(2026, 8, 13, 8, 35), last_scan_date="2026-08-12")
    assert allowed


def test_requires_an_aware_datetime():
    with pytest.raises(ValueError):
        should_scan(datetime(2026, 8, 12, 8, 35))


# ── Día hábil anterior ────────────────────────────────────────────────────────

def test_previous_trading_day_skips_the_weekend():
    assert previous_trading_day(date(2026, 8, 10)) == date(2026, 8, 7)  # lunes -> viernes


def test_previous_trading_day_skips_holidays():
    # 2026-12-25 es Navidad (viernes); el hábil anterior es el jueves 24.
    assert previous_trading_day(date(2026, 12, 28)) == date(2026, 12, 24)


# ── El incidente del 11 de agosto de 2026 ────────────────────────────────────

@pytest.mark.parametrize("hour,minute", [(7, 38), (8, 27)])
def test_the_two_runs_that_missed_the_window_now_scan(hour, minute):
    """Ese día el bot no alertó: sus dos ejecuciones cayeron fuera por poco.

    GitHub solo le concedió dos disparos, a las 07:38 y a las 08:27 de Nueva
    York. La ventana empezaba a las 08:30, así que el segundo se quedó fuera
    por DOS MINUTOS y no hubo escaneo en todo el día.

    Rechazarlos era un error de criterio: la señal se calcula con el cierre de
    ayer, que a las 07:38 es exactamente el mismo que a las 08:30.
    """
    allowed, reason = should_scan(ny(2026, 8, 11, hour, minute))
    assert allowed, reason


def test_the_market_open_still_closes_the_window():
    """Lo que sí hay que seguir rechazando: alertar con el mercado abierto."""
    allowed, reason = should_scan(ny(2026, 8, 11, 9, 31))
    assert not allowed and "ventana perdida" in reason
