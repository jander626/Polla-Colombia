"""Decide cuándo toca escanear.

El cron de GitHub Actions no sirve por sí solo para esto, por dos motivos que
se manifiestan en producción y no en pruebas:

1. **Se retrasa.** Bajo carga, un cron programado a las 12:30 puede dispararse
   a las 12:50. No es un reloj, es una sugerencia.
2. **Es UTC.** Las 08:30 de Nueva York son las 12:30 UTC en verano y las 13:30
   UTC en invierno. Un cron fijo se desfasa una hora dos veces al año, justo
   cuando nadie está mirando.

La solución es la misma que usaba el bot del Mundial: el cron dispara a menudo
dentro de una ventana amplia y es Python, con zonas horarias reales, quien
decide si hay que actuar. La deduplicación por fecha impide que varios
disparos dentro de la ventana produzcan varias alertas.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .config import (
    NY_TZ,
    SCAN_HOUR_NY,
    SCAN_MINUTE_NY,
    SCAN_WINDOW_MINUTES,
    is_trading_day,
)


def now_ny(now: datetime | None = None) -> datetime:
    """Hora actual en Nueva York, que es la que manda para la apertura."""
    reference = now or datetime.now(tz=NY_TZ)
    if reference.tzinfo is None:
        raise ValueError("se requiere un datetime con zona horaria")
    return reference.astimezone(NY_TZ)


def scan_window(day: date) -> tuple[datetime, datetime]:
    """Ventana [inicio, fin] del escaneo para un día, en hora de Nueva York."""
    start = datetime(
        day.year, day.month, day.day, SCAN_HOUR_NY, SCAN_MINUTE_NY, tzinfo=NY_TZ
    )
    return start, start + timedelta(minutes=SCAN_WINDOW_MINUTES)


def should_scan(
    now: datetime | None = None, last_scan_date: str = ""
) -> tuple[bool, str]:
    """¿Toca escanear ahora? Devuelve (decisión, motivo legible)."""
    local = now_ny(now)
    today = local.date()

    if not is_trading_day(today):
        return False, f"{today} no es día hábil de bolsa"

    if last_scan_date == today.isoformat():
        return False, f"ya se escaneó hoy ({today})"

    start, end = scan_window(today)
    if local < start:
        minutes = (start - local).total_seconds() / 60
        return False, f"aún es pronto: faltan {minutes:.0f} min para las {start:%H:%M} de Nueva York"

    if local > end:
        # Se perdió la ventana. No se escanea fuera de plazo: una alerta que
        # llega con el mercado ya abierto tiene precios de entrada obsoletos,
        # y es peor que ninguna alerta.
        return False, f"ventana perdida (terminó a las {end:%H:%M} de Nueva York)"

    return True, f"dentro de la ventana ({start:%H:%M}–{end:%H:%M} de Nueva York)"


def previous_trading_day(day: date) -> date:
    """Último día hábil anterior. La señal se calcula con su cierre."""
    cursor = day - timedelta(days=1)
    while not is_trading_day(cursor):
        cursor -= timedelta(days=1)
    return cursor
