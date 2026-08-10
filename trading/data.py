"""Capa de datos de mercado, con proveedor intercambiable.

Twelve Data es el proveedor primario: su plan gratuito da 800 llamadas al día
y 8 por minuto cubriendo acciones estadounidenses y forex con la misma API,
que es justo lo que necesita un escaneo diario de ~130 instrumentos.

yfinance queda como respaldo y no como fuente principal a propósito: Yahoo
bloquea de forma agresiva las IPs de centro de datos —y las de GitHub Actions
lo son—, así que depender de él en producción es construir sobre arena.

Cambiar de proveedor (o pasar a un plan de pago) no debería tocar la
estrategia: por eso todo el resto del bot consume `MarketData`, nunca una API
concreta.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Protocol

import pandas as pd
import requests

from .config import CACHE_DIR
from .universe import Instrument

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

TWELVEDATA_BASE = "https://api.twelvedata.com"
TWELVEDATA_MAX_PER_MINUTE = 8


class ProviderError(RuntimeError):
    """El proveedor no pudo servir el dato (red, cuota, símbolo desconocido)."""


class Provider(Protocol):
    name: str

    def fetch_daily(self, instrument: Instrument, bars: int) -> pd.DataFrame:
        """Velas diarias más recientes, orden ascendente por fecha."""
        ...


# ── Utilidades comunes ────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Deja el DataFrame en el contrato que espera el resto del bot.

    Índice de fechas ascendente y sin duplicados, columnas OHLCV en float, y
    sin filas incompletas en los precios. El volumen se rellena a 0 porque el
    forex no lo publica y no queremos que eso invalide la vela.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    for col in OHLCV_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out[OHLCV_COLUMNS]


class _RateLimiter:
    """Ventana deslizante: como mucho `limit` llamadas por `window` segundos."""

    def __init__(self, limit: int, window: float = 60.0) -> None:
        self.limit = limit
        self.window = window
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] >= self.window:
            self._calls.popleft()

        if len(self._calls) >= self.limit:
            sleep_for = self.window - (now - self._calls[0]) + 0.1
            if sleep_for > 0:
                print(f"[INFO] Límite de tasa alcanzado, esperando {sleep_for:.0f}s")
                time.sleep(sleep_for)
            return self.acquire()

        self._calls.append(time.monotonic())


# ── Twelve Data ───────────────────────────────────────────────────────────────

class TwelveDataProvider:
    name = "twelvedata"

    def __init__(self, api_key: str, max_per_minute: int = TWELVEDATA_MAX_PER_MINUTE):
        if not api_key:
            raise ProviderError("Falta TWELVEDATA_API_KEY")
        self.api_key = api_key
        self._limiter = _RateLimiter(max_per_minute)

    def fetch_daily(self, instrument: Instrument, bars: int) -> pd.DataFrame:
        params = {
            "symbol": instrument.twelvedata,
            "interval": "1day",
            "outputsize": min(bars, 5000),
            "apikey": self.api_key,
            "order": "ASC",
            "timezone": "UTC",
        }

        last_error = "sin detalle"
        for attempt in range(3):
            self._limiter.acquire()
            try:
                resp = requests.get(
                    f"{TWELVEDATA_BASE}/time_series", params=params, timeout=30
                )
            except requests.RequestException as exc:
                last_error = str(exc)
                time.sleep(2 * (attempt + 1))
                continue

            # Twelve Data devuelve 200 con {"status": "error"} en los fallos
            # de negocio, así que no basta con mirar el código HTTP.
            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("status") == "ok" and payload.get("values"):
                    return self._to_frame(payload["values"])
                last_error = str(payload.get("message", payload))[:200]
                # Cuota agotada: reintentar no ayuda hasta el minuto siguiente.
                if payload.get("code") == 429:
                    time.sleep(60)
                    continue
                break  # símbolo inválido u otro error permanente

            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code}"
                time.sleep(5 * (attempt + 1))
                continue

            last_error = f"HTTP {resp.status_code}: {resp.text[:150]}"
            break

        raise ProviderError(f"twelvedata {instrument.symbol}: {last_error}")

    @staticmethod
    def _to_frame(values: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(values)
        df = df.set_index("datetime")
        return _normalize(df)


# ── yfinance (respaldo) ───────────────────────────────────────────────────────

class YFinanceProvider:
    name = "yfinance"

    def fetch_daily(self, instrument: Instrument, bars: int) -> pd.DataFrame:
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ProviderError("yfinance no está instalado") from exc

        # Margen amplio: `period` cuenta días naturales, no sesiones.
        days = int(bars * 1.6) + 30
        try:
            raw = yfinance.Ticker(instrument.yfinance).history(
                period=f"{days}d", interval="1d", auto_adjust=False
            )
        except Exception as exc:  # yfinance lanza excepciones muy variadas
            raise ProviderError(f"yfinance {instrument.symbol}: {exc}") from exc

        if raw is None or raw.empty:
            raise ProviderError(f"yfinance {instrument.symbol}: sin datos")

        raw = raw.rename(columns=str.lower)
        return _normalize(raw).tail(bars)


# ── Caché en disco ────────────────────────────────────────────────────────────

class _Cache:
    """Caché CSV por símbolo.

    CSV y no parquet para no arrastrar pyarrow: son ficheros de pocos miles de
    filas y el coste de parseo es irrelevante frente a la latencia de red que
    evitan. La caché es lo que permite iterar sobre el backtest sin gastar la
    cuota diaria de la API en cada ejecución.
    """

    def __init__(self, directory: str = CACHE_DIR) -> None:
        self.directory = directory

    def _path(self, symbol: str) -> str:
        safe = symbol.replace("/", "_")
        return os.path.join(self.directory, f"{safe}_1day.csv")

    def read(self, symbol: str, max_age_hours: float) -> pd.DataFrame | None:
        path = self._path(symbol)
        if not os.path.exists(path):
            return None
        age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_hours > max_age_hours:
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        except (OSError, ValueError, pd.errors.ParserError):
            return None
        return _normalize(df) if not df.empty else None

    def write(self, symbol: str, df: pd.DataFrame) -> None:
        os.makedirs(self.directory, exist_ok=True)
        try:
            df.to_csv(self._path(symbol))
        except OSError as exc:  # disco lleno en CI: degradar, no romper
            print(f"[WARN] No se pudo cachear {symbol}: {exc}")


# ── Fachada ───────────────────────────────────────────────────────────────────

class MarketData:
    """Punto único de acceso a datos de mercado.

    Prueba los proveedores en orden y se queda con el primero que responda.
    Un símbolo que falla en todos no rompe el escaneo: se registra y se sigue,
    porque perder una acción de 130 no justifica quedarse sin alertas.
    """

    def __init__(
        self,
        providers: list[Provider] | None = None,
        cache: _Cache | None = None,
        cache_max_age_hours: float = 12.0,
    ) -> None:
        self.providers = providers if providers is not None else default_providers()
        self.cache = cache if cache is not None else _Cache()
        self.cache_max_age_hours = cache_max_age_hours

    def get_daily(
        self, instrument: Instrument, bars: int = 400, use_cache: bool = True
    ) -> pd.DataFrame | None:
        if use_cache:
            cached = self.cache.read(instrument.symbol, self.cache_max_age_hours)
            if cached is not None and len(cached) >= bars * 0.9:
                return cached.tail(bars)

        errors: list[str] = []
        for provider in self.providers:
            try:
                df = provider.fetch_daily(instrument, bars)
            except ProviderError as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            if df.empty:
                errors.append(f"{provider.name}: DataFrame vacío")
                continue
            self.cache.write(instrument.symbol, df)
            return df

        print(f"[WARN] Sin datos para {instrument.symbol} — {' | '.join(errors)}")
        return None

    def get_many(
        self, instruments: list[Instrument], bars: int = 400, use_cache: bool = True
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        total = len(instruments)
        for i, inst in enumerate(instruments, 1):
            df = self.get_daily(inst, bars=bars, use_cache=use_cache)
            if df is not None and not df.empty:
                out[inst.symbol] = df
            if i % 20 == 0 or i == total:
                print(f"[INFO] Datos descargados: {i}/{total} ({len(out)} válidos)")
        return out


def default_providers() -> list[Provider]:
    """Twelve Data si hay clave, con yfinance detrás como red de seguridad."""
    from .config import TWELVEDATA_API_KEY

    providers: list[Provider] = []
    if TWELVEDATA_API_KEY:
        providers.append(TwelveDataProvider(TWELVEDATA_API_KEY))
    else:
        print("[WARN] Sin TWELVEDATA_API_KEY — se usará solo yfinance (poco fiable en CI)")
    providers.append(YFinanceProvider())
    return providers
