"""Universo de instrumentos operables en Quantfury.

Quantfury no expone una API pública de trading, así que el bot no se conecta a
la plataforma: la usa como *restricción*. Solo se recomiendan instrumentos que
el usuario pueda comprar allí, porque una señal sobre algo no operable es una
señal inútil.

La lista de acciones es una semilla de nombres muy líquidos, no la verdad
absoluta: el filtro real de "mayor volumen" lo aplica `filter_by_liquidity()`
con datos de mercado, de modo que un valor que se seca sale solo del ranking
sin que nadie edite este archivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd

from . import indicators as ind
from .config import StrategyParams

AssetClass = Literal["stock", "etf", "forex"]


@dataclass(frozen=True)
class Instrument:
    symbol: str            # identificador canónico usado en todo el bot
    name: str              # nombre legible para el mensaje de Telegram
    asset_class: AssetClass
    twelvedata: str        # símbolo tal y como lo espera Twelve Data
    yfinance: str          # símbolo para el proveedor de respaldo

    @property
    def is_forex(self) -> bool:
        return self.asset_class == "forex"

    @property
    def price_decimals(self) -> int:
        """Decimales con los que formatear precios en las alertas.

        Los pares con yen cotizan en torno a 150 y se muestran con 3 decimales;
        el resto de divisas rondan el 1.xxxx y necesitan 5.
        """
        if not self.is_forex:
            return 2
        return 3 if "JPY" in self.symbol else 5


def _fx(pair: str, name: str) -> Instrument:
    base, quote = pair.split("/")
    return Instrument(
        symbol=pair,
        name=name,
        asset_class="forex",
        twelvedata=pair,
        yfinance=f"{base}{quote}=X",
    )


# ── Forex: los 14 pares fiat de Quantfury ─────────────────────────────────────
# Los siete primeros son los majors, donde se concentra el volumen; el scoring
# los favorece ligeramente frente a los cruces con yen.

MAJOR_PAIRS = ("EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD")

FOREX: tuple[Instrument, ...] = (
    _fx("EUR/USD", "Euro / Dólar"),
    _fx("GBP/USD", "Libra / Dólar"),
    _fx("USD/JPY", "Dólar / Yen"),
    _fx("AUD/USD", "Dólar australiano / Dólar"),
    _fx("USD/CAD", "Dólar / Dólar canadiense"),
    _fx("USD/CHF", "Dólar / Franco suizo"),
    _fx("NZD/USD", "Dólar neozelandés / Dólar"),
    _fx("EUR/GBP", "Euro / Libra"),
    _fx("EUR/JPY", "Euro / Yen"),
    _fx("GBP/JPY", "Libra / Yen"),
    _fx("CAD/JPY", "Dólar canadiense / Yen"),
    _fx("AUD/JPY", "Dólar australiano / Yen"),
    _fx("NZD/JPY", "Dólar neozelandés / Yen"),
    _fx("CHF/JPY", "Franco suizo / Yen"),
)


def _eq(symbol: str, name: str, asset_class: AssetClass = "stock") -> Instrument:
    return Instrument(
        symbol=symbol,
        name=name,
        asset_class=asset_class,
        twelvedata=symbol,
        yfinance=symbol,
    )


# ── ETFs ──────────────────────────────────────────────────────────────────────

ETFS: tuple[Instrument, ...] = (
    _eq("SPY", "S&P 500", "etf"),
    _eq("QQQ", "Nasdaq 100", "etf"),
    _eq("IWM", "Russell 2000", "etf"),
    _eq("DIA", "Dow Jones 30", "etf"),
)

BENCHMARK_SYMBOL = "SPY"  # referencia de fuerza relativa para acciones y ETFs


# ── Acciones: semilla de nombres líquidos de NYSE/NASDAQ ──────────────────────
# El tamaño de esta lista es directamente el consumo diario de la API
# (una llamada por símbolo), así que se mantiene deliberadamente acotada.

STOCKS: tuple[Instrument, ...] = tuple(
    _eq(sym, name)
    for sym, name in [
        # Tecnología
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"),
        ("GOOGL", "Alphabet"), ("AMZN", "Amazon"), ("META", "Meta"),
        ("TSLA", "Tesla"), ("AVGO", "Broadcom"), ("AMD", "AMD"),
        ("INTC", "Intel"), ("CRM", "Salesforce"), ("ORCL", "Oracle"),
        ("ADBE", "Adobe"), ("CSCO", "Cisco"), ("QCOM", "Qualcomm"),
        ("TXN", "Texas Instruments"), ("MU", "Micron"), ("AMAT", "Applied Materials"),
        ("LRCX", "Lam Research"), ("KLAC", "KLA"), ("NOW", "ServiceNow"),
        ("INTU", "Intuit"), ("PANW", "Palo Alto Networks"), ("PLTR", "Palantir"),
        ("UBER", "Uber"), ("ABNB", "Airbnb"), ("SHOP", "Shopify"),
        ("PYPL", "PayPal"), ("NFLX", "Netflix"), ("DIS", "Disney"),
        ("CMCSA", "Comcast"), ("VZ", "Verizon"), ("TMUS", "T-Mobile"),
        ("IBM", "IBM"), ("ACN", "Accenture"), ("DELL", "Dell"),
        ("MRVL", "Marvell"), ("ARM", "Arm Holdings"), ("CRWD", "CrowdStrike"),
        ("ANET", "Arista Networks"), ("SNOW", "Snowflake"), ("DDOG", "Datadog"),
        # Finanzas
        ("JPM", "JPMorgan"), ("BAC", "Bank of America"), ("WFC", "Wells Fargo"),
        ("C", "Citigroup"), ("GS", "Goldman Sachs"), ("MS", "Morgan Stanley"),
        ("SCHW", "Charles Schwab"), ("BLK", "BlackRock"), ("AXP", "American Express"),
        ("V", "Visa"), ("MA", "Mastercard"), ("COF", "Capital One"),
        ("SPGI", "S&P Global"), ("BX", "Blackstone"), ("PGR", "Progressive"),
        # Salud
        ("UNH", "UnitedHealth"), ("JNJ", "Johnson & Johnson"), ("LLY", "Eli Lilly"),
        ("PFE", "Pfizer"), ("MRK", "Merck"), ("ABBV", "AbbVie"),
        ("TMO", "Thermo Fisher"), ("ABT", "Abbott"), ("DHR", "Danaher"),
        ("BMY", "Bristol Myers"), ("AMGN", "Amgen"), ("GILD", "Gilead"),
        ("CVS", "CVS Health"), ("ISRG", "Intuitive Surgical"), ("VRTX", "Vertex"),
        ("REGN", "Regeneron"), ("SYK", "Stryker"), ("BSX", "Boston Scientific"),
        ("MDT", "Medtronic"),
        # Consumo
        ("WMT", "Walmart"), ("COST", "Costco"), ("TGT", "Target"),
        ("HD", "Home Depot"), ("LOW", "Lowe's"), ("MCD", "McDonald's"),
        ("SBUX", "Starbucks"), ("NKE", "Nike"), ("KO", "Coca-Cola"),
        ("PEP", "PepsiCo"), ("PG", "Procter & Gamble"), ("PM", "Philip Morris"),
        ("MDLZ", "Mondelez"), ("CL", "Colgate-Palmolive"), ("TJX", "TJX"),
        ("LULU", "Lululemon"), ("CMG", "Chipotle"), ("F", "Ford"),
        ("GM", "General Motors"),
        # Energía
        ("XOM", "Exxon Mobil"), ("CVX", "Chevron"), ("COP", "ConocoPhillips"),
        ("EOG", "EOG Resources"), ("SLB", "SLB"), ("MPC", "Marathon Petroleum"),
        ("OXY", "Occidental"),
        # Industria y transporte
        ("BA", "Boeing"), ("CAT", "Caterpillar"), ("DE", "Deere"),
        ("HON", "Honeywell"), ("GE", "GE Aerospace"), ("LMT", "Lockheed Martin"),
        ("RTX", "RTX"), ("UPS", "UPS"), ("UNP", "Union Pacific"),
        ("MMM", "3M"), ("ETN", "Eaton"),
        # Materiales, utilities e inmobiliario
        ("LIN", "Linde"), ("SHW", "Sherwin-Williams"), ("FCX", "Freeport-McMoRan"),
        ("NEM", "Newmont"), ("NEE", "NextEra Energy"), ("SO", "Southern Company"),
        ("DUK", "Duke Energy"), ("AMT", "American Tower"), ("PLD", "Prologis"),
        ("EQIX", "Equinix"),
    ]
)


ALL_INSTRUMENTS: tuple[Instrument, ...] = STOCKS + ETFS + FOREX

_BY_SYMBOL = {i.symbol: i for i in ALL_INSTRUMENTS}


def get(symbol: str) -> Instrument:
    """Instrumento por símbolo canónico. Lanza KeyError si no es operable."""
    return _BY_SYMBOL[symbol]


def all_symbols() -> list[str]:
    return [i.symbol for i in ALL_INSTRUMENTS]


def equities() -> tuple[Instrument, ...]:
    """Acciones y ETFs: los que tienen volumen y benchmark comparable."""
    return STOCKS + ETFS


# ── Filtro dinámico de liquidez ───────────────────────────────────────────────

def filter_by_liquidity(
    instruments: Iterable[Instrument],
    bars: dict[str, pd.DataFrame],
    params: StrategyParams,
    top_n: int | None = None,
) -> list[Instrument]:
    """Deja solo los instrumentos con volumen suficiente, ordenados por liquidez.

    Los pares de forex pasan siempre: el mercado de divisas no publica volumen
    centralizado, y los 14 pares de Quantfury son de por sí los más líquidos
    del mundo. Para acciones y ETFs se exige un volumen medio en dólares por
    encima del umbral, que es lo que traduce el "mayor volumen" que pide el
    usuario a un criterio medible.
    """
    scored: list[tuple[float, Instrument]] = []
    forex: list[Instrument] = []

    for inst in instruments:
        if inst.is_forex:
            forex.append(inst)
            continue

        df = bars.get(inst.symbol)
        if df is None or df.empty or "volume" not in df:
            continue

        dv = ind.dollar_volume(df["close"], df["volume"], params.volume_window)
        latest = dv.iloc[-1] if len(dv) else float("nan")
        if pd.isna(latest) or latest < params.min_dollar_volume:
            continue
        scored.append((float(latest), inst))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    liquid = [inst for _, inst in scored]
    if top_n is not None:
        liquid = liquid[:top_n]

    return liquid + forex
