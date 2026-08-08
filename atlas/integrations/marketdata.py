"""Market data with provider failover.

Why this exists: yfinance works perfectly from a laptop and returns nothing from
a cloud host, because Yahoo blocks datacenter IP ranges. That failure is silent,
so the bot looked fine locally and told users it "couldn't pull live data" in
production. Relying on one source was the bug.

Providers are tried in order and the first usable answer wins. Keyed providers
come first because they are contractual rather than best-effort; ones whose key
is absent are skipped, so the bot still runs with none configured.
"""

import logging

import httpx

from atlas.config import get_settings

log = logging.getLogger(__name__)

TIMEOUT = 12
UA = {"User-Agent": "Mozilla/5.0 (compatible; Atlas/1.0)"}


def _pct(price, prev):
    if price is None or not prev:
        return None
    return (price - prev) / prev * 100


def _quote(symbol, name, price, prev, currency, source):
    if price is None:
        return None
    return {
        "symbol": symbol.upper(),
        "name": name,
        "price": round(float(price), 4),
        "previous_close": round(float(prev), 4) if prev else None,
        "change_pct": _pct(float(price), float(prev) if prev else None),
        "currency": currency or "USD",
        "source": source,
    }


# --------------------------------------------------------------- providers


def finnhub_quote(symbol: str) -> dict | None:
    key = get_settings().finnhub_api_key
    if not key:
        return None
    r = httpx.get(
        "https://finnhub.io/api/v1/quote",
        params={"symbol": symbol.upper(), "token": key},
        timeout=TIMEOUT,
        headers=UA,
    )
    r.raise_for_status()
    d = r.json()
    # Finnhub answers 200 with zeroes for unknown symbols.
    if not d.get("c"):
        return None
    return _quote(symbol, None, d.get("c"), d.get("pc"), "USD", "Finnhub")


def fmp_quote(symbol: str) -> dict | None:
    key = get_settings().fmp_api_key
    if not key:
        return None
    r = httpx.get(
        f"https://financialmodelingprep.com/api/v3/quote/{symbol.upper()}",
        params={"apikey": key},
        timeout=TIMEOUT,
        headers=UA,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None
    d = rows[0]
    return _quote(
        symbol, d.get("name"), d.get("price"), d.get("previousClose"), "USD", "FMP"
    )


def yahoo_quote(symbol: str) -> dict | None:
    """Yahoo's chart endpoint. Keyless, but datacenter IPs are often blocked."""
    r = httpx.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"interval": "1d", "range": "2d"},
        timeout=TIMEOUT,
        headers=UA,
    )
    r.raise_for_status()
    result = (r.json().get("chart") or {}).get("result") or []
    if not result:
        return None
    meta = result[0].get("meta") or {}
    return _quote(
        symbol,
        meta.get("shortName") or meta.get("longName"),
        meta.get("regularMarketPrice"),
        meta.get("chartPreviousClose") or meta.get("previousClose"),
        meta.get("currency"),
        "Yahoo Finance",
    )


def alphavantage_quote(symbol: str) -> dict | None:
    key = get_settings().alphavantage_api_key
    if not key:
        return None
    r = httpx.get(
        "https://www.alphavantage.co/query",
        params={"function": "GLOBAL_QUOTE", "symbol": symbol.upper(), "apikey": key},
        timeout=TIMEOUT,
        headers=UA,
    )
    r.raise_for_status()
    d = r.json().get("Global Quote") or {}
    if not d.get("05. price"):
        return None
    return _quote(
        symbol, None, d.get("05. price"), d.get("08. previous close"), "USD",
        "Alpha Vantage",
    )


def yfinance_quote(symbol: str) -> dict | None:
    """Last resort. Reliable locally, usually blocked from cloud hosts."""
    import yfinance as yf

    info = yf.Ticker(symbol).info or {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None:
        return None
    return _quote(
        symbol,
        info.get("shortName"),
        price,
        info.get("previousClose"),
        info.get("currency"),
        "yfinance",
    )


# Keyed providers first: a contract beats best-effort scraping.
QUOTE_PROVIDERS = (
    ("finnhub", finnhub_quote),
    ("fmp", fmp_quote),
    ("yahoo", yahoo_quote),
    ("alphavantage", alphavantage_quote),
    ("yfinance", yfinance_quote),
)


def fetch_quote(symbol: str) -> dict | None:
    """First provider to return a usable quote wins."""
    for name, provider in QUOTE_PROVIDERS:
        try:
            quote = provider(symbol)
        except Exception as exc:
            log.debug("quote provider %s failed for %s: %s", name, symbol, exc)
            continue
        if quote is not None:
            return quote
    log.warning("every quote provider failed for %s", symbol)
    return None


def probe(symbol: str = "AAPL") -> dict:
    """Report which providers work from wherever this is running.

    Exists because provider availability depends on the host's IP, so it can only
    be answered from the deployed environment — not from a developer's laptop.
    """
    results = {}
    for name, provider in QUOTE_PROVIDERS:
        try:
            quote = provider(symbol)
            results[name] = (
                {"ok": True, "price": quote["price"]}
                if quote
                else {"ok": False, "reason": "no data (key missing or unknown symbol)"}
            )
        except Exception as exc:
            results[name] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:160]}
    return results
