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


# ----------------------------------------------------------- fundamentals
#
# Yahoo's fundamentals endpoint now requires a crumb, so unlike quotes there is
# no keyless option that survives a datacenter IP. A provider key is required in
# production; yfinance covers local development only.


def _fundamentals(symbol, name, source, **fields) -> dict:
    return {"symbol": symbol.upper(), "name": name, "source": source, **fields}


def finnhub_fundamentals(symbol: str) -> dict | None:
    key = get_settings().finnhub_api_key
    if not key:
        return None
    sym = symbol.upper()
    profile = httpx.get(
        "https://finnhub.io/api/v1/stock/profile2",
        params={"symbol": sym, "token": key},
        timeout=TIMEOUT,
        headers=UA,
    )
    profile.raise_for_status()
    p = profile.json()
    if not p:
        return None

    metrics = httpx.get(
        "https://finnhub.io/api/v1/stock/metric",
        params={"symbol": sym, "metric": "all", "token": key},
        timeout=TIMEOUT,
        headers=UA,
    )
    m = metrics.json().get("metric", {}) if metrics.status_code == 200 else {}

    cap = p.get("marketCapitalization")
    return _fundamentals(
        sym,
        p.get("name"),
        "Finnhub",
        # Finnhub reports market cap in millions.
        market_cap=int(cap * 1_000_000) if cap else None,
        trailing_pe=m.get("peTTM"),
        forward_pe=m.get("peBasicExclExtraTTM"),
        profit_margin=(m.get("netProfitMarginTTM") / 100)
        if m.get("netProfitMarginTTM") is not None
        else None,
        revenue_growth=(m.get("revenueGrowthTTMYoy") / 100)
        if m.get("revenueGrowthTTMYoy") is not None
        else None,
        dividend_yield=m.get("dividendYieldIndicatedAnnual"),
        sector=p.get("finnhubIndustry"),
        industry=p.get("finnhubIndustry"),
    )


def fmp_fundamentals(symbol: str) -> dict | None:
    key = get_settings().fmp_api_key
    if not key:
        return None
    r = httpx.get(
        f"https://financialmodelingprep.com/api/v3/profile/{symbol.upper()}",
        params={"apikey": key},
        timeout=TIMEOUT,
        headers=UA,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None
    p = rows[0]
    return _fundamentals(
        symbol,
        p.get("companyName"),
        "FMP",
        market_cap=p.get("mktCap"),
        trailing_pe=None,
        forward_pe=None,
        profit_margin=None,
        revenue_growth=None,
        dividend_yield=p.get("lastDiv"),
        sector=p.get("sector"),
        industry=p.get("industry"),
    )


def yfinance_fundamentals(symbol: str) -> dict | None:
    import yfinance as yf

    info = yf.Ticker(symbol).info or {}
    if not info.get("shortName"):
        return None
    return _fundamentals(
        symbol,
        info.get("shortName"),
        "yfinance",
        market_cap=info.get("marketCap"),
        trailing_pe=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        profit_margin=info.get("profitMargins"),
        revenue_growth=info.get("revenueGrowth"),
        dividend_yield=info.get("dividendYield"),
        sector=info.get("sector"),
        industry=info.get("industry"),
    )


FUNDAMENTAL_PROVIDERS = (
    ("finnhub", finnhub_fundamentals),
    ("fmp", fmp_fundamentals),
    ("yfinance", yfinance_fundamentals),
)


def fetch_fundamentals(symbol: str) -> dict | None:
    for name, provider in FUNDAMENTAL_PROVIDERS:
        try:
            data = provider(symbol)
        except Exception as exc:
            log.debug("fundamentals provider %s failed for %s: %s", name, symbol, exc)
            continue
        if data is not None:
            return data
    log.warning("every fundamentals provider failed for %s", symbol)
    return None


def probe(symbol: str = "AAPL") -> dict:
    """Report which providers work from wherever this is running.

    Exists because provider availability depends on the host's IP, so it can only
    be answered from the deployed environment — not from a developer's laptop.
    """
    results = {"quotes": {}, "fundamentals": {}}

    for name, provider in QUOTE_PROVIDERS:
        try:
            quote = provider(symbol)
            results["quotes"][name] = (
                {"ok": True, "price": quote["price"]}
                if quote
                else {"ok": False, "reason": "no data (key missing or unknown symbol)"}
            )
        except Exception as exc:
            results["quotes"][name] = {
                "ok": False,
                "reason": f"{type(exc).__name__}: {exc}"[:160],
            }

    for name, provider in FUNDAMENTAL_PROVIDERS:
        try:
            data = provider(symbol)
            results["fundamentals"][name] = (
                {"ok": True, "pe": data.get("trailing_pe")}
                if data
                else {"ok": False, "reason": "no data (key missing or unknown symbol)"}
            )
        except Exception as exc:
            results["fundamentals"][name] = {
                "ok": False,
                "reason": f"{type(exc).__name__}: {exc}"[:160],
            }

    return results
