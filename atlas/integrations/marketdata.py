"""Market data with provider failover.

Why this exists: yfinance works perfectly from a laptop and returns nothing from
a cloud host, because Yahoo blocks datacenter IP ranges. That failure is silent,
so the bot looked fine locally and told users it "couldn't pull live data" in
production. Relying on one source was the bug.

Providers are tried in order and the first usable answer wins. Keyed providers
come first because they are contractual rather than best-effort; ones whose key
is absent are skipped, so the bot still runs with none configured.

Field mappings were checked against live responses from production on
2026-09-16. Every percentage leaves this module as a percent, in a key ending
_pct, because one payload used to mix fractions and percents with no labels.
"""

import datetime as dt
import logging
import re
from functools import lru_cache
from zoneinfo import ZoneInfo

import httpx

from atlas.config import get_settings

log = logging.getLogger(__name__)

TIMEOUT = 12
UA = {"User-Agent": "Mozilla/5.0 (compatible; Atlas/1.0)"}

# FMP retired /api/v3 on 2025-08-31 — it now answers 403 "Legacy Endpoint" for
# anyone who subscribed after that date, which reads exactly like a bad key. The
# replacement is /stable, and it takes the symbol as a query parameter rather
# than in the path.
FMP_BASE = "https://financialmodelingprep.com/stable"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"

US_MARKET_TZ = "America/New_York"
# Plain US tickers and share classes (BRK.B). Exchange-suffixed listings (.NS,
# .L) and indices (^GSPC) are not, and the free keyed tiers do not cover them.
US_TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


def is_us_ticker(symbol: str) -> bool:
    return bool(US_TICKER.match((symbol or "").strip().upper()))


@lru_cache(maxsize=1)
def _http() -> httpx.Client:
    """One pooled client. A fresh client per call paid a TLS handshake each time,
    and the alert watcher alone makes dozens of calls an hour."""
    return httpx.Client(timeout=TIMEOUT, headers=UA)


def _pct(price, prev):
    if price is None or not prev:
        return None
    return (price - prev) / prev * 100


def _session(stamp, tz: str) -> tuple[str | None, str | None]:
    """(as_of in UTC, trading date in the exchange's own timezone) for a Unix time.

    Without these a Friday close fetched on Saturday was reported as current and
    keyed as Saturday's move.
    """
    if not stamp:
        return None, None
    moment = dt.datetime.fromtimestamp(int(stamp), dt.UTC)
    try:
        local = moment.astimezone(ZoneInfo(tz))
    except Exception:
        local = moment.astimezone(ZoneInfo(US_MARKET_TZ))
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ"), local.date().isoformat()


def _quote(
    symbol, name, price, prev, currency, source, *, as_of=None, session_date=None
):
    if price is None:
        return None
    return {
        "symbol": symbol.upper(),
        "name": name,
        "price": round(float(price), 4),
        "previous_close": round(float(prev), 4) if prev else None,
        "change_pct": _pct(float(price), float(prev) if prev else None),
        "currency": currency,
        "source": source,
        "as_of": as_of,
        "session_date": session_date,
    }


# --------------------------------------------------------------- providers


def finnhub_quote(symbol: str) -> dict | None:
    key = get_settings().finnhub_api_key
    if not key:
        return None
    r = _http().get(
        "https://finnhub.io/api/v1/quote", params={"symbol": symbol.upper(), "token": key}
    )
    r.raise_for_status()
    d = r.json()
    # Finnhub answers 200 with zeroes for unknown symbols.
    if not d.get("c"):
        return None
    as_of, session = _session(d.get("t"), US_MARKET_TZ)
    return _quote(
        symbol, None, d.get("c"), d.get("pc"), "USD", "Finnhub",
        as_of=as_of, session_date=session,
    )


def fmp_quote(symbol: str) -> dict | None:
    key = get_settings().fmp_api_key
    if not key:
        return None
    r = _http().get(FMP_BASE + "/quote", params={"symbol": symbol.upper(), "apikey": key})
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None
    d = rows[0]
    as_of, session = _session(d.get("timestamp"), US_MARKET_TZ)
    # Index levels are points, not dollars.
    currency = None if symbol.startswith("^") else "USD"
    return _quote(
        symbol, d.get("name"), d.get("price"), d.get("previousClose"), currency, "FMP",
        as_of=as_of, session_date=session,
    )


class _NoSuchSymbol:
    """Yahoo's definitive "this listing does not exist". Stops the chain, so a
    typo does not go on to spend Alpha Vantage's 25 requests a day."""


NO_SUCH_SYMBOL = _NoSuchSymbol()


def yahoo_quote(symbol: str):
    """Yahoo's chart endpoint. Keyless, but datacenter IPs are often blocked."""
    # range=1d, not 2d: over two days chartPreviousClose is the close BEFORE the
    # window, so the "daily" change silently spanned two sessions.
    r = _http().get(YAHOO_CHART + symbol, params={"interval": "1d", "range": "1d"})
    if r.status_code == 404:
        return NO_SUCH_SYMBOL
    r.raise_for_status()
    result = (r.json().get("chart") or {}).get("result") or []
    if not result:
        return None
    meta = result[0].get("meta") or {}
    as_of, session = _session(
        meta.get("regularMarketTime"), meta.get("exchangeTimezoneName") or US_MARKET_TZ
    )
    return _quote(
        symbol,
        meta.get("shortName") or meta.get("longName"),
        meta.get("regularMarketPrice"),
        meta.get("previousClose") or meta.get("chartPreviousClose"),
        None if symbol.startswith("^") else meta.get("currency"),
        "Yahoo Finance",
        as_of=as_of,
        session_date=session,
    )


def alphavantage_quote(symbol: str) -> dict | None:
    key = get_settings().alphavantage_api_key
    if not key:
        return None
    r = _http().get(
        "https://www.alphavantage.co/query",
        params={"function": "GLOBAL_QUOTE", "symbol": symbol.upper(), "apikey": key},
    )
    r.raise_for_status()
    d = r.json().get("Global Quote") or {}
    if not d.get("05. price"):
        return None
    return _quote(
        symbol, None, d.get("05. price"), d.get("08. previous close"), "USD",
        "Alpha Vantage", session_date=d.get("07. latest trading day") or None,
    )


# Keyed providers first: a contract beats best-effort scraping. yfinance is not
# here: it scrapes the same Yahoo backend yahoo_quote has just tried.
QUOTE_PROVIDERS = (
    ("finnhub", finnhub_quote),
    ("fmp", fmp_quote),
    ("yahoo", yahoo_quote),
    ("alphavantage", alphavantage_quote),
)

# Which symbols each keyed provider can answer on a free tier. Asking anyway
# costs a request that can only fail, and Alpha Vantage allows 25 a day.
_SERVES = {
    "finnhub": is_us_ticker,
    "fmp": lambda s: is_us_ticker(s) or s.startswith("^"),
    "alphavantage": is_us_ticker,
}


def fetch_quote(symbol: str) -> dict | None:
    """First provider to return a usable quote wins."""
    symbol = (symbol or "").strip().upper()
    for name, provider in QUOTE_PROVIDERS:
        serves = _SERVES.get(name)
        if serves is not None and not serves(symbol):
            continue
        try:
            quote = provider(symbol)
        except Exception as exc:
            log.debug("quote provider %s failed for %s: %s", name, symbol, scrub(str(exc)))
            continue
        if quote is NO_SUCH_SYMBOL:
            return None
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
    profile = _http().get(
        "https://finnhub.io/api/v1/stock/profile2", params={"symbol": sym, "token": key}
    )
    profile.raise_for_status()
    p = profile.json()
    if not p:
        return None

    metrics = _http().get(
        "https://finnhub.io/api/v1/stock/metric",
        params={"symbol": sym, "metric": "all", "token": key},
    )
    m = metrics.json().get("metric", {}) if metrics.status_code == 200 else {}

    cap = p.get("marketCapitalization")
    return _fundamentals(
        sym,
        p.get("name"),
        "Finnhub",
        currency=p.get("currency"),
        # Finnhub reports market cap in millions.
        market_cap=int(cap * 1_000_000) if cap else None,
        trailing_pe=m.get("peTTM"),
        # forwardPE, never peBasicExclExtraTTM: that one is a trailing multiple
        # and was reported here as forward for months.
        forward_pe=m.get("forwardPE"),
        # Finnhub already reports these three as percents.
        profit_margin_pct=m.get("netProfitMarginTTM"),
        revenue_growth_pct=m.get("revenueGrowthTTMYoy"),
        dividend_yield_pct=m.get("dividendYieldIndicatedAnnual"),
        sector=p.get("finnhubIndustry"),
        industry=p.get("finnhubIndustry"),
    )


def fmp_fundamentals(symbol: str) -> dict | None:
    key = get_settings().fmp_api_key
    if not key:
        return None
    r = _http().get(FMP_BASE + "/profile", params={"symbol": symbol.upper(), "apikey": key})
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None
    p = rows[0]
    # /stable renamed these: mktCap -> marketCap, lastDiv -> lastDividend. It is
    # the annual dividend per share in dollars, which used to be passed off as a
    # yield: Coca-Cola's 2.1 read as a 2.1 yield rather than about 2.4%.
    dividend, price = p.get("lastDividend"), p.get("price")
    return _fundamentals(
        symbol,
        p.get("companyName"),
        "FMP",
        currency=p.get("currency"),
        market_cap=p.get("marketCap"),
        trailing_pe=None,
        forward_pe=None,
        profit_margin_pct=None,
        revenue_growth_pct=None,
        dividend_yield_pct=round(dividend / price * 100, 4) if dividend and price else None,
        dividend_per_share=dividend,
        sector=p.get("sector"),
        industry=p.get("industry"),
    )


def yfinance_fundamentals(symbol: str) -> dict | None:
    import yfinance as yf

    info = yf.Ticker(symbol).info or {}
    if not info.get("shortName"):
        return None

    def _as_pct(fraction):
        return round(fraction * 100, 4) if fraction is not None else None

    return _fundamentals(
        symbol,
        info.get("shortName"),
        "yfinance",
        currency=info.get("currency"),
        market_cap=info.get("marketCap"),
        trailing_pe=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        # Yahoo gives margins and growth as fractions but dividendYield as a percent.
        profit_margin_pct=_as_pct(info.get("profitMargins")),
        revenue_growth_pct=_as_pct(info.get("revenueGrowth")),
        dividend_yield_pct=info.get("dividendYield"),
        sector=info.get("sector"),
        industry=info.get("industry"),
    )


FUNDAMENTAL_PROVIDERS = (
    ("finnhub", finnhub_fundamentals),
    ("fmp", fmp_fundamentals),
    ("yfinance", yfinance_fundamentals),
)


def fetch_fundamentals(symbol: str) -> dict | None:
    symbol = (symbol or "").strip().upper()
    for name, provider in FUNDAMENTAL_PROVIDERS:
        serves = _SERVES.get(name)
        if serves is not None and not serves(symbol):
            continue
        try:
            data = provider(symbol)
        except Exception as exc:
            log.debug(
                "fundamentals provider %s failed for %s: %s", name, symbol, scrub(str(exc))
            )
            continue
        if data is not None:
            return data
    log.warning("every fundamentals provider failed for %s", symbol)
    return None


# ----------------------------------------------------------------- history


def yahoo_history(symbol: str, period: str) -> dict | None:
    """Daily bars from Yahoo's chart endpoint, plus the close before the window.

    Plain HTTP, so price history no longer needs yfinance, which pulled pandas
    and numpy into RAM on a 1 GiB machine. The close before the window matters:
    measuring from the first bar inside it made "1d" always 0% and every longer
    period silently drop its first session's move.
    """
    interval = "1wk" if period == "5y" else "1d"
    r = _http().get(YAHOO_CHART + symbol, params={"interval": interval, "range": period})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    result = (r.json().get("chart") or {}).get("result") or []
    if not result:
        return None
    data = result[0]
    meta = data.get("meta") or {}
    tz = meta.get("exchangeTimezoneName") or US_MARKET_TZ
    bars = (data.get("indicators") or {}).get("quote") or [{}]
    closes, highs, lows = bars[0].get("close") or [], bars[0].get("high") or [], bars[0].get("low") or []
    rows = []
    for i, stamp in enumerate(data.get("timestamp") or []):
        close = closes[i] if i < len(closes) else None
        if close is None:
            continue  # holidays and the not-yet-closed session come back as null
        high = highs[i] if i < len(highs) and highs[i] is not None else close
        low = lows[i] if i < len(lows) and lows[i] is not None else close
        rows.append(
            {
                "date": _session(stamp, tz)[1],
                "close": round(float(close), 4),
                "high": round(float(high), 4),
                "low": round(float(low), 4),
            }
        )
    return {
        "rows": rows,
        "previous_close": meta.get("chartPreviousClose"),
        "currency": None if symbol.startswith("^") else meta.get("currency"),
        "source": "Yahoo Finance",
    }


def fetch_history(symbol: str, period: str) -> dict | None:
    symbol = (symbol or "").strip().upper()
    try:
        history = yahoo_history(symbol, period)
    except Exception as exc:
        log.warning("price history failed for %s: %s", symbol, scrub(str(exc)))
        return None
    if not history or not history["rows"]:
        return None
    return history


# ---------------------------------------------------------------- earnings


def finnhub_earnings(symbol: str) -> dict | None:
    """The next scheduled report, from Finnhub's free earnings calendar."""
    key = get_settings().finnhub_api_key
    if not key or not is_us_ticker(symbol):
        return None
    today = dt.datetime.now(dt.UTC).date()
    r = _http().get(
        "https://finnhub.io/api/v1/calendar/earnings",
        params={
            "symbol": symbol.upper(),
            "from": str(today),
            "to": str(today + dt.timedelta(days=120)),
            "token": key,
        },
    )
    r.raise_for_status()
    upcoming = sorted(
        (row for row in r.json().get("earningsCalendar") or [] if row.get("date")),
        key=lambda row: row["date"],
    )
    if not upcoming:
        return None
    row = upcoming[0]
    return {
        "dates": [row["date"]],
        "timing": {"bmo": "before the open", "amc": "after the close", "dmh": "during market hours"}.get(
            row.get("hour") or ""
        ),
        "eps_estimate": row.get("epsEstimate"),
        "eps_low": None,
        "eps_high": None,
        "revenue_estimate": row.get("revenueEstimate"),
        "source": "Finnhub",
    }


def yfinance_earnings(symbol: str) -> dict | None:
    """Yahoo's calendar, for listings Finnhub's free tier does not cover. It gives
    one date once the company confirms it and a two-date window until then."""
    import yfinance as yf

    calendar = yf.Ticker(symbol).calendar or {}
    dates = [str(d) for d in calendar.get("Earnings Date") or []]
    if not dates:
        return None
    return {
        "dates": dates,
        "timing": None,
        "eps_estimate": calendar.get("Earnings Average"),
        "eps_low": calendar.get("Earnings Low"),
        "eps_high": calendar.get("Earnings High"),
        "revenue_estimate": calendar.get("Revenue Average"),
        "source": "yfinance",
    }


def fetch_earnings(symbol: str) -> dict | None:
    symbol = (symbol or "").strip().upper()
    for name, provider in (("finnhub", finnhub_earnings), ("yfinance", yfinance_earnings)):
        try:
            data = provider(symbol)
        except Exception as exc:
            log.debug("earnings provider %s failed for %s: %s", name, symbol, scrub(str(exc)))
            continue
        if data is not None:
            return data
    return None


# ------------------------------------------------------------- credentials


_SECRET_PARAM = re.compile(
    r"((?:apikey|api_key|token|key)=)[^&\s'\"]+", re.IGNORECASE
)


def scrub(text: str) -> str:
    """Remove credentials from text that will be shown publicly.

    Provider errors quote the failing URL, and those URLs carry the API key as a
    query parameter — so an unscrubbed diagnostic page hands out every key it has.
    Both the parameter pattern and the exact configured values are removed,
    because a key can also appear in a body or header echo.
    """
    text = _SECRET_PARAM.sub(r"\1***", text)
    settings = get_settings()
    for secret in (
        settings.finnhub_api_key,
        settings.fmp_api_key,
        settings.alphavantage_api_key,
        settings.gemini_api_key,
        settings.groq_api_key,
        settings.telegram_token,
    ):
        if secret and len(secret) >= 8:
            text = text.replace(secret, "***")
    return text


# Daily-capped or scraping providers, which a routine health check should not
# spend. /diag?all=1 includes them.
_LIMITED = {"alphavantage", "yfinance"}


def probe(symbol: str = "AAPL", include_limited: bool = False) -> dict:
    """Report which providers work from wherever this is running.

    Exists because provider availability depends on the host's IP, so it can only
    be answered from the deployed environment — not from a developer's laptop.
    """
    results = {"quotes": {}, "fundamentals": {}}

    for name, provider in QUOTE_PROVIDERS:
        if name in _LIMITED and not include_limited:
            continue
        try:
            quote = provider(symbol)
            results["quotes"][name] = (
                {"ok": True, "price": quote["price"]}
                if isinstance(quote, dict)
                else {"ok": False, "reason": "no data (key missing or unknown symbol)"}
            )
        except Exception as exc:
            results["quotes"][name] = {
                "ok": False,
                "reason": scrub(f"{type(exc).__name__}: {exc}")[:160],
            }

    for name, provider in FUNDAMENTAL_PROVIDERS:
        if name in _LIMITED and not include_limited:
            continue
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
                "reason": scrub(f"{type(exc).__name__}: {exc}")[:160],
            }

    return results
