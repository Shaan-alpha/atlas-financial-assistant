"""Live market data. Keyless — yfinance needs no credentials."""

from atlas.tools.result import err, ok

SOURCE = "yfinance"

# Symbol -> display name for the headline indices.
INDICES = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones"}

VALID_PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "5y")


def _price_of(info: dict) -> float | None:
    """Equities report currentPrice; indices report only regularMarketPrice."""
    return info.get("currentPrice") or info.get("regularMarketPrice")


def _fetch_info(symbol: str) -> dict | None:
    """Network seam. Tests monkeypatch this so the suite stays offline."""
    import yfinance as yf

    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return None
    if not info or _price_of(info) is None:
        return None
    return info


def _fetch_history(symbol: str, period: str) -> list[dict]:
    """Network seam returning plain rows so tools stay pandas-free."""
    import yfinance as yf

    try:
        frame = yf.Ticker(symbol).history(period=period)
    except Exception:
        return []
    return [
        {
            "date": str(index.date()),
            "close": round(float(row["Close"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
        }
        for index, row in frame.iterrows()
    ]


def _fetch_calendar(symbol: str) -> dict:
    """Network seam. yfinance returns a dict of upcoming corporate dates."""
    import yfinance as yf

    try:
        return yf.Ticker(symbol).calendar or {}
    except Exception:
        return {}


def _quote_from_info(symbol: str, info: dict) -> dict:
    price = _price_of(info)
    prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
    change_pct = ((price - prev) / prev * 100) if price and prev else None
    return {
        "symbol": symbol.upper(),
        "name": info.get("shortName"),
        "price": price,
        "previous_close": prev,
        "change_pct": change_pct,
        "currency": info.get("currency", "USD"),
    }


def get_quote(symbol: str) -> dict:
    """Return the current price and daily move for one listed security or index.

    Args:
        symbol: Ticker symbol, for example "AAPL", "MSFT", or an index like "^GSPC".
    """
    info = _fetch_info(symbol)
    if info is None:
        return err("no_such_symbol", f"No listed security matches '{symbol}'.")
    return ok(_quote_from_info(symbol, info), source=SOURCE)


def get_fundamentals(symbol: str) -> dict:
    """Return valuation and profile fundamentals for one listed security.

    Args:
        symbol: Ticker symbol, for example "NVDA".
    """
    info = _fetch_info(symbol)
    if info is None:
        return err("no_such_symbol", f"No listed security matches '{symbol}'.")
    return ok(
        {
            "symbol": symbol.upper(),
            "name": info.get("shortName"),
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        },
        source=SOURCE,
    )


def compare_companies(symbols: list[str]) -> dict:
    """Return side-by-side fundamentals for two or more listed securities.

    Args:
        symbols: Two or more ticker symbols, for example ["MSFT", "GOOGL"].
    """
    if len(symbols) < 2:
        return err("need_two_symbols", "Comparison needs at least two ticker symbols.")

    companies: dict[str, dict] = {}
    unavailable: list[str] = []
    for symbol in symbols:
        result = get_fundamentals(symbol)
        if result["ok"]:
            companies[symbol.upper()] = result["data"]
        else:
            unavailable.append(symbol.upper())

    if not companies:
        return err("no_data", f"No data available for any of: {', '.join(symbols)}.")

    return ok({"companies": companies, "unavailable": unavailable}, source=SOURCE)


def market_overview() -> dict:
    """Return how the major US indices are trading right now.

    Use for broad questions like "how is the market today" or "what moved today".
    """
    rows = []
    for symbol, name in INDICES.items():
        info = _fetch_info(symbol)
        if info is None:
            continue
        quote = _quote_from_info(symbol, info)
        quote["name"] = name
        rows.append(quote)

    if not rows:
        return err("market_data_unavailable", "Index data is not available right now.")
    return ok({"indices": rows}, source=SOURCE)


def get_price_history(symbol: str, period: str = "1mo") -> dict:
    """Return how a security has traded over a period, for trend questions.

    Use for "how has X done this month" or comparing performance across days.

    Args:
        symbol: Ticker symbol, for example "NVDA".
        period: One of "1d", "5d", "1mo", "3mo", "6mo", "1y", "5y".
    """
    if period not in VALID_PERIODS:
        return err("bad_period", f"Period must be one of: {', '.join(VALID_PERIODS)}.")

    rows = _fetch_history(symbol, period)
    if not rows:
        return err("no_history", f"No price history available for '{symbol}'.")

    start, end = rows[0], rows[-1]
    change_pct = (
        ((end["close"] - start["close"]) / start["close"] * 100)
        if start["close"]
        else None
    )
    return ok(
        {
            "symbol": symbol.upper(),
            "period": period,
            "start_date": start["date"],
            "end_date": end["date"],
            "start_close": start["close"],
            "end_close": end["close"],
            "change_pct": change_pct,
            "period_high": max(r["high"] for r in rows),
            "period_low": min(r["low"] for r in rows),
            "points": len(rows),
        },
        source=SOURCE,
    )


def get_earnings_info(symbol: str) -> dict:
    """Return the next scheduled earnings date and analyst estimates.

    Use for "when does X report" or "what are they expected to earn".

    Args:
        symbol: Ticker symbol, for example "AAPL".
    """
    calendar = _fetch_calendar(symbol)
    dates = calendar.get("Earnings Date") or []
    if not dates:
        return err(
            "no_earnings_date", f"No scheduled earnings date published for '{symbol}'."
        )

    return ok(
        {
            "symbol": symbol.upper(),
            "next_earnings_date": str(dates[0]),
            "eps_estimate": calendar.get("Earnings Average"),
            "eps_low": calendar.get("Earnings Low"),
            "eps_high": calendar.get("Earnings High"),
            "revenue_estimate": calendar.get("Revenue Average"),
        },
        source=SOURCE,
    )
