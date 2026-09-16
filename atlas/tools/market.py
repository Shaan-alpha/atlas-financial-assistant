"""Live market data tools, backed by the provider chains in atlas.integrations.marketdata."""

import datetime as dt

from atlas.tools.result import err, ok

SOURCE = "market data"

# Symbol -> display name for the headline indices.
INDICES = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones"}

VALID_PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "5y")


def _sources(rows) -> str:
    """Name whichever providers actually answered, not a fixed label."""
    return ", ".join(sorted({row.get("source") or SOURCE for row in rows})) or SOURCE


def _fetch_quote(symbol: str) -> dict | None:
    """Normalized quote via the provider chain. Tests monkeypatch this."""
    from atlas.integrations import marketdata

    return marketdata.fetch_quote(symbol)


def get_quote(symbol: str) -> dict:
    """Return the current price and daily move for one listed security or index.

    Args:
        symbol: Ticker symbol, for example "AAPL", "MSFT", or an index like "^GSPC".
    """
    quote = _fetch_quote(symbol)
    if quote is None:
        return err("no_such_symbol", f"No listed security matches '{symbol}'.")
    # as_of is the provider's trade time: a Friday close read on Saturday must not
    # be stamped as a Saturday price. Alpha Vantage sends only a trading day, so
    # that is used rather than letting ok() stamp the current time.
    return ok(
        quote,
        source=quote.get("source") or SOURCE,
        as_of=quote.get("as_of") or quote.get("session_date"),
    )


def _fetch_fundamentals(symbol: str) -> dict | None:
    """Provider-chain seam for fundamentals. Tests monkeypatch this."""
    from atlas.integrations import marketdata

    return marketdata.fetch_fundamentals(symbol)


def get_fundamentals(symbol: str) -> dict:
    """Return valuation and profile fundamentals for one listed security.

    Percentages are percents, in fields ending _pct. Amounts are in `currency`.

    Args:
        symbol: Ticker symbol, for example "NVDA".
    """
    data = _fetch_fundamentals(symbol)
    if data is None:
        return err("no_such_symbol", f"No fundamentals available for '{symbol}'.")
    return ok(data, source=data.get("source") or SOURCE)


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

    return ok(
        {"companies": companies, "unavailable": unavailable},
        source=_sources(companies.values()),
    )


def market_overview() -> dict:
    """Return how the major US indices are trading right now.

    Use for broad questions like "how is the market today" or "what moved today".
    """
    rows = []
    for symbol, name in INDICES.items():
        quote = _fetch_quote(symbol)
        if quote is None:
            continue
        quote["name"] = name
        rows.append(quote)

    if not rows:
        return err("market_data_unavailable", "Index data is not available right now.")
    return ok({"indices": rows}, source=_sources(rows))


def _fetch_history(symbol: str, period: str) -> dict | None:
    """Provider seam: {"rows", "previous_close", "currency", "source"}. Tests
    monkeypatch this."""
    from atlas.integrations import marketdata

    return marketdata.fetch_history(symbol, period)


def get_price_history(symbol: str, period: str = "1mo") -> dict:
    """Return how a security has traded over a period, for trend questions.

    Use for "how has X done this month" or comparing performance across days.
    change_pct runs from the close before the period to the latest close, so
    "1d" is today's move. Closes are actual, not dividend-adjusted.

    Args:
        symbol: Ticker symbol, for example "NVDA".
        period: One of "1d", "5d", "1mo", "3mo", "6mo", "1y", "5y".
    """
    if period not in VALID_PERIODS:
        return err("bad_period", f"Period must be one of: {', '.join(VALID_PERIODS)}.")

    history = _fetch_history(symbol, period)
    rows = (history or {}).get("rows") or []
    if not rows:
        return err("no_history", f"No price history available for '{symbol}'.")

    start, end = rows[0], rows[-1]
    # The close before the window is the base. Measuring from the first bar
    # inside it made "1d" always 0% and dropped the first session of every period.
    base = history.get("previous_close") or start["close"]
    return ok(
        {
            "symbol": symbol.upper(),
            "period": period,
            "currency": history.get("currency"),
            "start_date": start["date"],
            "end_date": end["date"],
            "base_close": base,
            "end_close": end["close"],
            "change_pct": ((end["close"] - base) / base * 100) if base else None,
            "period_high": max(r["high"] for r in rows),
            "period_low": min(r["low"] for r in rows),
            "sessions": len(rows),
        },
        source=history.get("source") or SOURCE,
    )


def _fetch_calendar(symbol: str) -> dict | None:
    """Provider seam: {"dates", "timing", estimates..., "source"}. Tests
    monkeypatch this."""
    from atlas.integrations import marketdata

    return marketdata.fetch_earnings(symbol)


def _today() -> str:
    """Seam. Tests pin the date."""
    return dt.date.today().isoformat()


def get_earnings_info(symbol: str) -> dict:
    """Return the next scheduled earnings date and analyst estimates.

    Use for "when does X report" or "what are they expected to earn". When the
    company has not confirmed a date, a window comes back instead: say it is an
    estimate.

    Args:
        symbol: Ticker symbol, for example "AAPL".
    """
    calendar = _fetch_calendar(symbol) or {}
    today = _today()
    # Past dates are not the next report.
    dates = sorted(d[:10] for d in calendar.get("dates") or [] if d[:10] >= today)
    if not dates:
        return err(
            "no_earnings_date", f"No scheduled earnings date published for '{symbol}'."
        )

    window = len(dates) > 1 and dates[0] != dates[-1]
    return ok(
        {
            "symbol": symbol.upper(),
            # A window's first day is a guess, not a schedule; never present it as one.
            "next_earnings_date": None if window else dates[0],
            "estimated_window": [dates[0], dates[-1]] if window else None,
            "timing": calendar.get("timing"),
            "eps_estimate": calendar.get("eps_estimate"),
            "eps_low": calendar.get("eps_low"),
            "eps_high": calendar.get("eps_high"),
            "revenue_estimate": calendar.get("revenue_estimate"),
        },
        source=calendar.get("source") or SOURCE,
    )
