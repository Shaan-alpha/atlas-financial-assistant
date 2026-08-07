"""Live market data. Keyless — yfinance needs no credentials."""

from atlas.tools.result import err, ok

SOURCE = "yfinance"


def _fetch_info(symbol: str) -> dict | None:
    """Network seam. Tests monkeypatch this so the suite stays offline."""
    import yfinance as yf

    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return None
    if not info or info.get("currentPrice") is None:
        return None
    return info


def _quote_from_info(symbol: str, info: dict) -> dict:
    price = info.get("currentPrice")
    prev = info.get("previousClose")
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
    """Return the current price and daily move for one listed security.

    Args:
        symbol: Ticker symbol, for example "AAPL" or "MSFT".
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
