import atlas.tools.market as market

FAKE = {
    "AAPL": {
        "shortName": "Apple Inc.",
        "currentPrice": 231.4,
        "previousClose": 228.0,
        "currency": "USD",
        "marketCap": 3_500_000_000_000,
        "trailingPE": 34.2,
        "sector": "Technology",
    },
    # Indices report regularMarketPrice and leave currentPrice unset.
    "^GSPC": {
        "shortName": "S&P 500",
        "currentPrice": None,
        "regularMarketPrice": 7747.37,
        "previousClose": 7709.96,
        "currency": "USD",
    },
    "^IXIC": {
        "shortName": "Nasdaq",
        "regularMarketPrice": 26000.0,
        "previousClose": 25900.0,
    },
    "^DJI": {
        "shortName": "Dow Jones",
        "regularMarketPrice": 48000.0,
        "previousClose": 48100.0,
    },
}


def _fake_fetch(symbol: str) -> dict | None:
    """Fundamentals seam (yfinance-shaped)."""
    return FAKE.get(symbol.upper())


def _fake_quote(symbol: str) -> dict | None:
    """Provider-chain seam (already normalized)."""
    info = FAKE.get(symbol.upper())
    if info is None:
        return None
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    prev = info.get("previousClose")
    return {
        "symbol": symbol.upper(),
        "name": info.get("shortName"),
        "price": price,
        "previous_close": prev,
        "change_pct": ((price - prev) / prev * 100) if price and prev else None,
        "currency": info.get("currency", "USD"),
        "source": "TestProvider",
    }


def test_get_quote_computes_change(monkeypatch):
    monkeypatch.setattr(market, "_fetch_quote", _fake_quote)

    r = market.get_quote("aapl")

    assert r["ok"] is True
    assert r["data"]["symbol"] == "AAPL"
    assert r["data"]["price"] == 231.4
    assert round(r["data"]["change_pct"], 2) == 1.49
    assert r["source"] == "TestProvider"


def test_get_quote_unknown_symbol_returns_error_not_exception(monkeypatch):
    monkeypatch.setattr(market, "_fetch_quote", _fake_quote)

    r = market.get_quote("XYZQ")

    assert r["ok"] is False
    assert r["error"] == "no_such_symbol"


def test_get_quote_handles_an_index(monkeypatch):
    """Indices leave currentPrice unset — regularMarketPrice must be used."""
    monkeypatch.setattr(market, "_fetch_quote", _fake_quote)

    r = market.get_quote("^GSPC")

    assert r["ok"] is True
    assert r["data"]["price"] == 7747.37
    assert round(r["data"]["change_pct"], 2) == 0.49


def _fake_fundamentals(symbol: str) -> dict | None:
    info = FAKE.get(symbol.upper())
    if info is None:
        return None
    return {
        "symbol": symbol.upper(),
        "name": info.get("shortName"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "sector": info.get("sector"),
        "source": "TestProvider",
    }


def test_compare_companies_reports_partial_failure(monkeypatch):
    monkeypatch.setattr(market, "_fetch_fundamentals", _fake_fundamentals)

    r = market.compare_companies(["AAPL", "XYZQ"])

    assert r["ok"] is True
    assert "AAPL" in r["data"]["companies"]
    assert r["data"]["unavailable"] == ["XYZQ"]


def test_compare_companies_requires_at_least_two():
    r = market.compare_companies(["AAPL"])
    assert r["ok"] is False
    assert r["error"] == "need_two_symbols"


def test_market_overview_returns_the_three_major_indices(monkeypatch):
    monkeypatch.setattr(market, "_fetch_quote", _fake_quote)

    r = market.market_overview()

    assert r["ok"] is True
    names = {i["name"] for i in r["data"]["indices"]}
    assert names == {"S&P 500", "Nasdaq", "Dow Jones"}
    dow = next(i for i in r["data"]["indices"] if i["name"] == "Dow Jones")
    assert dow["change_pct"] < 0  # 48000 vs 48100 previous close


def _history(rows, previous_close=None):
    return lambda s, p: {
        "rows": rows, "previous_close": previous_close, "currency": "USD", "source": "Yahoo Finance"
    }


def test_price_history_measures_from_the_close_before_the_period(monkeypatch):
    """Measuring from the first bar inside the window dropped that session's
    move, and made "1d" a single bar that always read 0%."""
    monkeypatch.setattr(
        market,
        "_fetch_history",
        _history(
            [
                {"date": "2026-08-03", "close": 300.0, "high": 305.0, "low": 298.0},
                {"date": "2026-08-07", "close": 312.0, "high": 315.0, "low": 299.0},
            ],
            previous_close=240.0,
        ),
    )

    r = market.get_price_history("AAPL", "5d")

    assert r["ok"] is True
    assert r["data"]["base_close"] == 240.0
    assert r["data"]["end_close"] == 312.0
    assert round(r["data"]["change_pct"], 2) == 30.0
    assert r["data"]["period_high"] == 315.0
    assert r["data"]["period_low"] == 298.0
    assert r["data"]["currency"] == "USD"
    assert r["source"] == "Yahoo Finance"


def test_one_day_history_is_todays_move(monkeypatch):
    monkeypatch.setattr(
        market,
        "_fetch_history",
        _history([{"date": "2026-09-15", "close": 212.17, "high": 213.94, "low": 211.16}], 210.96),
    )

    r = market.get_price_history("NVDA", "1d")

    assert round(r["data"]["change_pct"], 2) == 0.57


def test_price_history_rejects_unknown_period():
    r = market.get_price_history("AAPL", "17y")
    assert r["ok"] is False
    assert r["error"] == "bad_period"


def test_price_history_with_no_data_returns_error(monkeypatch):
    monkeypatch.setattr(market, "_fetch_history", lambda s, p: None)

    r = market.get_price_history("XYZQ", "5d")

    assert r["ok"] is False
    assert r["error"] == "no_history"


def _calendar(dates, **extra):
    return lambda s: {"dates": dates, "source": "Finnhub", **extra}


def test_earnings_info_returns_next_date_and_estimates(monkeypatch):
    monkeypatch.setattr(market, "_today", lambda: "2026-09-16")
    monkeypatch.setattr(
        market,
        "_fetch_calendar",
        _calendar(["2026-10-30"], eps_estimate=1.97643, revenue_estimate=113256000000,
                  timing="after the close"),
    )

    r = market.get_earnings_info("AAPL")

    assert r["ok"] is True
    assert r["data"]["next_earnings_date"] == "2026-10-30"
    assert r["data"]["estimated_window"] is None
    assert r["data"]["timing"] == "after the close"
    assert r["data"]["eps_estimate"] == 1.97643
    assert r["data"]["revenue_estimate"] == 113256000000
    assert r["source"] == "Finnhub"


def test_an_unconfirmed_window_is_not_presented_as_a_date(monkeypatch):
    """Yahoo returns a two-date window until a company confirms; its first day
    used to be reported as the scheduled date."""
    monkeypatch.setattr(market, "_today", lambda: "2026-09-16")
    monkeypatch.setattr(market, "_fetch_calendar", _calendar(["2026-10-27", "2026-10-31"]))

    r = market.get_earnings_info("AAPL")

    assert r["data"]["next_earnings_date"] is None
    assert r["data"]["estimated_window"] == ["2026-10-27", "2026-10-31"]


def test_a_date_already_past_is_not_the_next_report(monkeypatch):
    monkeypatch.setattr(market, "_today", lambda: "2026-09-16")
    monkeypatch.setattr(market, "_fetch_calendar", _calendar(["2026-07-30"]))

    assert market.get_earnings_info("AAPL")["error"] == "no_earnings_date"


def test_earnings_info_when_unscheduled(monkeypatch):
    monkeypatch.setattr(market, "_fetch_calendar", lambda s: None)

    r = market.get_earnings_info("AAPL")

    assert r["ok"] is False
    assert r["error"] == "no_earnings_date"


def test_sources_name_the_providers_that_answered(monkeypatch):
    """market_overview and compare_companies said "yfinance" whoever answered."""
    monkeypatch.setattr(market, "_fetch_quote", _fake_quote)
    monkeypatch.setattr(market, "_fetch_fundamentals", _fake_fundamentals)

    assert market.market_overview()["source"] == "TestProvider"
    assert market.compare_companies(["AAPL", "^GSPC"])["source"] == "TestProvider"


def test_a_quote_carries_the_providers_trade_time(monkeypatch):
    def _stamped(symbol):
        quote = _fake_quote(symbol)
        quote["as_of"] = "2026-09-15T20:00:01Z"
        return quote

    monkeypatch.setattr(market, "_fetch_quote", _stamped)

    assert market.get_quote("AAPL")["as_of"] == "2026-09-15T20:00:01Z"


def test_a_quote_with_no_trade_time_is_stamped_with_its_trading_day(monkeypatch):
    """Alpha Vantage sends only "07. latest trading day". Stamping the answer with
    the current time would present Friday's close as a Saturday price."""

    def _daily_only(symbol):
        quote = _fake_quote(symbol)
        quote["as_of"] = None
        quote["session_date"] = "2026-09-15"
        return quote

    monkeypatch.setattr(market, "_fetch_quote", _daily_only)

    assert market.get_quote("AAPL")["as_of"] == "2026-09-15"
