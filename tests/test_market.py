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
    return FAKE.get(symbol.upper())


def test_get_quote_computes_change(monkeypatch):
    monkeypatch.setattr(market, "_fetch_info", _fake_fetch)

    r = market.get_quote("aapl")

    assert r["ok"] is True
    assert r["data"]["symbol"] == "AAPL"
    assert r["data"]["price"] == 231.4
    assert round(r["data"]["change_pct"], 2) == 1.49
    assert r["source"] == "yfinance"


def test_get_quote_unknown_symbol_returns_error_not_exception(monkeypatch):
    monkeypatch.setattr(market, "_fetch_info", _fake_fetch)

    r = market.get_quote("XYZQ")

    assert r["ok"] is False
    assert r["error"] == "no_such_symbol"


def test_get_quote_handles_an_index(monkeypatch):
    """Indices leave currentPrice unset — regularMarketPrice must be used."""
    monkeypatch.setattr(market, "_fetch_info", _fake_fetch)

    r = market.get_quote("^GSPC")

    assert r["ok"] is True
    assert r["data"]["price"] == 7747.37
    assert round(r["data"]["change_pct"], 2) == 0.49


def test_compare_companies_reports_partial_failure(monkeypatch):
    monkeypatch.setattr(market, "_fetch_info", _fake_fetch)

    r = market.compare_companies(["AAPL", "XYZQ"])

    assert r["ok"] is True
    assert "AAPL" in r["data"]["companies"]
    assert r["data"]["unavailable"] == ["XYZQ"]


def test_compare_companies_requires_at_least_two():
    r = market.compare_companies(["AAPL"])
    assert r["ok"] is False
    assert r["error"] == "need_two_symbols"


def test_market_overview_returns_the_three_major_indices(monkeypatch):
    monkeypatch.setattr(market, "_fetch_info", _fake_fetch)

    r = market.market_overview()

    assert r["ok"] is True
    names = {i["name"] for i in r["data"]["indices"]}
    assert names == {"S&P 500", "Nasdaq", "Dow Jones"}
    dow = next(i for i in r["data"]["indices"] if i["name"] == "Dow Jones")
    assert dow["change_pct"] < 0  # 48000 vs 48100 previous close


def test_price_history_summarizes_the_period(monkeypatch):
    monkeypatch.setattr(
        market,
        "_fetch_history",
        lambda s, p: [
            {"date": "2026-08-03", "close": 300.0, "high": 305.0, "low": 298.0},
            {"date": "2026-08-07", "close": 312.0, "high": 315.0, "low": 299.0},
        ],
    )

    r = market.get_price_history("AAPL", "5d")

    assert r["ok"] is True
    assert r["data"]["start_close"] == 300.0
    assert r["data"]["end_close"] == 312.0
    assert round(r["data"]["change_pct"], 2) == 4.0
    assert r["data"]["period_high"] == 315.0
    assert r["data"]["period_low"] == 298.0


def test_price_history_rejects_unknown_period():
    r = market.get_price_history("AAPL", "17y")
    assert r["ok"] is False
    assert r["error"] == "bad_period"


def test_price_history_with_no_data_returns_error(monkeypatch):
    monkeypatch.setattr(market, "_fetch_history", lambda s, p: [])

    r = market.get_price_history("XYZQ", "5d")

    assert r["ok"] is False
    assert r["error"] == "no_history"


def test_earnings_info_returns_next_date_and_estimates(monkeypatch):
    monkeypatch.setattr(
        market,
        "_fetch_calendar",
        lambda s: {
            "Earnings Date": ["2026-10-30"],
            "Earnings Average": 1.97643,
            "Revenue Average": 113256000000,
        },
    )

    r = market.get_earnings_info("AAPL")

    assert r["ok"] is True
    assert r["data"]["next_earnings_date"] == "2026-10-30"
    assert r["data"]["eps_estimate"] == 1.97643
    assert r["data"]["revenue_estimate"] == 113256000000


def test_earnings_info_when_unscheduled(monkeypatch):
    monkeypatch.setattr(market, "_fetch_calendar", lambda s: {})

    r = market.get_earnings_info("AAPL")

    assert r["ok"] is False
    assert r["error"] == "no_earnings_date"
