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
    }
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
