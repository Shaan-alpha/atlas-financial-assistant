"""Provider failover.

The production bug this guards: yfinance returns data from a laptop and nothing
from a cloud host, because Yahoo blocks datacenter IPs. One source is not enough.
"""

import pytest

import atlas.integrations.marketdata as md

pytestmark = pytest.mark.usefixtures("env")


def _chain(monkeypatch, providers):
    monkeypatch.setattr(md, "QUOTE_PROVIDERS", tuple(providers))


def _good(name, price=100.0):
    def provider(symbol):
        return md._quote(symbol, "Test Co", price, 90.0, "USD", name)

    return provider


def _none(symbol):
    return None


def _boom(symbol):
    raise RuntimeError("blocked from this IP")


def test_first_working_provider_wins(monkeypatch):
    _chain(monkeypatch, [("a", _good("A", 101)), ("b", _good("B", 202))])

    quote = md.fetch_quote("AAPL")

    assert quote["price"] == 101
    assert quote["source"] == "A"


def test_falls_through_a_provider_that_raises(monkeypatch):
    """A blocked datacenter IP raises; it must not take the whole chain down."""
    _chain(monkeypatch, [("blocked", _boom), ("backup", _good("Backup", 55))])

    quote = md.fetch_quote("AAPL")

    assert quote["source"] == "Backup"


def test_falls_through_a_provider_returning_nothing(monkeypatch):
    _chain(monkeypatch, [("keyless", _none), ("backup", _good("Backup", 77))])

    assert md.fetch_quote("AAPL")["price"] == 77


def test_all_providers_failing_returns_none(monkeypatch):
    _chain(monkeypatch, [("a", _boom), ("b", _none)])

    assert md.fetch_quote("AAPL") is None


def test_change_percent_is_computed():
    quote = md._quote("NVDA", "Nvidia", 110.0, 100.0, "USD", "T")
    assert round(quote["change_pct"], 2) == 10.0


def test_missing_previous_close_does_not_crash():
    quote = md._quote("NVDA", "Nvidia", 110.0, None, "USD", "T")
    assert quote["change_pct"] is None
    assert quote["price"] == 110.0


def test_no_price_means_no_quote():
    assert md._quote("NVDA", "Nvidia", None, 100.0, "USD", "T") is None


def test_keyed_providers_skip_themselves_when_unconfigured(monkeypatch):
    """With no keys set these must return None quietly, not raise."""
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    from atlas.config import get_settings

    get_settings.cache_clear()

    assert md.finnhub_quote("AAPL") is None
    assert md.fmp_quote("AAPL") is None
    assert md.alphavantage_quote("AAPL") is None


def test_probe_reports_every_provider(monkeypatch):
    _chain(monkeypatch, [("good", _good("G")), ("bad", _boom), ("empty", _none)])
    monkeypatch.setattr(md, "FUNDAMENTAL_PROVIDERS", (("fnd", _none),))

    report = md.probe("AAPL")

    assert report["quotes"]["good"]["ok"] is True
    assert report["quotes"]["bad"]["ok"] is False
    assert "blocked from this IP" in report["quotes"]["bad"]["reason"]
    assert report["quotes"]["empty"]["ok"] is False
    assert report["fundamentals"]["fnd"]["ok"] is False


def test_fundamentals_chain_falls_through(monkeypatch):
    """Fundamentals need a keyed provider in production: Yahoo now demands a
    crumb, so there is no keyless path that survives a datacenter IP."""

    def _blocked(symbol):
        raise RuntimeError("rate limited")

    def _works(symbol):
        return {"symbol": symbol, "name": "Test Co", "trailing_pe": 21.0,
                "source": "Backup"}

    monkeypatch.setattr(
        md, "FUNDAMENTAL_PROVIDERS", (("a", _blocked), ("b", _works))
    )

    data = md.fetch_fundamentals("AAPL")

    assert data["source"] == "Backup"
    assert data["trailing_pe"] == 21.0


def test_fundamentals_all_failing_returns_none(monkeypatch):
    monkeypatch.setattr(md, "FUNDAMENTAL_PROVIDERS", (("a", _none),))
    assert md.fetch_fundamentals("AAPL") is None


# ------------------------------------------------------------- credentials


def test_scrub_removes_api_keys_from_urls():
    """Provider errors quote the failing URL, and those URLs carry the key.
    /diag is a public endpoint, so an unscrubbed error hands out credentials."""
    # The real shape: on /stable the key trails a symbol parameter, so the
    # scrub has to catch &apikey=, not just ?apikey=.
    leaked = (
        "HTTPStatusError: Client error '403 Forbidden' for url "
        "'https://financialmodelingprep.com/stable/quote"
        "?symbol=AAPL&apikey=SUPERSECRETVALUE'"
    )

    cleaned = md.scrub(leaked)

    assert "SUPERSECRETVALUE" not in cleaned
    assert "apikey=***" in cleaned
    assert "403 Forbidden" in cleaned  # the useful part survives
    assert "symbol=AAPL" in cleaned  # and so does the context you need


@pytest.mark.parametrize("param", ["apikey", "api_key", "token", "KEY"])
def test_scrub_covers_common_parameter_names(param):
    assert "hunter2" not in md.scrub(f"https://x.test/a?{param}=hunter2&b=1")


def test_scrub_removes_configured_secret_values_anywhere(monkeypatch):
    """A key can leak in a body or header echo, not just a query string."""
    monkeypatch.setenv("FINNHUB_API_KEY", "abcd1234efgh5678")
    from atlas.config import get_settings

    get_settings.cache_clear()

    cleaned = md.scrub("upstream said: bad token abcd1234efgh5678 in header")

    assert "abcd1234efgh5678" not in cleaned
    assert "***" in cleaned


def test_probe_output_is_scrubbed(monkeypatch):
    def _leaky(symbol):
        raise RuntimeError("failed for url 'https://x.test/q?apikey=LEAKED_KEY_HERE'")

    monkeypatch.setattr(md, "QUOTE_PROVIDERS", (("leaky", _leaky),))
    monkeypatch.setattr(md, "FUNDAMENTAL_PROVIDERS", (("leaky", _leaky),))

    report = md.probe("AAPL")

    assert "LEAKED_KEY_HERE" not in str(report)


# --- provider mappings, checked against live responses on 2026-09-16 -------------

import httpx  # noqa: E402


class _Http:
    """Serves canned JSON by URL substring and records every request."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        request = httpx.Request("GET", url)
        for fragment, (status, body) in self.routes.items():
            if fragment in url:
                return httpx.Response(status, json=body, request=request)
        raise AssertionError(f"unexpected request to {url}")


@pytest.fixture
def keys(monkeypatch):
    for name in ("FINNHUB_API_KEY", "FMP_API_KEY", "ALPHAVANTAGE_API_KEY"):
        monkeypatch.setenv(name, f"{name.lower()}-value")
    from atlas.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _serve(monkeypatch, routes):
    http = _Http(routes)
    monkeypatch.setattr(md, "_http", lambda: http)
    return http


YAHOO_NVDA_1D = {
    "chart": {"result": [{"meta": {
        "regularMarketPrice": 212.17, "chartPreviousClose": 210.96, "previousClose": None,
        "regularMarketTime": 1789502401, "currency": "USD",
        "exchangeTimezoneName": "America/New_York", "shortName": "NVIDIA Corporation",
    }}]}
}


def test_yahoo_daily_change_is_one_session(monkeypatch):
    """Over range=2d, chartPreviousClose was 218.29, two sessions back; the real
    previous close was 210.96."""
    http = _serve(monkeypatch, {"finance.yahoo.com": (200, YAHOO_NVDA_1D)})

    quote = md.yahoo_quote("NVDA")

    assert http.calls[0][1]["range"] == "1d"
    assert quote["previous_close"] == 210.96
    assert round(quote["change_pct"], 2) == 0.57
    assert quote["as_of"] == "2026-09-15T20:00:01Z"
    assert quote["session_date"] == "2026-09-15"


def test_finnhub_quote_carries_its_trade_time(monkeypatch, keys):
    _serve(monkeypatch, {"finnhub.io/api/v1/quote": (200, {"c": 212.17, "pc": 210.96, "t": 1789502400})})

    quote = md.finnhub_quote("NVDA")

    assert quote["session_date"] == "2026-09-15"
    assert quote["currency"] == "USD"


def test_a_symbol_yahoo_does_not_know_stops_the_chain(monkeypatch, keys):
    """A typo used to fall through to Alpha Vantage, spending one of its 25
    requests a day on a listing that does not exist."""
    http = _serve(monkeypatch, {
        "finnhub.io": (200, {"c": 0}),
        "financialmodelingprep.com": (200, []),
        "finance.yahoo.com": (404, {"chart": {"result": None, "error": {"code": "Not Found"}}}),
    })

    assert md.fetch_quote("ZZZZQ") is None
    assert not [url for url, _ in http.calls if "alphavantage" in url]


def test_non_us_listings_skip_the_us_only_providers(monkeypatch, keys):
    """Finnhub's free tier answers 403 for them, and FMP and Alpha Vantage
    hardcoded a USD currency onto rupee prices."""
    rel = {"chart": {"result": [{"meta": {
        "regularMarketPrice": 1235.3, "chartPreviousClose": 1240.0, "currency": "INR",
        "regularMarketTime": 1789502401, "exchangeTimezoneName": "Asia/Kolkata",
    }}]}}
    http = _serve(monkeypatch, {"finance.yahoo.com": (200, rel)})

    quote = md.fetch_quote("reliance.ns")

    assert quote["currency"] == "INR"
    assert [url.split("/")[2] for url, _ in http.calls] == ["query1.finance.yahoo.com"]


def test_indices_go_to_fmp_but_never_to_alpha_vantage(monkeypatch, keys):
    http = _serve(monkeypatch, {
        "financialmodelingprep.com": (200, []),
        "finance.yahoo.com": (200, YAHOO_NVDA_1D),
    })

    quote = md.fetch_quote("^GSPC")

    hosts = [url.split("/")[2] for url, _ in http.calls]
    assert "finnhub.io" not in hosts and "www.alphavantage.co" not in hosts
    assert "financialmodelingprep.com" in hosts
    assert quote["currency"] is None  # index levels are points


FINNHUB_PROFILE = {"currency": "USD", "marketCapitalization": 5084135.84, "name": "NVIDIA Corp",
                   "finnhubIndustry": "Semiconductors"}
FINNHUB_METRIC = {"metric": {"peTTM": 26.2005, "peBasicExclExtraTTM": 26.2005, "forwardPE": 18.10034,
                             "netProfitMarginTTM": 63.66, "revenueGrowthTTMYoy": 83.38,
                             "dividendYieldIndicatedAnnual": 0.03349}}


def test_finnhub_fundamentals_are_labelled_truthfully(monkeypatch, keys):
    _serve(monkeypatch, {
        "stock/profile2": (200, FINNHUB_PROFILE),
        "stock/metric": (200, FINNHUB_METRIC),
    })

    data = md.finnhub_fundamentals("NVDA")

    assert data["trailing_pe"] == 26.2005
    # Was peBasicExclExtraTTM, a trailing multiple, reported as forward.
    assert data["forward_pe"] == 18.10034
    # Percents stay percents, and say so in the key.
    assert data["profit_margin_pct"] == 63.66
    assert data["revenue_growth_pct"] == 83.38
    assert data["dividend_yield_pct"] == 0.03349
    assert data["currency"] == "USD"
    assert "profit_margin" not in data and "dividend_yield" not in data


def test_fmp_dividend_is_turned_into_a_yield(monkeypatch, keys):
    """lastDividend is dollars per share; Coca-Cola's 2.1 read as a 2.1 yield."""
    _serve(monkeypatch, {"stable/profile": (200, [{
        "companyName": "The Coca-Cola Company", "price": 88.71, "lastDividend": 2.1,
        "currency": "USD", "marketCap": 381679115935, "sector": "Consumer Defensive",
    }])})

    data = md.fmp_fundamentals("KO")

    assert data["dividend_per_share"] == 2.1
    assert round(data["dividend_yield_pct"], 2) == 2.37


def test_yahoo_history_keeps_the_close_before_the_window(monkeypatch):
    body = {"chart": {"result": [{
        "meta": {"chartPreviousClose": 225.73, "currency": "USD",
                 "exchangeTimezoneName": "America/New_York"},
        "timestamp": [1789392600, 1789479000, 1789565400],
        "indicators": {"quote": [{"close": [210.96, 212.17, None],
                                  "high": [219.0, 213.94, None], "low": [209.1, 211.16, None]}]},
    }]}}
    _serve(monkeypatch, {"finance.yahoo.com": (200, body)})

    history = md.fetch_history("NVDA", "5d")

    assert history["previous_close"] == 225.73
    # A null close (a holiday, or the session still open) is skipped, not zeroed.
    assert [r["date"] for r in history["rows"]] == ["2026-09-14", "2026-09-15"]


def test_finnhub_earnings_reads_the_free_calendar(monkeypatch, keys):
    _serve(monkeypatch, {"calendar/earnings": (200, {"earningsCalendar": [
        {"symbol": "NVDA", "date": "2026-11-17", "hour": "amc", "epsEstimate": 2.5231,
         "revenueEstimate": 111273704778},
    ]})})

    data = md.finnhub_earnings("NVDA")

    assert data["dates"] == ["2026-11-17"]
    assert data["timing"] == "after the close"
    assert data["eps_estimate"] == 2.5231


def test_provider_failures_are_logged_without_keys(monkeypatch, keys, caplog):
    def _leaky(symbol):
        raise RuntimeError("403 for url 'https://x.test/q?symbol=A&apikey=finnhub_api_key-value'")

    monkeypatch.setattr(md, "QUOTE_PROVIDERS", (("leaky", _leaky),))

    with caplog.at_level("DEBUG", logger="atlas.integrations.marketdata"):
        md.fetch_quote("AAPL")

    assert "finnhub_api_key-value" not in caplog.text


def test_a_routine_probe_spares_the_daily_capped_providers(monkeypatch):
    called = []

    def _spy(name):
        def provider(symbol):
            called.append(name)
            return None

        return provider

    monkeypatch.setattr(md, "QUOTE_PROVIDERS", tuple((n, _spy(n)) for n in ("finnhub", "alphavantage")))
    monkeypatch.setattr(md, "FUNDAMENTAL_PROVIDERS", (("yfinance", _spy("yfinance")),))

    md.probe("AAPL")
    assert called == ["finnhub"]

    md.probe("AAPL", include_limited=True)
    assert called == ["finnhub", "finnhub", "alphavantage", "yfinance"]
