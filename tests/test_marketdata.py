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
    leaked = (
        "HTTPStatusError: Client error '403 Forbidden' for url "
        "'https://financialmodelingprep.com/api/v3/quote/AAPL?apikey=SUPERSECRETVALUE'"
    )

    cleaned = md.scrub(leaked)

    assert "SUPERSECRETVALUE" not in cleaned
    assert "apikey=***" in cleaned
    assert "403 Forbidden" in cleaned  # the useful part survives


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
