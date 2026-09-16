import pytest

from atlas.memory import store
from atlas.tools.registry import build_tools

pytestmark = pytest.mark.usefixtures("fresh_db")


def test_registry_exposes_every_expected_tool():
    uid = store.get_or_create_user(1, "Shaan")

    names = {t.__name__ for t in build_tools(uid)}

    assert names == {
        "get_quote",
        "get_fundamentals",
        "compare_companies",
        "market_overview",
        "get_price_history",
        "get_earnings_info",
        "get_recent_filings",
        "search_financial_news",
        "analyze_sheet",
        "clarify",
        "remember",
        "recall",
        "forget_about",
        "update_profile",
        "add_to_watchlist",
        "remove_from_watchlist",
        "create_alert",
        "list_alerts",
        "cancel_alert",
        "brief_me_now",
    }


def test_every_tool_has_a_docstring():
    """Gemini derives tool descriptions from docstrings. A missing one is a silent bug."""
    uid = store.get_or_create_user(2, "Shaan")

    for tool in build_tools(uid):
        assert tool.__doc__, f"{tool.__name__} has no docstring"



def test_a_replayed_turn_does_not_fetch_again(monkeypatch):
    """Failover to the next model reran every tool call from the start."""
    import inspect

    import atlas.tools.market as market

    fetched = []

    def _quote(symbol):
        fetched.append(symbol)
        return {"symbol": symbol, "price": 1.0, "source": "T"}

    monkeypatch.setattr(market, "_fetch_quote", _quote)
    uid = store.get_or_create_user(3, "Shaan")
    tool = {t.__name__: t for t in build_tools(uid)}["get_quote"]

    assert tool(symbol="NVDA")["ok"] is True
    assert tool(symbol="NVDA")["ok"] is True
    assert fetched == ["NVDA"]
    # The declaration the models see is unchanged.
    assert list(inspect.signature(tool).parameters) == ["symbol"]
    assert "Ticker symbol" in tool.__doc__

    # A new turn gets a fresh memo.
    {t.__name__: t for t in build_tools(uid)}["get_quote"](symbol="NVDA")
    assert fetched == ["NVDA", "NVDA"]


def test_failures_are_not_memoized(monkeypatch):
    import atlas.tools.market as market

    answers = iter([None, {"symbol": "NVDA", "price": 2.0, "source": "T"}])
    monkeypatch.setattr(market, "_fetch_quote", lambda symbol: next(answers))
    uid = store.get_or_create_user(4, "Shaan")
    tool = {t.__name__: t for t in build_tools(uid)}["get_quote"]

    assert tool(symbol="NVDA")["ok"] is False
    assert tool(symbol="NVDA")["ok"] is True
