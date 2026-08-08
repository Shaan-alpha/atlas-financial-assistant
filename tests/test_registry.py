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
