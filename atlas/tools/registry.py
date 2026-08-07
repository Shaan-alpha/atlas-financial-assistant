"""Assembles the tool list handed to Gemini for one user's turn."""

from collections.abc import Callable

from atlas.tools.clarify import clarify
from atlas.tools.filings import get_recent_filings
from atlas.tools.market import (
    compare_companies,
    get_earnings_info,
    get_fundamentals,
    get_price_history,
    get_quote,
    market_overview,
)
from atlas.tools.memory_tools import make_memory_tools
from atlas.tools.news import search_financial_news
from atlas.tools.sheets import analyze_sheet

STATELESS_TOOLS: list[Callable] = [
    get_quote,
    get_fundamentals,
    compare_companies,
    market_overview,
    get_price_history,
    get_earnings_info,
    get_recent_filings,
    search_financial_news,
    analyze_sheet,
    clarify,
]


def build_tools(user_id: int) -> list[Callable]:
    """Return every tool, with user-scoped ones bound to this user."""
    return [*STATELESS_TOOLS, *make_memory_tools(user_id)]
