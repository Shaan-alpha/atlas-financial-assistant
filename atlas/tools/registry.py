"""Assembles the tool list handed to Gemini for one user's turn."""

import functools
import json
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


# Read-only lookups whose answer cannot change within one turn. Model failover
# replays the whole tool loop on the next model (and on Groq), and each replay
# used to fetch every quote, filing and headline again.
_MEMOIZED = {
    get_quote, get_fundamentals, compare_companies, market_overview,
    get_price_history, get_earnings_info, get_recent_filings,
    search_financial_news, analyze_sheet,
}


def _memoize_for_turn(tool: Callable) -> Callable:
    """Reuse a successful result for identical arguments, for this turn only.

    functools.wraps keeps the name, docstring and signature, which is what both
    Gemini and the Groq fallback build the tool declaration from.
    """
    results: dict[str, dict] = {}

    @functools.wraps(tool)
    def wrapper(*args, **kwargs):
        key = json.dumps([args, kwargs], sort_keys=True, default=str)
        if key in results:
            return results[key]
        result = tool(*args, **kwargs)
        # Failures are not kept: the next attempt deserves a fresh try.
        if isinstance(result, dict) and result.get("ok"):
            results[key] = result
        return result

    return wrapper


def build_tools(user_id: int) -> list[Callable]:
    """Return every tool for one turn, with user-scoped ones bound to this user.

    Called once per turn, so the lookup memo lives exactly as long as the turn.
    """
    stateless = [_memoize_for_turn(t) if t in _MEMOIZED else t for t in STATELESS_TOOLS]
    return [*stateless, *make_memory_tools(user_id)]
