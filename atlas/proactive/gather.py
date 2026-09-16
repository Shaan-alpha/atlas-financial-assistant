"""Collect raw signals for one user's briefing.

Gathering is deliberately dumb: it casts a wide net and makes no judgement about
what deserves the user's attention. Deciding is the salience gate's job, and
keeping the two apart is what lets the gate stay silent honestly.
"""

import datetime as dt
import logging

from atlas.memory import store
from atlas.tools import filings, market, news

log = logging.getLogger(__name__)

# A watchlist name moving less than this is noise, not news.
NOTABLE_MOVE_PCT = 2.0

# Headlines handed to the gate with a move. Enough to explain it, few enough that
# one user's briefing prompt stays small on a free-tier token budget.
MOVE_HEADLINES = 3

# Filing types worth waking someone for. EDGAR renamed the beneficial-ownership
# forms from "SC 13D" to "SCHEDULE 13D" in December 2024; both spellings stay so
# older filings still match.
MATERIAL_FORMS = {
    "8-K", "8-K/A", "10-K", "10-Q", "S-1", "DEF 14A",
    "SC 13D", "SC 13G", "SCHEDULE 13D", "SCHEDULE 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A",
}
# Filings are checked over a window, not just "today": EDGAR dates are Eastern,
# a user ahead of Eastern is already on tomorrow, and a Friday-evening 8-K is
# first seen on Monday. The sent-signal ledger stops any of them repeating.
FILING_WINDOW_DAYS = 4
# One submissions request returns every recent filing, so reading more costs
# nothing. Three was the old limit, applied before the form filter, so routine
# Form 4s crowded out the 8-K that mattered.
FILINGS_SCANNED = 40


def _signal(key: str, kind: str, summary: str, detail: dict) -> dict:
    return {"key": key, "kind": kind, "summary": summary, "detail": detail}


def _headlines(symbol: str) -> list[dict]:
    """The last day's reporting on a name that moved, so the gate can say why.

    Fetched only for a move that already cleared the bar: a quiet name costs no
    search, and a quiet morning still short-circuits before any model call.
    """
    result = news.search_financial_news(f"{symbol} stock", symbol=symbol, days=1)
    if not result["ok"]:
        return []
    return [
        {key: article.get(key) for key in ("title", "source", "published", "summary")}
        for article in result["data"]["articles"][:MOVE_HEADLINES]
    ]


def gather(user_id: int, today: str) -> list[dict]:
    """Return every candidate signal for this user, unfiltered and unranked.

    Args:
        user_id: Internal user id.
        today: ISO date, used to key signals so each is offered only once.
    """
    signals: list[dict] = []
    watchlist = store.get_watchlist(user_id)

    for item in watchlist:
        symbol = item["symbol"]

        quote = market.get_quote(symbol)
        if quote["ok"]:
            change = quote["data"].get("change_pct")
            if change is not None and abs(change) >= NOTABLE_MOVE_PCT:
                direction = "up" if change > 0 else "down"
                # Keyed by trading session, not calendar day: a Friday close
                # read on Saturday is the same move, already offered once.
                session = quote["data"].get("session_date") or today
                signals.append(
                    _signal(
                        f"move:{symbol}:{session}",
                        "move",
                        f"{symbol} {direction} {abs(change):.1f}% to "
                        f"{quote['data']['price']}",
                        {**quote["data"], "headlines": _headlines(symbol)},
                    )
                )

        cutoff = (dt.date.fromisoformat(today) - dt.timedelta(days=FILING_WINDOW_DAYS)).isoformat()
        recent = filings.get_recent_filings(symbol, limit=FILINGS_SCANNED)
        if recent["ok"]:
            for filing in recent["data"]["filings"]:
                if filing["form"] not in MATERIAL_FORMS:
                    continue
                if not cutoff <= filing["filed_on"] <= today:
                    continue
                signals.append(
                    _signal(
                        f"filing:{symbol}:{filing['form']}:{filing['filed_on']}",
                        "filing",
                        f"{symbol} filed a {filing['form']}",
                        filing,
                    )
                )

        earnings = market.get_earnings_info(symbol)
        if earnings["ok"] and earnings["data"]["next_earnings_date"] == today:
            signals.append(
                _signal(
                    f"earnings:{symbol}:{today}",
                    "earnings",
                    f"{symbol} reports earnings today",
                    earnings["data"],
                )
            )

    return signals


def market_context() -> dict | None:
    """Index levels for colour. Never a signal on its own — it is always available,
    so treating it as news would mean never staying silent."""
    overview = market.market_overview()
    return overview["data"] if overview["ok"] else None
