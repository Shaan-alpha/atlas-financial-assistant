"""Collect raw signals for one user's briefing.

Gathering is deliberately dumb: it casts a wide net and makes no judgement about
what deserves the user's attention. Deciding is the salience gate's job, and
keeping the two apart is what lets the gate stay silent honestly.
"""

import logging

from atlas.memory import store
from atlas.tools import filings, market

log = logging.getLogger(__name__)

# A watchlist name moving less than this is noise, not news.
NOTABLE_MOVE_PCT = 2.0

# Filing types worth waking someone for.
MATERIAL_FORMS = {"8-K", "10-K", "10-Q", "S-1", "SC 13D", "SC 13G", "DEF 14A"}


def _signal(key: str, kind: str, summary: str, detail: dict) -> dict:
    return {"key": key, "kind": kind, "summary": summary, "detail": detail}


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
                signals.append(
                    _signal(
                        f"move:{symbol}:{today}",
                        "move",
                        f"{symbol} {direction} {abs(change):.1f}% to "
                        f"{quote['data']['price']}",
                        quote["data"],
                    )
                )

        recent = filings.get_recent_filings(symbol, limit=3)
        if recent["ok"]:
            for filing in recent["data"]["filings"]:
                if filing["form"] not in MATERIAL_FORMS:
                    continue
                if filing["filed_on"] != today:
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
