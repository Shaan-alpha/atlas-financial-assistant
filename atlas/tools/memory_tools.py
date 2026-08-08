"""Memory tools, bound to a single user by closure.

The model never supplies a user id, so it cannot reach another user's data.
"""

import re
from collections.abc import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from atlas.memory import store
from atlas.proactive import alerts
from atlas.tools.result import err, ok

SOURCE = "atlas-memory"

# Imported rather than restated so the tool and the evaluator cannot drift apart.
ALERT_KINDS = alerts.KINDS

_VALID_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _is_known_timezone(name: str) -> bool:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


def make_memory_tools(user_id: int) -> list[Callable]:
    def remember(fact: str, category: str = "general") -> dict:
        """Save a durable fact about the user for future conversations.

        Use for stable preferences, focus areas, and views — not for one-off questions.

        Args:
            fact: The fact to remember, written as a short third-person statement.
            category: One of "focus", "view", "preference", "role", "general".
        """
        store.add_fact(user_id, fact, category)
        return ok({"remembered": fact}, source=SOURCE)

    def recall() -> dict:
        """Return everything currently known about the user.

        Use when the user asks what you know or remember about them, and to ground
        personalized answers.
        """
        return ok(
            {
                "profile": store.profile_snapshot(user_id),
                "facts": store.all_facts(user_id),
                "watchlist": store.get_watchlist(user_id),
            },
            source=SOURCE,
        )

    def update_profile(
        role: str = "", timezone: str = "", briefing_time: str = ""
    ) -> dict:
        """Save the user's role, timezone, or preferred daily briefing time.

        Call this during onboarding as soon as they tell you any of these, and
        whenever they change one. Briefings only run for users who have set a time.

        Args:
            role: How they describe their work, e.g. "equity analyst", "founder".
            timezone: IANA name, e.g. "Asia/Kolkata". Ask if you are unsure.
            briefing_time: 24-hour local time as "HH:MM", e.g. "08:30".
        """
        if briefing_time and not _VALID_TIME.match(briefing_time):
            return err("bad_time", "Briefing time must look like '08:30' on a 24h clock.")
        if timezone and not _is_known_timezone(timezone):
            return err("bad_timezone", f"'{timezone}' is not a recognised IANA timezone.")

        fields = {
            k: v
            for k, v in {
                "role": role,
                "timezone": timezone,
                "briefing_time": briefing_time,
            }.items()
            if v
        }
        if not fields:
            return err("nothing_to_update", "Give at least one field to update.")

        store.set_profile(user_id, **fields)
        # Onboarding is done once we know their role.
        if fields.get("role"):
            store.set_profile(user_id, onboarding_state="done")
        return ok({"updated": fields, "profile": store.profile_snapshot(user_id)}, source=SOURCE)

    def add_to_watchlist(symbol: str, company: str = "") -> dict:
        """Start monitoring a security for the user.

        Use whenever the user says they follow, hold, track, or care about a company.
        Watchlist entries drive their daily briefing.

        Args:
            symbol: Ticker symbol, for example "NVDA".
            company: Optional company name for display.
        """
        added = store.add_watchlist(user_id, symbol, company or None)
        return ok(
            {
                "symbol": symbol.strip().upper(),
                "added": added,
                "already_present": not added,
                "watchlist": store.get_watchlist(user_id),
            },
            source=SOURCE,
        )

    def remove_from_watchlist(symbol: str) -> dict:
        """Stop monitoring a security for the user.

        Args:
            symbol: Ticker symbol to drop, for example "TSLA".
        """
        removed = store.remove_watchlist(user_id, symbol)
        return ok(
            {
                "symbol": symbol.strip().upper(),
                "removed": removed,
                "watchlist": store.get_watchlist(user_id),
            },
            source=SOURCE,
        )

    def forget_about(topic: str) -> dict:
        """Delete remembered facts matching a topic.

        Args:
            topic: Substring to match, for example "EV" or "briefing".
        """
        removed = store.forget(user_id, topic)
        return ok({"removed": removed, "topic": topic}, source=SOURCE)

    def brief_me_now() -> dict:
        """Compile the user's market briefing on demand, from their watchlist.

        Use when they ask for their briefing, what they missed, what's happening
        with their names, or to catch them up.

        When has_news is false, tell them plainly that nothing on their watchlist
        is worth their attention right now. Do not manufacture something to say.

        Args:
            None.
        """
        from atlas.proactive import briefing

        profile = store.profile_snapshot(user_id)
        result = briefing.build_now(user_id, profile.get("timezone") or "UTC")
        return ok(result, source=SOURCE)

    def create_alert(
        description: str, symbol: str, kind: str, threshold: float
    ) -> dict:
        """Set up a background watch on a security and notify the user when it hits.

        Use whenever they ask to be told, alerted, pinged, or notified about a price
        or a move. Do not promise to watch something without calling this.

        Args:
            description: What they asked for, in their own words.
            symbol: Ticker symbol, e.g. "TSLA".
            kind: "move_pct" for a daily move of at least threshold percent,
                "price_above" or "price_below" for a price level.
            threshold: Percent for move_pct, otherwise a price.
        """
        if kind not in ALERT_KINDS:
            return err("bad_kind", f"kind must be one of: {', '.join(ALERT_KINDS)}.")
        if threshold <= 0:
            return err("bad_threshold", "Threshold must be greater than zero.")

        alert_id = store.create_alert(user_id, description, symbol, kind, threshold)
        return ok(
            {"alert_id": alert_id, "symbol": symbol.upper(), "kind": kind,
             "threshold": threshold},
            source=SOURCE,
        )

    def list_alerts() -> dict:
        """Return the watches currently armed for this user."""
        return ok({"alerts": store.user_alerts(user_id)}, source=SOURCE)

    def cancel_alert(symbol: str) -> dict:
        """Turn off all watches on a security.

        Args:
            symbol: Ticker symbol to stop watching, e.g. "TSLA".
        """
        cancelled = store.cancel_alerts(user_id, symbol)
        return ok(
            {"symbol": symbol.upper(), "cancelled": cancelled}, source=SOURCE
        )

    return [
        remember,
        recall,
        forget_about,
        update_profile,
        add_to_watchlist,
        remove_from_watchlist,
        brief_me_now,
        create_alert,
        list_alerts,
        cancel_alert,
    ]
