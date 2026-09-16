"""Per-user admission for turns.

The bot is public: anyone who finds it can talk to it, and every turn spends
requests from a Gemini key shared with another project, plus free-tier
market-data quotas every user depends on. Nothing stopped one person, or one
script, from spending all of it. This bounds each Telegram user without an
allowlist, so the demo stays open to anyone who says hello.

In memory on purpose: a restart forgiving everyone is harmless, and a database
round trip in front of every message is not.
"""

import time
from collections import Counter, defaultdict, deque

# A person typing fast sends a few messages a minute; a script sends dozens.
BURST_TURNS = 6
BURST_WINDOW = 120.0
DAILY_TURNS = 150
DAY = 86_400.0
# One turn running and one waiting behind the per-user turn lock. More than that
# is someone firing messages faster than any of them can be answered.
MAX_IN_FLIGHT = 2

BUSY = "One at a time: I'm still working on your last message."
SLOW_DOWN = "You're sending these faster than I can research them. Give me a minute."
DAILY_LIMIT = "That's all I can take on for you today. Pick it up with me tomorrow."


class TurnGuard:
    """Admit or refuse a turn. Event-loop only, so no locking is needed."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._in_flight: Counter[int] = Counter()
        self._recent: defaultdict[int, deque] = defaultdict(deque)

    def admit(self, telegram_id: int) -> str | None:
        """Return None and count the turn, or return the reply that refuses it."""
        now = self._clock()
        if self._in_flight[telegram_id] >= MAX_IN_FLIGHT:
            return BUSY

        recent = self._recent[telegram_id]
        while recent and now - recent[0] >= DAY:
            recent.popleft()
        if len(recent) >= DAILY_TURNS:
            return DAILY_LIMIT
        burst = sum(1 for stamp in reversed(recent) if now - stamp < BURST_WINDOW)
        if burst >= BURST_TURNS:
            return SLOW_DOWN

        recent.append(now)
        self._in_flight[telegram_id] += 1
        return None

    def release(self, telegram_id: int) -> None:
        self._in_flight[telegram_id] -= 1
        if self._in_flight[telegram_id] <= 0:
            del self._in_flight[telegram_id]
