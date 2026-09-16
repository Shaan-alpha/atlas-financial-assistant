"""Proactive briefings, with emphasis on the silence path.

The brief requires the assistant to stay quiet when nothing matters. That is the
behaviour most likely to regress silently, so it is tested from every angle.
"""

import pytest

import atlas.proactive.briefing as briefing
import atlas.proactive.gather as gather
import atlas.proactive.salience as salience
from atlas.memory import store

pytestmark = pytest.mark.usefixtures("fresh_db")

TODAY = "2026-08-08"


def _quote(symbol, change):
    return {
        "ok": True,
        "data": {"symbol": symbol, "price": 100.0, "change_pct": change},
        "source": "yfinance",
    }


def _no_filings(symbol, limit=3):
    return {"ok": True, "data": {"filings": []}}


def _no_earnings(symbol):
    return {"ok": False, "error": "no_earnings_date"}


# ----------------------------------------------------------------- gather


def test_small_moves_are_not_signals(monkeypatch):
    uid = store.get_or_create_user(1, "Shaan")
    store.add_watchlist(uid, "NVDA")
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, 0.4))
    monkeypatch.setattr(gather.filings, "get_recent_filings", _no_filings)
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)

    assert gather.gather(uid, TODAY) == []


def test_notable_moves_become_signals(monkeypatch):
    uid = store.get_or_create_user(2, "Shaan")
    store.add_watchlist(uid, "NVDA")
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, -6.1))
    monkeypatch.setattr(gather.filings, "get_recent_filings", _no_filings)
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)

    signals = gather.gather(uid, TODAY)

    assert len(signals) == 1
    assert signals[0]["kind"] == "move"
    assert "down 6.1%" in signals[0]["summary"]
    assert signals[0]["key"] == f"move:NVDA:{TODAY}"


def test_a_notable_move_carries_the_headlines_behind_it(monkeypatch):
    """"NVDA down 6%" tells a user nothing they could not see on a ticker. The
    gate can only say why it matters if it is handed what was reported."""
    uid = store.get_or_create_user(20, "Shaan")
    store.add_watchlist(uid, "NVDA")
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, -6.1))
    monkeypatch.setattr(gather.filings, "get_recent_filings", _no_filings)
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)
    asked = {}

    def _news(query, symbol="", days=3):
        asked.update(symbol=symbol, days=days)
        return {
            "ok": True,
            "data": {
                "articles": [
                    {"title": f"Headline {i}", "source": "Reuters", "published": "t",
                     "url": "u", "summary": "s"}
                    for i in range(5)
                ]
            },
        }

    monkeypatch.setattr(gather.news, "search_financial_news", _news)

    detail = gather.gather(uid, TODAY)[0]["detail"]

    assert asked == {"symbol": "NVDA", "days": 1}
    assert [h["title"] for h in detail["headlines"]] == ["Headline 0", "Headline 1", "Headline 2"]


def test_quiet_names_never_fetch_news(monkeypatch):
    """An empty morning must still cost nothing, not a news search per name."""
    uid = store.get_or_create_user(21, "Shaan")
    store.add_watchlist(uid, "NVDA")
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, 0.3))
    monkeypatch.setattr(gather.filings, "get_recent_filings", _no_filings)
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)

    def _news(*args, **kwargs):
        raise AssertionError("no move, no news search")

    monkeypatch.setattr(gather.news, "search_financial_news", _news)

    assert gather.gather(uid, TODAY) == []


def test_a_news_outage_does_not_cost_the_move_signal(monkeypatch):
    uid = store.get_or_create_user(22, "Shaan")
    store.add_watchlist(uid, "NVDA")
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, 5.0))
    monkeypatch.setattr(gather.filings, "get_recent_filings", _no_filings)
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)

    signals = gather.gather(uid, TODAY)

    assert signals[0]["kind"] == "move"
    assert signals[0]["detail"]["headlines"] == []


def test_only_todays_material_filings_count(monkeypatch):
    uid = store.get_or_create_user(3, "Shaan")
    store.add_watchlist(uid, "TSLA")
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, 0.1))
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)
    monkeypatch.setattr(
        gather.filings,
        "get_recent_filings",
        lambda s, limit=3: {
            "ok": True,
            "data": {
                "filings": [
                    {"form": "8-K", "filed_on": TODAY, "url": "u1"},
                    {"form": "8-K", "filed_on": "2026-08-01", "url": "u2"},  # stale
                    {"form": "4", "filed_on": TODAY, "url": "u3"},  # immaterial
                ]
            },
        },
    )

    signals = gather.gather(uid, TODAY)

    assert [s["detail"]["url"] for s in signals] == ["u1"]


# ---------------------------------------------------------------- salience


async def test_no_signals_never_calls_the_model(monkeypatch):
    called = False

    async def _spy(prompt):
        nonlocal called
        called = True
        return {"send": True, "brief": "x"}

    monkeypatch.setattr(salience, "_decide", _spy)

    verdict = await salience.decide({}, [], [], None)

    assert verdict["send"] is False
    assert called is False, "an empty morning must not spend a request"


async def test_gate_can_refuse_to_send(monkeypatch):
    async def _refuse(prompt):
        return {"send": False, "brief": "", "used_keys": []}

    monkeypatch.setattr(salience, "_decide", _refuse)

    verdict = await salience.decide({}, [], [{"key": "k", "summary": "s"}], None)

    assert verdict["send"] is False


async def test_send_true_with_empty_brief_is_still_silence(monkeypatch):
    """A malformed yes must not produce an empty message."""

    async def _empty(prompt):
        return {"send": True, "brief": "   ", "used_keys": []}

    monkeypatch.setattr(salience, "_decide", _empty)

    verdict = await salience.decide({}, [], [{"key": "k", "summary": "s"}], None)

    assert verdict["send"] is False


async def test_model_failure_stays_silent(monkeypatch):
    async def _boom(prompt):
        raise RuntimeError("quota")

    monkeypatch.setattr(salience, "_decide", _boom)

    verdict = await salience.decide({}, [], [{"key": "k", "summary": "s"}], None)

    assert verdict["send"] is False, "a broken gate must not spam the user"


# ---------------------------------------------------------------- briefing


async def test_signals_are_never_offered_twice(monkeypatch):
    uid = store.get_or_create_user(4, "Shaan")
    store.add_watchlist(uid, "NVDA")
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, 7.0))
    monkeypatch.setattr(gather.filings, "get_recent_filings", _no_filings)
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)
    monkeypatch.setattr(gather, "market_context", lambda: None)

    async def _approve(prompt):
        return {"send": True, "brief": "*NVDA* up 7%", "used_keys": []}

    monkeypatch.setattr(salience, "_decide", _approve)
    monkeypatch.setattr(briefing, "local_today", lambda tz: TODAY)

    first = await briefing.build(uid, "UTC")
    second = await briefing.build(uid, "UTC")

    assert first == "*NVDA* up 7%"
    assert second is None, "the same move must not be re-sent tomorrow"


async def test_refused_signals_are_also_marked_seen(monkeypatch):
    """Otherwise the gate re-judges the same non-event every single morning."""
    uid = store.get_or_create_user(5, "Shaan")
    store.add_watchlist(uid, "NVDA")
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, 5.0))
    monkeypatch.setattr(gather.filings, "get_recent_filings", _no_filings)
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)
    monkeypatch.setattr(gather, "market_context", lambda: None)
    monkeypatch.setattr(briefing, "local_today", lambda tz: TODAY)

    calls = []

    async def _refuse(prompt):
        calls.append(1)
        return {"send": False, "brief": "", "used_keys": []}

    monkeypatch.setattr(salience, "_decide", _refuse)

    assert await briefing.build(uid, "UTC") is None
    assert await briefing.build(uid, "UTC") is None
    assert len(calls) == 1, "a refused signal must not be re-judged"


async def test_send_to_reports_false_when_silent(monkeypatch):
    uid = store.get_or_create_user(6, "Shaan")
    monkeypatch.setattr(briefing, "local_today", lambda tz: TODAY)

    class _Bot:
        sent = []

        async def send_message(self, **kw):
            self.sent.append(kw)

    bot = _Bot()
    assert await briefing.send_to(bot, uid, 999, "UTC") is False
    assert bot.sent == []


def test_local_today_respects_the_users_timezone():
    assert len(briefing.local_today("Asia/Kolkata")) == 10
    assert briefing.local_today("Not/AZone") == briefing.local_today("UTC")


def test_utc_conversion_shifts_by_the_offset():
    """08:30 in Kolkata is 03:00 UTC."""
    converted = briefing.utc_time_for("08:30", "Asia/Kolkata")
    assert (converted.hour, converted.minute) == (3, 0)


# ------------------------------------------------------- on-demand briefing
#
# Silence is the right answer to "should I interrupt them?" but the wrong answer
# to "tell me what's going on". The pull path therefore always says something.


def test_on_demand_reports_when_there_is_nothing(monkeypatch):
    uid = store.get_or_create_user(20, "Shaan")
    monkeypatch.setattr(briefing, "local_today", lambda tz: TODAY)
    monkeypatch.setattr(briefing.gather, "gather", lambda u, d: [])

    result = briefing.build_now(uid, "UTC")

    assert result == {"has_news": False, "signals": []}


def test_on_demand_hands_the_signals_to_the_chat_model(monkeypatch):
    """No gate and no model call of its own: it used to spend a request writing
    prose that the chat model then rewrote."""
    uid = store.get_or_create_user(21, "Shaan")
    monkeypatch.setattr(briefing, "local_today", lambda tz: TODAY)
    monkeypatch.setattr(
        briefing.gather,
        "gather",
        lambda u, d: [{"key": "k", "kind": "move", "summary": "NVDA up 7%", "detail": {"price": 1}}],
    )

    def _no_model(*args, **kwargs):
        raise AssertionError("the on-demand path must not call a model")

    monkeypatch.setattr(salience, "_decide_sync", _no_model)

    result = briefing.build_now(uid, "UTC")

    assert result["has_news"] is True
    assert result["signals"] == [{"kind": "move", "summary": "NVDA up 7%", "detail": {"price": 1}}]


def test_on_demand_does_not_consume_the_dedupe_ledger(monkeypatch):
    """A pull is not an interruption; consuming the ledger here would silence
    the scheduled briefing that follows."""
    uid = store.get_or_create_user(22, "Shaan")
    store.add_watchlist(uid, "NVDA")
    monkeypatch.setattr(briefing, "local_today", lambda tz: TODAY)
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, 7.0))
    monkeypatch.setattr(gather.filings, "get_recent_filings", _no_filings)
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)
    monkeypatch.setattr(gather, "market_context", lambda: None)
    monkeypatch.setattr(
        briefing.salience, "_decide_sync",
        lambda p, i=None: {"send": True, "brief": "b", "used_keys": []},
    )

    briefing.build_now(uid, "UTC")

    # The scheduled path must still see the signal as unsent.
    assert store.filter_unsent(uid, [f"move:NVDA:{TODAY}"]) == [f"move:NVDA:{TODAY}"]


def test_the_scheduled_gate_keeps_its_strict_bar():
    assert "did not ask for this" in salience.PUSH_INSTRUCTION
    assert "why it matters" in salience.PUSH_INSTRUCTION
    assert "Never invent or estimate a figure" in salience.PUSH_INSTRUCTION


# --- audit fixes, 2026-09-16 -----------------------------------------------------


def _filings(*rows):
    return lambda symbol, limit=5: {
        "ok": True,
        "data": {"filings": [{"form": f, "filed_on": d, "url": "u"} for f, d in rows]},
    }


def test_filings_are_read_over_a_window_and_past_routine_forms(monkeypatch):
    """Only the newest three filings were read, before the form filter, and only
    ones dated exactly the user's local today: an 8-K behind two Form 4s, or filed
    Friday evening, never surfaced."""
    uid = store.get_or_create_user(30, "Shaan")
    store.add_watchlist(uid, "NVDA")
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, 0.1))
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)
    asked = {}

    def _recent(symbol, limit=5):
        asked["limit"] = limit
        return _filings(
            ("4", TODAY), ("4", TODAY), ("4", TODAY),
            ("8-K", "2026-08-06"),
            ("SCHEDULE 13D", "2026-08-05"),
            ("10-Q", "2026-07-01"),
        )(symbol, limit)

    monkeypatch.setattr(gather.filings, "get_recent_filings", _recent)

    kinds = sorted(s["summary"] for s in gather.gather(uid, TODAY))

    assert asked["limit"] >= 20
    assert kinds == ["NVDA filed a 8-K", "NVDA filed a SCHEDULE 13D"]


def test_a_move_is_keyed_by_its_trading_session(monkeypatch):
    """Keyed by calendar day, Friday's move was offered again on Saturday,
    Sunday and Monday."""
    uid = store.get_or_create_user(31, "Shaan")
    store.add_watchlist(uid, "NVDA")

    def _friday(symbol):
        quote = _quote(symbol, -6.1)
        quote["data"]["session_date"] = "2026-08-07"
        return quote

    monkeypatch.setattr(gather.market, "get_quote", _friday)
    monkeypatch.setattr(gather.filings, "get_recent_filings", _no_filings)
    monkeypatch.setattr(gather.market, "get_earnings_info", _no_earnings)

    saturday = gather.gather(uid, "2026-08-08")
    sunday = gather.gather(uid, "2026-08-09")

    assert saturday[0]["key"] == sunday[0]["key"] == "move:NVDA:2026-08-07"


def test_an_unconfirmed_earnings_window_is_not_a_signal(monkeypatch):
    uid = store.get_or_create_user(32, "Shaan")
    store.add_watchlist(uid, "NVDA")
    monkeypatch.setattr(gather.market, "get_quote", lambda s: _quote(s, 0.1))
    monkeypatch.setattr(gather.filings, "get_recent_filings", _no_filings)
    monkeypatch.setattr(
        gather.market,
        "get_earnings_info",
        lambda s: {"ok": True, "data": {"next_earnings_date": None, "estimated_window": [TODAY, "2026-08-12"]}},
    )

    assert gather.gather(uid, TODAY) == []


class _JobQueue:
    def __init__(self, jobs=()):
        self._jobs = list(jobs)
        self.daily = []

    def jobs(self):
        return self._jobs

    def run_daily(self, callback, time, name, data, job_kwargs=None):
        self.daily.append({"name": name, "job_kwargs": job_kwargs})


class _Job:
    def __init__(self, name):
        self.name, self.removed = name, False

    def schedule_removal(self):
        self.removed = True


def test_a_failed_roster_read_keeps_the_briefings_scheduled(monkeypatch):
    from atlas.proactive import scheduler

    existing = _Job("briefing:1")
    queue = _JobQueue([existing])

    def _down():
        raise RuntimeError("the database is restarting")

    monkeypatch.setattr(scheduler.store, "users_with_briefings", _down)

    assert scheduler.sync_jobs(queue) == 0
    assert existing.removed is False


def test_a_briefing_survives_a_short_stall(monkeypatch):
    """APScheduler's default misfire grace is one second."""
    from atlas.proactive import scheduler

    uid = store.get_or_create_user(33, "Shaan")
    store.set_profile(uid, briefing_time="08:30", timezone="Asia/Kolkata")
    queue = _JobQueue()

    scheduler.sync_jobs(queue)

    assert queue.daily[0]["job_kwargs"]["misfire_grace_time"] >= 600
