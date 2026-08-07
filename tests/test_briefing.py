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
