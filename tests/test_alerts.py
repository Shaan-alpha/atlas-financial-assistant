import datetime as dt

import pytest

import atlas.proactive.alerts as alerts
from atlas.memory import store
from atlas.tools.memory_tools import make_memory_tools

pytestmark = pytest.mark.usefixtures("fresh_db")


def tools(user_id: int) -> dict:
    return {t.__name__: t for t in make_memory_tools(user_id)}


NOW = dt.datetime(2026, 8, 8, 12, 0, 0)


# ------------------------------------------------------------- evaluation


@pytest.mark.parametrize(
    "kind,threshold,quote,expected",
    [
        ("move_pct", 5.0, {"price": 100, "change_pct": 6.2}, True),
        ("move_pct", 5.0, {"price": 100, "change_pct": -6.2}, True),  # direction-agnostic
        ("move_pct", 5.0, {"price": 100, "change_pct": 4.9}, False),
        ("move_pct", 5.0, {"price": 100, "change_pct": None}, False),
        ("price_above", 250.0, {"price": 251.0, "change_pct": 1}, True),
        ("price_above", 250.0, {"price": 249.0, "change_pct": 1}, False),
        ("price_below", 200.0, {"price": 199.0, "change_pct": -1}, True),
        ("price_below", 200.0, {"price": 201.0, "change_pct": -1}, False),
        ("nonsense", 1.0, {"price": 1, "change_pct": 99}, False),
    ],
)
def test_trigger_evaluation(kind, threshold, quote, expected):
    assert alerts.is_triggered(kind, threshold, quote) is expected


def test_cooldown_suppresses_a_recent_fire():
    recent = NOW - dt.timedelta(hours=1)
    assert alerts.in_cooldown(recent, NOW) is True


def test_cooldown_expires():
    old = NOW - dt.timedelta(hours=7)
    assert alerts.in_cooldown(old, NOW) is False


def test_never_fired_is_not_in_cooldown():
    assert alerts.in_cooldown(None, NOW) is False


def test_message_echoes_the_users_own_words():
    text = alerts.describe(
        {
            "symbol": "TSLA",
            "kind": "move_pct",
            "threshold": 5.0,
            "description": "tell me if Tesla swings hard",
        },
        {"price": 326.9, "change_pct": -6.4},
    )
    assert "*TSLA*" in text
    assert "down 6.4%" in text
    assert "tell me if Tesla swings hard" in text


# ------------------------------------------------------------------ tools


def test_create_alert_rejects_unknown_kind():
    uid = store.get_or_create_user(1, "Shaan")
    r = tools(uid)["create_alert"]("watch it", "TSLA", "vibes", 5.0)
    assert r["ok"] is False
    assert r["error"] == "bad_kind"


def test_create_alert_rejects_nonpositive_threshold():
    uid = store.get_or_create_user(2, "Shaan")
    r = tools(uid)["create_alert"]("watch it", "TSLA", "move_pct", 0)
    assert r["ok"] is False
    assert r["error"] == "bad_threshold"


def test_create_then_list_then_cancel():
    uid = store.get_or_create_user(3, "Shaan")
    t = tools(uid)

    created = t["create_alert"]("alert me if TSLA moves 5%", "tsla", "move_pct", 5.0)
    assert created["ok"] is True
    assert created["data"]["symbol"] == "TSLA"

    listed = t["list_alerts"]()["data"]["alerts"]
    assert len(listed) == 1
    assert listed[0]["description"] == "alert me if TSLA moves 5%"

    cancelled = t["cancel_alert"]("TSLA")
    assert cancelled["data"]["cancelled"] == 1
    assert t["list_alerts"]()["data"]["alerts"] == []


def test_alerts_are_scoped_to_one_user():
    alice = store.get_or_create_user(4, "Alice")
    bob = store.get_or_create_user(5, "Bob")
    tools(alice)["create_alert"]("watch NVDA", "NVDA", "move_pct", 3.0)

    assert tools(bob)["list_alerts"]()["data"]["alerts"] == []


# ---------------------------------------------------------------- watcher


class _Bot:
    def __init__(self):
        self.sent = []

    async def send_message(self, **kw):
        self.sent.append(kw)


async def test_watcher_fires_and_then_respects_cooldown(monkeypatch):
    uid = store.get_or_create_user(6, "Shaan")
    tools(uid)["create_alert"]("ping me if TSLA moves 5%", "TSLA", "move_pct", 5.0)

    monkeypatch.setattr(
        alerts.market,
        "get_quote",
        lambda s: {"ok": True, "data": {"price": 300.0, "change_pct": -7.1},
                   "source": "yfinance"},
    )

    bot = _Bot()
    assert await alerts.check_all(bot, now=NOW) == 1
    assert "TSLA" in bot.sent[0]["text"]

    # Still past the threshold minutes later, but must not re-notify.
    assert await alerts.check_all(bot, now=NOW + dt.timedelta(minutes=15)) == 0
    assert len(bot.sent) == 1


async def test_watcher_stays_quiet_below_threshold(monkeypatch):
    uid = store.get_or_create_user(7, "Shaan")
    tools(uid)["create_alert"]("big moves only", "NVDA", "move_pct", 10.0)

    monkeypatch.setattr(
        alerts.market,
        "get_quote",
        lambda s: {"ok": True, "data": {"price": 100.0, "change_pct": 2.0},
                   "source": "yfinance"},
    )

    bot = _Bot()
    assert await alerts.check_all(bot, now=NOW) == 0
    assert bot.sent == []


async def test_watcher_skips_symbols_with_no_data(monkeypatch):
    uid = store.get_or_create_user(8, "Shaan")
    tools(uid)["create_alert"]("watch it", "XYZQ", "move_pct", 1.0)

    monkeypatch.setattr(
        alerts.market,
        "get_quote",
        lambda s: {"ok": False, "error": "no_such_symbol", "message": "nope"},
    )

    bot = _Bot()
    assert await alerts.check_all(bot, now=NOW) == 0


async def test_cancelled_alerts_never_fire(monkeypatch):
    uid = store.get_or_create_user(9, "Shaan")
    t = tools(uid)
    t["create_alert"]("watch TSLA", "TSLA", "move_pct", 1.0)
    t["cancel_alert"]("TSLA")

    monkeypatch.setattr(
        alerts.market,
        "get_quote",
        lambda s: {"ok": True, "data": {"price": 300.0, "change_pct": 50.0},
                   "source": "yfinance"},
    )

    bot = _Bot()
    assert await alerts.check_all(bot, now=NOW) == 0
