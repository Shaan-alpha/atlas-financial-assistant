import pytest

from atlas.memory import store
from atlas.tools.memory_tools import make_memory_tools

pytestmark = pytest.mark.usefixtures("fresh_db")


def tools(user_id: int) -> dict:
    """Look tools up by name; positional unpacking breaks whenever one is added."""
    return {t.__name__: t for t in make_memory_tools(user_id)}


def test_tools_are_bound_to_one_user():
    alice = store.get_or_create_user(1, "Alice")
    bob = store.get_or_create_user(2, "Bob")
    a = tools(alice)
    b = tools(bob)

    a["remember"]("Runs a long/short book", "focus")

    assert len(a["recall"]()["data"]["facts"]) == 1
    assert b["recall"]()["data"]["facts"] == []


def test_recall_includes_profile():
    uid = store.get_or_create_user(3, "Cara")
    store.set_profile(uid, role="PM")
    recall = tools(uid)["recall"]

    result = recall()

    assert result["ok"] is True
    assert result["data"]["profile"]["role"] == "PM"


def test_forget_about_reports_count():
    uid = store.get_or_create_user(4, "Dev")
    t = tools(uid); remember, forget_about = t["remember"], t["forget_about"]
    remember("Bearish on EV demand", "view")

    result = forget_about("EV")

    assert result["data"]["removed"] == 1


def test_watchlist_add_is_idempotent_and_normalizes_case():
    uid = store.get_or_create_user(5, "Eve")
    add_to_watchlist = tools(uid)["add_to_watchlist"]

    first = add_to_watchlist("nvda", "NVIDIA")
    second = add_to_watchlist("NVDA")

    assert first["data"]["added"] is True
    assert first["data"]["symbol"] == "NVDA"
    assert second["data"]["added"] is False
    assert second["data"]["already_present"] is True
    assert len(second["data"]["watchlist"]) == 1


def test_watchlist_remove_reports_when_absent():
    uid = store.get_or_create_user(6, "Fay")
    t = tools(uid); add_to_watchlist, remove_from_watchlist = t["add_to_watchlist"], t["remove_from_watchlist"]
    add_to_watchlist("TSLA")

    hit = remove_from_watchlist("tsla")
    miss = remove_from_watchlist("TSLA")

    assert hit["data"]["removed"] is True
    assert miss["data"]["removed"] is False
    assert hit["data"]["watchlist"] == []


def test_watchlist_is_scoped_to_one_user():
    alice = store.get_or_create_user(7, "Alice")
    bob = store.get_or_create_user(8, "Bob")
    alice_add = tools(alice)["add_to_watchlist"]
    bob_recall = tools(bob)["recall"]

    alice_add("AAPL")

    assert bob_recall()["data"]["watchlist"] == []


def test_recall_surfaces_the_watchlist():
    uid = store.get_or_create_user(9, "Gus")
    t = tools(uid); recall, add_to_watchlist = t["recall"], t["add_to_watchlist"]
    add_to_watchlist("MSFT", "Microsoft")

    assert recall()["data"]["watchlist"] == [
        {"symbol": "MSFT", "company": "Microsoft"}
    ]


# --- audit fixes, 2026-09-16 -----------------------------------------------------


def test_forget_about_needs_a_topic():
    """An empty topic used to be an ILIKE '%%' that deleted every fact."""
    uid = store.get_or_create_user(90, "Dev")
    t = tools(uid)
    t["remember"]("Covers semiconductors", "focus")

    result = t["forget_about"]("  ")

    assert result["ok"] is False
    assert result["error"] == "need_topic"
    assert len(t["recall"]()["data"]["facts"]) == 1


def test_forget_about_says_what_it_removed():
    uid = store.get_or_create_user(91, "Dev")
    t = tools(uid)
    t["remember"]("Bearish on EV demand", "view")

    assert t["forget_about"]("EV")["data"]["removed_facts"] == ["Bearish on EV demand"]


def test_a_long_role_does_not_lose_the_briefing_time():
    """Postgres rejected a role over 80 characters and the rollback took the
    briefing time and timezone saved in the same call down with it."""
    uid = store.get_or_create_user(92, "Dev")

    result = tools(uid)["update_profile"](
        role="portfolio manager " * 10, timezone="Asia/Kolkata", briefing_time="08:30"
    )

    assert result["ok"] is True
    profile = store.profile_snapshot(uid)
    assert profile["briefing_time"] == "08:30"
    assert len(profile["role"]) <= 80


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t["update_profile"](briefing_time="08:30"),
        lambda t: t["add_to_watchlist"]("NVDA"),
    ],
    ids=["briefing-time", "watchlist"],
)
def test_onboarding_ends_once_the_user_gives_anything_durable(call):
    """Only a role used to end onboarding, so someone who skipped it was greeted
    as a stranger on every turn forever."""
    uid = store.get_or_create_user(93, "Dev")

    call(tools(uid))

    assert store.profile_snapshot(uid)["onboarding_state"] == "done"


def test_the_watchlist_is_capped(monkeypatch):
    import atlas.tools.memory_tools as memory_tools

    monkeypatch.setattr(memory_tools, "MAX_WATCHLIST", 2)
    uid = store.get_or_create_user(94, "Dev")
    t = tools(uid)
    t["add_to_watchlist"]("NVDA")
    t["add_to_watchlist"]("AMD")

    result = t["add_to_watchlist"]("TSLA")

    assert result["ok"] is False
    assert result["error"] == "watchlist_full"
    # Re-adding a name already there is not growth and must still succeed.
    assert t["add_to_watchlist"]("NVDA")["ok"] is True
