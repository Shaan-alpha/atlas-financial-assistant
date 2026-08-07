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
