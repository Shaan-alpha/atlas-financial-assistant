import pytest

from atlas.memory import store
from atlas.tools.memory_tools import make_memory_tools

pytestmark = pytest.mark.usefixtures("fresh_db")


def test_tools_are_bound_to_one_user():
    alice = store.get_or_create_user(1, "Alice")
    bob = store.get_or_create_user(2, "Bob")
    a_remember, a_recall, *_ = make_memory_tools(alice)
    _, b_recall, *_ = make_memory_tools(bob)

    a_remember("Runs a long/short book", "focus")

    assert len(a_recall()["data"]["facts"]) == 1
    assert b_recall()["data"]["facts"] == []


def test_recall_includes_profile():
    uid = store.get_or_create_user(3, "Cara")
    store.set_profile(uid, role="PM")
    _, recall, *_ = make_memory_tools(uid)

    result = recall()

    assert result["ok"] is True
    assert result["data"]["profile"]["role"] == "PM"


def test_forget_about_reports_count():
    uid = store.get_or_create_user(4, "Dev")
    remember, _, forget_about, *_ = make_memory_tools(uid)
    remember("Bearish on EV demand", "view")

    result = forget_about("EV")

    assert result["data"]["removed"] == 1


def test_watchlist_add_is_idempotent_and_normalizes_case():
    uid = store.get_or_create_user(5, "Eve")
    *_, add_to_watchlist, _ = make_memory_tools(uid)

    first = add_to_watchlist("nvda", "NVIDIA")
    second = add_to_watchlist("NVDA")

    assert first["data"]["added"] is True
    assert first["data"]["symbol"] == "NVDA"
    assert second["data"]["added"] is False
    assert second["data"]["already_present"] is True
    assert len(second["data"]["watchlist"]) == 1


def test_watchlist_remove_reports_when_absent():
    uid = store.get_or_create_user(6, "Fay")
    *_, add_to_watchlist, remove_from_watchlist = make_memory_tools(uid)
    add_to_watchlist("TSLA")

    hit = remove_from_watchlist("tsla")
    miss = remove_from_watchlist("TSLA")

    assert hit["data"]["removed"] is True
    assert miss["data"]["removed"] is False
    assert hit["data"]["watchlist"] == []


def test_watchlist_is_scoped_to_one_user():
    alice = store.get_or_create_user(7, "Alice")
    bob = store.get_or_create_user(8, "Bob")
    *_, alice_add, _ = make_memory_tools(alice)
    _, bob_recall, *_ = make_memory_tools(bob)

    alice_add("AAPL")

    assert bob_recall()["data"]["watchlist"] == []


def test_recall_surfaces_the_watchlist():
    uid = store.get_or_create_user(9, "Gus")
    _, recall, _, add_to_watchlist, _ = make_memory_tools(uid)
    add_to_watchlist("MSFT", "Microsoft")

    assert recall()["data"]["watchlist"] == [
        {"symbol": "MSFT", "company": "Microsoft"}
    ]
