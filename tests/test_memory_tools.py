import pytest

from atlas.memory import store
from atlas.tools.memory_tools import make_memory_tools

pytestmark = pytest.mark.usefixtures("fresh_db")


def test_tools_are_bound_to_one_user():
    alice = store.get_or_create_user(1, "Alice")
    bob = store.get_or_create_user(2, "Bob")
    a_remember, a_recall, _ = make_memory_tools(alice)
    _, b_recall, _ = make_memory_tools(bob)

    a_remember("Runs a long/short book", "focus")

    assert len(a_recall()["data"]["facts"]) == 1
    assert b_recall()["data"]["facts"] == []


def test_recall_includes_profile():
    uid = store.get_or_create_user(3, "Cara")
    store.set_profile(uid, role="PM")
    _, recall, _ = make_memory_tools(uid)

    result = recall()

    assert result["ok"] is True
    assert result["data"]["profile"]["role"] == "PM"


def test_forget_about_reports_count():
    uid = store.get_or_create_user(4, "Dev")
    remember, _, forget_about = make_memory_tools(uid)
    remember("Bearish on EV demand", "view")

    result = forget_about("EV")

    assert result["data"]["removed"] == 1
