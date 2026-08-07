import pytest

from atlas.memory import store

pytestmark = pytest.mark.usefixtures("fresh_db")


def test_get_or_create_is_idempotent():
    first = store.get_or_create_user(7, name="Shaan")
    second = store.get_or_create_user(7, name="Shaan")
    assert first == second


def test_profile_snapshot_reflects_updates():
    uid = store.get_or_create_user(7, name="Shaan")
    store.set_profile(uid, role="equity analyst", briefing_time="08:30")

    snap = store.profile_snapshot(uid)

    assert snap["role"] == "equity analyst"
    assert snap["briefing_time"] == "08:30"


def test_duplicate_facts_are_not_stored_twice():
    uid = store.get_or_create_user(7, name="Shaan")
    store.add_fact(uid, "Covers semiconductors", "focus")
    store.add_fact(uid, "covers semiconductors", "focus")

    assert len(store.all_facts(uid)) == 1


def test_forget_removes_matching_facts():
    uid = store.get_or_create_user(7, name="Shaan")
    store.add_fact(uid, "Bearish on EV demand", "view")
    store.add_fact(uid, "Covers semiconductors", "focus")

    removed = store.forget(uid, "EV")

    assert removed == 1
    assert len(store.all_facts(uid)) == 1


def test_recent_messages_returns_chronological_tail():
    uid = store.get_or_create_user(7, name="Shaan")
    for i in range(5):
        store.append_message(uid, "user", f"m{i}")

    tail = store.recent_messages(uid, limit=3)

    assert [m["content"] for m in tail] == ["m2", "m3", "m4"]
