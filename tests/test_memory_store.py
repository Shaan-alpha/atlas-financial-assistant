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


# --- audit fixes, 2026-09-16 -----------------------------------------------------


def test_telegram_ids_past_32_bits_can_register():
    """Telegram ids passed 2**31 years ago. Production stored them in a 32-bit
    column, so every newer account crashed on its very first message."""
    uid = store.get_or_create_user(8_123_456_789, "New Account")

    assert store.get_or_create_user(8_123_456_789) == uid


def test_the_telegram_id_column_is_64_bit():
    from sqlalchemy import BigInteger

    from atlas.db.models import User

    assert isinstance(User.__table__.c.telegram_id.type, BigInteger)


class _FakeConn:
    def __init__(self, dialect, data_type):
        self.dialect = type("D", (), {"name": dialect})()
        self.data_type, self.executed = data_type, []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.executed.append(sql)
        data_type = self.data_type

        class _Result:
            def scalar(self):
                return data_type

        return _Result()


def test_startup_widens_a_32_bit_telegram_id_on_postgres():
    from atlas.db.session import upgrade_schema

    conn = _FakeConn("postgresql", "integer")
    upgrade_schema(conn)

    assert any("ALTER TABLE users ALTER COLUMN telegram_id TYPE BIGINT" in s for s in conn.executed)


@pytest.mark.parametrize("dialect,data_type", [("postgresql", "bigint"), ("sqlite", None)])
def test_startup_leaves_a_correct_schema_alone(dialect, data_type):
    from atlas.db.session import upgrade_schema

    conn = _FakeConn(dialect, data_type)
    upgrade_schema(conn)

    assert not any("ALTER" in s for s in conn.executed)


@pytest.mark.parametrize("pattern", ["", "   ", "%", "_"])
def test_forget_never_wipes_everything_on_a_wildcard(pattern):
    uid = store.get_or_create_user(801, "Shaan")
    store.add_fact(uid, "Covers semiconductors", "focus")
    store.add_fact(uid, "Bearish on EV demand", "view")

    assert store.forget(uid, pattern) == 0
    assert len(store.all_facts(uid)) == 2


def test_forget_matches_words_not_letters():
    """"EV" must not delete a fact that merely contains the letters e-v."""
    uid = store.get_or_create_user(802, "Shaan")
    store.add_fact(uid, "Bearish on EV demand", "view")
    store.add_fact(uid, "Reviews every earnings call", "preference")

    assert store.forget(uid, "ev") == 1
    assert [f["fact"] for f in store.all_facts(uid)] == ["Reviews every earnings call"]


def test_facts_with_wildcard_characters_are_still_stored():
    """add_fact used the fact itself as an ILIKE pattern, so "Owns 5% of X"
    matched "Owns 50% of X" and was silently dropped as a duplicate."""
    uid = store.get_or_create_user(803, "Shaan")
    store.add_fact(uid, "Owns 50% of the fund in semis", "focus")
    store.add_fact(uid, "Owns 5_% of the fund in semis", "focus")
    store.add_fact(uid, "Owns 5% of the fund in semis", "focus")

    assert len(store.all_facts(uid)) == 3


def test_identical_alerts_are_not_armed_twice():
    """Model failover replays the whole tool loop, which used to create the same
    alert once per model that tried."""
    uid = store.get_or_create_user(804, "Shaan")

    first = store.create_alert(uid, "tell me if TSLA moves 5%", "tsla", "move_pct", 5.0)
    again = store.create_alert(uid, "ping me on a 5% TSLA move", "TSLA", "move_pct", 5.0)

    assert again == first
    assert len(store.user_alerts(uid)) == 1


def test_concurrent_first_messages_from_a_new_user_do_not_crash(monkeypatch):
    """Two updates from a brand-new user both miss the SELECT, and the second
    INSERT hits the unique constraint."""
    from atlas.db.models import User
    from atlas.db.session import session_scope

    with session_scope() as s:
        s.add(User(telegram_id=555, name="Racer"))

    original = store._find_user
    calls = {"n": 0}

    def _miss_once(session, telegram_id):
        calls["n"] += 1
        return None if calls["n"] == 1 else original(session, telegram_id)

    monkeypatch.setattr(store, "_find_user", _miss_once)

    uid = store.get_or_create_user(555, "Racer")

    with session_scope() as s:
        assert uid == s.query(User).filter_by(telegram_id=555).one().id


def test_forget_still_matches_the_plural_the_user_typed():
    """"stop remembering my briefing preferences" reaches forget_about("briefing")."""
    uid = store.get_or_create_user(805, "Shaan")
    store.add_fact(uid, "Prefers morning briefings", "preference")
    store.add_fact(uid, "Reviews every earnings call", "preference")

    assert store.forget(uid, "briefing") == 1
    assert [f["fact"] for f in store.all_facts(uid)] == ["Reviews every earnings call"]


def test_timestamps_are_stored_the_way_they_are_compared():
    """The columns are TIMESTAMP WITHOUT TIME ZONE; psycopg would hand Postgres an
    aware value to convert with the session timezone."""
    from atlas.db.models import _utcnow

    assert _utcnow().tzinfo is None
