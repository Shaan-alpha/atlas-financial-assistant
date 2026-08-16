"""Concurrent turns.

Updates are now processed in parallel, which removes the accidental protection
sequential processing gave us. Two things have to hold at once:

  - one user's turns stay strictly ordered, because respond() reads the history
    before it appends to it;
  - different users genuinely overlap, which is the entire point.

The blocking-database tests are the other half: a turn that sits on the event
loop holding a synchronous query stalls every other user, so concurrency without
them buys nothing.
"""

import asyncio
import contextlib
import threading
import time
from types import SimpleNamespace

import pytest

import atlas.engine.conversation as conversation
import atlas.ingress.handlers as handlers
from atlas.memory import store

pytestmark = pytest.mark.usefixtures("fresh_db")


class _Resp:
    def __init__(self, text):
        self.text = text


def _traced(events, delay=0.0):
    """A model seam that records when each turn enters and leaves the call."""

    async def _fake(model, contents, system_prompt, tools):
        text = contents[-1].parts[0].text
        events.append(("start", text))
        if delay:
            await asyncio.sleep(delay)
        events.append(("end", text))
        return _Resp(f"reply-to-{text}")

    return _fake


async def test_two_turns_from_one_user_do_not_interleave(monkeypatch):
    events = []
    monkeypatch.setattr(conversation, "_generate", _traced(events, delay=0.05))
    uid = store.get_or_create_user(1, "Shaan")

    await asyncio.gather(
        conversation.respond(uid, "first"),
        conversation.respond(uid, "second"),
    )

    # Without the per-user lock this is start/start/end/end.
    assert events == [
        ("start", "first"),
        ("end", "first"),
        ("start", "second"),
        ("end", "second"),
    ]


async def test_different_users_are_not_serialized(monkeypatch):
    events = []
    monkeypatch.setattr(conversation, "_generate", _traced(events, delay=0.05))
    one = store.get_or_create_user(2, "Ada")
    two = store.get_or_create_user(3, "Linus")

    await asyncio.gather(
        conversation.respond(one, "from-ada"),
        conversation.respond(two, "from-linus"),
    )

    # Both must be in flight together, or the lock is too coarse.
    assert [kind for kind, _ in events] == ["start", "start", "end", "end"]


async def test_the_second_turn_sees_the_first_turns_reply(monkeypatch):
    """The reason the lock exists: overlapping turns each build a prompt from a
    history the other has already invalidated."""
    monkeypatch.setattr(conversation, "_generate", _traced([], delay=0.05))
    uid = store.get_or_create_user(4, "Shaan")

    await asyncio.gather(
        conversation.respond(uid, "first"),
        conversation.respond(uid, "second"),
    )

    rows = [(m["role"], m["content"]) for m in store.recent_messages(uid)]
    assert rows == [
        ("user", "first"),
        ("model", "reply-to-first"),
        ("user", "second"),
        ("model", "reply-to-second"),
    ]


async def test_a_failed_turn_releases_the_lock(monkeypatch):
    """A lock still held after an error would wedge that one user permanently,
    and they would have no way to tell — every later message just never answers."""

    async def _boom(model, contents, system_prompt, tools):
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(conversation, "_generate", _boom)
    uid = store.get_or_create_user(5, "Shaan")

    assert await conversation.respond(uid, "hello") == conversation.FAILURE_REPLY

    monkeypatch.setattr(conversation, "_generate", _traced([]))
    reply = await asyncio.wait_for(conversation.respond(uid, "again"), timeout=2)
    assert reply == "reply-to-again"


async def test_database_reads_run_off_the_event_loop(monkeypatch):
    """Every store.* call is synchronous SQLAlchemy. Left on the loop thread it
    blocks every other user's turn, which is the whole point of going concurrent."""
    threads = []
    real = store.profile_snapshot

    def _record(user_id):
        threads.append(threading.get_ident())
        return real(user_id)

    monkeypatch.setattr(store, "profile_snapshot", _record)
    monkeypatch.setattr(conversation, "_generate", _traced([]))
    uid = store.get_or_create_user(6, "Shaan")

    await conversation.respond(uid, "hello")

    assert threads, "profile_snapshot was never called"
    assert threading.get_ident() not in threads


async def test_a_slow_query_does_not_freeze_the_other_users(monkeypatch):
    """The honest version: while one turn sits in the database, the loop must
    still be running everybody else."""

    def _slow(user_id):
        time.sleep(0.2)
        return {
            "name": "S",
            "role": None,
            "timezone": None,
            "briefing_time": None,
            "onboarding_state": "new",
        }

    monkeypatch.setattr(store, "profile_snapshot", _slow)
    monkeypatch.setattr(conversation, "_generate", _traced([]))
    uid = store.get_or_create_user(7, "Shaan")

    ticks = 0

    async def _tick():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    ticker = asyncio.create_task(_tick())
    await conversation.respond(uid, "hello")
    ticker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ticker

    # 200ms of database work at 5ms a tick. Deliberately loose: this is a
    # liveness check, not a benchmark.
    assert ticks >= 5


class _FakeChat:
    async def send_action(self, action):
        pass


class _FakeMessage:
    def __init__(self, text):
        self.text = text
        self.replies = []

    async def reply_text(self, text, parse_mode=None):
        self.replies.append(text)


class _FakeUpdate:
    def __init__(self, telegram_id, text):
        self.effective_user = SimpleNamespace(id=telegram_id, first_name="Shaan")
        self.effective_chat = _FakeChat()
        self.message = _FakeMessage(text)


async def test_the_user_lookup_in_a_handler_runs_off_the_event_loop(monkeypatch):
    """_user_id() is the first thing every handler does, and it is a database
    round trip — so it blocked the loop before respond() even started."""
    threads = []
    real = store.get_or_create_user

    def _record(telegram_id, name=None):
        threads.append(threading.get_ident())
        return real(telegram_id, name)

    async def _echo(user_id, text, attachments=None):
        return "ok"

    monkeypatch.setattr(store, "get_or_create_user", _record)
    monkeypatch.setattr(handlers, "respond", _echo)

    update = _FakeUpdate(99, "hello")
    await handlers.on_text(update, None)

    assert update.message.replies == ["ok"]
    assert threads and threading.get_ident() not in threads
