"""Rate-limit resilience.

Free-tier Gemini allows 5 requests per minute PER MODEL, and automatic function
calling spends one request per tool round-trip. A single judge question can
exhaust one model, so falling back across models is what keeps the bot answering.
"""

import pytest

import atlas.engine.conversation as conversation
from atlas.integrations.gemini import (
    CHAT_CHAIN,
    is_rate_limited,
    retry_after_seconds,
)
from atlas.memory import store
from atlas.memory.extract import looks_durable

pytestmark = pytest.mark.usefixtures("fresh_db")

QUOTA_ERROR = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
    "current quota', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'retryDelay': '15s'}]}}"
)


class _Resp:
    text = "done"


def test_quota_errors_are_recognised():
    assert is_rate_limited(RuntimeError(QUOTA_ERROR)) is True


def test_ordinary_errors_are_not_treated_as_quota():
    assert is_rate_limited(RuntimeError("connection reset")) is False


def test_retry_delay_is_read_from_the_error():
    assert retry_after_seconds(RuntimeError(QUOTA_ERROR)) == 15.0


def test_retry_delay_falls_back_when_absent():
    assert retry_after_seconds(RuntimeError("boom"), default=3.0) == 3.0


async def test_falls_back_to_the_next_model_on_quota(monkeypatch):
    tried = []

    async def _fake(model, contents, system_prompt, tools):
        tried.append(model)
        if model == CHAT_CHAIN[0]:
            raise RuntimeError(QUOTA_ERROR)
        return _Resp()

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(1, "Shaan")

    reply = await conversation.respond(uid, "hello")

    assert reply == "done"
    assert tried == [CHAT_CHAIN[0], CHAT_CHAIN[1]]


async def test_non_quota_errors_abort_immediately_without_burning_the_chain(monkeypatch):
    tried = []

    async def _fake(model, contents, system_prompt, tools):
        tried.append(model)
        raise RuntimeError("malformed request")

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(2, "Shaan")

    reply = await conversation.respond(uid, "hello")

    assert reply == conversation.FAILURE_REPLY
    assert tried == [CHAT_CHAIN[0]], "a real fault must not retry every model"


async def test_total_exhaustion_tells_the_user_it_is_temporary(monkeypatch):
    async def _always_limited(model, contents, system_prompt, tools):
        raise RuntimeError(QUOTA_ERROR)

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(conversation, "_generate", _always_limited)
    monkeypatch.setattr(conversation.asyncio, "sleep", _no_sleep)
    uid = store.get_or_create_user(3, "Shaan")

    reply = await conversation.respond(uid, "hello")

    assert reply == conversation.BUSY_REPLY
    assert "rate limited" in reply.lower()


async def test_quota_wait_is_capped(monkeypatch):
    """A judge should not stare at a typing indicator for a full minute."""
    slept = []

    async def _always_limited(model, contents, system_prompt, tools):
        raise RuntimeError("429 RESOURCE_EXHAUSTED retryDelay: '600'")

    async def _record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(conversation, "_generate", _always_limited)
    monkeypatch.setattr(conversation.asyncio, "sleep", _record)
    uid = store.get_or_create_user(4, "Shaan")

    await conversation.respond(uid, "hello")

    assert slept == [conversation.MAX_QUOTA_WAIT]


@pytest.mark.parametrize(
    "text",
    [
        "I cover semiconductors for a hedge fund",
        "my fund is long NVDA",
        "we're bearish on EV demand",
        "Add Apple to the things you watch for me please and also track Microsoft too",
    ],
)
def test_durable_turns_trigger_extraction(text):
    assert looks_durable(text) is True


@pytest.mark.parametrize(
    "text", ["nvda price?", "compare msft and googl", "what moved today", ""]
)
def test_throwaway_turns_skip_extraction(text):
    assert looks_durable(text) is False
