"""Rate-limit resilience.

Free-tier Gemini allows 5 requests per minute PER MODEL, and automatic function
calling spends one request per tool round-trip. A single judge question can
exhaust one model, so falling back across models is what keeps the bot answering.
"""

import pytest

import atlas.engine.conversation as conversation
from atlas.integrations.gemini import (
    CHAT_CHAIN,
    EXTRACT_CHAIN,
    is_model_unavailable,
    is_rate_limited,
    is_transient,
    retry_after_seconds,
)
from atlas.memory import store
from atlas.memory.extract import _extract as _real_extract
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
        "Our desk runs a long/short book with a heavy tilt toward AI infrastructure names",
    ],
)
def test_durable_turns_trigger_extraction(text):
    assert looks_durable(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "nvda price?",
        "compare msft and googl",
        "what moved today",
        "",
        "tell me about NVDA",
        "how is the US market doing",
        "what should I buy today?",
        "I'm back.",
    ],
)
def test_throwaway_turns_skip_extraction(text):
    assert looks_durable(text) is False


# --- transient upstream faults ------------------------------------------------
#
# A 499 CANCELLED on the first request of a fresh process cost a real user their
# whole turn: the chain aborted on the preferred model with two untried models
# still in it, and they got "I hit trouble reaching my data sources just then."


class _Cancelled(Exception):
    """Shaped like google.genai.errors.ClientError, which carries .code."""

    code = 499

    def __str__(self):
        return "499 CANCELLED. {'error': {'status': 'CANCELLED'}}"


class _Answer:
    def __init__(self, text):
        self.text = text


def test_a_cancelled_request_is_transient():
    assert is_transient(_Cancelled()) is True
    assert is_transient(Exception("503 UNAVAILABLE")) is True
    # Not transient: a real fault that merely mentions a number.
    assert is_transient(RuntimeError("upstream 503")) is False
    assert is_transient(ValueError("bad request")) is False


async def test_a_transient_fault_falls_through_to_the_next_model(monkeypatch):
    tried = []

    async def _fake(model, contents, system_prompt, tools):
        tried.append(model)
        if len(tried) == 1:
            raise _Cancelled()
        return _Answer("NVDA is at 225.16.")

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(700, "Shaan")

    reply = await conversation.respond(uid, "what's NVDA at?")

    assert "225.16" in reply
    assert tried == list(conversation.CHAT_CHAIN[:2])


async def test_a_genuine_fault_still_aborts_the_chain(monkeypatch):
    """Only transient and quota errors are worth another model. Retrying a real
    fault three times just makes the user wait three times as long for it."""
    tried = []

    async def _fake(model, contents, system_prompt, tools):
        tried.append(model)
        raise ValueError("malformed request")

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(701, "Shaan")

    reply = await conversation.respond(uid, "hello")

    assert reply == conversation.FAILURE_REPLY
    assert tried == [conversation.CHAT_CHAIN[0]]


# --- retired models -----------------------------------------------------------
#
# Production outage, 2026-09-15: gemini-3.6-flash and gemini-3-flash-preview both
# answered 503 "high demand", and the last model in the chain, gemini-2.5-flash,
# had been retired by Google. Its 404 was treated as a genuine fault, so the turn
# died and the user was told their data sources were unreachable. Every data
# source was healthy; the model chain had quietly shrunk to two entries.


class _ApiError(Exception):
    """Shaped like google.genai.errors.APIError: a .code and the raw payload."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _retired(model):
    return _ApiError(
        404,
        "404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model "
        f"models/{model} is no longer available to new users. Please update your "
        "code to use models/gemini-3.6-flash for the latest features and "
        "improvements.', 'status': 'NOT_FOUND'}}",
    )


def _overloaded():
    return _ApiError(
        503,
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is "
        "currently experiencing high demand. Spikes in demand are usually "
        "temporary. Please try again later.', 'status': 'UNAVAILABLE'}}",
    )


# Retired for this project's key, verified live from production on 2026-09-16.
RETIRED_MODELS = {"gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"}


def test_a_retired_model_is_recognised_as_unavailable():
    assert is_model_unavailable(_retired("gemini-2.5-flash"), "gemini-2.5-flash")
    # The 404 must name the model we asked for, not a longer sibling of it.
    assert not is_model_unavailable(
        _retired("gemini-2.5-flash-lite"), "gemini-2.5-flash"
    )
    # A 404 about something else in the request says nothing about the model.
    assert not is_model_unavailable(
        _ApiError(404, "404 NOT_FOUND. File files/abc123 was not found."),
        "gemini-3.6-flash",
    )
    assert not is_model_unavailable(_overloaded(), "gemini-3.6-flash")


@pytest.mark.parametrize(
    "chain", [CHAT_CHAIN, EXTRACT_CHAIN], ids=["chat", "extract"]
)
def test_no_chain_leans_on_a_retired_model(chain):
    assert not set(chain) & RETIRED_MODELS
    assert len(set(chain)) >= 2, "a one-model chain has nothing to fail over to"


async def test_a_retired_model_is_skipped_rather_than_fatal(monkeypatch):
    tried = []

    async def _fake(model, contents, system_prompt, tools):
        tried.append(model)
        if len(tried) == 1:
            raise _retired(model)
        return _Answer("NVDA is at 212.17.")

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(702, "Shaan")

    reply = await conversation.respond(uid, "what's NVDA at?")

    assert "212.17" in reply
    assert tried == list(CHAT_CHAIN[:2])


async def test_the_september_outage_recovers_after_one_short_wait(monkeypatch):
    """503, 503, retired: the exact sequence from production. After a brief
    wait the best model that is still served gets one more try."""
    tried = []
    slept = []

    async def _fake(model, contents, system_prompt, tools):
        tried.append(model)
        if len(tried) < len(CHAT_CHAIN):
            raise _overloaded()
        if len(tried) == len(CHAT_CHAIN):
            raise _retired(model)
        return _Answer("NVDA is at 212.17.")

    async def _record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(conversation, "_generate", _fake)
    monkeypatch.setattr(conversation.asyncio, "sleep", _record)
    uid = store.get_or_create_user(703, "Shaan")

    reply = await conversation.respond(uid, "today nvidia stocks")

    assert "212.17" in reply
    assert len(slept) == 1
    assert tried == [*CHAT_CHAIN, CHAT_CHAIN[0]]


async def test_a_retired_model_is_never_the_final_attempt(monkeypatch):
    tried = []

    async def _fake(model, contents, system_prompt, tools):
        tried.append(model)
        if model == CHAT_CHAIN[0]:
            raise _retired(model)
        raise _overloaded()

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(conversation, "_generate", _fake)
    monkeypatch.setattr(conversation.asyncio, "sleep", _no_sleep)
    uid = store.get_or_create_user(704, "Shaan")

    await conversation.respond(uid, "hello")

    assert tried == [*CHAT_CHAIN, CHAT_CHAIN[1]]


async def test_every_model_retired_fails_fast_without_waiting(monkeypatch):
    slept = []

    async def _fake(model, contents, system_prompt, tools):
        raise _retired(model)

    async def _record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(conversation, "_generate", _fake)
    monkeypatch.setattr(conversation.asyncio, "sleep", _record)
    uid = store.get_or_create_user(705, "Shaan")

    reply = await conversation.respond(uid, "hello")

    assert reply == conversation.FAILURE_REPLY
    assert slept == [], "waiting does not bring a retired model back"


async def test_persistent_overload_tells_the_user_it_is_temporary(monkeypatch):
    async def _fake(model, contents, system_prompt, tools):
        raise _overloaded()

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(conversation, "_generate", _fake)
    monkeypatch.setattr(conversation.asyncio, "sleep", _no_sleep)
    uid = store.get_or_create_user(706, "Shaan")

    reply = await conversation.respond(uid, "hello")

    assert reply == conversation.OVERLOADED_REPLY


async def test_groq_answers_when_no_gemini_model_does(monkeypatch):
    import atlas.engine.fallback as fallback

    slept = []
    seen = {}

    async def _gemini(model, contents, system_prompt, tools):
        raise _overloaded()

    async def _groq(history, text, system_prompt, tools):
        seen.update(text=text, tools={t.__name__ for t in tools}, prompt=system_prompt)
        return "*NVDA* is at *$212.17*."

    async def _record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(conversation, "_generate", _gemini)
    monkeypatch.setattr(fallback, "respond", _groq)
    monkeypatch.setattr(conversation.asyncio, "sleep", _record)
    uid = store.get_or_create_user(707, "Shaan")

    reply = await conversation.respond(uid, "today nvidia stocks")

    assert reply == "*NVDA* is at *$212.17*."
    assert slept == [], "a second provider answering beats waiting on the first"
    assert seen["text"] == "today nvidia stocks"
    assert "get_quote" in seen["tools"]
    assert "Atlas" in seen["prompt"]
    # The fallback's reply is part of the conversation like any other.
    assert [m["role"] for m in store.recent_messages(uid)] == ["user", "model"]


async def test_turns_with_attachments_never_go_to_groq(monkeypatch):
    """Uploaded files live in Gemini's Files API; no other provider can read them."""
    import atlas.engine.fallback as fallback

    tried = []

    async def _gemini(model, contents, system_prompt, tools):
        tried.append(model)
        if len(tried) <= len(CHAT_CHAIN):
            raise _overloaded()
        return _Answer("The chart shows a breakout.")

    async def _groq(*args):
        raise AssertionError("an image turn must not reach groq")

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(conversation, "_generate", _gemini)
    monkeypatch.setattr(fallback, "respond", _groq)
    monkeypatch.setattr(conversation.asyncio, "sleep", _no_sleep)
    uid = store.get_or_create_user(708, "Shaan")

    reply = await conversation.respond(
        uid, "what is this chart?", [{"kind": "image", "bytes": b"x", "mime": "image/jpeg"}]
    )

    assert reply == "The chart shows a breakout."


async def test_a_failed_fallback_still_waits_and_retries_gemini(monkeypatch):
    import atlas.engine.fallback as fallback

    tried = []
    slept = []

    async def _gemini(model, contents, system_prompt, tools):
        tried.append(model)
        if len(tried) <= len(CHAT_CHAIN):
            raise _overloaded()
        return _Answer("back on gemini")

    async def _groq(*args):
        raise ConnectionError("groq down too")

    async def _record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(conversation, "_generate", _gemini)
    monkeypatch.setattr(fallback, "respond", _groq)
    monkeypatch.setattr(conversation.asyncio, "sleep", _record)
    uid = store.get_or_create_user(709, "Shaan")

    assert await conversation.respond(uid, "hello") == "back on gemini"
    assert len(slept) == 1


def test_the_briefing_gate_falls_back_to_groq(monkeypatch):
    """A briefing the gate could not decide on is a briefing that never arrives."""
    import atlas.engine.fallback as fallback
    from atlas.proactive import salience

    class _Models:
        def generate_content(self, model, contents, config):
            raise _overloaded()

    class _Client:
        models = _Models()

    monkeypatch.setattr(salience, "get_client", lambda: _Client())
    monkeypatch.setattr(
        fallback, "complete_json", lambda system, prompt: {"send": True, "brief": "*AMD* -4%"}
    )

    assert salience._decide_sync("{}")["brief"] == "*AMD* -4%"


def test_no_failure_reply_blames_data_sources():
    """Tool failures go back to the model as results and never reach these
    replies, so naming data sources sends whoever reads it the wrong way."""
    for reply in (
        conversation.FAILURE_REPLY,
        conversation.OVERLOADED_REPLY,
        conversation.BUSY_REPLY,
    ):
        assert "data source" not in reply.lower()


# --- the other chains ---------------------------------------------------------
#
# Fact extraction and the briefing gate each walk their own chain, and each used
# to fail over on quota alone. A 503 or a retired model ended them.


@pytest.mark.parametrize("fault", [_overloaded, _retired], ids=["overloaded", "retired"])
async def test_fact_extraction_fails_over(monkeypatch, fault):
    import atlas.memory.extract as extract

    tried = []

    class _Response:
        text = '[{"fact": "Covers semiconductors", "category": "focus"}]'

    class _Models:
        async def generate_content(self, model, contents, config):
            tried.append(model)
            if len(tried) == 1:
                raise fault(model) if fault is _retired else fault()
            return _Response()

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    monkeypatch.setattr(extract, "get_client", lambda: _Client())

    # The suite stubs _extract globally; exercise the real one captured at import.
    items = await _real_extract("I cover semis", "Noted.")

    assert items[0]["fact"] == "Covers semiconductors"
    assert tried == list(EXTRACT_CHAIN[:2])


@pytest.mark.parametrize("fault", [_overloaded, _retired], ids=["overloaded", "retired"])
def test_the_briefing_gate_fails_over(monkeypatch, fault):
    from atlas.proactive import salience

    tried = []

    class _Response:
        text = '{"send": true, "brief": "*NVDA* -4%", "used_keys": []}'

    class _Models:
        def generate_content(self, model, contents, config):
            tried.append(model)
            if len(tried) == 1:
                raise fault(model) if fault is _retired else fault()
            return _Response()

    class _Client:
        models = _Models()

    monkeypatch.setattr(salience, "get_client", lambda: _Client())

    raw = salience._decide_sync("{}")

    assert raw["send"] is True
    assert tried == list(EXTRACT_CHAIN[:2])


def test_the_gemini_client_never_waits_forever(env):
    """A hung request held the user's turn lock indefinitely."""
    import atlas.integrations.gemini as gemini

    gemini.get_client.cache_clear()
    try:
        client = gemini.get_client()
        assert client._api_client._http_options.timeout == gemini.REQUEST_TIMEOUT_MS
    finally:
        gemini.get_client.cache_clear()


def test_a_client_side_timeout_moves_to_the_next_model():
    import httpx

    assert is_transient(httpx.ReadTimeout("timed out")) is True
    assert is_transient(TimeoutError()) is True
