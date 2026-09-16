"""Groq as a second provider for chat turns.

The 2026-09-15 outage was two Gemini models answering 503 within seconds of each
other. Another Gemini model shares that provider's capacity and this key's quota;
Groq shares neither, and its free tier serves open-weight models that call tools.
"""

import json
from types import SimpleNamespace

import pytest

import atlas.engine.fallback as fallback
from atlas.memory import store
from atlas.tools.registry import build_tools

pytestmark = pytest.mark.usefixtures("fresh_db")


def _call(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _completion(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


class _Script:
    """Plays back completions in order and records every request."""

    def __init__(self, *steps):
        self.steps, self.requests = list(steps), []

    async def __call__(self, model, messages, tools):
        self.requests.append(
            {"model": model, "messages": json.loads(json.dumps(messages, default=str)), "tools": tools}
        )
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class _StatusError(Exception):
    def __init__(self, status_code, message=""):
        super().__init__(f"Error code: {status_code} {message}")
        self.status_code = status_code


def get_quote(symbol: str) -> dict:
    """Return the current price for one listed security.

    Args:
        symbol: Ticker symbol, for example "NVDA".
    """
    return {"ok": True, "data": {"symbol": symbol.upper(), "price": 212.17}}


def test_every_registered_tool_becomes_a_function_spec():
    uid = store.get_or_create_user(1, "Shaan")
    tools = build_tools(uid)

    specs, table = fallback.tool_specs(tools)

    assert {s["function"]["name"] for s in specs} == {t.__name__ for t in tools}
    assert set(table) == {t.__name__ for t in tools}
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["function"]["parameters"]["type"] == "object"
        assert spec["function"]["description"]


async def test_a_tool_round_trip_answers_the_question(monkeypatch):
    script = _Script(
        _completion(tool_calls=[_call("get_quote", '{"symbol": "nvda"}')]),
        _completion(content="*NVDA* is at *$212.17*."),
    )
    monkeypatch.setattr(fallback, "_complete", script)

    reply = await fallback.respond([], "what's nvda at?", "SYSTEM", [get_quote])

    assert reply == "*NVDA* is at *$212.17*."
    tool_message = script.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"])["data"]["price"] == 212.17
    assert script.requests[0]["messages"][0] == {"role": "system", "content": "SYSTEM"}


async def test_malformed_arguments_go_back_to_the_model_as_an_error(monkeypatch):
    script = _Script(
        _completion(tool_calls=[_call("get_quote", "{not json")]),
        _completion(tool_calls=[_call("no_such_tool", "{}", "call_2")]),
        _completion(content="Sorry, I could not look that up."),
    )
    monkeypatch.setattr(fallback, "_complete", script)

    reply = await fallback.respond([], "nvda?", "SYSTEM", [get_quote])

    assert reply == "Sorry, I could not look that up."
    assert json.loads(script.requests[1]["messages"][-1]["content"])["ok"] is False
    assert json.loads(script.requests[2]["messages"][-1]["content"])["error"] == "unknown_tool"


async def test_wrong_argument_names_do_not_crash_the_turn(monkeypatch):
    script = _Script(
        _completion(tool_calls=[_call("get_quote", '{"ticker": "NVDA"}')]),
        _completion(content="done"),
    )
    monkeypatch.setattr(fallback, "_complete", script)

    assert await fallback.respond([], "nvda?", "SYSTEM", [get_quote]) == "done"
    assert json.loads(script.requests[1]["messages"][-1]["content"])["error"] == "bad_arguments"


async def test_a_refused_model_hands_over_to_the_next(monkeypatch):
    script = _Script(_StatusError(429), _completion(content="answer"))
    monkeypatch.setattr(fallback, "_complete", script)

    reply = await fallback.respond([], "hello", "SYSTEM", [get_quote])

    assert reply == "answer"
    assert [r["model"] for r in script.requests] == list(fallback.GROQ_CHAT_CHAIN[:2])


async def test_a_bad_key_does_not_burn_the_chain(monkeypatch):
    script = _Script(_StatusError(401))
    monkeypatch.setattr(fallback, "_complete", script)

    with pytest.raises(_StatusError):
        await fallback.respond([], "hello", "SYSTEM", [get_quote])
    assert len(script.requests) == 1


async def test_history_is_trimmed_and_roles_translated(monkeypatch):
    script = _Script(_completion(content="ok"))
    monkeypatch.setattr(fallback, "_complete", script)
    history = [
        {"role": "user" if i % 2 == 0 else "model", "content": f"m{i}"} for i in range(30)
    ]

    await fallback.respond(history, "latest", "SYSTEM", [get_quote])

    messages = script.requests[0]["messages"]
    assert len(messages) == 1 + fallback.HISTORY_MESSAGES + 1
    first = 30 - fallback.HISTORY_MESSAGES
    # Stored roles are "user" and "model"; Groq's API wants "user" and "assistant".
    assert messages[1] == {"role": "user", "content": f"m{first}"}
    assert messages[2] == {"role": "assistant", "content": f"m{first + 1}"}
    assert messages[-1] == {"role": "user", "content": "latest"}


async def test_tool_loops_are_bounded(monkeypatch):
    looping = [
        _completion(tool_calls=[_call("get_quote", '{"symbol": "NVDA"}', f"c{i}")])
        for i in range(fallback.MAX_TOOL_ROUNDS)
    ]
    script = _Script(*looping, _completion(content="final"))
    monkeypatch.setattr(fallback, "_complete", script)

    reply = await fallback.respond([], "nvda?", "SYSTEM", [get_quote])

    assert reply == "final"
    # The last request withholds tools, so the model has to answer.
    assert script.requests[-1]["tools"] is None
    assert len(script.requests) == fallback.MAX_TOOL_ROUNDS + 1


def test_json_completion_falls_through_models(monkeypatch):
    calls = []

    def _fake(model, system, prompt):
        calls.append(model)
        if len(calls) == 1:
            raise _StatusError(503)
        return '{"send": false}'

    monkeypatch.setattr(fallback, "_complete_json", _fake)

    assert fallback.complete_json("SYSTEM", "{}") == {"send": False}
    assert calls == list(fallback.GROQ_JSON_CHAIN[:2])


async def test_a_short_token_wait_is_waited_out_on_the_same_model(monkeypatch):
    """Groq's free tier allows 8,000 tokens a minute per model; a second tool round
    often needs a second or two of that window back."""
    slept = []

    async def _sleep(seconds):
        slept.append(seconds)

    script = _Script(
        _StatusError(429, "Rate limit reached ... Please try again in 1.875s."),
        _completion(content="answer"),
    )
    monkeypatch.setattr(fallback, "_complete", script)
    monkeypatch.setattr(fallback.asyncio, "sleep", _sleep)

    assert await fallback.respond([], "hello", "SYSTEM", [get_quote]) == "answer"
    assert [r["model"] for r in script.requests] == [fallback.GROQ_CHAT_CHAIN[0]] * 2
    assert slept and slept[0] < fallback.MAX_TPM_WAIT + 1


async def test_a_long_token_wait_moves_to_the_next_model(monkeypatch):
    """The SDK used to sleep 32 seconds here while the user watched a typing dot."""
    async def _sleep(seconds):
        raise AssertionError("must not wait half a minute")

    script = _Script(
        _StatusError(429, "Please try again in 32.5s."),
        _completion(content="answer"),
    )
    monkeypatch.setattr(fallback, "_complete", script)
    monkeypatch.setattr(fallback.asyncio, "sleep", _sleep)

    assert await fallback.respond([], "hello", "SYSTEM", [get_quote]) == "answer"
    assert [r["model"] for r in script.requests] == list(fallback.GROQ_CHAT_CHAIN[:2])


def test_tool_descriptions_sent_to_groq_are_lean():
    uid = store.get_or_create_user(9, "Shaan")

    specs, _ = fallback.tool_specs(build_tools(uid))

    for spec in specs:
        description = spec["function"]["description"]
        assert len(description) <= fallback.TOOL_DESCRIPTION_CHARS
        assert "Args:" not in description


def test_the_sdk_never_sleeps_on_its_own(env):
    fallback._async_client.cache_clear()
    fallback._sync_client.cache_clear()
    try:
        assert fallback._async_client().max_retries == 0
        assert fallback._sync_client().max_retries == 0
    finally:
        fallback._async_client.cache_clear()
        fallback._sync_client.cache_clear()


async def test_a_large_tool_result_stays_valid_json(monkeypatch):
    """Slicing the JSON landed mid-number: "212.17" reached the model as "212"."""
    def big_news(query: str) -> dict:
        """Search recent financial news.

        Args:
            query: What to search for.
        """
        return {
            "ok": True,
            "source": "Google News",
            "as_of": "2026-09-16T00:00:00Z",
            "data": {"articles": [
                {"title": f"Story {i}", "summary": "x" * 400, "price": 212.17} for i in range(40)
            ]},
        }

    script = _Script(
        _completion(tool_calls=[_call("big_news", '{"query": "nvda"}')]),
        _completion(content="done"),
    )
    monkeypatch.setattr(fallback, "_complete", script)

    await fallback.respond([], "news?", "SYSTEM", [big_news])

    sent = script.requests[1]["messages"][-1]["content"]
    assert len(sent) <= fallback.TOOL_RESULT_CHARS
    payload = json.loads(sent)  # whole JSON, not a cut string
    assert payload["source"] == "Google News"  # attribution survives the trim
    assert payload["data"]["truncated"] is True
    assert all(a["price"] == 212.17 for a in payload["data"]["articles"])
