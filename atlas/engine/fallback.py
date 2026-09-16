"""Groq as a second provider, for when no Gemini model answers.

The 2026-09-15 outage was gemini-3.6-flash and gemini-3-flash-preview both
answering 503 "high demand" within seconds of each other. A third Gemini model
shares that provider's capacity and this key's quota; a different provider shares
neither. Groq's free tier serves open-weight models that call tools, so a text
turn can still be answered with the same system prompt and the same tools, at $0.

Text only. Images and uploaded documents live in Gemini's Files API, which no
other provider can read, so those turns wait for Gemini instead.
"""

import asyncio
import copy
import json
import logging
import re
from functools import lru_cache

from google.genai import types
from groq import APIConnectionError, APITimeoutError, AsyncGroq, Groq

from atlas.config import get_settings
from atlas.tools.result import err

log = logging.getLogger(__name__)

# Both verified serving tool calls from production on 2026-09-16.
GROQ_CHAT_CHAIN = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
# The briefing gate needs judgement more than breadth; the small model is plenty
# and leaves the large one's daily token budget for chat.
GROQ_JSON_CHAIN = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")

MAX_TOOL_ROUNDS = 6
# Groq's free tier allows 8,000 tokens a minute per model, and every round of the
# tool loop resends the whole conversation. Measured on 2026-09-16: full tool
# docstrings alone made the first request 3,400 tokens, the second round hit the
# limit, and the SDK slept 32 seconds. So everything sent here is kept lean.
HISTORY_MESSAGES = 6
TOOL_RESULT_CHARS = 3000
TOOL_DESCRIPTION_CHARS = 240
# A token-per-minute refusal says how long until the window has room. Worth
# waiting that out when it is a moment; beyond it, the next model's own bucket
# answers sooner.
MAX_TPM_WAIT = 4.0
TIMEOUT = 30.0

# A key Groq rejects is rejected for every model. Anything else (quota, overload,
# a model retired or unable to form a tool call) is worth the next model.
_FATAL_STATUSES = frozenset({401, 403})


@lru_cache(maxsize=1)
def _async_client() -> AsyncGroq:
    # No SDK retries: they sleep for as long as Groq's retry-after says (32s was
    # observed), while the user waits. _complete_patiently decides instead.
    return AsyncGroq(api_key=get_settings().groq_api_key, timeout=TIMEOUT, max_retries=0)


@lru_cache(maxsize=1)
def _sync_client() -> Groq:
    return Groq(api_key=get_settings().groq_api_key, timeout=TIMEOUT, max_retries=0)


def _worth_next_model(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        # ValueError: a model that returned unparseable JSON says nothing about the next.
        return isinstance(exc, (APIConnectionError, APITimeoutError, ValueError))
    return status not in _FATAL_STATUSES


_TRY_AGAIN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")


def _retry_after(exc: Exception) -> float | None:
    """Seconds Groq asked us to wait on a rate limit, when it said."""
    if getattr(exc, "status_code", None) != 429:
        return None
    match = _TRY_AGAIN.search(str(exc))
    if not match:
        return None
    return int(match.group(1) or 0) * 60 + float(match.group(2))


def _short(description: str | None, name: str) -> str:
    """The first paragraph of a docstring, which says what the tool is for. The
    rest (Args, examples) is already in the parameter schema, and on this budget
    every tool's full docstring is resent on every round."""
    text = (description or "").strip().split("\n\n")[0]
    text = " ".join(text.split())
    return text[:TOOL_DESCRIPTION_CHARS] or name


def tool_specs(tools: list) -> tuple[list[dict], dict]:
    """OpenAI-style function specs, built the way Gemini builds its own.

    Using the Gemini SDK's own declaration keeps both providers reading the same
    names, parameter types, and docstrings, so the two cannot drift apart.
    """
    specs, table = [], {}
    for fn in tools:
        declaration = types.FunctionDeclaration.from_callable_with_api_option(
            callable=fn, api_option="GEMINI_API"
        )
        parameters = (
            declaration.parameters.json_schema.model_dump(mode="json", exclude_none=True)
            if declaration.parameters
            else {"type": "object", "properties": {}}
        )
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": declaration.name,
                    "description": _short(declaration.description, declaration.name),
                    "parameters": parameters,
                },
            }
        )
        table[declaration.name] = fn
    return specs, table


def _messages(system_prompt: str, history: list[dict], text: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for row in history[-HISTORY_MESSAGES:]:
        role = "assistant" if row["role"] == "model" else "user"
        messages.append({"role": role, "content": row["content"]})
    messages.append({"role": "user", "content": text})
    return messages


async def _complete(model: str, messages: list[dict], tools: list[dict] | None):
    """Network seam. Tests replace this."""
    kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
    return await _async_client().chat.completions.create(
        model=model, messages=messages, **kwargs
    )


async def _call_tool(table: dict, name: str, arguments: str | None) -> str:
    fn = table.get(name)
    if fn is None:
        result = err("unknown_tool", f"There is no tool named {name}.")
    else:
        try:
            args = json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:
            result = err("bad_arguments", f"Arguments were not valid JSON: {exc}")
        else:
            try:
                # Tools make blocking HTTP and database calls.
                result = await asyncio.to_thread(fn, **args)
            except TypeError as exc:
                result = err("bad_arguments", str(exc))
            except Exception:
                log.exception("fallback tool %s failed", name)
                result = err("tool_failed", f"{name} failed. Tell the user it is unavailable.")
    return _fit(result)


def _fit(result) -> str:
    """Serialize a tool result whole, dropping list items until it fits.

    Never a cut string: slicing the JSON landed mid-number, so "212.17" could
    reach the model as "212" while the system prompt tells it every figure comes
    from a tool result.
    """
    text = json.dumps(result, default=str)
    if len(text) <= TOOL_RESULT_CHARS:
        return text

    trimmed = copy.deepcopy(result)
    data = trimmed.get("data") if isinstance(trimmed, dict) else None
    if isinstance(data, dict):
        lists = [key for key, value in data.items() if isinstance(value, list) and value]
        while lists and len(text) > TOOL_RESULT_CHARS:
            longest = max(lists, key=lambda k: len(json.dumps(data[k], default=str)))
            data[longest] = data[longest][:-1]
            if not data[longest]:
                lists.remove(longest)
            data["truncated"] = True
            text = json.dumps(trimmed, default=str)
        if len(text) <= TOOL_RESULT_CHARS:
            return text

    return json.dumps(
        err("result_too_large", "That result was too big to read. Narrow the question."),
        default=str,
    )


async def _complete_patiently(model: str, messages: list[dict], tools: list[dict] | None):
    try:
        return await _complete(model, messages, tools)
    except Exception as exc:
        wait = _retry_after(exc)
        if wait is None or wait > MAX_TPM_WAIT:
            raise
        log.info("groq %s asked for %.1fs; waiting it out", model, wait)
        await asyncio.sleep(wait + 0.25)
        return await _complete(model, messages, tools)


async def _run(model: str, messages: list[dict], specs: list[dict], table: dict) -> str:
    for _ in range(MAX_TOOL_ROUNDS):
        message = (await _complete_patiently(model, messages, specs)).choices[0].message
        if not message.tool_calls:
            return message.content or ""
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            content = await _call_tool(table, call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    # Out of rounds: withhold the tools so the model has to answer with what it has.
    return (await _complete_patiently(model, messages, None)).choices[0].message.content or ""


async def respond(history: list[dict], text: str, system_prompt: str, tools: list) -> str:
    """Answer one text turn on Groq, walking its model chain."""
    specs, table = tool_specs(tools)
    last: Exception | None = None
    for model in GROQ_CHAT_CHAIN:
        try:
            reply = await _run(model, _messages(system_prompt, history, text), specs, table)
        except Exception as exc:
            if not _worth_next_model(exc):
                raise
            last = exc
            log.warning("groq %s failed (%s), trying next model", model, exc)
            continue
        log.warning("answered on groq %s because no Gemini model did", model)
        return reply
    raise last if last is not None else RuntimeError("no groq chat model configured")


def _complete_json(model: str, system: str, prompt: str) -> str:
    """Network seam. Tests replace this."""
    completion = _sync_client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        reasoning_effort="low",
    )
    return completion.choices[0].message.content or "{}"


def complete_json(system: str, prompt: str) -> dict:
    """One JSON-object completion on Groq. Synchronous, like its only caller."""
    last: Exception | None = None
    for model in GROQ_JSON_CHAIN:
        try:
            return json.loads(_complete_json(model, system, prompt))
        except Exception as exc:
            if not _worth_next_model(exc):
                raise
            last = exc
            log.warning("groq %s failed (%s), trying next model", model, exc)
    raise last if last is not None else RuntimeError("no groq json model configured")
