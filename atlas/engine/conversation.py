"""The turn loop.

Gemini's automatic function calling runs the tool cycle; we supply the tools,
the hydrated history, and the system prompt.
"""

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass
from functools import partial

from google.genai import types

from atlas.engine import fallback, turnlock
from atlas.engine.prompt import build_system_prompt
from atlas.integrations.gemini import (
    CHAT_CHAIN,
    failover_reason,
    get_client,
    is_rate_limited,
    is_transient,
    log_failover,
    retry_after_seconds,
)
from atlas.memory import store
from atlas.memory.extract import extract_and_store
from atlas.tools.registry import build_tools

log = logging.getLogger(__name__)

HISTORY_TURNS = 20
# How long an uploaded document stays in front of the model for follow-ups. Long
# enough for a conversation about it, short enough that an unrelated question an
# hour later is not paying for a PDF's worth of tokens on every request.
DOCUMENT_FOLLOW_UP = dt.timedelta(minutes=30)
MAX_REPLY_CHARS = 1400  # far below Telegram's 4096; concision is a requirement
MAX_QUOTA_WAIT = 20.0  # seconds; beyond this a user would rather get an answer back
# Tool failures go back to the model as results and never reach these replies,
# so none of them may blame data sources: that sent the 2026-09-15 diagnosis
# looking at healthy market-data APIs while the model chain was the fault.
FAILURE_REPLY = "Something went wrong on my side just then. Try me again?"
BUSY_REPLY = (
    "I'm being rate limited right now — give me about a minute and ask me again."
)
OVERLOADED_REPLY = (
    "My AI model is overloaded right now — give me about a minute and ask me again."
)
EMPTY_REPLY = "I did not get that — could you say it another way?"
# A turn that already saved memory through these tools has nothing left for the
# background extractor to find, so it does not spend a request looking.
MEMORY_WRITES = frozenset({"remember", "update_profile", "add_to_watchlist"})


@dataclass(frozen=True)
class _Reply:
    """A fallback provider's answer, shaped like the Gemini response _turn reads."""

    text: str


# Strong refs so background tasks are not garbage collected mid-flight.
_BACKGROUND: set[asyncio.Task] = set()


def _to_contents(history: list[dict], text: str, attachments: list[dict] | None):
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
        for m in history
    ]

    parts = [types.Part(text=text)]
    for item in attachments or []:
        if item.get("note"):
            parts.append(types.Part(text=item["note"]))
        if item["kind"] == "file":
            parts.append(
                types.Part(
                    file_data=types.FileData(
                        file_uri=item["uri"], mime_type=item["mime"]
                    )
                )
            )
        elif item["kind"] == "image":
            parts.append(
                types.Part(
                    inline_data=types.Blob(data=item["bytes"], mime_type=item["mime"])
                )
            )
    contents.append(types.Content(role="user", parts=parts))
    return contents


async def _generate(model: str, contents, system_prompt: str, tools: list):
    """Network seam. Tests monkeypatch this."""
    return await get_client().aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=8
            ),
        ),
    )


async def _generate_resilient(
    contents, system_prompt: str, tools: list, second_provider=None
):
    """Try each model in the chain, then a second provider, then wait once.

    Free-tier quota is per model, so a 429 on the preferred model does not mean the
    next one is also exhausted. Transient upstream faults are also worth carrying
    to the next model — a 499 CANCELLED on the first request of a fresh process
    used to abort the whole turn with two untried models still in the chain. So is
    a model Google has retired, which is skipped and never retried: waiting does
    not bring it back. Anything else is a genuine fault: retrying it just wastes
    the user's time.

    `second_provider`, when given, is tried as soon as the chain is spent: a provider
    that answers now beats waiting on one that has just refused three times.
    """
    last: Exception | None = None
    # Models that refused for a reason a short wait can outlast, best first.
    recoverable: list[tuple[str, Exception]] = []

    for model in CHAT_CHAIN:
        try:
            return await _generate(model, contents, system_prompt, tools)
        except Exception as exc:
            last = exc
            reason = failover_reason(exc, model)
            if reason is None:
                raise
            log_failover(log, model, reason, exc)
            if reason != "retired":
                recoverable.append((model, exc))

    if second_provider is not None:
        try:
            answer = await second_provider()
        except Exception as exc:
            log.warning("fallback provider failed too: %s", exc)
        else:
            if answer and answer.strip():
                return _Reply(answer)
            # Empty is a failure, not an answer: fall through to the one retry
            # on the model whose quota window is about to reopen.
            log.warning("fallback provider returned nothing; retrying the chain")

    if not recoverable:
        raise last if last is not None else RuntimeError("no chat model configured")

    model = recoverable[0][0]
    delay = min(retry_after_seconds(recoverable[-1][1]), MAX_QUOTA_WAIT)
    log.warning("no model answered; waiting %.1fs before one final try on %s", delay, model)
    await asyncio.sleep(delay)
    return await _generate(model, contents, system_prompt, tools)


def _called_tools(response) -> set[str]:
    """Names of the tools this turn called. Stored history is text only, so any
    function call in the SDK's record belongs to this turn."""
    names = set()
    for content in getattr(response, "automatic_function_calling_history", None) or []:
        for part in getattr(content, "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call is not None and getattr(call, "name", None):
                names.add(call.name)
    return names


def _load_turn(user_id: int, text: str, want_document: bool):
    """Everything the turn needs, in one hop off the loop.

    Grouped rather than awaited one by one: against a networked Postgres each
    store call is its own round trip plus a commit, and this is the path the
    user is actually waiting on.
    """
    profile = store.profile_snapshot(user_id)
    facts = store.all_facts(user_id)
    history = store.recent_messages(user_id, limit=HISTORY_TURNS)
    document = store.recent_document(user_id, DOCUMENT_FOLLOW_UP) if want_document else None
    # Appended after the read, so this turn is not its own history.
    store.append_message(user_id, "user", text)
    return profile, facts, history, document


async def respond(
    user_id: int, text: str, attachments: list[dict] | None = None
) -> str:
    # One turn at a time per user. The history read and the model append straddle
    # the Gemini await, so overlapping turns from the same person would answer
    # each other's prompts and interleave their rows in the log.
    async with turnlock.user_turn(user_id):
        return await _turn(user_id, text, attachments)


async def _turn(
    user_id: int, text: str, attachments: list[dict] | None = None
) -> str:
    profile, facts, history, document = await asyncio.to_thread(
        _load_turn, user_id, text, not attachments
    )
    system_prompt = build_system_prompt(profile, facts)
    tools = build_tools(user_id)

    # New attachments live in Gemini's Files API, which no other provider can
    # read. A document carried over from earlier does not block the fallback: the
    # question itself is plain text, and an answer without the file beats none.
    groq = None if attachments else partial(
        fallback.respond, history, text, system_prompt, tools
    )
    if document is not None:
        # History is stored as text, so without this the model answers a
        # follow-up about a PDF with no PDF.
        attachments = [
            {
                "kind": "file",
                "uri": document["uri"],
                "mime": document["mime"],
                "note": "(For reference: the document they sent recently, "
                f"{document['name']}, is attached again.)",
            }
        ]

    try:
        response = await _generate_resilient(
            _to_contents(history, text, attachments), system_prompt, tools, groq
        )
    except Exception as exc:
        log.exception("generation failed for user %s", user_id)
        # Quota and overload are self-healing; say so rather than implying a fault.
        if is_rate_limited(exc):
            return BUSY_REPLY
        if is_transient(exc):
            return OVERLOADED_REPLY
        return FAILURE_REPLY

    reply = (getattr(response, "text", "") or "").strip()
    if not reply:
        return EMPTY_REPLY

    if len(reply) > MAX_REPLY_CHARS:
        reply = reply[:MAX_REPLY_CHARS].rsplit(" ", 1)[0] + "…"

    await asyncio.to_thread(store.append_message, user_id, "model", reply)

    if not _called_tools(response) & MEMORY_WRITES:
        # Detached: memory writes must not add latency to the reply.
        task = asyncio.create_task(extract_and_store(user_id, text, reply))
        _BACKGROUND.add(task)
        task.add_done_callback(_BACKGROUND.discard)

    return reply
