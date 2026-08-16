"""The turn loop.

Gemini's automatic function calling runs the tool cycle; we supply the tools,
the hydrated history, and the system prompt.
"""

import asyncio
import logging

from google.genai import types

from atlas.engine import turnlock
from atlas.engine.prompt import build_system_prompt
from atlas.integrations.gemini import (
    CHAT_CHAIN,
    get_client,
    is_rate_limited,
    is_transient,
    retry_after_seconds,
)
from atlas.memory import store
from atlas.memory.extract import extract_and_store
from atlas.tools.registry import build_tools

log = logging.getLogger(__name__)

HISTORY_TURNS = 20
MAX_REPLY_CHARS = 1400  # far below Telegram's 4096; concision is a requirement
MAX_QUOTA_WAIT = 20.0  # seconds; beyond this a user would rather get an answer back
FAILURE_REPLY = "I hit trouble reaching my data sources just then. Try me again?"
BUSY_REPLY = (
    "I'm being rate limited right now — give me about a minute and ask me again."
)
EMPTY_REPLY = "I did not get that — could you say it another way?"

# Strong refs so background tasks are not garbage collected mid-flight.
_BACKGROUND: set[asyncio.Task] = set()


def _to_contents(history: list[dict], text: str, attachments: list[dict] | None):
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
        for m in history
    ]

    parts = [types.Part(text=text)]
    for item in attachments or []:
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


async def _generate_resilient(contents, system_prompt: str, tools: list):
    """Try each model in the chain, then wait out the quota window once.

    Free-tier quota is per model, so a 429 on the preferred model does not mean the
    next one is also exhausted. Transient upstream faults are also worth carrying
    to the next model — a 499 CANCELLED on the first request of a fresh process
    used to abort the whole turn with two untried models still in the chain.
    Anything else is a genuine fault: retrying it just wastes the user's time.
    """
    last: Exception | None = None

    for model in CHAT_CHAIN:
        try:
            return await _generate(model, contents, system_prompt, tools)
        except Exception as exc:
            last = exc
            if is_rate_limited(exc):
                log.warning("%s rate limited, trying next model", model)
            elif is_transient(exc):
                log.warning("%s failed transiently (%s), trying next model", model, exc)
            else:
                raise

    delay = min(retry_after_seconds(last) if last else 5.0, MAX_QUOTA_WAIT)
    log.warning("all models rate limited; waiting %.1fs before one final attempt", delay)
    await asyncio.sleep(delay)
    return await _generate(CHAT_CHAIN[0], contents, system_prompt, tools)


def _load_turn(user_id: int, text: str):
    """Everything the turn needs, in one hop off the loop.

    Grouped rather than awaited one by one: against a networked Postgres each
    store call is its own round trip plus a commit, and this is the path the
    user is actually waiting on.
    """
    profile = store.profile_snapshot(user_id)
    facts = store.all_facts(user_id)
    history = store.recent_messages(user_id, limit=HISTORY_TURNS)
    # Appended after the read, so this turn is not its own history.
    store.append_message(user_id, "user", text)
    return profile, facts, history


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
    profile, facts, history = await asyncio.to_thread(_load_turn, user_id, text)

    try:
        response = await _generate_resilient(
            _to_contents(history, text, attachments),
            build_system_prompt(profile, facts),
            build_tools(user_id),
        )
    except Exception as exc:
        log.exception("generation failed for user %s", user_id)
        # Quota is transient and self-healing; say so rather than implying a fault.
        return BUSY_REPLY if is_rate_limited(exc) else FAILURE_REPLY

    reply = (getattr(response, "text", "") or "").strip()
    if not reply:
        return EMPTY_REPLY

    if len(reply) > MAX_REPLY_CHARS:
        reply = reply[:MAX_REPLY_CHARS].rsplit(" ", 1)[0] + "…"

    await asyncio.to_thread(store.append_message, user_id, "model", reply)

    # Detached: memory writes must not add latency to the reply.
    task = asyncio.create_task(extract_and_store(user_id, text, reply))
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)

    return reply
