"""The turn loop.

Gemini's automatic function calling runs the tool cycle; we supply the tools,
the hydrated history, and the system prompt.
"""

import asyncio
import logging

from google.genai import types

from atlas.engine.prompt import build_system_prompt
from atlas.integrations.gemini import MODEL_CHAT, get_client
from atlas.memory import store
from atlas.memory.extract import extract_and_store
from atlas.tools.registry import build_tools

log = logging.getLogger(__name__)

HISTORY_TURNS = 20
MAX_REPLY_CHARS = 1400  # far below Telegram's 4096; concision is a requirement
FAILURE_REPLY = "I hit trouble reaching my data sources just then. Try me again?"
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


async def respond(
    user_id: int, text: str, attachments: list[dict] | None = None
) -> str:
    profile = store.profile_snapshot(user_id)
    facts = store.all_facts(user_id)
    history = store.recent_messages(user_id, limit=HISTORY_TURNS)

    store.append_message(user_id, "user", text)

    try:
        response = await _generate(
            MODEL_CHAT,
            _to_contents(history, text, attachments),
            build_system_prompt(profile, facts),
            build_tools(user_id),
        )
    except Exception:
        log.exception("generation failed for user %s", user_id)
        return FAILURE_REPLY

    reply = (getattr(response, "text", "") or "").strip()
    if not reply:
        return EMPTY_REPLY

    if len(reply) > MAX_REPLY_CHARS:
        reply = reply[:MAX_REPLY_CHARS].rsplit(" ", 1)[0] + "…"

    store.append_message(user_id, "model", reply)

    # Detached: memory writes must not add latency to the reply.
    task = asyncio.create_task(extract_and_store(user_id, text, reply))
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)

    return reply
