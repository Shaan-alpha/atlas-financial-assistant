"""Background extraction of durable facts from a completed turn."""

import json
import logging

from google.genai import types

from atlas.integrations.gemini import get_client
from atlas.memory import store

log = logging.getLogger(__name__)

MODEL = "gemini-3.1-flash-lite"  # cheap: this runs after every turn

INSTRUCTION = """\
Extract only DURABLE facts about the user from this exchange — things still true next
month. Their role, coverage areas, holdings, investment views, and stated preferences.

Ignore one-off questions. "What's Apple trading at" reveals nothing durable.

Return a JSON array. Each item has "fact" (short, third person) and "category"
(one of: focus, view, preference, role, general). Return [] if nothing durable appeared.
"""


async def _extract(user_text: str, reply: str) -> list[dict]:
    """Network seam. Tests monkeypatch this."""
    response = await get_client().aio.models.generate_content(
        model=MODEL,
        contents=f"User said: {user_text}\n\nAssistant replied: {reply}",
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTION,
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text or "[]")


async def extract_and_store(user_id: int, user_text: str, reply: str) -> int:
    """Extract durable facts and persist them. Never raises — this runs detached."""
    try:
        items = await _extract(user_text, reply)
    except Exception:
        log.exception("fact extraction failed for user %s", user_id)
        return 0

    stored = 0
    for item in items or []:
        fact = (item.get("fact") or "").strip() if isinstance(item, dict) else ""
        if not fact:
            continue
        store.add_fact(user_id, fact, item.get("category", "general"))
        stored += 1
    return stored
