"""Background extraction of durable facts from a completed turn."""

import json
import logging
import re

from google.genai import types

from atlas.integrations.gemini import EXTRACT_CHAIN, get_client, is_rate_limited
from atlas.memory import store

log = logging.getLogger(__name__)

# Durable facts almost always arrive in the first person: "I cover semis",
# "my fund is long NVDA", "we're bearish on EV". Gating on that skips the model
# call entirely for the common case ("nvda price?"), which matters because
# free-tier quota is only 5 requests per minute per model.
_FIRST_PERSON = re.compile(r"\b(i|i'm|im|my|mine|me|we|we're|our|us)\b", re.IGNORECASE)
_LONG_ENOUGH = 80


def looks_durable(user_text: str) -> bool:
    """Cheap pre-filter deciding whether a turn is worth an extraction call."""
    text = (user_text or "").strip()
    if not text:
        return False
    return bool(_FIRST_PERSON.search(text)) or len(text) >= _LONG_ENOUGH

INSTRUCTION = """\
Extract only DURABLE facts about the user from this exchange — things still true next
month. Their role, coverage areas, holdings, investment views, and stated preferences.

Ignore one-off questions. "What's Apple trading at" reveals nothing durable.

Return a JSON array. Each item has "fact" (short, third person) and "category"
(one of: focus, view, preference, role, general). Return [] if nothing durable appeared.
"""


async def _extract(user_text: str, reply: str) -> list[dict]:
    """Network seam. Tests monkeypatch this."""
    last: Exception | None = None
    for model in EXTRACT_CHAIN:
        try:
            response = await get_client().aio.models.generate_content(
                model=model,
                contents=f"User said: {user_text}\n\nAssistant replied: {reply}",
                config=types.GenerateContentConfig(
                    system_instruction=INSTRUCTION,
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text or "[]")
        except Exception as exc:
            last = exc
            if not is_rate_limited(exc):
                raise
    raise last if last is not None else RuntimeError("no extraction model configured")


async def extract_and_store(user_id: int, user_text: str, reply: str) -> int:
    """Extract durable facts and persist them. Never raises — this runs detached."""
    if not looks_durable(user_text):
        return 0
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
