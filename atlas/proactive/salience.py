"""The salience gate.

The brief is explicit: "If there is nothing important to share, the assistant
should remain silent rather than sending unnecessary notifications."

So silence is enforced control flow, not a prompt suggestion:

  - No signals at all short-circuits before any model call.
  - The model returns send=true/false, and a false answer sends nothing.
  - A model failure is treated as "stay silent". A briefing nobody asked for is
    worse than a briefing that never arrives.

Deciding and composing happen in ONE call rather than two. Free-tier quota is 5
requests per minute per model, and a second call per user per morning buys
nothing the first cannot do.
"""

import json
import logging

from google.genai import types

from atlas.integrations.gemini import EXTRACT_CHAIN, get_client, is_rate_limited

log = logging.getLogger(__name__)

INSTRUCTION = """\
You decide whether a financial assistant should interrupt its user this morning,
and if so, you write the briefing.

BE STRICT. This person is busy. Only send when something genuinely changes what
they might think or do today. Routine drift, small moves, and stale items are
NOT worth an interruption. Silence is a good outcome and costs you nothing.

If nothing clears that bar, return {"send": false, "brief": "", "used_keys": []}.

If something does, write the briefing:
- Open with the single most important thing. No greeting, no preamble.
- 6 lines maximum. One line per item.
- EVERY item says why it matters to THIS user, given their role and interests.
- Telegram markdown only: *bold* with single asterisks. No headings or tables.
- Use only numbers present in the data. Never invent or estimate a figure.
- Mention index levels only if they add to a signal you are already reporting.

Return JSON: {"send": bool, "brief": string, "used_keys": [signal keys used]}
"""


def _payload(profile: dict, facts: list[dict], signals: list[dict], context) -> str:
    return json.dumps(
        {
            "user": {
                "role": profile.get("role"),
                "timezone": profile.get("timezone"),
                "known_facts": [f["fact"] for f in facts],
            },
            "signals": signals,
            "market_context": context,
        },
        default=str,
    )


async def _decide(prompt: str) -> dict:
    last: Exception | None = None
    for model in EXTRACT_CHAIN:
        try:
            response = await get_client().aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=INSTRUCTION,
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text or "{}")
        except Exception as exc:
            last = exc
            if not is_rate_limited(exc):
                raise
    raise last if last is not None else RuntimeError("no salience model configured")


async def decide(
    profile: dict, facts: list[dict], signals: list[dict], context: dict | None
) -> dict:
    """Return {"send": bool, "brief": str, "used_keys": list[str]}."""
    silent = {"send": False, "brief": "", "used_keys": []}

    # Nothing happened. Do not spend a request confirming that.
    if not signals:
        return silent

    try:
        verdict = await _decide(_payload(profile, facts, signals, context))
    except Exception:
        log.exception("salience gate failed; staying silent")
        return silent

    brief = (verdict.get("brief") or "").strip()
    if not verdict.get("send") or not brief:
        return silent

    return {
        "send": True,
        "brief": brief,
        "used_keys": verdict.get("used_keys") or [s["key"] for s in signals],
    }
