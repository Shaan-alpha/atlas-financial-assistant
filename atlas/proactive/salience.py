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

import asyncio
import json
import logging

from google.genai import types

from atlas.integrations.gemini import EXTRACT_CHAIN, get_client, is_rate_limited

log = logging.getLogger(__name__)

SILENT = {"send": False, "brief": "", "used_keys": []}

PUSH_PREAMBLE = """\
You decide whether a financial assistant should interrupt its user this morning,
and if so, you write the briefing.

BE STRICT. This person is busy and did not ask for this. Only send when something
genuinely changes what they might think or do today. Routine drift, small moves,
and stale items are NOT worth an interruption. Silence is a good outcome here and
costs you nothing.

If nothing clears that bar, return {"send": false, "brief": "", "used_keys": []}.
"""

PULL_PREAMBLE = """\
The user has just ASKED what is happening with the names they follow. Answer them.

The bar is different from an unprompted alert: they want to know, so report
anything genuinely notable rather than only what would justify interrupting them.
A meaningful move on a name they follow qualifies.

Return {"send": false, ...} only when truly nothing has moved and there is no
news at all — never to avoid bothering someone who asked.
"""

_BODY = """\
If something is worth reporting, write the briefing:
- Open with the single most important thing. No greeting, no preamble.
- 6 lines maximum. One line per item.
- EVERY item says why it matters to THIS user, given their role and interests.
- Telegram markdown only: *bold* with single asterisks. No headings or tables.
- Use only numbers present in the data. Never invent or estimate a figure.
- Mention index levels only if they add to a signal you are already reporting.

Return JSON: {"send": bool, "brief": string, "used_keys": [signal keys used]}
"""

PUSH_INSTRUCTION = PUSH_PREAMBLE + "\n" + _BODY
PULL_INSTRUCTION = PULL_PREAMBLE + "\n" + _BODY


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


def _decide_sync(prompt: str, instruction: str = PUSH_INSTRUCTION) -> dict:
    """The actual model call. Synchronous so it can serve both the scheduled
    briefing and the on-demand tool, which runs inside a sync tool call."""
    last: Exception | None = None
    for model in EXTRACT_CHAIN:
        try:
            response = get_client().models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=instruction,
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text or "{}")
        except Exception as exc:
            last = exc
            if not is_rate_limited(exc):
                raise
    raise last if last is not None else RuntimeError("no salience model configured")


async def _decide(prompt: str, instruction: str = PUSH_INSTRUCTION) -> dict:
    """Async wrapper. Off-thread so the scheduled path never blocks the loop."""
    return await asyncio.to_thread(_decide_sync, prompt, instruction)


def _verdict_from(raw: dict, signals: list[dict]) -> dict:
    """Shared interpretation of the model's answer, used by both paths."""
    brief = (raw.get("brief") or "").strip()
    if not raw.get("send") or not brief:
        return SILENT
    return {
        "send": True,
        "brief": brief,
        "used_keys": raw.get("used_keys") or [s["key"] for s in signals],
    }


def decide_sync(
    profile: dict, facts: list[dict], signals: list[dict], context: dict | None
) -> dict:
    """Synchronous gate for the on-demand tool.

    Uses the pull instruction: the user asked, so the bar is "is this notable"
    rather than "does this justify interrupting them".
    """
    if not signals:
        return SILENT
    try:
        raw = _decide_sync(
            _payload(profile, facts, signals, context), PULL_INSTRUCTION
        )
    except Exception:
        log.exception("salience gate failed; staying silent")
        return SILENT
    return _verdict_from(raw, signals)


async def decide(
    profile: dict, facts: list[dict], signals: list[dict], context: dict | None
) -> dict:
    """Return {"send": bool, "brief": str, "used_keys": list[str]}."""
    # Nothing happened. Do not spend a request confirming that.
    if not signals:
        return SILENT

    try:
        raw = await _decide(_payload(profile, facts, signals, context))
    except Exception:
        log.exception("salience gate failed; staying silent")
        return SILENT

    return _verdict_from(raw, signals)
