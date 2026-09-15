"""Gemini client, model selection, and rate-limit resilience.

Model IDs verified live from production 2026-09-16. Re-check before relying on
them: on that date the whole 2.5 family (flash, flash-lite, pro) answered 404 for
this key, which had silently shortened every chain by one.

Free-tier quota is enforced per project PER MODEL (quotaId
`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`, limit 5 RPM). Automatic
function calling spends one request per tool round-trip, so a single question can
exhaust one model's budget. Falling back to a different model therefore buys a
fresh bucket rather than hitting the same wall — which is why these are chains,
not single names.
"""

import logging
import re
from functools import lru_cache

from google import genai

from atlas.config import get_settings

log = logging.getLogger(__name__)

# Ordered best-first. Each entry is an independent quota bucket. Stable releases
# sit ahead of previews, which Google retires sooner.
CHAT_CHAIN = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash-preview")
# Kept to two: grounding quota is shared across every Gemini 3 model, so a third
# entry would mostly spend another request confirming the same refusal.
GROUNDED_CHAIN = ("gemini-3-flash-preview", "gemini-3.6-flash")
EXTRACT_CHAIN = ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite")

# Kept for readability at call sites that only need the preferred model.
MODEL_CHAT = CHAT_CHAIN[0]
MODEL_GROUNDED = GROUNDED_CHAIN[0]

_RETRY_SECONDS = re.compile(r"retryDelay['\"]?:\s*['\"]?(\d+)")


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client(api_key=get_settings().gemini_api_key)


def is_rate_limited(exc: Exception) -> bool:
    """True when Gemini refused for quota reasons rather than a real fault."""
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


# Transient upstream faults. Unlike a malformed request, these say nothing about
# the call itself, so the next model in the chain is worth trying. 499 CANCELLED
# is the one seen in practice: it turns up on the first request a fresh process
# makes, and aborting on it costs the user their whole turn while two untried
# models sit in the chain.
_TRANSIENT_CODES = frozenset({499, 500, 502, 503, 504})
_TRANSIENT_STATUSES = ("CANCELLED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED")


def is_transient(exc: Exception) -> bool:
    """True when Gemini failed for a reason a retry could plausibly survive.

    Deliberately keyed on the API error code and status rather than on loose
    substrings: matching "503" anywhere in a message would swallow genuine
    faults that happen to mention it.
    """
    if getattr(exc, "code", None) in _TRANSIENT_CODES:
        return True
    text = str(exc)
    return any(status in text for status in _TRANSIENT_STATUSES)


def is_model_unavailable(exc: Exception, model: str) -> bool:
    """True when Gemini no longer serves `model` at all.

    Google retires models on its own schedule, and the retired one answers 404
    NOT_FOUND naming itself: "This model models/gemini-2.5-flash is no longer
    available to new users". That is a fact about the model, not the request, so
    the next model is worth trying. Keyed on the model's own name so a 404 about
    something else in the request (an expired file upload) still counts as a
    genuine fault.
    """
    text = str(exc)
    if getattr(exc, "code", None) != 404 and "NOT_FOUND" not in text:
        return False
    return re.search(rf"models/{re.escape(model)}(?![\w.-])", text) is not None


def failover_reason(exc: Exception, model: str) -> str | None:
    """Why the next model in a chain is worth trying, or None for a genuine fault.

    One rule for every chain. News, extraction, and the briefing gate each used to
    fail over on quota alone, so a 503 or a retired model ended them outright.
    """
    if is_rate_limited(exc):
        return "rate limited"
    if is_model_unavailable(exc, model):
        return "retired"
    if is_transient(exc):
        return "transient"
    return None


def log_failover(logger: logging.Logger, model: str, reason: str, exc: Exception) -> None:
    """A retired model needs a code change, so it is an error, not a routine hop."""
    if reason == "retired":
        logger.error("%s is no longer served; remove it from its chain (%s)", model, exc)
    else:
        logger.warning("%s %s (%s), trying next model", model, reason, exc)


def retry_after_seconds(exc: Exception, default: float = 5.0) -> float:
    """Seconds Gemini asked us to wait, when it says so."""
    match = _RETRY_SECONDS.search(str(exc))
    return float(match.group(1)) if match else default
