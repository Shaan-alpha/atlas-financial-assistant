"""Gemini client, model selection, and rate-limit resilience.

Model IDs verified 2026-08-07. Re-check before relying on them.

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

# Ordered best-first. Each entry is an independent quota bucket.
CHAT_CHAIN = ("gemini-3.6-flash", "gemini-3-flash-preview", "gemini-2.5-flash")
GROUNDED_CHAIN = ("gemini-3-flash-preview", "gemini-3.6-flash", "gemini-2.5-flash")
EXTRACT_CHAIN = ("gemini-3.1-flash-lite", "gemini-2.5-flash-lite")

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


def retry_after_seconds(exc: Exception, default: float = 5.0) -> float:
    """Seconds Gemini asked us to wait, when it says so."""
    match = _RETRY_SECONDS.search(str(exc))
    return float(match.group(1)) if match else default
