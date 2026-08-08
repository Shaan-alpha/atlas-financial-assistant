"""Live financial news via Gemini search grounding.

Grounding is used rather than a headline API because it returns synthesized answers
with real citations, which is what the accuracy requirement actually needs.
"""

import logging

from google.genai import types

from atlas.integrations.gemini import GROUNDED_CHAIN, get_client, is_rate_limited
from atlas.tools.result import err, ok

log = logging.getLogger(__name__)

SOURCE = "Google Search (grounded)"


def _generate_grounded(query: str):
    """Network seam. Tests monkeypatch this.

    Free-tier quota is per model, so an exhausted preview model does not mean the
    next one is also spent. This is the same failover the conversation engine
    uses; without it a single quota wall silently removes live news entirely.
    """
    last: Exception | None = None
    for model in GROUNDED_CHAIN:
        try:
            return get_client().models.generate_content(
                model=model,
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
            )
        except Exception as exc:
            last = exc
            if not is_rate_limited(exc):
                raise
            log.warning("grounded model %s rate limited, trying next", model)
    raise last if last is not None else RuntimeError("no grounded model configured")


def _extract_citations(response) -> list[dict]:
    citations: list[dict] = []
    for candidate in getattr(response, "candidates", []) or []:
        metadata = getattr(candidate, "grounding_metadata", None)
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is not None and getattr(web, "uri", None):
                citations.append({"uri": web.uri, "title": getattr(web, "title", None)})
    return citations


def search_financial_news(query: str) -> dict:
    """Search the live web for financial news and return a cited summary.

    Use for anything time-sensitive: why a stock moved, breaking news, recent
    announcements, analyst activity, macro events.

    Args:
        query: A specific natural-language question, not a bare keyword.
    """
    try:
        response = _generate_grounded(query)
    except Exception:
        return err("search_unavailable", "Live search is not responding right now.")

    summary = (getattr(response, "text", "") or "").strip()
    if not summary:
        return err("no_results", f"No current reporting found for: {query}")

    return ok(
        {"summary": summary, "citations": _extract_citations(response)}, source=SOURCE
    )
