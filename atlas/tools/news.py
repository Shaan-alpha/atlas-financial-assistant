"""Live financial news via Gemini search grounding.

Grounding is used rather than a headline API because it returns synthesized answers
with real citations, which is what the accuracy requirement actually needs.
"""

from google.genai import types

from atlas.integrations.gemini import MODEL_GROUNDED, get_client
from atlas.tools.result import err, ok

SOURCE = "Google Search (grounded)"


def _generate_grounded(query: str):
    """Network seam. Tests monkeypatch this."""
    return get_client().models.generate_content(
        model=MODEL_GROUNDED,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        ),
    )


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
