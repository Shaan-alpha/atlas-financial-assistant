"""Gemini client and model selection.

Model IDs verified 2026-08-07. Re-check before relying on them.
"""

from functools import lru_cache

from google import genai

from atlas.config import get_settings

MODEL_CHAT = "gemini-3.6-flash"          # agentic conversation + tool calling
MODEL_RESEARCH = "gemini-3.1-pro-preview"  # deep comparison and document reasoning
MODEL_GROUNDED = "gemini-3-flash-preview"  # optimized for search grounding


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client(api_key=get_settings().gemini_api_key)
