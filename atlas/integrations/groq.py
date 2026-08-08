"""Voice transcription via Groq Whisper."""

import logging
from functools import lru_cache

from groq import Groq

from atlas.config import get_settings

log = logging.getLogger(__name__)

WHISPER_MODEL = "whisper-large-v3-turbo"

# Spoken tickers are short, unfamiliar letter strings — exactly what speech
# recognition guesses at. "AMD" came back as "EMD" and the bot answered
# confidently about the wrong security. Whisper accepts a prompt to bias its
# vocabulary, so we hand it the terms this user is actually likely to say.
COMMON_TICKERS = (
    "AMD, NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA, INTC, TSM, AVGO, QCOM, "
    "MU, ARM, SMCI, NFLX, JPM, NIFTY, SENSEX, BANKNIFTY, RELIANCE, TCS, INFY"
)


@lru_cache(maxsize=1)
def _client() -> Groq:
    return Groq(api_key=get_settings().groq_api_key)


def build_hint(watchlist_symbols: list[str] | None = None) -> str:
    """Bias transcription toward finance vocabulary and this user's tickers."""
    terms = COMMON_TICKERS
    if watchlist_symbols:
        # The user's own watchlist first: it is the strongest prior available.
        terms = ", ".join(watchlist_symbols) + ", " + terms
    return (
        "A conversation about stock markets and company financials. "
        f"Ticker symbols that may be spoken: {terms}."
    )


def _call_whisper(audio: bytes, filename: str, hint: str) -> str:
    """Network seam. Tests monkeypatch this."""
    result = _client().audio.transcriptions.create(
        file=(filename, audio), model=WHISPER_MODEL, prompt=hint
    )
    return result.text


def transcribe(
    audio: bytes,
    filename: str = "voice.ogg",
    watchlist_symbols: list[str] | None = None,
) -> str | None:
    """Transcribe a voice note. Returns None when nothing usable came back."""
    try:
        text = _call_whisper(audio, filename, build_hint(watchlist_symbols))
    except Exception:
        log.exception("transcription failed")
        return None
    text = (text or "").strip()
    return text or None
