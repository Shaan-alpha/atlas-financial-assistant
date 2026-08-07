"""Voice transcription via Groq Whisper."""

import logging
from functools import lru_cache

from groq import Groq

from atlas.config import get_settings

log = logging.getLogger(__name__)

WHISPER_MODEL = "whisper-large-v3-turbo"


@lru_cache(maxsize=1)
def _client() -> Groq:
    return Groq(api_key=get_settings().groq_api_key)


def _call_whisper(audio: bytes, filename: str) -> str:
    """Network seam. Tests monkeypatch this."""
    result = _client().audio.transcriptions.create(
        file=(filename, audio), model=WHISPER_MODEL
    )
    return result.text


def transcribe(audio: bytes, filename: str = "voice.ogg") -> str | None:
    """Transcribe a voice note. Returns None when nothing usable came back."""
    try:
        text = _call_whisper(audio, filename)
    except Exception:
        log.exception("transcription failed")
        return None
    text = (text or "").strip()
    return text or None
