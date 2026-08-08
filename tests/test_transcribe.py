import pytest

import atlas.integrations.groq as groq

pytestmark = pytest.mark.usefixtures("env")


def test_returns_trimmed_transcript(monkeypatch):
    monkeypatch.setattr(
        groq, "_call_whisper", lambda a, f, h: "  compare msft and googl  "
    )

    assert groq.transcribe(b"audio") == "compare msft and googl"


def test_blank_transcript_becomes_none(monkeypatch):
    monkeypatch.setattr(groq, "_call_whisper", lambda a, f, h: "   ")

    assert groq.transcribe(b"audio") is None


def test_upstream_failure_becomes_none(monkeypatch):
    def _boom(a, f, h):
        raise RuntimeError("groq down")

    monkeypatch.setattr(groq, "_call_whisper", _boom)

    assert groq.transcribe(b"audio") is None


# ------------------------------------------------------- ticker recognition
#
# Real failure this guards: a spoken "AMD" transcribed as "EMD", after which the
# bot answered confidently about an entirely different security.


def test_hint_names_common_tickers():
    hint = groq.build_hint()

    assert "AMD" in hint
    assert "NVDA" in hint
    assert "stock markets" in hint


def test_watchlist_symbols_lead_the_hint():
    """The user's own holdings are the strongest prior for what they will say."""
    hint = groq.build_hint(["PLTR", "SMCI"])

    assert hint.index("PLTR") < hint.index("AMD")
    assert "SMCI" in hint


def test_hint_is_passed_through_to_whisper(monkeypatch):
    seen = {}

    def _capture(audio, filename, hint):
        seen["hint"] = hint
        return "what is amd doing"

    monkeypatch.setattr(groq, "_call_whisper", _capture)

    groq.transcribe(b"audio", "voice.ogg", ["PLTR"])

    assert "PLTR" in seen["hint"]
