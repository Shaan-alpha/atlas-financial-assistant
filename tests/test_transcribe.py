import pytest

import atlas.integrations.groq as groq

pytestmark = pytest.mark.usefixtures("env")


def test_returns_trimmed_transcript(monkeypatch):
    monkeypatch.setattr(groq, "_call_whisper", lambda a, f: "  compare msft and googl  ")

    assert groq.transcribe(b"audio") == "compare msft and googl"


def test_blank_transcript_becomes_none(monkeypatch):
    monkeypatch.setattr(groq, "_call_whisper", lambda a, f: "   ")

    assert groq.transcribe(b"audio") is None


def test_upstream_failure_becomes_none(monkeypatch):
    def _boom(a, f):
        raise RuntimeError("groq down")

    monkeypatch.setattr(groq, "_call_whisper", _boom)

    assert groq.transcribe(b"audio") is None
