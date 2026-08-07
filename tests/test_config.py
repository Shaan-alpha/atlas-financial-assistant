import pytest

from atlas.config import get_settings


def test_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("GEMINI_API_KEY", "gem")
    monkeypatch.setenv("GROQ_API_KEY", "grq")
    get_settings.cache_clear()

    s = get_settings()

    assert s.telegram_token == "tok"
    assert s.gemini_api_key == "gem"
    assert s.database_url == "sqlite:///atlas.db"


def test_missing_required_setting_raises(monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gem")
    monkeypatch.setenv("GROQ_API_KEY", "grq")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="TELEGRAM_TOKEN"):
        get_settings()
