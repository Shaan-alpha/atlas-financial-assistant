import pytest

REQUIRED_ENV = {
    "TELEGRAM_TOKEN": "test-token",
    "GEMINI_API_KEY": "test-gemini",
    "GROQ_API_KEY": "test-groq",
}


@pytest.fixture
def env(monkeypatch):
    """Required settings present, no database."""
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    from atlas.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fresh_db(env, tmp_path, monkeypatch):
    """Point the app at an empty per-test SQLite file and create the schema."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")

    from atlas.config import get_settings
    import atlas.db.session as sess

    get_settings.cache_clear()
    sess.reset_engine()
    sess.init_db()
    yield
    sess.reset_engine()


@pytest.fixture(autouse=True)
def _no_live_extraction(monkeypatch):
    """Background fact extraction runs after every turn. Keep the suite offline."""
    import atlas.memory.extract as extract

    async def _none(user_text, reply):
        return []

    monkeypatch.setattr(extract, "_extract", _none)
