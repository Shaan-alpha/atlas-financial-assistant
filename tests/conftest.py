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


@pytest.fixture(autouse=True)
def _no_live_news(monkeypatch):
    """News feeds are plain HTTP, reachable from anything that gathers a briefing.
    Offline by default; tests/test_news.py replaces the seam with canned feeds."""
    import atlas.tools.news as news

    def _offline(url, params):
        raise ConnectionError("news feeds are offline in tests")

    monkeypatch.setattr(news, "_http_get", _offline)


@pytest.fixture(autouse=True)
def _no_live_fallback(monkeypatch):
    """The Groq fallback runs whenever a test exhausts the Gemini chain. Keep it
    offline, and failing, so those tests see the Gemini-only behaviour they assert
    unless they opt in by replacing these seams themselves."""
    import atlas.engine.fallback as fallback

    async def _offline(model, messages, tools):
        raise ConnectionError("groq is offline in tests")

    def _offline_json(model, system, prompt):
        raise ConnectionError("groq is offline in tests")

    monkeypatch.setattr(fallback, "_complete", _offline)
    monkeypatch.setattr(fallback, "_complete_json", _offline_json)


@pytest.fixture(autouse=True)
def _symbols_are_quotable(monkeypatch):
    """create_alert checks a symbol can be quoted before arming it. Offline, treat
    every symbol as quotable; tests of the refusal replace this themselves."""
    import atlas.tools.memory_tools as memory_tools

    monkeypatch.setattr(memory_tools, "_symbol_is_quotable", lambda symbol: True)
