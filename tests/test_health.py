import httpx
import pytest

from atlas.config import _normalize_database_url
from atlas.health import ping, start_health_server


def test_health_server_answers_200():
    server = start_health_server(0)  # port 0 -> OS picks a free port
    try:
        port = server.server_address[1]
        resp = httpx.get(f"http://127.0.0.1:{port}/", timeout=5)
        assert resp.status_code == 200
        assert "atlas" in resp.text
    finally:
        server.shutdown()


def test_ping_reaches_a_live_server():
    server = start_health_server(0)
    try:
        port = server.server_address[1]
        assert ping(f"http://127.0.0.1:{port}/") is True
    finally:
        server.shutdown()


def test_ping_without_a_url_is_a_noop():
    assert ping(None) is False
    assert ping("") is False


def test_ping_swallows_a_dead_target():
    """Keep-alive must never take the bot down with it."""
    assert ping("http://127.0.0.1:9/") is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Hosts hand out postgres:// and postgresql://; SQLAlchemy 2 needs a driver.
        ("postgres://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
        ("postgresql://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
        # Already-qualified and SQLite URLs pass through untouched.
        ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("sqlite:///atlas.db", "sqlite:///atlas.db"),
    ],
)
def test_database_url_is_normalized_for_sqlalchemy(raw, expected):
    assert _normalize_database_url(raw) == expected
