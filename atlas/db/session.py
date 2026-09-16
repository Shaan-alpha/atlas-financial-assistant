import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from atlas.config import get_settings
from atlas.db.models import Base

log = logging.getLogger(__name__)

_engine = None
_Session: sessionmaker | None = None


def reset_engine() -> None:
    """Drop cached engine. Tests use this after repointing DATABASE_URL."""
    global _engine, _Session
    _engine = None
    _Session = None


def _get_session_factory() -> sessionmaker:
    global _engine, _Session
    if _Session is None:
        # pre_ping: a pooled connection outlives a Postgres restart (an apt
        # upgrade, the nightly backup's neighbour, a VM reboot) and the first
        # query on it would fail. Checking costs one trivial round trip.
        _engine = create_engine(get_settings().database_url, future=True, pool_pre_ping=True)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _Session


def upgrade_schema(conn) -> None:
    """In-place fixes create_all cannot make, since it never alters a table.

    There is no migration tool on purpose: one small schema, one production
    database. Each step checks before it acts, so running it on every start is
    safe and a fresh database needs none of them.
    """
    if conn.dialect.name != "postgresql":
        return  # SQLite's INTEGER is already 64-bit.
    data_type = conn.execute(
        text(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'users' AND column_name = 'telegram_id'"
        )
    ).scalar()
    if data_type == "integer":
        log.warning("widening users.telegram_id to BIGINT")
        conn.execute(text("ALTER TABLE users ALTER COLUMN telegram_id TYPE BIGINT"))


def init_db() -> None:
    _get_session_factory()
    Base.metadata.create_all(_engine)
    with _engine.begin() as conn:
        upgrade_schema(conn)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
