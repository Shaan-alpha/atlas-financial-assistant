from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from atlas.config import get_settings
from atlas.db.models import Base

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
        _engine = create_engine(get_settings().database_url, future=True)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _Session


def init_db() -> None:
    _get_session_factory()
    Base.metadata.create_all(_engine)


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
