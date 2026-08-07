from datetime import datetime, timezone

from atlas.db.models import (
    Alert,
    MemoryFact,
    Message,
    SentSignal,
    User,
    WatchlistItem,
)
from atlas.db.session import session_scope

PROFILE_FIELDS = {"name", "role", "timezone", "briefing_time", "onboarding_state"}


def get_or_create_user(telegram_id: int, name: str | None = None) -> int:
    with session_scope() as s:
        user = s.query(User).filter_by(telegram_id=telegram_id).one_or_none()
        if user is None:
            user = User(telegram_id=telegram_id, name=name)
            s.add(user)
            s.flush()
        return user.id


def set_profile(user_id: int, **fields) -> None:
    unknown = set(fields) - PROFILE_FIELDS
    if unknown:
        raise ValueError(f"Unknown profile fields: {sorted(unknown)}")
    with session_scope() as s:
        user = s.get(User, user_id)
        for key, value in fields.items():
            if value is not None:
                setattr(user, key, value)


def profile_snapshot(user_id: int) -> dict:
    with session_scope() as s:
        user = s.get(User, user_id)
        return {
            "name": user.name,
            "role": user.role,
            "timezone": user.timezone,
            "briefing_time": user.briefing_time,
            "onboarding_state": user.onboarding_state,
        }


def add_fact(user_id: int, fact: str, category: str = "general") -> None:
    """Store a fact, skipping case-insensitive duplicates.

    Reconciliation is intentionally simple: the fact set per user stays small enough
    that exact-match dedupe is sufficient. Semantic dedupe would be ceremony here.
    """
    normalized = fact.strip()
    if not normalized:
        return
    with session_scope() as s:
        existing = (
            s.query(MemoryFact)
            .filter(MemoryFact.user_id == user_id)
            .filter(MemoryFact.fact.ilike(normalized))
            .one_or_none()
        )
        if existing is None:
            s.add(MemoryFact(user_id=user_id, fact=normalized, category=category))


def all_facts(user_id: int) -> list[dict]:
    with session_scope() as s:
        rows = (
            s.query(MemoryFact)
            .filter_by(user_id=user_id)
            .order_by(MemoryFact.updated_at.desc())
            .all()
        )
        return [{"fact": r.fact, "category": r.category} for r in rows]


def forget(user_id: int, needle: str) -> int:
    with session_scope() as s:
        rows = (
            s.query(MemoryFact)
            .filter(MemoryFact.user_id == user_id)
            .filter(MemoryFact.fact.ilike(f"%{needle}%"))
            .all()
        )
        for row in rows:
            s.delete(row)
        return len(rows)


def add_watchlist(user_id: int, symbol: str, company: str | None = None) -> bool:
    """Add a symbol to the watchlist. Returns False if it was already there."""
    ticker = symbol.strip().upper()
    if not ticker:
        return False
    with session_scope() as s:
        existing = (
            s.query(WatchlistItem)
            .filter_by(user_id=user_id, symbol=ticker)
            .one_or_none()
        )
        if existing is not None:
            return False
        s.add(WatchlistItem(user_id=user_id, symbol=ticker, company=company))
        return True


def remove_watchlist(user_id: int, symbol: str) -> bool:
    """Remove a symbol from the watchlist. Returns False if it was not there."""
    ticker = symbol.strip().upper()
    with session_scope() as s:
        row = (
            s.query(WatchlistItem)
            .filter_by(user_id=user_id, symbol=ticker)
            .one_or_none()
        )
        if row is None:
            return False
        s.delete(row)
        return True


def get_watchlist(user_id: int) -> list[dict]:
    with session_scope() as s:
        rows = (
            s.query(WatchlistItem)
            .filter_by(user_id=user_id)
            .order_by(WatchlistItem.id)
            .all()
        )
        return [{"symbol": r.symbol, "company": r.company} for r in rows]


def users_with_briefings() -> list[dict]:
    """Everyone who has asked for a daily briefing, for the scheduler."""
    with session_scope() as s:
        rows = s.query(User).filter(User.briefing_time.isnot(None)).all()
        return [
            {
                "user_id": r.id,
                "telegram_id": r.telegram_id,
                "briefing_time": r.briefing_time,
                "timezone": r.timezone,
            }
            for r in rows
        ]


def filter_unsent(user_id: int, keys: list[str]) -> list[str]:
    """Return only the keys this user has not already been shown."""
    if not keys:
        return []
    with session_scope() as s:
        seen = {
            row.signal_key
            for row in s.query(SentSignal)
            .filter(SentSignal.user_id == user_id)
            .filter(SentSignal.signal_key.in_(keys))
            .all()
        }
    return [k for k in keys if k not in seen]


def mark_sent(user_id: int, keys: list[str]) -> None:
    with session_scope() as s:
        for key in keys:
            s.add(SentSignal(user_id=user_id, signal_key=key))


def create_alert(
    user_id: int, description: str, symbol: str, kind: str, threshold: float
) -> int:
    with session_scope() as s:
        alert = Alert(
            user_id=user_id,
            description=description.strip(),
            symbol=symbol.strip().upper(),
            kind=kind,
            threshold=threshold,
        )
        s.add(alert)
        s.flush()
        return alert.id


def user_alerts(user_id: int) -> list[dict]:
    with session_scope() as s:
        rows = (
            s.query(Alert)
            .filter_by(user_id=user_id, active=True)
            .order_by(Alert.id)
            .all()
        )
        return [
            {
                "id": r.id,
                "description": r.description,
                "symbol": r.symbol,
                "kind": r.kind,
                "threshold": r.threshold,
            }
            for r in rows
        ]


def active_alerts() -> list[dict]:
    """Every armed alert across all users, for the watcher."""
    with session_scope() as s:
        rows = (
            s.query(Alert, User.telegram_id)
            .join(User, Alert.user_id == User.id)
            .filter(Alert.active.is_(True))
            .all()
        )
        return [
            {
                "id": a.id,
                "user_id": a.user_id,
                "telegram_id": tg,
                "description": a.description,
                "symbol": a.symbol,
                "kind": a.kind,
                "threshold": a.threshold,
                "last_fired_at": a.last_fired_at,
            }
            for a, tg in rows
        ]


def mark_alert_fired(alert_id: int, when: datetime | None = None) -> None:
    """Stamp the fire time.

    `when` is passed in by the watcher so evaluation and bookkeeping share one
    clock; taking wall-clock here would let cooldown disagree with the check that
    just fired.
    """
    stamp = when or datetime.now(timezone.utc).replace(tzinfo=None)
    with session_scope() as s:
        alert = s.get(Alert, alert_id)
        if alert is not None:
            alert.last_fired_at = stamp


def cancel_alerts(user_id: int, symbol: str) -> int:
    with session_scope() as s:
        rows = (
            s.query(Alert)
            .filter_by(user_id=user_id, symbol=symbol.strip().upper(), active=True)
            .all()
        )
        for row in rows:
            row.active = False
        return len(rows)


def append_message(user_id: int, role: str, content: str) -> None:
    with session_scope() as s:
        s.add(Message(user_id=user_id, role=role, content=content))


def recent_messages(user_id: int, limit: int = 20) -> list[dict]:
    with session_scope() as s:
        rows = (
            s.query(Message)
            .filter_by(user_id=user_id)
            .order_by(Message.id.desc())
            .limit(limit)
            .all()
        )
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]
