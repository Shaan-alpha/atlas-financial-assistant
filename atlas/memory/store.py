from atlas.db.models import MemoryFact, Message, User
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
