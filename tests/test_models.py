import pytest

from atlas.db.models import MemoryFact, User
from atlas.db.session import session_scope

pytestmark = pytest.mark.usefixtures("fresh_db")


def test_user_and_fact_round_trip():
    with session_scope() as s:
        s.add(User(telegram_id=42, name="Shaan", role="analyst"))

    with session_scope() as s:
        user = s.query(User).filter_by(telegram_id=42).one()
        assert user.role == "analyst"
        assert user.onboarding_state == "new"
        s.add(MemoryFact(user_id=user.id, fact="covers semis", category="focus"))

    with session_scope() as s:
        facts = s.query(MemoryFact).all()
        assert [f.fact for f in facts] == ["covers semis"]
