"""Memory tools, bound to a single user by closure.

The model never supplies a user id, so it cannot reach another user's data.
"""

from collections.abc import Callable

from atlas.memory import store
from atlas.tools.result import ok

SOURCE = "atlas-memory"


def make_memory_tools(user_id: int) -> list[Callable]:
    def remember(fact: str, category: str = "general") -> dict:
        """Save a durable fact about the user for future conversations.

        Use for stable preferences, focus areas, and views — not for one-off questions.

        Args:
            fact: The fact to remember, written as a short third-person statement.
            category: One of "focus", "view", "preference", "role", "general".
        """
        store.add_fact(user_id, fact, category)
        return ok({"remembered": fact}, source=SOURCE)

    def recall() -> dict:
        """Return everything currently known about the user.

        Use when the user asks what you know or remember about them, and to ground
        personalized answers.
        """
        return ok(
            {"profile": store.profile_snapshot(user_id), "facts": store.all_facts(user_id)},
            source=SOURCE,
        )

    def forget_about(topic: str) -> dict:
        """Delete remembered facts matching a topic.

        Args:
            topic: Substring to match, for example "EV" or "briefing".
        """
        removed = store.forget(user_id, topic)
        return ok({"removed": removed, "topic": topic}, source=SOURCE)

    return [remember, recall, forget_about]
