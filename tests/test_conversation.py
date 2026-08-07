import pytest

import atlas.engine.conversation as conversation
from atlas.memory import store

pytestmark = pytest.mark.usefixtures("fresh_db")


class _Resp:
    def __init__(self, text):
        self.text = text


async def test_respond_persists_both_sides_of_the_turn(monkeypatch):
    captured = {}

    async def _fake(model, contents, system_prompt, tools):
        captured["system_prompt"] = system_prompt
        captured["tool_count"] = len(tools)
        return _Resp("Apple trades at 231.40, up 1.5% today.")

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(1, "Shaan")

    reply = await conversation.respond(uid, "how is apple doing")

    assert "231.40" in reply
    history = store.recent_messages(uid)
    assert [m["role"] for m in history] == ["user", "model"]
    assert captured["tool_count"] == 10


async def test_history_is_passed_to_the_model(monkeypatch):
    seen = {}

    async def _fake(model, contents, system_prompt, tools):
        seen["contents"] = contents
        return _Resp("ok")

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(2, "Shaan")
    store.append_message(uid, "user", "earlier question")
    store.append_message(uid, "model", "earlier answer")

    await conversation.respond(uid, "follow up")

    texts = [part.text for c in seen["contents"] for part in c.parts]
    assert "earlier question" in texts
    assert "follow up" in texts


async def test_model_failure_returns_honest_message(monkeypatch):
    async def _boom(model, contents, system_prompt, tools):
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(conversation, "_generate", _boom)
    uid = store.get_or_create_user(3, "Shaan")

    reply = await conversation.respond(uid, "hello")

    assert "trouble" in reply.lower()
    # The failed turn must not be persisted as a model reply.
    assert [m["role"] for m in store.recent_messages(uid)] == ["user"]


async def test_empty_model_text_does_not_send_blank_message(monkeypatch):
    async def _blank(model, contents, system_prompt, tools):
        return _Resp("")

    monkeypatch.setattr(conversation, "_generate", _blank)
    uid = store.get_or_create_user(4, "Shaan")

    reply = await conversation.respond(uid, "hello")

    assert reply.strip()
