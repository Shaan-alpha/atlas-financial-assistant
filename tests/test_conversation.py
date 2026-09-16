import asyncio

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
    assert captured["tool_count"] == 20


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

    assert reply == conversation.FAILURE_REPLY
    # The failed turn must not be persisted as a model reply.
    assert [m["role"] for m in store.recent_messages(uid)] == ["user"]


async def test_empty_model_text_does_not_send_blank_message(monkeypatch):
    async def _blank(model, contents, system_prompt, tools):
        return _Resp("")

    monkeypatch.setattr(conversation, "_generate", _blank)
    uid = store.get_or_create_user(4, "Shaan")

    reply = await conversation.respond(uid, "hello")

    assert reply.strip()


async def test_a_recent_upload_stays_attached_for_follow_ups(monkeypatch):
    """History is stored as text, so a follow-up about a PDF used to reach the
    model with no PDF at all."""
    import datetime as dt

    seen = []

    async def _fake(model, contents, system_prompt, tools):
        seen.append(list(contents[-1].parts))
        return _Resp("Segment margins widened.")

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(40, "Shaan")
    store.add_document(uid, "https://files/abc", "results.pdf", "application/pdf")

    await conversation.respond(uid, "and the segment margins?")

    uris = [p.file_data.file_uri for p in seen[0] if p.file_data]
    assert uris == ["https://files/abc"]
    assert any("results.pdf" in (p.text or "") for p in seen[0])

    # Past the window, the document is no longer paid for on every request.
    monkeypatch.setattr(conversation, "DOCUMENT_FOLLOW_UP", dt.timedelta(seconds=-1))
    await conversation.respond(uid, "what is NVDA at?")
    assert not [p for p in seen[1] if p.file_data]



async def test_a_turn_that_saved_memory_skips_background_extraction(monkeypatch):
    """remember or add_to_watchlist already stored what the turn revealed; the
    extractor then spent another request finding the same fact."""
    from types import SimpleNamespace

    import atlas.memory.extract as extract

    spent = []

    async def _extract(user_text, reply):
        spent.append(user_text)
        return []

    class _WithCalls:
        text = "Noted, I'll watch NVDA for you."
        automatic_function_calling_history = [
            SimpleNamespace(parts=[SimpleNamespace(function_call=SimpleNamespace(name="add_to_watchlist"))])
        ]

    async def _fake(model, contents, system_prompt, tools):
        return _WithCalls()

    monkeypatch.setattr(extract, "_extract", _extract)
    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(41, "Shaan")

    await conversation.respond(uid, "I hold a big position in NVDA")
    await asyncio.sleep(0)

    assert spent == []
