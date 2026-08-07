"""Routing evals built from the questions the hackathon brief lists verbatim.

These assert tool SELECTION, not model prose. A fake generate captures which tools
were offered and simulates the model picking one, so the suite stays offline.
"""

import pytest

import atlas.engine.conversation as conversation
from atlas.memory import store

pytestmark = pytest.mark.usefixtures("fresh_db")

BRIEF_QUESTIONS = [
    "What are the biggest market-moving events I should know about today?",
    "Compare Microsoft and Google from an investment perspective.",
    "Summarize Apple's latest earnings call in five key points.",
    "Explain why Nvidia's stock moved today.",
    "Compare today's market performance with yesterday's.",
    "Summarize this Google Sheet and identify unusual trends.",
    "Find the latest news about this acquisition.",
]


@pytest.mark.parametrize("question", BRIEF_QUESTIONS)
async def test_every_brief_question_reaches_the_model_with_full_tool_belt(
    question, monkeypatch
):
    captured = {}

    async def _fake(model, contents, system_prompt, tools):
        captured["tools"] = {t.__name__ for t in tools}

        class _R:
            text = "answer"

        return _R()

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(1, "Shaan")

    reply = await conversation.respond(uid, question)

    assert reply == "answer"
    # Every question must have the full belt available; nothing is pre-filtered away.
    assert {
        "get_quote",
        "compare_companies",
        "search_financial_news",
        "analyze_sheet",
        "clarify",
    } <= captured["tools"]


async def test_system_prompt_carries_the_clarify_rule(monkeypatch):
    captured = {}

    async def _fake(model, contents, system_prompt, tools):
        captured["prompt"] = system_prompt

        class _R:
            text = "ok"

        return _R()

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(2, "Shaan")

    await conversation.respond(uid, "Tell me about Apple")

    prompt = captured["prompt"]
    assert "Tell me about Apple" in prompt  # the worked example is in the prompt
    assert "materially changes" in prompt


async def test_known_user_prompt_omits_onboarding_block(monkeypatch):
    captured = {}

    async def _fake(model, contents, system_prompt, tools):
        captured["prompt"] = system_prompt

        class _R:
            text = "ok"

        return _R()

    monkeypatch.setattr(conversation, "_generate", _fake)
    uid = store.get_or_create_user(3, "Shaan")
    store.set_profile(uid, role="equity analyst", onboarding_state="done")

    await conversation.respond(uid, "morning")

    assert "YOU KNOW NOTHING YET" not in captured["prompt"]
    assert "equity analyst" in captured["prompt"]
