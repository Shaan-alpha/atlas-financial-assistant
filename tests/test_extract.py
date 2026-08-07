import pytest

import atlas.memory.extract as extract
from atlas.memory import store

pytestmark = pytest.mark.usefixtures("fresh_db")


async def test_stores_extracted_facts(monkeypatch):
    async def _fake(user_text, reply):
        return [{"fact": "Covers semiconductors", "category": "focus"}]

    monkeypatch.setattr(extract, "_extract", _fake)
    uid = store.get_or_create_user(1, "Shaan")

    stored = await extract.extract_and_store(uid, "I cover semis", "Noted.")

    assert stored == 1
    assert store.all_facts(uid)[0]["fact"] == "Covers semiconductors"


async def test_extraction_failure_is_swallowed(monkeypatch):
    async def _boom(user_text, reply):
        raise RuntimeError("model down")

    monkeypatch.setattr(extract, "_extract", _boom)
    uid = store.get_or_create_user(2, "Shaan")

    assert await extract.extract_and_store(uid, "hi", "hello") == 0


async def test_malformed_entries_are_skipped(monkeypatch):
    async def _messy(user_text, reply):
        return [
            {"category": "focus"},
            {"fact": "  ", "category": "x"},
            {"fact": "Runs a long/short book", "category": "role"},
        ]

    monkeypatch.setattr(extract, "_extract", _messy)
    uid = store.get_or_create_user(3, "Shaan")

    assert await extract.extract_and_store(uid, "t", "r") == 1
