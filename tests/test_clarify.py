from atlas.tools.clarify import clarify


def test_clarify_returns_question_and_options():
    r = clarify("What would you like on Apple?", ["latest news", "valuation", "filings"])

    assert r["ok"] is True
    assert r["data"]["question"].startswith("What would you like")
    assert r["data"]["options"] == ["latest news", "valuation", "filings"]


def test_clarify_rejects_too_many_options():
    r = clarify("Which?", ["a", "b", "c", "d", "e", "f"])

    assert r["ok"] is False
    assert r["error"] == "too_many_options"
