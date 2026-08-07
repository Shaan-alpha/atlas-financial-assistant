import atlas.tools.news as news


class _FakeResponse:
    text = "Nvidia fell 4% on datacenter guidance."

    class _Candidate:
        class _Meta:
            class _Chunk:
                class _Web:
                    uri = "https://example.com/nvda"
                    title = "Nvidia guidance"

                web = _Web()

            grounding_chunks = [_Chunk()]

        grounding_metadata = _Meta()

    candidates = [_Candidate()]


def test_returns_summary_with_citations(monkeypatch):
    monkeypatch.setattr(news, "_generate_grounded", lambda q: _FakeResponse())

    r = news.search_financial_news("why did nvidia move today")

    assert r["ok"] is True
    assert "Nvidia fell 4%" in r["data"]["summary"]
    assert r["data"]["citations"][0]["uri"] == "https://example.com/nvda"


def test_reports_when_grounding_finds_nothing(monkeypatch):
    class _Empty:
        text = ""
        candidates = []

    monkeypatch.setattr(news, "_generate_grounded", lambda q: _Empty())

    r = news.search_financial_news("obscure query")

    assert r["ok"] is False
    assert r["error"] == "no_results"
