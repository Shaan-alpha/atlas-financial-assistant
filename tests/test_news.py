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


# --------------------------------------------------------------- failover
#
# Real production failure: the grounded preview model hit its quota and news.py
# had no fallback chain, so live news silently disappeared while every other
# feature kept working.


class _Grounded:
    text = "Nvidia rose 2.3% on datacenter demand."
    candidates = []


def test_falls_through_a_rate_limited_grounded_model(monkeypatch):
    import atlas.tools.news as news_mod
    from atlas.integrations.gemini import GROUNDED_CHAIN

    tried = []

    class _FakeModels:
        def generate_content(self, model, contents, config):
            tried.append(model)
            if model == GROUNDED_CHAIN[0]:
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
            return _Grounded()

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setattr(news_mod, "get_client", lambda: _FakeClient())

    result = news_mod.search_financial_news("why did nvidia move")

    assert result["ok"] is True
    assert tried == [GROUNDED_CHAIN[0], GROUNDED_CHAIN[1]]


def test_a_real_fault_does_not_burn_the_grounded_chain(monkeypatch):
    import atlas.tools.news as news_mod

    tried = []

    class _FakeModels:
        def generate_content(self, model, contents, config):
            tried.append(model)
            raise RuntimeError("malformed request")

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setattr(news_mod, "get_client", lambda: _FakeClient())

    result = news_mod.search_financial_news("q")

    assert result["ok"] is False
    assert len(tried) == 1
