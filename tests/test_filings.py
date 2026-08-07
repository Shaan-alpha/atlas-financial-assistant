import atlas.tools.filings as filings

CIK_MAP = {"AAPL": "0000320193"}
SUBMISSIONS = {
    "0000320193": {
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "10-Q"],
                "filingDate": ["2026-07-30", "2026-07-01", "2026-05-02"],
                "accessionNumber": [
                    "0000320193-26-0001",
                    "0000320193-26-0002",
                    "0000320193-26-0003",
                ],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
            }
        },
    }
}


def _patch(monkeypatch):
    monkeypatch.setattr(filings, "_fetch_cik_map", lambda: CIK_MAP)
    monkeypatch.setattr(filings, "_fetch_submissions", lambda cik: SUBMISSIONS.get(cik))


def test_returns_recent_filings_newest_first(monkeypatch):
    _patch(monkeypatch)

    r = filings.get_recent_filings("AAPL")

    assert r["ok"] is True
    assert r["data"]["filings"][0]["form"] == "10-K"
    assert r["data"]["filings"][0]["url"].startswith("https://www.sec.gov/Archives/")
    assert r["source"] == "SEC EDGAR"


def test_filters_by_form_type(monkeypatch):
    _patch(monkeypatch)

    r = filings.get_recent_filings("AAPL", form_type="8-K")

    assert [f["form"] for f in r["data"]["filings"]] == ["8-K"]


def test_unknown_symbol_returns_error(monkeypatch):
    _patch(monkeypatch)

    r = filings.get_recent_filings("XYZQ")

    assert r["ok"] is False
    assert r["error"] == "no_such_issuer"
