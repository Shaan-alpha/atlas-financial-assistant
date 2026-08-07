import atlas.tools.sheets as sheets

CSV = "Quarter,Revenue,Costs\nQ1,1000,600\nQ2,1200,700\nQ3,900,1500\n"


def test_parses_standard_sheet_url():
    parsed = sheets._parse_sheet_url(
        "https://docs.google.com/spreadsheets/d/1AbC_def/edit#gid=42"
    )
    assert parsed == ("1AbC_def", "42")


def test_defaults_gid_to_zero():
    parsed = sheets._parse_sheet_url("https://docs.google.com/spreadsheets/d/1AbC_def/edit")
    assert parsed == ("1AbC_def", "0")


def test_rejects_non_sheet_url():
    r = sheets.analyze_sheet("https://example.com/not-a-sheet")
    assert r["ok"] is False
    assert r["error"] == "not_a_sheet_url"


def test_returns_headers_rows_and_numeric_summary(monkeypatch):
    monkeypatch.setattr(sheets, "_fetch_csv", lambda sid, gid: CSV)

    r = sheets.analyze_sheet("https://docs.google.com/spreadsheets/d/1AbC_def/edit#gid=0")

    assert r["ok"] is True
    assert r["data"]["headers"] == ["Quarter", "Revenue", "Costs"]
    assert r["data"]["row_count"] == 3
    assert r["data"]["numeric_summary"]["Revenue"]["max"] == 1200.0
    assert r["data"]["rows"][2] == ["Q3", "900", "1500"]


def test_private_sheet_returns_actionable_error(monkeypatch):
    def _denied(sid, gid):
        raise PermissionError

    monkeypatch.setattr(sheets, "_fetch_csv", _denied)

    r = sheets.analyze_sheet("https://docs.google.com/spreadsheets/d/1AbC_def/edit")

    assert r["ok"] is False
    assert r["error"] == "sheet_not_shared"
    assert "anyone with the link" in r["message"]
