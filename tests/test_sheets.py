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


def test_finance_formatting_is_read_as_numbers(monkeypatch):
    """Parenthesised negatives, currency symbols and percents were skipped, so
    a loss-making quarter vanished from min and mean."""
    csv_text = 'Quarter,Net income,Margin\nQ1,"$1,200",12.5%\nQ2,"(300)",-3%\nQ3,"$900",9%\n'
    monkeypatch.setattr(sheets, "_fetch_csv", lambda sid, gid: csv_text)

    summary = sheets.analyze_sheet(
        "https://docs.google.com/spreadsheets/d/1AbC_def/edit"
    )["data"]["numeric_summary"]

    assert summary["Net income"]["min"] == -300.0
    assert summary["Net income"]["count"] == 3
    assert summary["Margin"]["max"] == 12.5


def test_a_huge_sheet_is_bounded(monkeypatch):
    wide = ",".join(f"c{i}" for i in range(80))
    body = wide + "\n" + "\n".join(",".join("x" * 500 for _ in range(80)) for _ in range(700))
    monkeypatch.setattr(sheets, "_fetch_csv", lambda sid, gid: body)

    data = sheets.analyze_sheet("https://docs.google.com/spreadsheets/d/1AbC_def/edit")["data"]

    assert len(data["headers"]) == sheets.MAX_COLUMNS
    assert len(data["rows"]) == sheets.ROWS_RETURNED
    assert data["row_count"] == sheets.MAX_ROWS
    assert data["truncated"] is True
    assert max(len(cell) for cell in data["rows"][0]) == sheets.MAX_CELL_CHARS


def test_a_row_cut_in_half_by_the_byte_cap_is_dropped(monkeypatch):
    """The download stops mid-row, so its last number is half a value."""
    rows = "\n".join(f"Q{i},{1000 + i}" for i in range(1, 40))
    body = ("Quarter,Revenue\n" + rows + "\nQ40,987654")[: sheets.MAX_BYTES]
    monkeypatch.setattr(sheets, "MAX_BYTES", len(body.encode()))
    monkeypatch.setattr(sheets, "_fetch_csv", lambda sid, gid: body)

    data = sheets.analyze_sheet("https://docs.google.com/spreadsheets/d/1AbC_def/edit")["data"]

    assert data["truncated"] is True
    assert [r[0] for r in data["rows"]][-1] == "Q39"
    assert data["numeric_summary"]["Revenue"]["max"] == 1039.0
