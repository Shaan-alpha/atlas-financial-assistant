"""Google Sheets analysis via public CSV export. No OAuth required."""

import csv
import io
import re

import httpx

from atlas.tools.result import err, ok

SOURCE = "Google Sheets (CSV export)"
MAX_ROWS = 500

_SHEET_RE = re.compile(r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)")
_GID_RE = re.compile(r"[#&]gid=([0-9]+)")


def _parse_sheet_url(url: str) -> tuple[str, str] | None:
    match = _SHEET_RE.search(url)
    if not match:
        return None
    gid_match = _GID_RE.search(url)
    return match.group(1), (gid_match.group(1) if gid_match else "0")


def _fetch_csv(sheet_id: str, gid: str) -> str:
    """Network seam. Raises PermissionError when the sheet is not link-shared."""
    export = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    )
    resp = httpx.get(export, timeout=30, follow_redirects=True)
    if resp.status_code in (401, 403) or "text/html" in resp.headers.get("content-type", ""):
        # Google serves a sign-in HTML page rather than a 403 for private sheets.
        raise PermissionError
    resp.raise_for_status()
    return resp.text


def _numeric_summary(headers: list[str], rows: list[list[str]]) -> dict:
    summary: dict[str, dict] = {}
    for index, header in enumerate(headers):
        values = []
        for row in rows:
            if index >= len(row):
                continue
            try:
                values.append(float(row[index].replace(",", "").strip()))
            except (ValueError, AttributeError):
                continue
        if len(values) >= 2:
            summary[header] = {
                "min": min(values),
                "max": max(values),
                "mean": round(sum(values) / len(values), 4),
                "count": len(values),
            }
    return summary


def analyze_sheet(url: str) -> dict:
    """Read a link-shared Google Sheet and return its contents for analysis.

    The sheet must be shared as "anyone with the link can view".

    Args:
        url: Full Google Sheets URL.
    """
    parsed = _parse_sheet_url(url)
    if parsed is None:
        return err("not_a_sheet_url", f"'{url}' is not a Google Sheets link.")

    sheet_id, gid = parsed
    try:
        raw = _fetch_csv(sheet_id, gid)
    except PermissionError:
        return err(
            "sheet_not_shared",
            "That sheet is private. Set sharing to 'anyone with the link' and resend it.",
        )
    except Exception:
        return err("sheet_unavailable", "Could not read that sheet right now.")

    table = list(csv.reader(io.StringIO(raw)))
    if not table:
        return err("empty_sheet", "That sheet has no data in it.")

    headers, rows = table[0], table[1 : MAX_ROWS + 1]
    return ok(
        {
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "truncated": len(table) - 1 > MAX_ROWS,
            "numeric_summary": _numeric_summary(headers, rows),
        },
        source=SOURCE,
    )
