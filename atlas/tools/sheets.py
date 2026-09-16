"""Google Sheets analysis via public CSV export. No OAuth required."""

import csv
import io
import itertools
import re

import httpx

from atlas.tools.result import err, ok

SOURCE = "Google Sheets (CSV export)"
# Read at most this much of the export. The whole file used to be downloaded and
# parsed before anything was cut, on a 1 GiB machine.
MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 500  # rows the numeric summary covers
ROWS_RETURNED = 100  # raw rows handed to the model, which re-reads them every AFC round
MAX_COLUMNS = 30
MAX_CELL_CHARS = 200

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
    with httpx.stream("GET", export, timeout=30, follow_redirects=True) as resp:
        if resp.status_code in (401, 403) or "text/html" in resp.headers.get("content-type", ""):
            # Google serves a sign-in HTML page rather than a 403 for private sheets.
            raise PermissionError
        resp.raise_for_status()
        chunks, size = [], 0
        for chunk in resp.iter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= MAX_BYTES:
                break
    return b"".join(chunks)[:MAX_BYTES].decode("utf-8", errors="replace")


_NOISE = re.compile(r"[\s,$€£₹¥%]")


def _parse_number(cell) -> float | None:
    """A number as a finance sheet writes it: "$1,234", "12.5%", or "(1,234)"
    for a negative. Plain float() skipped all three, skewing min and mean."""
    if not isinstance(cell, str):
        return None
    text = _NOISE.sub("", cell).replace("\u2212", "-")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def _numeric_summary(headers: list[str], rows: list[list[str]]) -> dict:
    summary: dict[str, dict] = {}
    for index, header in enumerate(headers):
        values = []
        for row in rows:
            if index >= len(row):
                continue
            value = _parse_number(row[index])
            if value is not None:
                values.append(value)
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

    # One row past the limit is enough to know the sheet was cut.
    table = [
        [cell[:MAX_CELL_CHARS] for cell in row[:MAX_COLUMNS]]
        for row in itertools.islice(csv.reader(io.StringIO(raw)), MAX_ROWS + 2)
    ]
    cut = len(raw.encode()) >= MAX_BYTES
    if cut and len(table) > 1:
        # The download stopped mid-row, so its last cell is half a value. Summing
        # it would invent a number that is nowhere in the user's sheet.
        table.pop()
    if not table:
        return err("empty_sheet", "That sheet has no data in it.")

    headers, rows = table[0], table[1 : MAX_ROWS + 1]
    return ok(
        {
            "headers": headers,
            "rows": rows[:ROWS_RETURNED],
            "row_count": len(rows),
            "rows_returned": min(len(rows), ROWS_RETURNED),
            "truncated": len(table) - 1 > MAX_ROWS or cut,
            "numeric_summary": _numeric_summary(headers, rows),
        },
        source=SOURCE,
    )
