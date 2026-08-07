"""SEC EDGAR filings. Keyless, but requires an identifying User-Agent header."""

from functools import lru_cache

import httpx

from atlas.tools.result import err, ok

SOURCE = "SEC EDGAR"
# SEC requires a descriptive User-Agent identifying the caller or returns 403.
HEADERS = {"User-Agent": "Atlas Financial Assistant (contact: shaansatsangi@gmail.com)"}


@lru_cache(maxsize=1)
def _fetch_cik_map() -> dict[str, str]:
    """Map ticker -> zero-padded CIK. Cached; the file is large and changes rarely."""
    resp = httpx.get(
        "https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30
    )
    resp.raise_for_status()
    return {
        row["ticker"].upper(): str(row["cik_str"]).zfill(10)
        for row in resp.json().values()
    }


def _fetch_submissions(cik: str) -> dict | None:
    resp = httpx.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS, timeout=30
    )
    if resp.status_code != 200:
        return None
    return resp.json()


def _archive_url(cik: str, accession: str, document: str) -> str:
    bare = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{bare}/{document}"


def get_recent_filings(symbol: str, form_type: str = "", limit: int = 5) -> dict:
    """Return recent SEC filings for a US-listed company.

    Args:
        symbol: Ticker symbol, for example "TSLA".
        form_type: Optional exact form filter, for example "10-K", "10-Q", "8-K".
        limit: Maximum filings to return.
    """
    try:
        cik = _fetch_cik_map().get(symbol.upper())
    except Exception:
        return err("edgar_unavailable", "SEC EDGAR is not responding right now.")

    if cik is None:
        return err("no_such_issuer", f"No SEC filer matches ticker '{symbol}'.")

    submissions = _fetch_submissions(cik)
    if submissions is None:
        return err("edgar_unavailable", f"Could not load filings for '{symbol}'.")

    recent = submissions.get("filings", {}).get("recent", {})
    rows = []
    for form, date, accession, document in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        strict=False,
    ):
        if form_type and form != form_type:
            continue
        rows.append(
            {
                "form": form,
                "filed_on": date,
                "url": _archive_url(cik, accession, document),
            }
        )
        if len(rows) >= limit:
            break

    return ok(
        {"symbol": symbol.upper(), "company": submissions.get("name"), "filings": rows},
        source=SOURCE,
    )
