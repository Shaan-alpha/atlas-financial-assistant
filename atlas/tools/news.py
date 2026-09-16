"""Live financial news from free headline feeds.

Gemini search grounding used to answer this with a cited summary. It stopped on
2026-09-16: grounding returns 429 on every Gemini 3 model for a free key, and the
one model that grounded for free (gemini-2.5-flash) had been retired. Paying for
grounding would break the zero-spend constraint, so this returns recent headlines,
each with its outlet, time, and link, and the chat model explains what they say.
That is also one Gemini request cheaper per search, since grounding was a model
call hidden inside the tool.

Two free feeds, merged:
  - Finnhub company news. Per ticker, US listings only on the free tier, and each
    story carries a paragraph of summary, which is what actually explains a move.
  - Google News RSS. Keyless, any market, and it ranks a free-text question well.
"""

import datetime as dt
import email.utils
import logging
import re
import xml.etree.ElementTree as ET

import httpx

from atlas.config import get_settings
from atlas.integrations.marketdata import is_us_ticker, scrub
from atlas.tools.result import err, ok

log = logging.getLogger(__name__)

TIMEOUT = 12
UA = {"User-Agent": "Mozilla/5.0 (compatible; Atlas/1.0)"}
MAX_ARTICLES = 8
# Summaries lead, because they explain; Google's relevance ranking fills the rest.
FINNHUB_SHARE = 3
SUMMARY_CHARS = 280
DEFAULT_DAYS = 3

_WORD = re.compile(r"[a-z0-9]{4,}")
# Words that say nothing about WHICH story is wanted, so they cannot vouch for one.
_STOPWORDS = frozenset(
    "about after before what when where which with from that this they them today "
    "yesterday week month stock stocks share shares news latest recent move moved "
    "moving market markets price fall fell rise rose drop dropped jump jumped gain "
    "gained down higher lower does happened happening".split()
)


def _http_get(url: str, params: dict) -> httpx.Response:
    """Network seam. Tests replace this."""
    response = httpx.get(url, params=params, timeout=TIMEOUT, headers=UA)
    response.raise_for_status()
    return response


def _iso_minutes(moment: dt.datetime) -> str:
    return moment.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%MZ")


def _google_news(query: str, days: int) -> list[dict]:
    response = _http_get(
        "https://news.google.com/rss/search",
        {"q": f"{query} when:{days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"},
    )
    articles = []
    for item in ET.fromstring(response.content).iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip() or None
        # Google appends " - Outlet" to every title; the outlet has its own field.
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].rstrip()
        if not title:
            continue
        published = None
        if item.findtext("pubDate"):
            try:
                published = _iso_minutes(
                    email.utils.parsedate_to_datetime(item.findtext("pubDate"))
                )
            except (TypeError, ValueError):
                published = None
        articles.append(
            {
                "title": title,
                "source": source,
                "published": published,
                "url": item.findtext("link"),
                "summary": None,
            }
        )
    return articles


def _finnhub_news(symbol: str, days: int, keywords: set[str]) -> list[dict]:
    key = get_settings().finnhub_api_key
    # Finnhub's free tier covers US listings only; anything else answers 403.
    if not key or not is_us_ticker(symbol):
        return []
    today = dt.datetime.now(dt.UTC).date()
    response = _http_get(
        "https://finnhub.io/api/v1/company-news",
        {
            "symbol": symbol,
            "from": str(today - dt.timedelta(days=days)),
            "to": str(today),
            "token": key,
        },
    )
    # A ticker's feed is mostly screener lists and roundups that never name it,
    # so a story has to mention the ticker or a word from the question to count.
    wanted = [
        re.compile(rf"\b{re.escape(word)}\b") for word in (keywords | {symbol.lower()})
    ]
    rows = sorted(response.json() or [], key=lambda d: d.get("datetime") or 0, reverse=True)
    articles = []
    for row in rows:
        title = (row.get("headline") or "").strip()
        summary = (row.get("summary") or "").strip()
        text = f"{title} {summary}".lower()
        if not title or not any(pattern.search(text) for pattern in wanted):
            continue
        stamp = row.get("datetime")
        articles.append(
            {
                "title": title,
                "source": row.get("source") or None,
                "published": _iso_minutes(dt.datetime.fromtimestamp(stamp, dt.UTC))
                if stamp
                else None,
                "url": row.get("url") or None,
                "summary": summary[:SUMMARY_CHARS] or None,
            }
        )
        if len(articles) == FINNHUB_SHARE:
            break
    return articles


def _story_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def search_financial_news(query: str, symbol: str = "", days: int = DEFAULT_DAYS) -> dict:
    """Search recent financial news. Returns headlines with outlet, time, and link.

    Use for anything time-sensitive: why a stock moved, breaking news, deals,
    analyst actions, macro events. Results are headlines, some with a short
    summary, not an explanation: report what they say, name the outlet, and say
    plainly when they do not explain what was asked.

    Args:
        query: A specific natural-language question, e.g. "why did Nvidia fall today".
        symbol: The ticker when the question is about one company, e.g. "NVDA".
        days: How far back to look, from 1 to 30. Defaults to 3.
    """
    symbol = (symbol or "").strip().upper()
    query = (query or "").strip() or symbol
    if not query:
        return err("empty_query", "Say what news to look for.")
    try:
        days = min(max(int(days), 1), 30)
    except (TypeError, ValueError):
        days = DEFAULT_DAYS

    keywords = {w for w in _WORD.findall(query.lower()) if w not in _STOPWORDS}
    feeds = (
        ("Finnhub", lambda: _finnhub_news(symbol, days, keywords)),
        ("Google News", lambda: _google_news(query, days)),
    )

    articles: list[dict] = []
    seen: set[str] = set()
    used: list[str] = []
    failed: list[str] = []
    for name, fetch in feeds:
        try:
            found = fetch()
        except Exception as exc:
            log.warning("%s news search failed: %s", name, scrub(str(exc)))
            failed.append(name)
            continue
        # Checked and recorded in one pass: one wire story syndicated by three
        # outlets arrives three times in a single feed, and none of them was in
        # `seen` yet, so all three used to survive.
        fresh = []
        for article in found:
            key = _story_key(article["title"])
            if key in seen:
                continue
            seen.add(key)
            fresh.append(article)
            if len(articles) + len(fresh) >= MAX_ARTICLES:
                break
        if fresh:
            used.append(name)
        articles.extend(fresh)

    if not articles:
        # Google News runs for every query; with it down, nothing was searched.
        if "Google News" in failed:
            log.warning("every news search failed for %r", query)
            return err("search_unavailable", "Live news is not responding right now.")
        return err("no_results", f"No reporting in the last {days} days for: {query}")

    return ok({"query": query, "days": days, "articles": articles}, source=" + ".join(used))
