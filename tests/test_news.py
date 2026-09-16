"""Live news from free headline feeds.

Gemini grounding stopped working for a free key on 2026-09-16, so news comes from
Google News RSS and Finnhub company news. Both are network calls behind one
seam, _http_get, which these tests replace with canned responses.
"""

import datetime as dt
import logging

import httpx
import pytest

import atlas.tools.news as news

pytestmark = pytest.mark.usefixtures("env")

NOW = int(dt.datetime(2026, 9, 15, 18, 0, tzinfo=dt.UTC).timestamp())


def _rss(*items):
    body = "".join(
        f"<item><title>{title} - {source}</title><link>{link}</link>"
        f"<pubDate>{date}</pubDate><source url='https://x.test'>{source}</source></item>"
        for title, source, link, date in items
    )
    return f"<?xml version='1.0'?><rss><channel>{body}</channel></rss>".encode()


GOOGLE_ITEMS = (
    ("Nvidia slips as chip stocks cool", "Reuters", "https://news.google.com/a", "Tue, 15 Sep 2026 17:09:29 GMT"),
    ("Cisco and Nvidia expand AI partnership", "Yahoo Finance", "https://news.google.com/b", "Tue, 15 Sep 2026 13:30:00 GMT"),
)


def _finnhub(*rows):
    return [
        {
            "headline": headline,
            "summary": summary,
            "source": source,
            "url": f"https://finnhub.test/{i}",
            "datetime": NOW - i * 3600,
            "related": "NVDA",
        }
        for i, (headline, summary, source) in enumerate(rows)
    ]


class _Net:
    """Serves canned responses by host and records every request."""

    def __init__(self, google=None, finnhub=None, fail=()):
        self.google, self.finnhub, self.fail, self.calls = google, finnhub, fail, []

    def __call__(self, url, params):
        self.calls.append((url, params))
        request = httpx.Request("GET", url, params=params)
        for host in self.fail:
            if host in url:
                raise httpx.ConnectError(f"boom for url '{request.url}'", request=request)
        if "news.google.com" in url:
            return httpx.Response(200, content=self.google or _rss(), request=request)
        return httpx.Response(200, json=self.finnhub or [], request=request)

    def hosts(self):
        return [url.split("/")[2] for url, _ in self.calls]


@pytest.fixture
def finnhub_key(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "finnhub-secret-token")
    from atlas.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_google_news_headlines_come_back_clean(monkeypatch):
    net = _Net(google=_rss(*GOOGLE_ITEMS))
    monkeypatch.setattr(news, "_http_get", net)

    r = news.search_financial_news("why did nvidia move today")

    assert r["ok"] is True
    first = r["data"]["articles"][0]
    # Google appends " - Source" to every title; it belongs in its own field.
    assert first["title"] == "Nvidia slips as chip stocks cool"
    assert first["source"] == "Reuters"
    assert first["published"] == "2026-09-15T17:09Z"
    assert first["url"] == "https://news.google.com/a"
    assert net.calls[0][1]["q"] == "why did nvidia move today when:3d"
    assert r["source"] == "Google News"


def test_finnhub_adds_relevant_summaries_first(monkeypatch, finnhub_key):
    net = _Net(
        google=_rss(*GOOGLE_ITEMS),
        finnhub=_finnhub(
            ("Nvidia guides datacenter revenue higher", "NVDA raised its outlook.", "CNBC"),
            ("Discover which Dow Jones stocks are making waves", "A screener list.", "ChartMill"),
            ("NVDA options activity spikes", "Unusual call buying in Nvidia.", "Benzinga"),
        ),
    )
    monkeypatch.setattr(news, "_http_get", net)

    r = news.search_financial_news("why did nvidia fall", symbol="nvda")

    titles = [a["title"] for a in r["data"]["articles"]]
    # Finnhub's feed for a ticker is full of screener spam that never names it.
    assert "Discover which Dow Jones stocks are making waves" not in titles
    assert titles[:2] == ["Nvidia guides datacenter revenue higher", "NVDA options activity spikes"]
    assert r["data"]["articles"][0]["summary"] == "NVDA raised its outlook."
    assert "Nvidia slips as chip stocks cool" in titles
    assert r["source"] == "Finnhub + Google News"


def test_the_same_story_from_both_feeds_appears_once(monkeypatch, finnhub_key):
    net = _Net(
        google=_rss(("Nvidia slips as chip stocks cool", "Reuters", "https://g/a", GOOGLE_ITEMS[0][3])),
        finnhub=_finnhub(("Nvidia slips as chip stocks cool!", "Chips cooled.", "Reuters")),
    )
    monkeypatch.setattr(news, "_http_get", net)

    r = news.search_financial_news("nvidia", symbol="NVDA")

    assert len(r["data"]["articles"]) == 1


def test_results_are_capped(monkeypatch):
    many = [(f"Nvidia story {i}", "Outlet", f"https://g/{i}", GOOGLE_ITEMS[0][3]) for i in range(40)]
    monkeypatch.setattr(news, "_http_get", _Net(google=_rss(*many)))

    r = news.search_financial_news("nvidia")

    assert len(r["data"]["articles"]) == news.MAX_ARTICLES


def test_non_us_tickers_and_a_missing_key_skip_finnhub(monkeypatch, finnhub_key):
    net = _Net(google=_rss(*GOOGLE_ITEMS))
    monkeypatch.setattr(news, "_http_get", net)

    news.search_financial_news("reliance results", symbol="RELIANCE.NS")
    news.search_financial_news("s&p 500 today", symbol="^GSPC")

    # Finnhub's free tier answers 403 for non-US listings; do not spend the call.
    assert "finnhub.io" not in net.hosts()


def test_share_classes_still_use_finnhub(monkeypatch, finnhub_key):
    net = _Net(google=_rss(*GOOGLE_ITEMS))
    monkeypatch.setattr(news, "_http_get", net)

    news.search_financial_news("berkshire", symbol="BRK.B")

    assert "finnhub.io" in net.hosts()


def test_one_feed_down_still_answers_and_logs_without_the_key(
    monkeypatch, finnhub_key, caplog
):
    net = _Net(google=_rss(*GOOGLE_ITEMS), fail=("finnhub.io",))
    monkeypatch.setattr(news, "_http_get", net)

    with caplog.at_level(logging.WARNING, logger="atlas.tools.news"):
        r = news.search_financial_news("nvidia", symbol="NVDA")

    assert r["ok"] is True
    assert "Finnhub" in caplog.text
    assert "finnhub-secret-token" not in caplog.text


def test_every_feed_down_is_reported_and_logged(monkeypatch, finnhub_key, caplog):
    net = _Net(fail=("finnhub.io", "news.google.com"))
    monkeypatch.setattr(news, "_http_get", net)

    with caplog.at_level(logging.WARNING, logger="atlas.tools.news"):
        r = news.search_financial_news("nvidia", symbol="NVDA")

    assert r["ok"] is False
    assert r["error"] == "search_unavailable"
    assert "news search failed" in caplog.text


def test_nothing_found_is_not_an_outage(monkeypatch):
    monkeypatch.setattr(news, "_http_get", _Net(google=_rss()))

    r = news.search_financial_news("obscure query")

    assert r["ok"] is False
    assert r["error"] == "no_results"


def test_lookback_is_clamped(monkeypatch):
    net = _Net(google=_rss(*GOOGLE_ITEMS))
    monkeypatch.setattr(news, "_http_get", net)

    news.search_financial_news("nvidia", days=400)
    news.search_financial_news("nvidia", days=0)

    assert net.calls[0][1]["q"].endswith("when:30d")
    assert net.calls[1][1]["q"].endswith("when:1d")


def test_news_never_calls_gemini(monkeypatch):
    """Grounding was a model request hidden inside a tool, on a quota-starved key."""
    import atlas.integrations.gemini as gemini

    def _no(*a, **k):
        raise AssertionError("news must not touch Gemini")

    monkeypatch.setattr(gemini, "get_client", _no)
    monkeypatch.setattr(news, "_http_get", _Net(google=_rss(*GOOGLE_ITEMS)))

    assert news.search_financial_news("nvidia")["ok"] is True


def test_one_story_syndicated_by_three_outlets_appears_once(monkeypatch):
    """Google News returns the same wire story from Reuters, Yahoo and MSN; all
    three used to take a slot, and a briefing could be three copies of one line."""
    same = [
        ("Nvidia falls 4% after guidance", outlet, f"https://g/{i}", GOOGLE_ITEMS[0][3])
        for i, outlet in enumerate(("Reuters", "Yahoo Finance", "MSN"))
    ]
    monkeypatch.setattr(news, "_http_get", _Net(google=_rss(*same, *GOOGLE_ITEMS)))

    titles = [a["title"] for a in news.search_financial_news("nvidia")["data"]["articles"]]

    assert titles.count("Nvidia falls 4% after guidance") == 1
    assert len(titles) == 3
