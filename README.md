# Atlas

An AI financial analyst that lives in Telegram.

**Try it:** [@AtlasAnalyst_bot](https://t.me/AtlasAnalyst_bot), just say hello. There is nothing to configure and no command to learn.

---

## What it does

Atlas holds a conversation. It learns who you are as you talk, pulls live market data, reads the documents you send it, and speaks up on its own when something on your watchlist actually matters.

- **Conversation only.** No slash commands, no inline buttons, no menus. Onboarding happens by talking.
- **Live market data** — quotes, fundamentals, price history, earnings dates, SEC filings, and the day's headlines.
- **Documents that keep their shape** — send a PDF, a spreadsheet, a Google Sheet link, or a photo of a chart.
- **Voice** — send a voice note instead of typing.
- **Memory that persists** — role, timezone, watchlist, and durable facts, across restarts.
- **Proactive briefings and alerts**, including the decision *not* to send one.

## Seven decisions worth reading the code for

### 1. Documents are handed to the model whole

Most PDF pipelines run `pypdf`, get a wall of text, and lose the tables, which in a financial filing is where the answer usually lives. A segment margin table flattens into an unlabelled column of numbers, and the model then confidently misreads it.

Atlas uploads the file to Gemini directly ([`atlas/ingress/handlers.py`](atlas/ingress/handlers.py)) and lets native document understanding read the layout. Tables stay tables. Ask "which segment carried the quarter?" of a results PDF and it reads the margin column correctly rather than guessing from prose.

The same path handles images, so a photographed chart works too.

### 2. Silence is enforced control flow, not a prompt suggestion

The brief asks the assistant to stay quiet when nothing is important. Saying "only message when it matters" in a system prompt does not survive contact with a model that wants to be helpful.

So the salience gate ([`atlas/proactive/salience.py`](atlas/proactive/salience.py)) is real code:

- No signals at all short-circuits **before** any model call; an empty morning costs nothing.
- The gate returns `send: true/false`, and `false` sends nothing.
- A malformed yes with an empty body is still silence.
- A gate failure defaults to silence. A briefing nobody asked for is worse than one that never arrives.

Silence is the right answer to "should I interrupt them?" and the wrong answer to "what's happening with my names?", so only the scheduled path has a gate. Asked directly, the tool hands the raw signals to the model already in the conversation rather than spending a second model call writing prose for the first one to rewrite.

### 3. Market data fails over, and the failover is measured from the host

`yfinance` works perfectly from a laptop and is silently rate-limited from a datacenter IP. You cannot discover that locally; the deploy target is the only place the question has an answer.

Atlas chains four quote providers and three fundamentals providers ([`atlas/integrations/marketdata.py`](atlas/integrations/marketdata.py)), and exposes `/diag` so provider health can be read **from the running host**:

```
/diag  ->  quotes:       finnhub ok  fmp ok  yahoo ok
           fundamentals: finnhub ok  fmp ok
```

That endpoint is why the deployed bot answers instead of apologising. Provider
errors are scrubbed of API keys before they are shown, because an unscrubbed
error hands out credentials. `/diag` answers on loopback unless the host has to
reach it over the network, caches its answer for an hour, and leaves the
daily-capped providers out of a routine check: probing cost Alpha Vantage 8 of
its 25 daily requests per hit.

Each provider is asked only about symbols its free tier actually covers. Finnhub
and Alpha Vantage answer for US tickers, FMP for those and for indices, Yahoo for
everything, which is what keeps a rupee-denominated quote from being labelled in
dollars and stops a typo spending a daily-capped request.

It also catches the failure mode that looks like a working system. Two providers
once sat dead in the chain for weeks: one key was never set, and FMP answered
`403` on every call because it had retired `/api/v3` and served only a *"Legacy
Endpoint"* message to keys issued after August 2025. A 403 reads exactly like a
rejected key, so the key got blamed. The endpoint was the problem. Nothing broke
visibly, because the provider *below* them in the chain answered, which is the
whole point of a chain, and precisely why it needs a health endpoint to see
inside it.

### 4. Rate limits are routed around, not waited out

Gemini's free tier meters **per model**: 5 requests/minute for each. So a 429 on one model is not a global stop, falling back to a *different* model draws from a fresh bucket.

Atlas walks a model chain per workload ([`atlas/integrations/gemini.py`](atlas/integrations/gemini.py)) and caps total wait so a user never watches a spinner for a minute.

Quota is not the only thing worth surviving. Transient upstream faults (`499
CANCELLED`, `503 UNAVAILABLE`) also carry to the next model, because they say
nothing about the request itself. That distinction was learned the hard way: a
`499` on the first call of a freshly deployed process aborted the whole turn with
two untried models still in the chain, and the user got an apology instead of a
quote. Genuine faults still stop at the first model, since retrying a malformed
request three times only makes someone wait three times as long to be told no.

Models also do not live forever. Google retires them on its own schedule, and on
2026-09-15 the last model in the chat chain began answering `404 NOT_FOUND`. The
two live models ahead of it hit a `503` demand spike at the same moment, the 404
was treated as a genuine fault, and the turn died. A 404 that names the requested
model now counts as "retired": the chain skips it, logs it as an error because
only a code change fixes it, and never spends the post-wait retry on it. Every
chain (chat, fact extraction, the briefing gate) follows the same rule.

When no Gemini model answers at all, the turn moves to a different provider
rather than waiting on the one that just refused three times. Groq's free tier
serves open-weight models that call tools, so chat continues with the same system
prompt and the same tools ([`atlas/engine/fallback.py`](atlas/engine/fallback.py)).
Two things make it usable inside an 8,000-token-per-minute budget: Groq gets the
first paragraph of each tool docstring rather than all twenty in full, and a
token-window refusal is either waited out (when it asks for a second) or handed
to the next model's own bucket, instead of the SDK sleeping half a minute. A
measured fallback turn with two tool calls answers in about six seconds.

Turns with an image or an uploaded document skip it: those files live in Gemini's
Files API and no other provider can read them.

### 5. An alert answers once, not every six hours

A daily change does not move between the close and the next open, so "tell me if
TSLA moves 5%" used to re-announce Friday's move overnight and all weekend: the
condition still held, and a six-hour cooldown kept expiring. Alerts are now
bounded by what actually changes. A price level is one-shot and says so when it
fires; a move fires at most once per trading session, keyed on the session date
the provider reports rather than on the calendar day here
([`atlas/proactive/alerts.py`](atlas/proactive/alerts.py)).

The same timestamp fixes the briefing: a Friday close read on a Saturday morning
is the same move, already reported, not a new one.

### 6. The bot is public, so a turn has a budget

Anyone who finds the bot can talk to it, and every turn spends requests from a
Gemini key shared with another project plus market-data quotas every user
depends on. Each Telegram user gets one turn running and one queued, a burst
limit, and a daily ceiling ([`atlas/ingress/guard.py`](atlas/ingress/guard.py)),
refused in one sentence before any model or provider is called. It is in memory
on purpose: a restart forgiving everyone is harmless, and a database round trip
in front of every message is not.

### 7. Concurrency is per-user ordered, not free-for-all

Updates used to be processed strictly one at a time, so one slow turn (a PDF
upload, a voice note) stalled every other user behind it.

Running them concurrently removes that, but it also removes the accidental
protection serial processing gave: `respond()` reads the conversation history,
*then* appends to it, with a multi-second model call in between. Two overlapping
turns from one person would each answer a prompt the other had already
invalidated, then interleave their rows in the log. Because history is ordered by
row id, a tangled pair stays tangled for the next twenty turns.

So [`atlas/engine/turnlock.py`](atlas/engine/turnlock.py) holds one lock per user
in a `WeakValueDictionary`; different people run in parallel, one person's turns
never overlap, and the registry cannot grow without bound because a lock exists
only while someone holds or waits on it. What that buys is mutual exclusion, not
arrival ordering; the docstring says so plainly rather than promising a guarantee
the design does not make.

## Architecture

Agentic core, deterministic edges. Gemini's automatic function calling runs the loop; there is no intent classifier to misroute a question.

```mermaid
flowchart TD
    TG(["<b>Telegram</b><br/>text · voice note · photo · PDF · sheet"])

    ING["<b>ingress/</b><br/>normalise every input shape<br/>voice → Groq Whisper large-v3-turbo<br/>documents &amp; images → handed to Gemini whole"]

    LOCK["<b>engine/turnlock.py</b><br/>one lock per user, WeakValueDictionary<br/><i>different people run in parallel;<br/>one person's turns never overlap</i>"]

    ENG["<b>engine/</b><br/>conversation loop · prompt assembly<br/>walks a model chain per workload"]

    subgraph WORK ["Gemini automatic function calling — no intent classifier to misroute"]
        direction LR
        TOOLS["<b>tools/</b> · 20 tools<br/>quotes · fundamentals · comparisons<br/>price history · earnings · SEC filings<br/>headlines · sheets · clarify<br/><i>each closure-bound to one user id</i>"]
        MEM["<b>memory/</b><br/>profile · durable facts<br/>watchlist · conversation history"]
    end

    subgraph PRO ["proactive/ — the part that decides not to speak"]
        direction TB
        SCHED["<b>scheduler</b> · APScheduler<br/>briefings &amp; alert sweeps"]
        SIG{"any signals<br/>at all?"}
        GATE{"<b>salience gate</b><br/>push and pull run different<br/>instructions against the same body"}
        SCHED --> SIG
        SIG -->|"no"| SHORT(["short-circuit<br/><i>an empty morning costs nothing</i>"])
        SIG -->|"yes"| GATE
    end

    SILENCE(["<b>silence</b><br/><i>malformed yes → silence<br/>gate failure → silence</i>"])
    REPLY(["reply to the user"])

    subgraph FAIL ["Failover, measured from the host"]
        direction TB
        Q["<b>quotes</b> · finnhub → fmp → yahoo<br/>→ alphavantage"]
        F["<b>fundamentals</b> · finnhub → fmp → yfinance"]
        M["<b>models</b> · gemini 3.6 → 3.5 → 3 preview<br/>→ groq gpt-oss, a different provider"]
        DIAG["<b>/diag</b> · provider health read from<br/>the running host, keys scrubbed"]
    end

    PG[("<b>Postgres 18</b> · SQLAlchemy 2.0 / psycopg3<br/>nightly pg_dump on a systemd timer")]
    WD["<b>watchdog</b><br/>force-exits when the Application is up<br/>but the poller underneath has finished<br/><i>back polling 6 s after kill -9</i>"]

    TG --> ING --> LOCK --> ENG
    ENG --> TOOLS
    ENG --> MEM
    TOOLS --> Q
    TOOLS --> F
    Q -.-> DIAG
    F -.-> DIAG
    MEM <--> PG
    TOOLS --> REPLY
    GATE -->|"send: true"| REPLY
    GATE -->|"send: false"| SILENCE
    WD -.->|"systemd Restart=always"| ENG

    classDef gate fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fee2e2
    classDef quiet fill:#0f172a,stroke:#475569,stroke-width:1.5px,color:#94a3b8
    classDef core fill:#312e81,stroke:#818cf8,stroke-width:2px,color:#e2e8f0
    classDef store fill:#1e293b,stroke:#475569,stroke-width:1.5px,color:#cbd5e1
    class GATE,SIG gate
    class SILENCE,SHORT quiet
    class ENG,TOOLS,MEM,LOCK core
    class PG,Q,F,DIAG,WD store
```

Read the red path first. Everything else is a conversation loop; the gate is the part that had to be code, because "only message when it matters" in a system prompt does not survive contact with a model that wants to be helpful.

**20 tools**, each bound to one user by closure; the model never supplies a user id, so it cannot reach another user's data.

`get_quote` · `get_fundamentals` · `compare_companies` · `market_overview` · `get_price_history` · `get_earnings_info` · `get_recent_filings` · `search_financial_news` · `analyze_sheet` · `clarify` · `remember` · `recall` · `forget_about` · `update_profile` · `add_to_watchlist` · `remove_from_watchlist` · `brief_me_now` · `create_alert` · `list_alerts` · `cancel_alert`

### Ambiguity is a tool call

"Tell me about Apple" is underspecified. Rather than picking one reading and writing three paragraphs the user did not want, the model calls `clarify` ([`atlas/tools/clarify.py`](atlas/tools/clarify.py)) and offers concrete options in plain prose; no buttons, because the brief forbids them.

## Stack

Python 3.13 · python-telegram-bot 22 · Gemini (chat, vision, documents) · Groq (`gpt-oss` fallback chat, Whisper `large-v3-turbo`) · PostgreSQL 18 + SQLAlchemy 2.0 / psycopg3 · APScheduler · Finnhub · FMP · Yahoo · Alpha Vantage · Google News · SEC EDGAR · Azure VM under systemd

## Tests

```bash
pip install -e ".[dev]"
pytest          # 191 tests
```

Weighted toward the behaviour most likely to regress quietly: the silence path. A briefing that fires when it shouldn't is the failure a user actually notices, and it is invisible in a happy-path test.

The same bias explains the newer files. `tests/test_concurrency.py` pins the
per-user ordering above; a test that two turns interleave is worthless unless it
also proves two *different* users still overlap. `tests/test_main_wiring.py`
exists because `main()` had no coverage at all, which is exactly how a bot that
replayed a day-old backlog on every restart went unnoticed. And
`tests/test_env_docs.py` derives its expectations from `config.py` rather than a
hardcoded list, so a provider added later fails the suite instead of quietly
going undocumented.

## Running it

```bash
cp .env.example .env      # TELEGRAM_TOKEN, GEMINI_API_KEY, GROQ_API_KEY are required; market-data keys are optional
pip install -e .
python -m atlas.main
```

### Deployment

Runs on an Azure VM under `systemd`, with Postgres 18 on the same host and a
nightly `pg_dump` on a systemd timer.

The hosting choice is load-bearing rather than incidental. Atlas is a long-lived
polling process: it holds a `getUpdates` long poll and runs an in-process
scheduler for briefings and alerts. On a free tier that sleeps when idle, that
shape fails badly, polling is *outbound*, so a sleeping bot is never woken by a
Telegram message, only by unrelated HTTP traffic.

Worse, the failure is silent. The health server runs on its own thread, so when
polling dies the process keeps answering `200` while fetching nothing, and a
platform health check sees a service in perfect health. Atlas answered nobody
for days that way.

Two things fix it. `atlas/main.py` runs a watchdog that force-exits when the
Application is up but the poller underneath it has finished, and `/` returns
`503` once polling has stopped instead of a cheerful `200`. `systemd` then does
what a health check could not:

```ini
Restart=always
RestartSec=5
StartLimitIntervalSec=0    # in [Unit] — systemd ignores it under [Service]
```

Verified by `kill -9`: back and polling in six seconds.

[`render.yaml`](render.yaml) is kept for one-command Blueprint deploys. If you
use it, set `PUBLIC_URL` so the self-ping keeps the service awake, and keep the
database in the web service's region; Render's internal database hostname is
region-scoped and will not resolve across regions.
