# Atlas — AI Financial Assistant for Telegram

**Design spec — 2026-08-07**

Hackathon: Humanity Founders "Atlas AI Financial Assistant"
Deadline: Sunday 2026-08-09 EOD · hard cutoff Monday 2026-08-10 12:00
Prize: full-time Founding Engineer role

---

## 1. Objective

Build an AI financial assistant living inside Telegram that feels like an experienced
analyst rather than a chatbot. It must converse naturally, remember the user, reason over
live financial data and documents, and proactively surface only what matters.

The brief is explicit that this is **not** a ChatGPT wrapper, **not** a news reader with
summaries, and **not** a command-driven bot. Product thinking and conversation quality
outweigh feature count.

## 2. Constraints

Non-negotiable, taken from the brief:

- Telegram is the only interface.
- Users interact with **text, voice, and images only**.
- **No** slash commands, inline buttons, menus, quick replies, or command navigation.
  - Exception: `/start` is sent automatically by Telegram's own UI on first open. It is
    handled invisibly as "conversation begins" and no other command is ever exposed.
- Finance is the primary vertical. Other verticals are optional and must not dilute it.
- Responses stay concise and immediately useful.
- Accuracy matters: never present unverified information as fact.
- Source code submission is **not required** (explicitly optional, not prohibited).

Two requirements the brief states plainly and that competing submissions have ignored:

1. **Ask clarifying questions** when a request is ambiguous, rather than assuming.
2. **Stay silent** when there is nothing important to say.

Both are treated here as first-class components, not prompt suggestions.

## 3. Competitive context

Submissions already posted to the hackathon Telegram group converge on the same shape:
Groq Llama 3.1 8B for chat, Llama 3.3 70B for research, Whisper for voice, a bolted-on
vision model via OpenRouter, `yfinance`, `pypdf`, SQLite. Their weaknesses:

- Intent-classifier routing, which makes unanticipated phrasings fall through to a
  generic fallback — the bot feels command-driven despite having no commands.
- `pypdf` text extraction discards tables and charts from financial documents.
- Google Sheets support is link-based at best; no real Gmail / Calendar / Drive OAuth.
- Briefings fire unconditionally on a cron. No silence behavior.
- No clarifying-question behavior.

## 4. Stack

| Concern | Choice | Rationale |
| --- | --- | --- |
| Language | Python 3.13 | Brief prefers it; best ecosystem for this problem |
| Bot | `python-telegram-bot` v22 | Mature; handles voice/photo/document natively |
| Reasoning | Gemini via `google-genai` | Native tool-calling loop, PDF, vision, grounding |
| Chat / router | `gemini-3.6-flash` | Built for sustained agentic performance |
| Deep research | `gemini-3.1-pro-preview` | SOTA reasoning for comparisons and analysis |
| Grounded news | `gemini-3-flash-preview` | Explicitly optimized for search + grounding |
| Voice | Groq Whisper large-v3-turbo | Fast, free, accurate |
| Fallback | OpenRouter | Provider outage / quota |
| Storage | SQLite + SQLAlchemy | Postgres-compatible if scaled later |
| Scheduling | APScheduler | Timezone-aware per-user jobs |
| Web | FastAPI | OAuth callback + health endpoint |

Using Gemini natively for documents and images collapses four models into one **and**
improves quality: `client.files.upload()` reads real PDFs including tables and charts,
where `pypdf` flattens them to lossy text.

Model IDs above are current as of 2026-08-07 and must be re-verified at implementation
time rather than trusted from memory.

## 5. Architecture

```
Telegram  (text · voice · image · document)
     │
┌────▼──────────────┐
│ INGRESS           │  voice → Whisper → text
│ normalize inbound │  image → inline_data
└────┬──────────────┘  document → Gemini Files API
     │
┌────▼───────────────────────────────────────┐
│ CONVERSATION ENGINE                        │
│  hydrate profile + memory + recent turns   │
│  Gemini automatic function calling         │
│  clarify-or-answer decision                │
└────┬───────────────────────────────────────┘
     │
┌────▼───────────────────────────────────────┐
│ TOOL BELT  (plain Python fns)              │
│  market · filings · news · sheets · drive  │
│  gmail · calendar · memory · alerts        │
└────┬───────────────────────────────────────┘
     │
┌────▼─────────┐   ┌──────────────────────────────┐
│ PERSISTENCE  │◄──┤ PROACTIVE (APScheduler)      │
│ SQLite       │   │ gather → SALIENCE GATE →     │
└──────────────┘   │ send  ·or·  stay silent      │
                   └──────────────────────────────┘
```

### Module layout

| Package | Responsibility |
| --- | --- |
| `atlas/ingress/` | Telegram handlers, media normalization to one `InboundMessage` |
| `atlas/engine/` | Conversation loop, prompt assembly, clarify logic |
| `atlas/tools/` | One module per family; pure, Telegram-unaware |
| `atlas/integrations/` | Google OAuth, Groq, market data providers |
| `atlas/memory/` | Profile, fact store, background extraction |
| `atlas/proactive/` | Briefing pipeline, salience gate, alert watcher |
| `atlas/db/` | Models, session, migrations |

### Key boundaries

**Tools never know about Telegram.** `market.compare("MSFT", "GOOGL")` returns structured
data, not a formatted message. The engine owns presentation. Consequences: every tool is
unit-testable without a bot token, and the same tool serves both a chat reply and the
morning briefing.

**The salience gate is control flow, not a prompt.** See §8.

## 6. Conversation engine

Gemini's automatic function calling drives the loop: Python functions are passed as
`tools=[...]` and the SDK executes the call cycle. No hand-written intent classifier —
this is what keeps the bot from feeling command-driven.

Each turn hydrates: user profile, full fact set, recent message history, and any active
document context.

### Clarifying questions

The model is given an explicit `clarify(question, options)` tool so choosing to ask is a
first-class, loggable action rather than a prompt hope. It renders as plain conversational
text — never buttons.

Bounded by one rule: **clarify only when the ambiguity materially changes the answer.**

- "Tell me about Apple" → clarify (news? financials? valuation? filings?)
- "Apple's P/E" → just answer

Over-clarifying is as bad as never clarifying; the eval suite (§10) covers both directions.

## 7. Memory

Two tiers.

**Structured** — role, timezone, briefing time, watchlist. Deterministic fields written by
tools during onboarding and as conversations reveal them.

**Free-form facts** — extracted by a background pass *after* the reply is sent, so memory
writes never add latency. Examples: "covers semis for a long/short fund", "prefers terse
answers", "bearish on EV demand".

On write, a new fact is reconciled against existing ones (update vs. insert) to prevent
near-duplicate accumulation.

On read, the user's **full** fact set is loaded. At this scale (dozens of facts) vector
retrieval would be pure ceremony. If the store ever outgrows the context window, that is
the moment to add retrieval — not before.

"What do you know about me?" calls `memory.recall()` and narrates the result naturally.
This is a named demo moment in the hackathon video.

### Onboarding

Conversational and progressive, never a form. `onboarding_state` tracks which topics have
been covered, but the questions are generated naturally in-conversation. Every question is
skippable, and the user can start using the assistant immediately at any point. Remaining
preferences are learned over time rather than demanded upfront.

## 8. Proactive intelligence

Per user, at their local briefing time:

```
gather ──► dedupe ──► SALIENCE GATE ──► compose ──► send
   │        against       "worth waking            │
   │        sent_signals   them for?"              └─► or: SILENT
   │                                                    (logged, nothing sent)
   └─ watchlist moves · new filings · earnings today · grounded news
```

The gate is a separate model call whose only job is to return the subset of signals that
clears the bar for interrupting this specific user — or nothing at all. An empty return
means **no message is sent**, and the decision is logged.

This implements *"if there is nothing important to share, the assistant should remain
silent"* as an enforced branch. The log makes the behavior demonstrable to a judge.

**Alerts** are created from natural language ("alert me if TSLA moves 5%"), stored as a
parsed condition, polled by a watcher, fired once, then cooled down to prevent spam.

## 9. Integrations

### Google Sheets — no OAuth

Link-shared sheets export as CSV without credentials. A judge pastes a link and it works
instantly with zero setup. The brief calls Sheets integration vital; gating it behind a
consent screen would be a product mistake.

### Gmail / Calendar / Drive — real OAuth

FastAPI callback endpoint; signed `state` parameter carries the Telegram user id; tokens
encrypted at rest. Narrower non-restricted scopes (`drive.file`, `calendar.events`) are
preferred wherever they still do the job.

**Known limitation, accepted deliberately.** `gmail.readonly` and `drive.readonly` are
restricted Google scopes. An unverified app authorizes only explicitly-added test users;
full verification takes weeks, well past the deadline. Therefore:

- Depth is proven in the **demo video** using the developer's own account.
- If a judge attempts OAuth and is blocked, the bot **explains the limitation plainly and
  offers the link-based path instead**. A clear explanation reads as thoughtful; a raw
  Google error screen reads as broken.

### Financial data

SEC EDGAR (no key) for filings and `yfinance` for quotes and fundamentals — neither needs
credentials, so both work regardless of which keys are on hand.

One open item, resolved at implementation rather than guessed here: the developer holds
financial-data API keys but the specific providers are not yet confirmed. Whichever they
are (Finnhub, Alpha Vantage, FMP, Polygon), they slot in behind a single provider
interface as an enrichment layer. The product does not depend on any one of them — the
keyless sources above cover the core, and the provider interface means a missing or
rate-limited key degrades one tool rather than breaking the assistant.

Gemini search grounding supplies live news **with citations**, which directly serves the
accuracy requirement.

## 10. Error handling and testing

Every tool returns a structured result or a **typed error the model can reason about** —
never a raw exception. `market.quote("XYZQ")` returns `{error: "no_such_symbol"}` and the
model says so, rather than inventing a price.

Every data-bearing result carries a **source and timestamp**; the system prompt requires
attribution. When grounded search finds nothing, the bot says so.

Provider chain: Gemini → OpenRouter on 5xx/quota. Voice transcription failure asks for a
retype. Telegram's 4096-char limit is enforced as a much lower soft cap — a response that
wants to be long should be tightened, not split.

### Eval harness

The brief lists **15 example questions verbatim**. These are the most likely judge inputs,
so they become the test suite: assert the correct tool fires for each, and that `clarify`
triggers on ambiguous inputs but not on specific ones.

Table-driven tests cover the salience gate (given signal sets, assert send vs. silent) —
the behavior most likely to regress silently.

Tools are pure, so the suite runs on recorded fixtures with no live API calls.

## 11. Deployment

Single process: FastAPI (OAuth callback + health) alongside python-telegram-bot.

The bot must stay warm. A cold start means a judge messages and waits ~50 seconds, which
reads as broken. A 10-minute self-ping defeats free-tier sleep.

## 12. Build order

Each stage is independently demoable, so the project is submittable at any cut point.

1. Skeleton — ingress, engine loop, one tool (`market.quote`), SQLite, echo round-trip
2. Conversation quality — system prompt, clarify tool, memory read/write, onboarding
3. Finance depth — fundamentals, compare, SEC filings, grounded news with citations
4. Multimodal — voice via Whisper, images and PDFs via Gemini native
5. Sheets by link — the zero-setup integration win
6. Proactive — scheduler, gather, salience gate, alerts
7. Google OAuth — Gmail, Calendar, Drive, plus graceful blocked-user fallback
8. Deploy, warm, and record the demo video

## 13. Out of scope

Deliberately excluded as ceremony at this scale or timeline:

- Vector search / embeddings for memory (§7)
- Multi-user tenancy beyond per-Telegram-user isolation
- Production Google OAuth verification (§9)
- Non-finance verticals — optional in the brief, and explicitly not worth diluting finance
- Streaming token-by-token replies; Telegram message edits are noisier than a typing
  indicator followed by one clean message
