"""Assemble and deliver one user's morning briefing."""

import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from atlas.engine import turnlock
from atlas.memory import store
from atlas.proactive import gather, salience

log = logging.getLogger(__name__)


def local_today(tz_name: str) -> str:
    """The user's local date, since 'filed today' means their today, not UTC's."""
    try:
        tz = ZoneInfo(tz_name or "UTC")
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz).date().isoformat()


def _gate_inputs(user_id: int):
    """The three blocking reads the salience gate needs, in one hop.

    market_context() is an HTTP call evaluated inline as an argument, so it
    blocked the loop even though salience.decide is properly async.
    """
    return (
        store.profile_snapshot(user_id),
        store.all_facts(user_id),
        gather.market_context(),
    )


async def build(user_id: int, tz_name: str) -> str | None:
    """Return the briefing text, or None when nothing is worth sending.

    None is a first-class outcome, not a failure.
    """
    today = local_today(tz_name)

    # gather() is synchronous and does three blocking HTTP calls per watchlist
    # name. On the event loop that freezes every in-flight turn at once, which
    # only became survivable-looking because updates used to be serial.
    signals = await asyncio.to_thread(gather.gather, user_id, today)
    fresh_keys = await asyncio.to_thread(
        store.filter_unsent, user_id, [s["key"] for s in signals]
    )
    signals = [s for s in signals if s["key"] in set(fresh_keys)]

    if not signals:
        log.info("user %s: nothing new today, staying silent", user_id)
        return None

    profile, facts, context = await asyncio.to_thread(_gate_inputs, user_id)
    verdict = await salience.decide(profile, facts, signals, context)

    if not verdict["send"]:
        # Mark them seen anyway: the gate judged them unworthy, and re-offering
        # the same signals tomorrow would just ask the same question again.
        await asyncio.to_thread(
            store.mark_sent, user_id, [s["key"] for s in signals]
        )
        log.info("user %s: gate chose silence over %d signals", user_id, len(signals))
        return None

    await asyncio.to_thread(store.mark_sent, user_id, [s["key"] for s in signals])
    return verdict["brief"]


def build_now(user_id: int, tz_name: str) -> dict:
    """Build a briefing because the user asked for one, right now.

    Differs from the scheduled path in two ways, both deliberate:

    - It always returns something to say. Silence is the correct answer to
      "should I interrupt them?", but not to "tell me what's going on".
    - It does not mark signals as sent. The dedupe ledger exists so the assistant
      never *interrupts* twice with the same item; a pull is not an interruption,
      and consuming the ledger here would silence tomorrow's briefing.
    """
    today = local_today(tz_name)

    signals = gather.gather(user_id, today)
    if not signals:
        return {"has_news": False, "brief": None, "signals_considered": 0}

    verdict = salience.decide_sync(
        store.profile_snapshot(user_id),
        store.all_facts(user_id),
        signals,
        gather.market_context(),
    )
    return {
        "has_news": verdict["send"],
        "brief": verdict["brief"] or None,
        "signals_considered": len(signals),
    }


async def send_to(bot, user_id: int, telegram_id: int, tz_name: str) -> bool:
    """Build and deliver. Returns whether anything was actually sent."""
    try:
        brief = await build(user_id, tz_name)
    except Exception:
        log.exception("briefing failed for user %s", user_id)
        return False

    if brief is None:
        return False

    from atlas.ingress.reply import to_telegram_markdown

    # Delivery only. build() stays outside the lock: it is multi-second work
    # that touches no conversation state, and holding the lock across it would
    # make a user's next message wait on their own briefing.
    async with turnlock.user_turn(user_id):
        try:
            await bot.send_message(
                chat_id=telegram_id,
                text=to_telegram_markdown(brief),
                parse_mode="Markdown",
            )
        except Exception:
            log.warning("markdown briefing rejected; sending plain")
            try:
                await bot.send_message(chat_id=telegram_id, text=brief)
            except Exception:
                log.exception("briefing delivery failed for %s", telegram_id)
                return False

        await asyncio.to_thread(store.append_message, user_id, "model", brief)
    return True


def utc_time_for(briefing_time: str, tz_name: str):
    """Convert a user's local HH:MM into a UTC time for the scheduler."""
    hour, minute = (int(p) for p in briefing_time.split(":"))
    try:
        tz = ZoneInfo(tz_name or "UTC")
    except Exception:
        tz = ZoneInfo("UTC")
    today = datetime.now(tz).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return today.astimezone(timezone.utc).timetz()
