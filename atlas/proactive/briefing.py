"""Assemble and deliver one user's morning briefing."""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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


async def build(user_id: int, tz_name: str) -> str | None:
    """Return the briefing text, or None when nothing is worth sending.

    None is a first-class outcome, not a failure.
    """
    today = local_today(tz_name)

    signals = gather.gather(user_id, today)
    fresh_keys = store.filter_unsent(user_id, [s["key"] for s in signals])
    signals = [s for s in signals if s["key"] in set(fresh_keys)]

    if not signals:
        log.info("user %s: nothing new today, staying silent", user_id)
        return None

    verdict = await salience.decide(
        store.profile_snapshot(user_id),
        store.all_facts(user_id),
        signals,
        gather.market_context(),
    )

    if not verdict["send"]:
        # Mark them seen anyway: the gate judged them unworthy, and re-offering
        # the same signals tomorrow would just ask the same question again.
        store.mark_sent(user_id, [s["key"] for s in signals])
        log.info("user %s: gate chose silence over %d signals", user_id, len(signals))
        return None

    store.mark_sent(user_id, [s["key"] for s in signals])
    return verdict["brief"]


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

    store.append_message(user_id, "model", brief)
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
