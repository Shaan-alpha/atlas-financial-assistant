"""Natural-language alerts and the watcher that fires them.

The user says "tell me if TSLA moves 5%" and the model turns that into a stored
condition via the create_alert tool. This module owns evaluation and delivery.
"""

import asyncio
import datetime as dt
import logging

from atlas.engine import turnlock
from atlas.memory import store
from atlas.tools import market

log = logging.getLogger(__name__)

CHECK_INTERVAL = dt.timedelta(minutes=15)

# After firing, stay quiet for a while. A stock past a threshold is still past it
# fifteen minutes later, and re-sending would turn one alert into a stream.
COOLDOWN = dt.timedelta(hours=6)

KINDS = ("move_pct", "price_above", "price_below")


def is_triggered(kind: str, threshold: float, quote: dict) -> bool:
    """Pure evaluation against a quote payload."""
    price = quote.get("price")
    change = quote.get("change_pct")

    if kind == "move_pct":
        return change is not None and abs(change) >= threshold
    if kind == "price_above":
        return price is not None and price >= threshold
    if kind == "price_below":
        return price is not None and price <= threshold
    return False


def in_cooldown(last_fired_at, now: dt.datetime) -> bool:
    if last_fired_at is None:
        return False
    return (now - last_fired_at) < COOLDOWN


def describe(alert: dict, quote: dict) -> str:
    symbol = alert["symbol"]
    price = quote.get("price")
    change = quote.get("change_pct")

    if alert["kind"] == "move_pct":
        direction = "up" if (change or 0) > 0 else "down"
        headline = f"*{symbol}* is {direction} {abs(change):.1f}% at {price}"
    elif alert["kind"] == "price_above":
        headline = f"*{symbol}* has crossed above {alert['threshold']} — now {price}"
    else:
        headline = f"*{symbol}* has dropped below {alert['threshold']} — now {price}"

    # Echo their own wording so the alert reads like the thing they asked for.
    return f"{headline}.\nYou asked me to watch for this: _{alert['description']}_"


def _record_fired(alert: dict, now: dt.datetime, text: str) -> None:
    """Cooldown stamp and conversation log, in one hop off the loop."""
    store.mark_alert_fired(alert["id"], now)
    store.append_message(alert["user_id"], "model", text)


async def check_all(bot, now: dt.datetime | None = None) -> int:
    """Evaluate every armed alert. Returns how many fired."""
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    fired = 0

    for alert in await asyncio.to_thread(store.active_alerts):
        if in_cooldown(alert["last_fired_at"], now):
            continue

        # Blocking HTTP, once per armed alert. Off-thread, or the watcher
        # freezes every conversation in flight for the length of the sweep.
        quote = await asyncio.to_thread(market.get_quote, alert["symbol"])
        if not quote["ok"]:
            continue
        if not is_triggered(alert["kind"], alert["threshold"], quote["data"]):
            continue

        text = describe(alert, quote["data"])
        # Lock only the delivery and the log append — never the fetch above.
        async with turnlock.user_turn(alert["user_id"]):
            try:
                await bot.send_message(
                    chat_id=alert["telegram_id"], text=text, parse_mode="Markdown"
                )
            except Exception:
                try:
                    await bot.send_message(chat_id=alert["telegram_id"], text=text)
                except Exception:
                    log.exception("alert delivery failed for %s", alert["telegram_id"])
                    continue

            await asyncio.to_thread(_record_fired, alert, now, text)
        fired += 1

    if fired:
        log.info("fired %d alert(s)", fired)
    return fired


async def _job(context) -> None:
    await check_all(context.bot)


def install(job_queue) -> None:
    job_queue.run_repeating(_job, interval=CHECK_INTERVAL, first=CHECK_INTERVAL)
    log.info("alert watcher running every %s", CHECK_INTERVAL)
