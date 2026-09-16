import datetime as dt
import logging
import os
import threading
import time

from telegram import Update
from telegram.error import Conflict, NetworkError
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from atlas.config import get_settings
from atlas.db.session import init_db
from atlas.health import mark_polling_stopped, ping, start_health_server
from atlas.ingress import handlers
from atlas.proactive import alerts, scheduler

log = logging.getLogger(__name__)

KEEPALIVE_INTERVAL = dt.timedelta(minutes=10)

# How often to check that polling is still alive. Short enough that a dead bot is
# restarted within a minute or so; long enough to be free.
WATCHDOG_INTERVAL = 45.0

# Bounded on purpose. Without this, updates are processed strictly one at a
# time and a single slow turn (a PDF upload, a voice note) stalls every other
# user. Unbounded is the wrong answer too: .concurrent_updates(True) means 256,
# which is the entire connection pool and far past what a 5-request-per-minute
# model quota can serve. Per-user ordering is kept by atlas.engine.turnlock.
MAX_CONCURRENT_UPDATES = 16


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # httpx logs full request URLs at INFO, which puts the bot token in plaintext
    # in every log line. Anyone reading a log or a screen share could hijack the bot.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext.Updater").setLevel(logging.WARNING)
    _quiet_sdk_chatter()


class _DropSdkChatter(logging.Filter):
    """google-genai logs an INFO and a WARNING on every model call. The warning
    recommends a chat API that would change nothing for a stateless turn, and the
    pair made up most of the journal, burying the lines that matter."""

    NOISE = (
        "AFC is enabled with max remote calls",
        "Direct use of automatic function calling",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(noise in message for noise in self.NOISE)


def _quiet_sdk_chatter() -> None:
    sdk = logging.getLogger("google_genai.models")
    if not any(isinstance(f, _DropSdkChatter) for f in sdk.filters):
        sdk.addFilter(_DropSdkChatter())


async def _keepalive(context) -> None:
    ping(get_settings().public_url)


async def _on_error(update, context) -> None:
    """Catch anything a handler let escape.

    Without this, python-telegram-bot logs a full traceback and the user simply
    never hears back. Redeploys also surface a transient getUpdates Conflict
    while the old instance drains, which is noise rather than a fault.
    """
    error = context.error
    if isinstance(error, Conflict):
        log.warning("another instance is polling; this resolves on its own")
        return
    if isinstance(error, NetworkError) and update is None:
        # getUpdates hit a network blip (Telegram answers Bad Gateway some
        # nights). PTB retries on its own and no user is waiting on it.
        log.warning("telegram polling network error, retrying: %s", error)
        return

    log.exception("unhandled error while processing an update", exc_info=error)

    message = getattr(update, "effective_message", None)
    if message is not None:
        try:
            await message.reply_text(
                "Something went wrong on my side just then. Try me again?"
            )
        except Exception:
            log.debug("could not deliver the error notice")


def _polling_is_dead(app) -> bool:
    """True when the Application is up but the poller underneath it has finished.

    Observed on Render: a redeploy overlap raises getUpdates Conflict, the
    Updater's polling task ends, and the Application, its job queue and the
    health-server thread all carry on. The process then answers 200 forever
    while fetching no messages, so the host never restarts it.

    Both flags drop together on a genuine shutdown; only the zombie shows a
    running Application with a finished polling task. The private attribute is
    read defensively — if a future version renames it, this degrades to never
    firing rather than to a crash.
    """
    updater = getattr(app, "updater", None)
    if updater is None or not app.running:
        return False
    task = getattr(updater, "_Updater__polling_task", None)
    return task is not None and task.done()


def _watch_polling(app) -> None:
    while True:
        time.sleep(WATCHDOG_INTERVAL)
        try:
            if not _polling_is_dead(app):
                continue
            log.error("telegram polling has died; exiting so the host restarts us")
        except Exception:  # noqa: BLE001 - the watchdog must never take the bot down
            log.debug("watchdog check failed", exc_info=True)
            continue
        _die()


def _die(code: int = 1) -> None:
    """Report unhealthy, flush, and go — so the platform restarts a fresh process."""
    mark_polling_stopped()
    # Flush by hand: _exit skips atexit, and losing these lines would leave the
    # restart looking unexplained in the host's log.
    for handler in logging.getLogger().handlers:
        handler.flush()
    # _exit, not sys.exit: a hung shutdown must not keep the zombie alive, and
    # that hang is the exact failure being fixed.
    os._exit(code)


def main() -> None:
    settings = get_settings()
    _configure_logging(settings.log_level)

    # Bind the port BEFORE touching the database. Hosts kill a web service that
    # never opens a port, so doing this second turns any database problem into
    # two misleading errors instead of one useful one.
    # A host that probes health over the network (a PaaS, with PUBLIC_URL set)
    # needs every interface. On the VM nothing outside needs it, and /diag spends
    # provider quota, so it answers on loopback only.
    start_health_server(settings.port, "0.0.0.0" if settings.public_url else "127.0.0.1")

    # No yfinance prewarm any more: price history and earnings come over plain
    # HTTP, so pandas and numpy stay out of a 1 GiB machine's memory unless a
    # rare fallback actually needs them.

    init_db()

    app = (
        ApplicationBuilder()
        .token(settings.telegram_token)
        .concurrent_updates(MAX_CONCURRENT_UPDATES)
        .build()
    )

    # New messages in private chats only. Edits used to reach handlers that read
    # update.message, which is None for an edit. And a reply in a group is built
    # from the sender's private history and facts, which the group would read.
    direct = filters.UpdateType.MESSAGE & filters.ChatType.PRIVATE

    # /start only: Telegram's UI sends it on first open. No other command exists.
    app.add_handler(CommandHandler("start", handlers.start, filters=direct))
    app.add_handler(MessageHandler(direct & filters.TEXT & ~filters.COMMAND, handlers.on_text))
    app.add_handler(MessageHandler(direct & (filters.VOICE | filters.AUDIO), handlers.on_voice))
    app.add_handler(MessageHandler(direct & filters.PHOTO, handlers.on_photo))
    app.add_handler(MessageHandler(direct & filters.Document.ALL, handlers.on_document))
    app.add_error_handler(_on_error)

    if app.job_queue is not None:
        scheduler.install(app.job_queue)
        alerts.install(app.job_queue)
        if settings.public_url:
            app.job_queue.run_repeating(
                _keepalive, interval=KEEPALIVE_INTERVAL, first=KEEPALIVE_INTERVAL
            )
            log.info("keep-alive scheduled every %s", KEEPALIVE_INTERVAL)
        else:
            # Only a problem on a host that sleeps when idle. On an always-on
            # box the self-ping is pointless, so state the condition rather
            # than predicting a cold start that will never come.
            log.info(
                "PUBLIC_URL not set: self-ping disabled (correct on an "
                "always-on host; on a sleeping free tier expect ~50s cold starts)"
            )
    else:
        log.warning("no job queue: briefings and keep-alive are disabled")

    threading.Thread(
        target=_watch_polling, args=(app,), name="polling-watchdog", daemon=True
    ).start()

    log.info("Atlas is up")
    try:
        # Drop the backlog. Telegram queues updates for 24h while the bot is
        # down, and answering yesterday's "what's AAPL at?" on restart is worse
        # than not answering it — stale prices, and the burst spends the quota.
        # Messages only: every other update type is one Atlas has no handler for,
        # and fetching it costs a round trip for nothing.
        app.run_polling(drop_pending_updates=True, allowed_updates=[Update.MESSAGE])
    except Exception:
        log.exception("telegram polling crashed; exiting so the host restarts us")
        _die(1)
        return
    # run_polling returns only after a stop signal: PTB catches SIGTERM and shuts
    # the Application down in order. Nothing in Atlas calls stop_running, and a
    # poller that dies underneath a running Application never returns here at
    # all; the watchdog above catches that. So this is a deploy or a restart, and
    # exiting 1 made every one of them read as a crash in the journal.
    log.info("stopped on request; exiting")
    _die(0)


if __name__ == "__main__":
    main()
