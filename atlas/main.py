import datetime as dt
import logging

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from atlas.config import get_settings
from atlas.db.session import init_db
from atlas.health import ping, start_health_server
from atlas.ingress import handlers

log = logging.getLogger(__name__)

KEEPALIVE_INTERVAL = dt.timedelta(minutes=10)


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


async def _keepalive(context) -> None:
    ping(get_settings().public_url)


def main() -> None:
    settings = get_settings()
    _configure_logging(settings.log_level)
    init_db()

    # Free hosting tiers require an HTTP listener and sleep without traffic.
    start_health_server(settings.port)

    app = ApplicationBuilder().token(settings.telegram_token).build()

    # /start only: Telegram's UI sends it on first open. No other command exists.
    app.add_handler(CommandHandler("start", handlers.start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.on_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handlers.on_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handlers.on_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handlers.on_document))

    if settings.public_url and app.job_queue is not None:
        app.job_queue.run_repeating(
            _keepalive, interval=KEEPALIVE_INTERVAL, first=KEEPALIVE_INTERVAL
        )
        log.info("keep-alive scheduled every %s", KEEPALIVE_INTERVAL)

    log.info("Atlas is up")
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
