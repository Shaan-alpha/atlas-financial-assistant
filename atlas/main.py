import logging

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from atlas.config import get_settings
from atlas.db.session import init_db
from atlas.ingress import handlers


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    init_db()

    app = ApplicationBuilder().token(settings.telegram_token).build()

    # /start only: Telegram's UI sends it on first open. No other command exists.
    app.add_handler(CommandHandler("start", handlers.start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.on_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handlers.on_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handlers.on_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handlers.on_document))

    app.run_polling()


if __name__ == "__main__":
    main()
