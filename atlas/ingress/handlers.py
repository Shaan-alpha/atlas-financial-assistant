"""Telegram handlers.

Only /start is handled, because Telegram's own UI sends it on first open. No other
command exists — the brief forbids a command surface.
"""

import asyncio
import io
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from atlas.engine.conversation import respond
from atlas.ingress.normalize import caption_or_default
from atlas.ingress.reply import send_reply
from atlas.integrations.gemini import get_client
from atlas.integrations.groq import transcribe
from atlas.memory import store

log = logging.getLogger(__name__)

GREETING = (
    "I'm Atlas — I follow markets so you don't have to.\n\n"
    "Before we get going: what best describes what you do?"
)
VOICE_FAILED = "I couldn't make out that voice note. Mind typing it?"


async def _typing(update: Update) -> None:
    await update.effective_chat.send_action(ChatAction.TYPING)


async def _user_id(update: Update) -> int:
    user = update.effective_user
    # Blocking SQLAlchemy, and it runs ahead of the typing indicator on every
    # single update — off-thread for the same reason as transcribe() below.
    return await asyncio.to_thread(
        store.get_or_create_user, user.id, name=user.first_name
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Telegram sends /start automatically on first open. Greet, never mention commands."""
    uid = await _user_id(update)
    profile = await asyncio.to_thread(store.profile_snapshot, uid)
    if profile["onboarding_state"] == "new":
        await send_reply(update.message, GREETING)
    else:
        await _typing(update)
        await send_reply(update.message, await respond(uid, "I'm back."))


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = await _user_id(update)
    await _typing(update)
    await send_reply(update.message, await respond(uid, update.message.text))


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = await _user_id(update)
    await _typing(update)

    voice = update.message.voice or update.message.audio
    handle = await context.bot.get_file(voice.file_id)
    audio = bytes(await handle.download_as_bytearray())

    # Bias transcription toward the tickers this user actually follows; spoken
    # ticker letters are the most error-prone thing in a finance voice note.
    symbols = [
        item["symbol"] for item in await asyncio.to_thread(store.get_watchlist, uid)
    ]

    # transcribe() is a blocking HTTP call — off-thread so one voice note does not
    # stall every other user's turn.
    text = await asyncio.to_thread(transcribe, audio, "voice.ogg", symbols)
    if text is None:
        await send_reply(update.message, VOICE_FAILED)
        return

    await send_reply(update.message, await respond(uid, text))


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = await _user_id(update)
    await _typing(update)

    photo = update.message.photo[-1]  # last entry is the largest rendition
    handle = await context.bot.get_file(photo.file_id)
    image = bytes(await handle.download_as_bytearray())

    prompt = caption_or_default(update.message.caption, "image")
    attachments = [{"kind": "image", "bytes": image, "mime": "image/jpeg"}]
    await send_reply(update.message, await respond(uid, prompt, attachments))


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upload to the Gemini Files API so the model reads the real document.

    Native ingest preserves tables and charts that text extraction would discard.
    """
    uid = await _user_id(update)
    await _typing(update)

    document = update.message.document
    handle = await context.bot.get_file(document.file_id)
    blob = bytes(await handle.download_as_bytearray())

    mime = document.mime_type or "application/pdf"
    try:
        # Blocking upload — off-thread for the same reason as voice.
        uploaded = await asyncio.to_thread(
            lambda: get_client().files.upload(
                file=io.BytesIO(blob),
                config={"mime_type": mime, "display_name": document.file_name},
            )
        )
    except Exception:
        log.exception("file upload failed")
        await send_reply(update.message, "I couldn't read that file. Try resending it?")
        return

    prompt = caption_or_default(update.message.caption, "document")
    attachments = [{"kind": "file", "uri": uploaded.uri, "mime": mime}]
    await send_reply(update.message, await respond(uid, prompt, attachments))
