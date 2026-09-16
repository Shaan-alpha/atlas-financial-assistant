"""Telegram handlers.

Only /start is handled, because Telegram's own UI sends it on first open. No other
command exists — the brief forbids a command surface.
"""

import asyncio
import functools
import io
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from atlas.engine.conversation import respond
from atlas.ingress.guard import TurnGuard
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
VOICE_TOO_LONG = "That voice note is longer than I can take. Could you keep it under ten minutes?"
UNSUPPORTED_DOCUMENT = (
    "I can read PDFs, images, and CSV or text files up to 10 MB. For a spreadsheet, "
    "export it to CSV or send me a Google Sheets link."
)

# Checked against Telegram's own metadata before anything is downloaded: the
# bytes are held in RAM for the whole turn, on a 1 GiB machine.
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_VOICE_SECONDS = 600
# Albums arrive as one update per photo, a few hundred milliseconds apart.
ALBUM_WAIT = 1.5
MAX_ALBUM_IMAGES = 10

GUARD = TurnGuard()
_albums: dict[str, list] = {}


def _readable(mime: str | None) -> bool:
    return bool(mime) and (
        mime == "application/pdf" or mime.startswith("text/") or mime.startswith("image/")
    )


def _guarded(handler):
    """Admit the turn before any work, and always give the slot back."""

    @functools.wraps(handler)
    async def wrapper(update: Update, context, *args):
        telegram_id = update.effective_user.id
        refusal = GUARD.admit(telegram_id)
        if refusal:
            await send_reply(update.message, refusal)
            return
        try:
            await handler(update, context, *args)
        finally:
            GUARD.release(telegram_id)

    return wrapper


async def _typing(update: Update) -> None:
    await update.effective_chat.send_action(ChatAction.TYPING)


async def _user_id(update: Update) -> int:
    user = update.effective_user
    # Blocking SQLAlchemy, and it runs ahead of the typing indicator on every
    # single update — off-thread for the same reason as transcribe() below.
    return await asyncio.to_thread(
        store.get_or_create_user, user.id, name=user.first_name
    )


@_guarded
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Telegram sends /start automatically on first open. Greet, never mention commands."""
    uid = await _user_id(update)
    profile = await asyncio.to_thread(store.profile_snapshot, uid)
    if profile["onboarding_state"] == "new":
        await send_reply(update.message, GREETING)
    else:
        await _typing(update)
        await send_reply(update.message, await respond(uid, "I'm back."))


@_guarded
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = await _user_id(update)
    await _typing(update)
    await send_reply(update.message, await respond(uid, update.message.text))


@_guarded
async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    voice = update.message.voice or update.message.audio
    if (voice.duration or 0) > MAX_VOICE_SECONDS:
        await send_reply(update.message, VOICE_TOO_LONG)
        return

    uid = await _user_id(update)
    await _typing(update)

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
    del audio
    if text is None:
        await send_reply(update.message, VOICE_FAILED)
        return

    await send_reply(update.message, await respond(uid, text))


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """One photo, or the first of an album, starts a turn; the rest of an album
    join it. Deliberately unguarded here: an album's later photos are not turns."""
    group = update.message.media_group_id
    if group and group in _albums:
        _albums[group].append(update.message)
        return
    if group:
        _albums[group] = [update.message]
    try:
        await _photo_turn(update, context, group)
    finally:
        # A refused or failed turn never collected its album.
        if group:
            _albums.pop(group, None)


@_guarded
async def _photo_turn(update: Update, context, group: str | None) -> None:
    uid = await _user_id(update)
    await _typing(update)

    if group:
        await asyncio.sleep(ALBUM_WAIT)
        messages = _albums.pop(group, [update.message])
    else:
        messages = [update.message]

    attachments = []
    for message in messages[:MAX_ALBUM_IMAGES]:
        photo = message.photo[-1]  # last entry is the largest rendition
        handle = await context.bot.get_file(photo.file_id)
        image = bytes(await handle.download_as_bytearray())
        attachments.append({"kind": "image", "bytes": image, "mime": "image/jpeg"})

    caption = next((m.caption for m in messages if m.caption and m.caption.strip()), None)
    kind = "image" if len(attachments) == 1 else f"set of {len(attachments)} images"
    prompt = caption_or_default(caption, kind)
    await send_reply(update.message, await respond(uid, prompt, attachments))


@_guarded
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upload to the Gemini Files API so the model reads the real document.

    Native ingest preserves tables and charts that text extraction would discard.
    """
    document = update.message.document
    mime = document.mime_type
    if not _readable(mime) or (document.file_size or 0) > MAX_DOCUMENT_BYTES:
        await send_reply(update.message, UNSUPPORTED_DOCUMENT)
        return

    uid = await _user_id(update)
    await _typing(update)

    handle = await context.bot.get_file(document.file_id)
    blob = bytes(await handle.download_as_bytearray())

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
    finally:
        del blob

    # Kept so a follow-up question ("and the segment margins?") still has the
    # document in front of the model, not just this turn's reply about it.
    await asyncio.to_thread(
        store.add_document, uid, uploaded.uri, document.file_name or "document", mime
    )

    prompt = caption_or_default(update.message.caption, "document")
    attachments = [{"kind": "file", "uri": uploaded.uri, "mime": mime}]
    await send_reply(update.message, await respond(uid, prompt, attachments))
