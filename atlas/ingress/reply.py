"""Sending replies to Telegram.

Telegram's legacy Markdown parser rejects malformed input with BadRequest, which
would otherwise surface to the user as no reply at all. Every send therefore falls
back to plain text rather than failing.
"""

import logging
import re

from telegram.constants import ParseMode
from telegram.error import BadRequest

log = logging.getLogger(__name__)

# Models emit CommonMark **bold**; Telegram legacy Markdown wants *bold*.
_DOUBLE_ASTERISK = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_DOUBLE_UNDERSCORE = re.compile(r"__(.+?)__", re.DOTALL)


def to_telegram_markdown(text: str) -> str:
    """Normalize CommonMark emphasis into Telegram's legacy Markdown dialect."""
    text = _DOUBLE_ASTERISK.sub(r"*\1*", text)
    return _DOUBLE_UNDERSCORE.sub(r"_\1_", text)


async def send_reply(message, text: str) -> None:
    """Reply with Markdown, degrading to plain text if Telegram rejects it."""
    formatted = to_telegram_markdown(text)
    try:
        await message.reply_text(formatted, parse_mode=ParseMode.MARKDOWN)
    except BadRequest:
        log.warning("markdown rejected by Telegram; resending as plain text")
        await message.reply_text(text)
