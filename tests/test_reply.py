import pytest
from telegram.error import BadRequest

from atlas.ingress.reply import send_reply, to_telegram_markdown


class _FakeMessage:
    """Records reply_text calls; optionally rejects the first (markdown) attempt."""

    def __init__(self, reject_markdown: bool = False):
        self.calls: list[dict] = []
        self.reject_markdown = reject_markdown

    async def reply_text(self, text, parse_mode=None):
        self.calls.append({"text": text, "parse_mode": parse_mode})
        if parse_mode is not None and self.reject_markdown:
            raise BadRequest("Can't parse entities")


def test_double_asterisks_become_telegram_single():
    assert to_telegram_markdown("**NVDA** is up") == "*NVDA* is up"


def test_double_underscores_become_single():
    assert to_telegram_markdown("__note__") == "_note_"


def test_plain_text_is_untouched():
    assert to_telegram_markdown("no markup here") == "no markup here"


async def test_markdown_is_used_when_telegram_accepts_it():
    message = _FakeMessage()

    await send_reply(message, "**NVDA** up 2%")

    assert len(message.calls) == 1
    assert message.calls[0]["text"] == "*NVDA* up 2%"
    assert message.calls[0]["parse_mode"] == "Markdown"


async def test_falls_back_to_plain_text_when_markdown_is_rejected():
    message = _FakeMessage(reject_markdown=True)

    await send_reply(message, "unbalanced *markup")

    assert len(message.calls) == 2
    assert message.calls[1]["parse_mode"] is None
    # The fallback sends the original text, not the transformed one.
    assert message.calls[1]["text"] == "unbalanced *markup"


async def test_a_rejected_send_still_delivers_something():
    message = _FakeMessage(reject_markdown=True)

    await send_reply(message, "hello")

    assert message.calls[-1]["text"] == "hello"
