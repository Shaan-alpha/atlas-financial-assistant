"""Telegram ingress: who gets a turn, and what is read before one starts."""

import asyncio
import datetime as dt
from types import SimpleNamespace

import pytest
from telegram import Chat, Message, Update, User

import atlas.ingress.guard as guard
import atlas.ingress.handlers as handlers
from atlas.memory import store

pytestmark = pytest.mark.usefixtures("fresh_db")


# ------------------------------------------------------------------ guard


class _Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


def test_one_turn_running_and_one_queued_is_the_limit():
    g = guard.TurnGuard(clock=_Clock())

    assert g.admit(1) is None
    assert g.admit(1) is None
    assert g.admit(1) == guard.BUSY
    # Other people are unaffected.
    assert g.admit(2) is None

    g.release(1)
    assert g.admit(1) is None


def test_bursts_are_slowed_and_then_forgiven():
    clock = _Clock()
    g = guard.TurnGuard(clock=clock)
    for _ in range(guard.BURST_TURNS):
        assert g.admit(1) is None
        g.release(1)

    assert g.admit(1) == guard.SLOW_DOWN

    clock.now += guard.BURST_WINDOW
    assert g.admit(1) is None


def test_a_daily_ceiling_holds_until_the_window_rolls(monkeypatch):
    monkeypatch.setattr(guard, "DAILY_TURNS", 3)
    clock = _Clock()
    g = guard.TurnGuard(clock=clock)
    for _ in range(3):
        assert g.admit(1) is None
        g.release(1)
        clock.now += guard.BURST_WINDOW

    assert g.admit(1) == guard.DAILY_LIMIT

    clock.now += guard.DAY
    assert g.admit(1) is None


# ----------------------------------------------------------- update routing


def _update(chat_type="private", edited=False, text="hello"):
    message = Message(
        message_id=1,
        date=dt.datetime(2026, 9, 16, tzinfo=dt.UTC),
        chat=Chat(id=10, type=chat_type),
        from_user=User(id=7, first_name="Shaan", is_bot=False),
        text=text,
    )
    if edited:
        return Update(update_id=1, edited_message=message)
    return Update(update_id=1, message=message)


def _text_handler(app):
    from telegram.ext import MessageHandler

    return next(
        h
        for group in app.handlers.values()
        for h in group
        if isinstance(h, MessageHandler) and h.callback is handlers.on_text
    )


@pytest.fixture
def app(monkeypatch):
    import atlas.main as main
    from telegram.ext import Application

    seen = {}

    def _run_polling(self, **kwargs):
        seen["app"], seen["polling"] = self, kwargs

    monkeypatch.setattr(Application, "run_polling", _run_polling)
    monkeypatch.setattr(main.os, "_exit", lambda code: None)
    monkeypatch.setattr(main, "mark_polling_stopped", lambda: None)
    monkeypatch.setattr(main, "start_health_server", lambda port, host=None: None)
    monkeypatch.setattr(main, "_configure_logging", lambda level: None)
    main.main()
    return seen


def test_only_new_messages_are_fetched(app):
    """An edit reached handlers that read update.message, which is None for an
    edit, and the user got a false "something went wrong"."""
    assert app["polling"]["allowed_updates"] == [Update.MESSAGE]


def test_edits_never_reach_the_text_handler(app):
    assert not _text_handler(app["app"]).check_update(_update(edited=True))


def test_group_chats_are_ignored(app):
    """A reply in a group is built from the sender's private history and facts,
    which the rest of the group would read."""
    handler = _text_handler(app["app"])

    assert handler.check_update(_update(chat_type="private"))
    assert not handler.check_update(_update(chat_type="group"))
    assert not handler.check_update(_update(chat_type="supergroup"))


# --------------------------------------------------------------- handlers


class _Msg:
    def __init__(self, **fields):
        self.replies = []
        self.caption = None
        self.media_group_id = None
        self.__dict__.update(fields)

    async def reply_text(self, text, parse_mode=None):
        self.replies.append(text)


def _handler_update(message, telegram_id=7):
    async def _action(action):
        return None

    return SimpleNamespace(
        effective_user=SimpleNamespace(id=telegram_id, first_name="Shaan"),
        effective_chat=SimpleNamespace(send_action=_action),
        effective_message=message,
        message=message,
    )


class _Bot:
    def __init__(self):
        self.fetched = []

    async def get_file(self, file_id):
        self.fetched.append(file_id)

        async def _download():
            return bytearray(b"bytes-of-" + file_id.encode())

        return SimpleNamespace(download_as_bytearray=_download)


@pytest.fixture
def turns(monkeypatch):
    seen = []

    async def _respond(user_id, text, attachments=None):
        seen.append({"text": text, "attachments": attachments or []})
        return "answered"

    monkeypatch.setattr(handlers, "respond", _respond)
    monkeypatch.setattr(handlers, "GUARD", guard.TurnGuard())
    return seen


async def test_a_refused_turn_never_reaches_the_model(monkeypatch, turns):
    class _Full:
        def admit(self, telegram_id):
            return guard.SLOW_DOWN

        def release(self, telegram_id):
            raise AssertionError("nothing was admitted")

    monkeypatch.setattr(handlers, "GUARD", _Full())
    message = _Msg(text="nvda?")

    await handlers.on_text(_handler_update(message), None)

    assert turns == []
    assert message.replies == [guard.SLOW_DOWN]


async def test_an_oversized_document_is_refused_before_download(turns):
    bot = _Bot()
    message = _Msg(
        document=SimpleNamespace(
            file_id="big", file_size=50 * 1024 * 1024, mime_type="application/pdf", file_name="x.pdf"
        )
    )

    await handlers.on_document(_handler_update(message), SimpleNamespace(bot=bot))

    assert bot.fetched == []
    assert turns == []
    assert message.replies == [handlers.UNSUPPORTED_DOCUMENT]


@pytest.mark.parametrize("mime", [None, "application/zip", "application/vnd.ms-excel"])
async def test_an_unreadable_document_type_gets_a_plain_answer(turns, mime):
    bot = _Bot()
    message = _Msg(
        document=SimpleNamespace(file_id="f", file_size=1000, mime_type=mime, file_name="x")
    )

    await handlers.on_document(_handler_update(message), SimpleNamespace(bot=bot))

    assert bot.fetched == []
    assert message.replies == [handlers.UNSUPPORTED_DOCUMENT]


async def test_a_very_long_voice_note_is_refused_before_download(turns):
    bot = _Bot()
    message = _Msg(voice=SimpleNamespace(file_id="v", duration=3600), audio=None)

    await handlers.on_voice(_handler_update(message), SimpleNamespace(bot=bot))

    assert bot.fetched == []
    assert message.replies == [handlers.VOICE_TOO_LONG]


async def test_a_photo_album_is_one_turn(monkeypatch, turns):
    """Telegram delivers an album as one update per photo; each used to start
    its own full model turn, and only the first carried the caption."""
    monkeypatch.setattr(handlers, "ALBUM_WAIT", 0.05)
    bot = _Bot()
    context = SimpleNamespace(bot=bot)
    photos = [
        _Msg(media_group_id="g1", photo=[SimpleNamespace(file_id=f"p{i}")],
             caption="compare these charts" if i == 1 else None)
        for i in range(3)
    ]

    await asyncio.gather(*(handlers.on_photo(_handler_update(p), context) for p in photos))

    assert len(turns) == 1
    assert turns[0]["text"] == "compare these charts"
    assert len(turns[0]["attachments"]) == 3
    assert photos[0].replies == ["answered"]


async def test_an_uploaded_document_is_remembered_for_follow_ups(monkeypatch, turns):
    uploaded = SimpleNamespace(uri="https://files/abc", name="files/abc")

    class _Files:
        def upload(self, file, config):
            return uploaded

    monkeypatch.setattr(handlers, "get_client", lambda: SimpleNamespace(files=_Files()))
    message = _Msg(
        document=SimpleNamespace(
            file_id="d", file_size=2048, mime_type="application/pdf", file_name="results.pdf"
        )
    )

    await handlers.on_document(_handler_update(message), SimpleNamespace(bot=_Bot()))

    uid = store.get_or_create_user(7)
    doc = store.recent_document(uid, within=dt.timedelta(minutes=30))
    assert doc["uri"] == "https://files/abc"
    assert doc["name"] == "results.pdf"
