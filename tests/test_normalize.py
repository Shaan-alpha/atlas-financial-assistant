from atlas.ingress.normalize import InboundMessage, caption_or_default


def test_inbound_defaults_to_empty_attachments():
    m = InboundMessage(telegram_id=5, text="hi")
    assert m.attachments == []


def test_caption_fallback_used_when_no_caption():
    assert caption_or_default(None, "image") == "The user sent an image with no caption."
    assert (
        caption_or_default("   ", "document")
        == "The user sent a document with no caption."
    )


def test_caption_preserved_when_present():
    assert caption_or_default("what is this chart", "image") == "what is this chart"
