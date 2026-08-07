from dataclasses import dataclass, field


@dataclass
class InboundMessage:
    telegram_id: int
    text: str
    attachments: list[dict] = field(default_factory=list)


def caption_or_default(caption: str | None, kind: str) -> str:
    """Media with no caption still needs a prompt for the model to act on."""
    if caption and caption.strip():
        return caption.strip()
    article = "an" if kind[:1].lower() in "aeiou" else "a"
    return f"The user sent {article} {kind} with no caption."
