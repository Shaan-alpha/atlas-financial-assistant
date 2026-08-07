from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ok(data: dict, source: str, as_of: str | None = None) -> dict:
    """Successful tool result.

    `source` and `as_of` exist so the model can attribute claims. The system prompt
    requires it to cite them rather than stating figures bare.
    """
    return {"ok": True, "data": data, "source": source, "as_of": as_of or _now_iso()}


def err(error: str, message: str) -> dict:
    """Failed tool result.

    Deliberately returned rather than raised: the model reads `message` and tells the
    user the truth instead of inventing a plausible answer.
    """
    return {"ok": False, "error": error, "message": message}
