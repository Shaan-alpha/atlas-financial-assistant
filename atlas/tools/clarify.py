"""The clarify tool.

Asking a follow-up is a first-class, loggable action rather than a prompt hope. The
result renders as plain conversational text — never buttons, which the brief forbids.
"""

from atlas.tools.result import err, ok

MAX_OPTIONS = 4


def clarify(question: str, options: list[str]) -> dict:
    """Ask the user one short follow-up question before answering.

    Use ONLY when the ambiguity materially changes the answer. "Tell me about Apple"
    warrants a clarification; "Apple's P/E" does not — just answer that.

    Args:
        question: One short question, phrased conversationally.
        options: Two to four concrete interpretations to offer, as plain phrases.
    """
    if len(options) > MAX_OPTIONS:
        return err("too_many_options", f"Offer at most {MAX_OPTIONS} interpretations.")
    if len(options) < 2:
        return err("too_few_options", "A clarification needs at least two interpretations.")
    return ok({"question": question, "options": options}, source="atlas-clarify")
