import re

# Two claim shapes a support draft gets wrong: a warranty statement, and a fix attributed to a
# firmware version. A person chose these words; the pattern finds nothing it was not given.
CHECKABLE_PATTERN = re.compile(r"\bwarranty\b|\bfirmware\s+\d+(\.\d+)?\b", re.IGNORECASE)


def flag_unverified_claims(text: str) -> list[str]:
    """Return the sentences that make a claim someone could look up.

    This only *finds* them. It does not check them, and it cannot: nothing in this repository
    holds the warranty policy or a firmware changelog. An empty list means no sentence matched
    the pattern, not that the draft is true.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence for sentence in sentences if CHECKABLE_PATTERN.search(sentence)]


def build_claim_prompt(ticket_text: str) -> str:
    return (
        "Draft a reply for a Raspberry Pi support agent to review, to this ticket: "
        f"{ticket_text} State plainly whether the issue is covered under warranty, and "
        "whether it's fixed in a specific firmware version."
    )
