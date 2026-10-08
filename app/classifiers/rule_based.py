from app.classifiers.categories import OTHER_CATEGORY

KEYWORD_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("warranty", ("warranty", "rma", "replacement", "refund", "dead on arrival")),
    ("camera", ("camera", "libcamera", "picamera", "csi", "imx")),
    ("networking", ("wifi", "wi-fi", "ethernet", "network", "ssh", "ip address")),
    ("storage", ("sd card", "sdcard", "microsd", "ssd", "usb drive", "corrupt")),
    ("boot", ("boot", "won't start", "wont start", "no display", "rainbow screen")),
    ("power", ("power", "psu", "supply", "voltage", "amps", "undervolt")),
]


def classify_by_rules(ticket_text: str) -> str:
    """Return the category this ticket's words match, or "other" if none do.

    Case-insensitive substring matching, which is deliberately crude: it is what the
    real thing looks like before anyone has time to make it better, and its failures are
    the argument for the next section of the lab.
    """
    haystack = ticket_text.lower()
    for category, keywords in KEYWORD_RULES:
        if any(keyword in haystack for keyword in keywords):
            return category

    return OTHER_CATEGORY
