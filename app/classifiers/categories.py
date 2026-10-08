"""The one list of teams a ticket can be routed to.

Every classifier in this repository (rules, language model, decision model) answers from
this list, so it lives in one place. The descriptions are not decoration: the decision
model is given them as the meaning of each option.
"""

OTHER_CATEGORY = "other"

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "warranty": "Warranty claims, returns, refunds, replacements or dead-on-arrival units.",
    "camera": "Camera modules, ribbon cables, libcamera or picamera problems.",
    "networking": "Wi-Fi, ethernet, SSH or IP address problems.",
    "storage": "SD cards, SSDs, USB drives or corrupted storage.",
    "boot": "The board does not start, shows no display or stops during start-up.",
    "power": "Power supplies, voltage, undervoltage, or unexpected resets and shutdowns.",
    OTHER_CATEGORY: "Anything that is not a technical support request for the above teams.",
}

CATEGORIES = tuple(CATEGORY_DESCRIPTIONS)

URGENCY_DESCRIPTIONS: dict[str, str] = {
    "low": "Informational or not time-sensitive.",
    "normal": "A real problem, but the customer can keep working.",
    "high": "The customer is blocked, losing money, or says it is urgent.",
}

URGENCIES = tuple(URGENCY_DESCRIPTIONS)
