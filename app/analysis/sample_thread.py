import json

EXCHANGES_PATH = "fixtures/sample_thread.json"

SYSTEM_TURN = {
    "role": "system",
    "content": "You are Frontline, drafting replies to Raspberry Pi support tickets "
    "for an agent to review before it reaches the customer.",
}


def load_exchanges(path: str = EXCHANGES_PATH) -> list[dict]:
    """Every exchange, each a `{"user": ..., "assistant": ...}` pair, in the order they happened."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def thread_after(exchange_count: int) -> list[dict]:
    """The turns exactly as they would be sent after `exchange_count` exchanges."""
    turns = [SYSTEM_TURN]
    for exchange in load_exchanges()[:exchange_count]:
        turns.append({"role": "user", "content": exchange["user"]})
        turns.append({"role": "assistant", "content": exchange["assistant"]})
    return turns
