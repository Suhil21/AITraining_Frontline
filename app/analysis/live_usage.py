"""Ask each configured language-model slot what a set of messages really costs.

The count a tokenizer gives is an estimate for any model it was not built for. The number a
provider bills is the `usage.prompt_tokens` in its own response, so this sends the messages for
real, once per slot, and reads that field. `max_tokens=1` keeps the reply (and the bill) to a
token or two, though some hosts ignore the limit. A slot that does not answer is left out of the
result and named on stderr; nothing here falls back to a guess.

Used by `scripts/capture_token_growth.py` (to record it) and by `main.py tokens` (to show it).
"""

import sys
from collections.abc import Mapping, Sequence
from typing import cast

from openai.types.chat import ChatCompletionMessageParam

from app.clients.llm_client import call_slot, model_slots


def live_usage(messages: Sequence[Mapping[str, str]]) -> dict[str, dict]:
    """slot label -> {"prompt_tokens": ..., "model_returned": ...}, for each slot that answered."""
    usage = {}
    for slot in model_slots():
        try:
            response = call_slot(
                slot, cast(list[ChatCompletionMessageParam], list(messages)), max_tokens=1
            )
        except Exception as exc:  # noqa: BLE001 - any failure means "this slot is left out"
            print(
                f"  slot {slot.label} did not answer ({type(exc).__name__}): left out",
                file=sys.stderr,
            )
            continue
        if response.usage is None:
            print(f"  slot {slot.label} returned no usage: left out", file=sys.stderr)
            continue
        usage[slot.label] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "model_returned": response.model,
        }
    return usage
