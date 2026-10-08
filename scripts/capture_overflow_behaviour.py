"""Capture fixtures/overflow_behaviour.json: what each slot really does past its window.

Run from the repository root, with at least one slot configured in .env:

    uv run python -m scripts.capture_overflow_behaviour                    # every configured slot
    uv run python -m scripts.capture_overflow_behaviour --slot fallback_2  # just that slot

**There is no expected outcome here and the script does not encode one.** Past its window a
provider does one of three things, and all three are real:

1. **raises**: a 400 that says the prompt is too long (or, on some hosts, a nonsense limit such
   as "max_tokens must be at least 1, got -261169", because the host subtracted the prompt from
   the window and went negative);
2. **answers with a signal**: a response whose `finish_reason` is something other than `stop`
   (`length`, `content_filter`), or an `incomplete_details` block;
3. **answers silently**: a normal-looking reply, `finish_reason: stop`, on a prompt the host
   quietly cut down to fit or summarized behind the scene. The only evidence is the usage it reports: far fewer prompt tokens
   than were sent. This is the dangerous one, because nothing fails.

Which one you get is a fact about the slot, not about context windows in general, so each
configured slot is probed on its own (never through the fallback chain, which would hide all but
the last) and every outcome is recorded as it happened.

The combination that demonstrates them
--------------------------------------

* **Raises, with a clear message:** Ollama Cloud serving `gpt-oss:20b`. A 400, "The prompt is too
  long: 159203, model maximum context length: 131072". Its real limit is 131,072, not the 128,000
  you may have configured.
* **Raises, with a misleading message:** NVIDIA's hosted API serving `openai/gpt-oss-20b`. A 400,
  "max_tokens must be at least 1, got -31842". The host subtracted the prompt from the window and
  went negative, and never says "too long".
* **Silent truncation:** local Ollama serving `llama2:7b` with a 4,096-token window (base URL
  `http://localhost:11434/v1`, no key needed; the model tag is `llama2:7b`). It answered normally
  with `finish_reason: stop` on a prompt it had cut to 2,051 tokens out of about 5,128 sent.
  Nothing failed; only the usage figure shows it. It takes about 17 seconds.

So two of the three behaviours can be shown, and the failure messages come in two kinds. A
`finish_reason` of `length` is not evidence of overflow on its own (a `max_tokens` cap produces it
too); it counts here only because it arrives on a prompt that was past the window.

The padding is sized from each slot's own `context_window`, never from a constant, so one that
is small is not under-filled and one that is large is not drowned. It aims `OVERSHOOT` times past
the window, measured with the token counter, which is enough to be unambiguous without sending a
prompt that costs real money: four times a million-token window would be four million tokens.
Raise it only on a slot you do not pay for.
"""

import argparse
import json
import math
import os
import time
from datetime import date

from app.analysis.token_count import count_tokens
from app.clients.llm_client import ModelSlot, call_slot, model_slots
from app.config.settings import TokenCountSettings, load_token_count_settings

OUTPUT_PATH = "fixtures/overflow_behaviour.json"
PADDING_SENTENCE = "Hi! My name is Ishak. This sentence is padding to force an overflow. "
QUESTION = "\nWhat is my name?"
OVERSHOOT = 1.25
# A reply whose reported prompt is under this share of what was sent came from a cut-down prompt.
SILENT_TRUNCATION_BELOW = 0.9
DEFAULT_TIMEOUT_SECONDS = 300  # a prompt this large takes a while even when the host accepts it


def tokens_per_padding_sentence(settings: TokenCountSettings) -> float:
    """Calibrated once with the token counter, not guessed at four characters per token."""
    repeats = 200
    messages = [{"role": "user", "content": PADDING_SENTENCE * repeats}]
    return count_tokens(messages, settings).total / repeats


def oversized_prompt(context_window: int, per_sentence: float) -> tuple[str, int]:
    """A prompt about `OVERSHOOT` times the window, and the token count it was sized to."""
    repeats = math.ceil(context_window * OVERSHOOT / per_sentence)
    return PADDING_SENTENCE * repeats + QUESTION, round(repeats * per_sentence)


def verdict(response, sent_tokens: int) -> str:
    choice = response.choices[0]
    reported = response.usage.prompt_tokens if response.usage else None
    if reported is not None and reported < SILENT_TRUNCATION_BELOW * sent_tokens:
        return "silent truncation"
    if choice.finish_reason != "stop":
        return f"signalled: finish_reason {choice.finish_reason}"
    return "accepted in full"


def probe(slot: ModelSlot, per_sentence: float) -> dict:
    prompt, sent_tokens = oversized_prompt(slot.context_window, per_sentence)
    record: dict = {
        "model": slot.model_name,
        "context_window": slot.context_window,
        "prompt_tokens_sent": sent_tokens,
        "prompt_chars": len(prompt),
    }
    try:
        response = call_slot(slot, [{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001 - the provider's own error shape is what is recorded
        # Narrowing this would mean deciding in advance which failures count, which is the
        # assumption this capture exists to avoid.
        record.update(
            outcome="raised",
            exception_type=type(exc).__name__,
            status_code=getattr(exc, "status_code", None),
            message=str(exc)[:600],
        )
        return record
    usage = response.usage
    record.update(
        outcome="answered",
        verdict=verdict(response, sent_tokens),
        finish_reason=response.choices[0].finish_reason,
        usage_prompt_tokens=usage.prompt_tokens if usage else None,
        usage_completion_tokens=usage.completion_tokens if usage else None,
        model_returned=response.model,
        content_prefix=(response.choices[0].message.content or "")[:300],
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slot", action="append", help="probe only this slot (repeatable)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    os.environ["LLM_TIMEOUT_SECONDS"] = str(args.timeout)

    slots = [s for s in model_slots() if not args.slot or s.label in args.slot]
    if not slots:
        raise SystemExit(
            "no model slot is configured: copy .env.example to .env and fill in at least one "
            "slot's BASE_URL, NAME and CONTEXT_WINDOW before capturing"
        )
    settings = TokenCountSettings(model=load_token_count_settings().model, api_key="")
    per_sentence = tokens_per_padding_sentence(settings)

    result: dict = {"slots": {}}
    if args.slot and os.path.exists(OUTPUT_PATH):  # re-probing one slot keeps the others
        with open(OUTPUT_PATH, encoding="utf-8") as file:
            result = json.load(file)
    result["header"] = {
        "captured_date": date.today().isoformat(),
        "capture_script": "scripts/capture_overflow_behaviour.py",
        "overshoot": OVERSHOOT,
        "tokens_counted_for": settings.model,
        "note": "prompt_tokens_sent is the token counter's estimate for the padding, not a "
        "provider's count; usage_prompt_tokens is what the provider says it processed",
    }
    for slot in slots:
        print(f"probing {slot.label} ({slot.model_name}, window {slot.context_window}) ...")
        started = time.monotonic()
        found = probe(slot, per_sentence)
        found["seconds"] = round(time.monotonic() - started, 1)
        found["captured_date"] = date.today().isoformat()
        result["slots"][slot.label] = found
        outcome = found.get("verdict") or f"raised: {found.get('status_code')}"
        print(f"  -> {outcome} ({found['seconds']} s)", flush=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
