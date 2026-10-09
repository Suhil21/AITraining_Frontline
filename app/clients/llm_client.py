import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Seconds a single slot may take. 20 has worked for fast models; a reasoning model that spends
# hundreds of hidden tokens on a short reply needs more (observed: 25 s for a two-sentence
# draft). Read per call so a corrected .env takes effect immediately.
DEFAULT_TIMEOUT_SECONDS = 20.0

SLOT_PREFIXES = (
    ("primary", "MODEL"),
    ("fallback", "FALLBACK_MODEL"),
    ("fallback_2", "FALLBACK_2_MODEL"),
    ("fallback_3", "FALLBACK_3_MODEL"),
)


@dataclass(frozen=True)
class ModelSlot:
    label: str
    base_url: str
    api_key: str
    model_name: str
    context_window: int


def model_slots() -> list[ModelSlot]:
    """Ordered model slots, read fresh from os.environ on every call — never cached,
    so a corrected .env or a seeded failure takes effect on the very next call. A
    slot only counts as configured once its base URL, model name and context window
    are all present; a blank API key is fine for a local endpoint that doesn't need
    one.

    The context window is per slot and is read from the slot, never from a constant.
    The last slot in .env.example has a 4,096-token window against the first slot's
    262,000 — a 64x drop. A chain that quietly degrades to it changes what the code
    can honestly claim about how much thread it can send.
    """
    slots = []
    for label, prefix in SLOT_PREFIXES:
        base_url = os.environ.get(f"{prefix}_BASE_URL")
        model_name = os.environ.get(f"{prefix}_NAME")
        context_window = os.environ.get(f"{prefix}_CONTEXT_WINDOW")
        if base_url and model_name and context_window:
            slots.append(
                ModelSlot(
                    label=label,
                    base_url=base_url,
                    api_key=os.environ.get(f"{prefix}_API_KEY", ""),
                    model_name=model_name,
                    context_window=int(context_window),
                )
            )
    return slots


class AllSlotsFailedError(RuntimeError):
    """Every configured slot failed. Carries each attempt's slot label, model name,
    exception type and message, so a caller can report exactly what happened rather
    than just that something did — which is what turns a seeded failure into a
    diagnosis instead of a guess."""

    def __init__(self, attempts: list[dict]):
        self.attempts = attempts
        super().__init__(f"No configured model slot produced a response: {attempts}")


def request_timeout_seconds() -> float:
    return float(os.environ.get("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))


def client_for(slot: ModelSlot, max_retries: int = 0) -> OpenAI:
    """The SDK client for one slot: the per-call timeout, no redirects, and no retries unless
    asked. Retrying the same slot is for a caller that must stay on one model."""
    return OpenAI(
        base_url=slot.base_url,
        api_key=slot.api_key,
        timeout=request_timeout_seconds(),
        max_retries=max_retries,
    )


def call_slot(
    slot: ModelSlot, messages: list[dict], max_retries: int = 0, **create_kwargs
) -> object:
    """Ask ONE slot, with no fallback: for a caller that must know which endpoint answered.

    Raises whatever the SDK raises. `call_with_fallback` is the one to use when any slot will do.
    """
    return client_for(slot, max_retries).chat.completions.create(
        model=slot.model_name, messages=messages, **create_kwargs
    )


def call_with_fallback(
    messages: list[dict], **create_kwargs
) -> tuple[ModelSlot, object, list[dict]]:
    """Try each configured slot in order until one answers a real chat completion.

    Returns (slot, response, attempts) — the slot that answered, its raw response
    (callers pull out whatever field they need: content, system_fingerprint,
    finish_reason, ...), and every earlier slot's failed attempt, even on success.

    A per-call timeout (20 seconds unless LLM_TIMEOUT_SECONDS says otherwise) is required,
    not a precaution: an unresponsive
    endpoint hangs with no error at all otherwise, blocking the whole chain instead
    of falling through to the slot that would have answered.

    Raises AllSlotsFailedError (carrying the same attempts) if no slot produces a
    response.
    """
    attempts: list[dict] = []
    for slot in model_slots():
        # max_retries=0: the SDK's default of 2 silently multiplies the timeout by up
        # to 3x on a slow slot. Retrying the same slot is redundant anyway — falling
        # through to the next slot is this function's job, not the client's.
        client = OpenAI(
            base_url=slot.base_url,
            api_key=slot.api_key,
            timeout=request_timeout_seconds(),
            max_retries=0,
            # A redirect on a POST is followed as a GET, and the server then answers 405: a
            # misleading symptom (observed live: a subdomain that 301s to its parent domain).
            # Not following it makes the failure say what it is.
            http_client=httpx.Client(follow_redirects=False),
        )
        try:
            response = client.chat.completions.create(
                model=slot.model_name,
                messages=messages,
                **create_kwargs,
            )
            return slot, response, attempts
        except Exception as exc:  # noqa: BLE001 — deliberate: any failure (auth,
            # connection, rate limit, malformed response) means "this slot didn't
            # work," and the response is the same either way — record it and try the
            # next slot. Narrowing this to specific exception types would mean naming
            # every provider's exception hierarchy up front, and missing one would
            # silently break the fallback chain instead of trying the next slot.
            attempts.append(
                {
                    "slot": slot.label,
                    "model": slot.model_name,
                    "error": type(exc).__name__,
                    "message": str(exc)[:300],
                }
            )

    raise AllSlotsFailedError(attempts)
