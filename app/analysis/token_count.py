import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.config.settings import TokenCountSettings, load_token_count_settings


def flatten_turns(turns: list[dict]) -> str:

    parts = []
    for turn in turns:
        parts.append(f"<|im_start|>{turn['role']}<|im_sep|>{turn['content']}<|im_end|>")

    return "".join(parts)


def naive_estimate(text: str) -> int:
    """
    Divide the length by 4
    Example: text = "Hello, world!" -> Hell     , wo    rld! --> 3 tokens
    """
    return round(len(text) / 4)


@dataclass
class TokenCount:
    total: int
    tokenizer: str
    model: str


async def _count(
    messages: Sequence[Mapping[str, object]], settings: TokenCountSettings
) -> TokenCount:
    import litellm  # imported here: it is a heavy import, and only counting needs it

    if not settings:
        settings = load_token_count_settings()

    if not settings.api_key:
        # Counting locally is the intent, so LiteLLM's "no key found" warning is not news.
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)

    result = await litellm.acount_tokens(
        model=settings.model,
        messages=[dict(message) for message in messages],
        api_key=settings.api_key,  # blank on purpose: see the module docstring
    )
    if result.error:
        raise RuntimeError(result.error_message)

    return TokenCount(
        total=result.total_tokens,
        tokenizer=result.tokenizer_type,
        model=settings.model,
    )


def count_tokens(
    messages: Sequence[Mapping[str, object]], settings: TokenCountSettings
) -> TokenCount:

    return asyncio.run(_count(messages, settings))


def window_share(token_count: int, context_window: int) -> float:
    """Fraction of a model's context window a request uses. Above 1.0 it does not fit."""
    return token_count / context_window
