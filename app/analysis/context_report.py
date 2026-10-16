from app.analysis.live_usage import live_usage
from app.clients.llm_client import model_slots
from app.config.settings import load_decision_settings, load_token_count_settings

from .sample_thread import load_exchanges, thread_after
from .token_count import count_tokens, flatten_turns, naive_estimate, window_share


def growth_lines() -> list[str]:
    settings = load_token_count_settings()
    threads = [thread_after(n) for n in range(1, len(load_exchanges()) + 1)]

    usage, note = [live_usage(turns) for turns in threads], "live, just now"

    slots = list(dict.fromkeys(slot for row in usage for slot in row))  # configured order
    lines = ["turns  litellm  naive  error" + "".join(f"  api:{slot}" for slot in slots)]

    for exchange_count, turns in enumerate(threads, start=1):
        real, estimate = count_tokens(turns, settings).total, naive_estimate(flatten_turns(turns))
        row = f"{exchange_count * 2:>5}  {real:>7}  {estimate:>5}  {real - estimate:>+5}"
        for slot in slots:
            billed = usage[exchange_count - 1].get(slot, {}).get("prompt_tokens")
            row += f"  {billed if billed is not None else '-':>{len('api:' + slot)}}"
        lines.append(row)
    if slots:
        lines.append(f"(api: = prompt tokens the slot reported for these messages, {note})")
    return lines


def window_lines() -> list[str]:
    counted = count_tokens(thread_after(len(load_exchanges())), load_token_count_settings())
    windows = {"decision": load_decision_settings().context_window}
    windows.update({slot.label: slot.context_window for slot in model_slots()})
    lines = [
        f"\nfull thread = {counted.total} tokens ({counted.tokenizer}, counted for {counted.model})"
    ]
    for name, window in windows.items():
        if window:
            lines.append(
                f"  {name:<12} window {window:>8}  uses {window_share(counted.total, window):.2%}"
            )
    return lines


def report(recorded_usage: bool = False) -> str:
    return "\n".join([*growth_lines(), *window_lines()])