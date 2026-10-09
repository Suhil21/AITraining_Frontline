from app.clients.llm_client import ModelSlot, call_slot, model_slots
from app.config.settings import ConfigError

RUNS_PER_SETTING = 5
RETRIES = 2

SAMPLING_SETTINGS: dict[str, dict] = {
    "default": {},
    "temperature_0": {"temperature": 0},
    "temperature_0_seed_42": {"temperature": 0, "seed": 42},
}


def build_drafting_prompt(ticket_text: str) -> str:
    return (
        "Draft a two-to-three sentence reply for a Raspberry Pi support agent to review, "
        f"to this ticket: {ticket_text}"
    )


def sample_drafts(prompt: str, runs: int = RUNS_PER_SETTING) -> tuple[ModelSlot, dict]:
    """Send the same prompt `runs` times under each setting, to ONE model.

    Not call_with_fallback: a fallback would quietly move a run to another model, and then
    "different replies" could just mean "different models". The SDK retries a failed call on
    the same slot instead.
    """
    slots = model_slots()
    if not slots:
        raise ConfigError("No model slot is configured (MODEL_BASE_URL, MODEL_NAME, ...)")
    slot = slots[0]
    messages = [{"role": "user", "content": prompt}]
    drafts: dict[str, list[str]] = {}
    for label, settings in SAMPLING_SETTINGS.items():
        drafts[label] = []
        for _ in range(runs):
            response = call_slot(slot, messages, max_retries=RETRIES, **settings)
            drafts[label].append(response.choices[0].message.content or "")
    return slot, drafts
