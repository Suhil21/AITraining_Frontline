import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or invalid."""


@dataclass
class DecisionSettings:
    base_url: str
    model_name: str
    context_window: int
    api_key: str


def _require_env_var(var_name: str) -> str:
    """Retrieve an environment variable and raise an error if it's not set."""
    value = os.environ.get(var_name)
    if value is None:
        raise ConfigError(f"Missing required environment variable: {var_name}")
    return value


def load_decision_settings() -> DecisionSettings:
    """Load decision model settings from environment variables."""
    base_url = _require_env_var("DECISION_MODEL_BASE_URL")
    model_name = _require_env_var("DECISION_MODEL_NAME")
    context_window = _require_env_var("DECISION_MODEL_CONTEXT_WINDOW")
    api_key = _require_env_var("DECISION_MODEL_API_KEY")

    return DecisionSettings(
        base_url=base_url,
        model_name=model_name,
        context_window=int(context_window) if context_window else 0,
        api_key=api_key,
    )
