from dataclasses import dataclass

from langchain_typesafe import Choice

from app.classifiers.categories import CATEGORY_DESCRIPTIONS, URGENCY_DESCRIPTIONS
from app.clients.decision_client import DecisionClient


@dataclass(frozen=True)
class DecisionTriage:
    category: str
    category_confidence: float

    # Chaining multiple teams routing
    category_probabilities: dict[str, float]

    urgency: str
    urgency_confidence: float


def build_questions() -> dict[str, Choice]:
    return {
        "category": Choice(
            instructions="Which team should handle this support ticket?",
            criteria=CATEGORY_DESCRIPTIONS,  # type: ignore
        ),
        "urgency": Choice(
            instructions="How urgent is this support ticket for the customer?",
            criteria=URGENCY_DESCRIPTIONS,  # type: ignore
        ),
    }


def classify_with_decision_model(
    ticket_text: str, client: DecisionClient | None = None
) -> DecisionTriage:
    """Category and urgency in one call. Raises if the endpoint fails; never guesses."""
    if not client:
        client = DecisionClient()
    response = client.decide(ticket_text, build_questions())  # type: ignore
    category = response.choices["category"]
    urgency = response.choices["urgency"]
    return DecisionTriage(
        category=category.choice,
        category_confidence=category.confidence,
        category_probabilities=category.probabilities,
        urgency=urgency.choice,
        urgency_confidence=urgency.confidence,
    )
