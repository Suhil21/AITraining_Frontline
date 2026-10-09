from app.classifiers.decision_based import build_questions
from app.clients.decision_client import DecisionClient


def repeat_decision(
    ticket_text: str, count: int, client: DecisionClient | None = None
) -> list[dict]:
    """Ask the decision model the same question `count` times; keep each category answer."""
    client = client or DecisionClient()
    answers = []
    for _ in range(count):
        category = client.decide(ticket_text, build_questions()).choices["category"]
        answers.append(
            {
                "choice": category.choice,
                "confidence": category.confidence,
                "probabilities": category.probabilities,
            }
        )
    return answers


def summarise_spread(answers: list[dict]) -> dict:
    """How far the repeated answers moved: distinct choices, distinct probability vectors,
    and the lowest and highest confidence."""
    confidences = [answer["confidence"] for answer in answers]
    return {
        "calls": len(answers),
        "distinct_choices": sorted({answer["choice"] for answer in answers}),
        "distinct_probability_vectors": len(
            {tuple(sorted(answer["probabilities"].items())) for answer in answers}
        ),
        "confidence_min": min(confidences),
        "confidence_max": max(confidences),
    }
