from app.classifiers import CATEGORIES
from app.clients import call_with_fallback

SYSTEM_PROMPT = """Route this support ticket to one team.

Teams: {categories}

Reply with the team name and nothing else."""


def classify_by_llm(ticket_text: str) -> str:
    """
    Return the category this ticket's words match, or "other" if none do.
    """
    prompt = SYSTEM_PROMPT.format(categories=", ".join(CATEGORIES))
    prompt = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": ticket_text,
        },
    ]
    slot, response, _ = call_with_fallback(prompt)
    result = response.choices[0].message.content or ""

    return result.strip().lower()
