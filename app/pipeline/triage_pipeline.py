"""Frontline v0.1: triage a support ticket and draft a reply.

Two models, each doing the job it suits. The decision model routes the ticket (category and
urgency, typed, with confidence). The language model drafts the reply, told where the ticket
went. A support agent reads the reply before a customer does.
"""

from dataclasses import asdict, dataclass

from app.analysis.claim_flagging import flag_unverified_claims
from app.analysis.token_count import count_tokens
from app.classifiers import classify_by_rules
from app.classifiers.decision_based import DecisionTriage, classify_with_decision_model
from app.clients import call_with_fallback
from app.clients.llm_client import ModelSlot
from app.config.settings import load_token_count_settings

SYSTEM_PROMPT = (
    "You are Frontline, the first line of support at a company selling Raspberry Pi "
    "hardware and accessories. A support agent reads everything you write before a "
    "customer does."
)

DRAFT_PROMPT = """Draft a reply to this support ticket, in two or three sentences.

The ticket has been routed to the {category} team (urgency: {urgency}).

Ticket: {ticket_text}"""


@dataclass(frozen=True)
class TriageResult:
    rule_based_category: str
    category: str
    category_confidence: float
    urgency: str
    draft_reply: str
    token_count: int
    flagged_claims: list[str]


@dataclass(frozen=True)
class TriageDiagnostics:
    """What a person needs to trust the result. Printed to stderr, never mixed into the JSON."""

    answering_slot: ModelSlot
    failed_attempts: list[dict]


def build_draft_messages(ticket_text: str, decision: DecisionTriage) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": DRAFT_PROMPT.format(
                category=decision.category, urgency=decision.urgency, ticket_text=ticket_text
            ),
        },
    ]


def triage_ticket(ticket_text: str) -> tuple[TriageResult, TriageDiagnostics]:
    """Route with the decision model, then draft with the language model.

    Failures from either model propagate. A pipeline that quietly substituted a default
    category would hide exactly the failure you are trying to find.
    """
    decision = classify_with_decision_model(ticket_text)
    messages = build_draft_messages(ticket_text, decision)
    # Counted on the exact messages that are sent, before they are sent.
    token_count = count_tokens(messages, load_token_count_settings()).total

    slot, response, attempts = call_with_fallback(messages, temperature=0)
    draft_reply = (response.choices[0].message.content or "").strip()

    result = TriageResult(
        rule_based_category=classify_by_rules(ticket_text),
        category=decision.category,
        category_confidence=decision.category_confidence,
        urgency=decision.urgency,
        draft_reply=draft_reply,
        token_count=token_count,
        flagged_claims=flag_unverified_claims(draft_reply),
    )
    return result, TriageDiagnostics(answering_slot=slot, failed_attempts=attempts)


def result_to_dict(result: TriageResult) -> dict:
    return asdict(result)
