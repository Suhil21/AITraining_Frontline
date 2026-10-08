"""Run all three classifiers on the same tickets and print one scoreboard.

Block 1.1's point is the comparison, so it is a function you can call from a check and a
command you can run: `uv run python main.py bakeoff`. The language-model column is optional
(`--with-llm`) because it needs a live key and its answers can differ run to run.
"""

import json
from dataclasses import dataclass

from app.classifiers.decision_based import classify_with_decision_model
from app.classifiers.llm_based import classify_by_llm
from app.classifiers.rule_based import classify_by_rules
from app.clients.decision_client import DecisionClient

TICKETS_PATH = "fixtures/tickets.json"


@dataclass(frozen=True)
class BakeoffRow:
    ticket_id: int
    human_label: str
    rule: str
    decision: str
    decision_confidence: float
    llm: str | None


def load_tickets(path: str = TICKETS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


LLM_OFF, LLM_LIVE, LLM_RECORDED = "off", "live", "recorded"
LLM_CLASSIFICATIONS_PATH = "fixtures/llm_classifications.json"


def _recorded_llm_answers() -> dict[int, str]:
    with open(LLM_CLASSIFICATIONS_PATH, encoding="utf-8") as file:
        results = json.load(file)["results"]
    return {int(ticket_id): record["category"] for ticket_id, record in results.items()}


def run_bakeoff(client: DecisionClient | None = None, llm: str = LLM_LIVE) -> list[BakeoffRow]:
    if not client:
        client = DecisionClient()
    recorded = _recorded_llm_answers() if llm == LLM_RECORDED else {}
    rows = []
    for ticket in load_tickets():
        decision = classify_with_decision_model(ticket["text"], client)
        rows.append(
            BakeoffRow(
                ticket_id=ticket["id"],
                human_label=ticket["human_label"],
                rule=classify_by_rules(ticket["text"]),
                decision=decision.category,
                decision_confidence=decision.category_confidence,
                llm=_llm_answer(ticket, llm, recorded),
            )
        )
    return rows


def _llm_answer(ticket: dict, llm: str, recorded: dict[int, str]) -> str | None:
    if llm == LLM_LIVE:
        return classify_by_llm(ticket["text"])
    if llm == LLM_RECORDED:
        return recorded[ticket["id"]]
    return None


def format_scoreboard(rows: list[BakeoffRow]) -> str:
    show_llm = any(row.llm is not None for row in rows)
    header = f"{'id':>3}  {'human':<10} {'rule':<10} {'decision':<10} {'conf':>5}"
    header += f"  {'llm':<10}" if show_llm else ""
    lines = [header]
    for row in rows:
        line = (
            f"{row.ticket_id:>3}  {row.human_label:<10} {row.rule:<10} "
            f"{row.decision:<10} {row.decision_confidence:>5.2f}"
        )
        line += f"  {row.llm or '':<10}" if show_llm else ""
        lines.append(line)
    total = len(rows)
    summary = [
        f"rule     {sum(r.rule == r.human_label for r in rows)}/{total}",
        f"decision {sum(r.decision == r.human_label for r in rows)}/{total}",
    ]
    if show_llm:
        summary.append(f"llm      {sum(r.llm == r.human_label for r in rows)}/{total}")
    return "\n".join([*lines, "", *summary])
