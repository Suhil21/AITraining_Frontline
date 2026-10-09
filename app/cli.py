import argparse
from collections.abc import Sequence

from app.analysis.context_report import report
from app.analysis.sample_thread import ticket_text
from app.analysis.sampling import build_drafting_prompt, sample_drafts
from app.classifiers import classify_by_llm, classify_by_rules
from app.classifiers.decision_based import classify_with_decision_model
from app.scripts.bakeoff import format_scoreboard, run_bakeoff

CLASSIFY_METHODS = ("rule", "llm", "decision")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Frontline v0.1 triage")
    commands = parser.add_subparsers(dest="command", required=True)

    classify = commands.add_parser("classify", help="route one ticket to a team")
    classify.add_argument("ticket_text")
    classify.add_argument("--method", choices=CLASSIFY_METHODS, default="rule")

    bakeoff = commands.add_parser("bakeoff", help="compare all classifiers on the fixtures")
    bakeoff.add_argument(
        "--with-llm", action="store_true", help="also call the language model (live)"
    )

    commands.add_parser("calculate_tokens", help="calculate token and print report")

    sample = commands.add_parser("sample", help="same prompt several times, three settings (live)")
    sample.add_argument("--ticket-id", type=int, default=3)
    sample.add_argument("--runs", type=int, default=5, help="calls per setting")

    return parser


def run_classify(ticket_text: str, method: str) -> None:
    if method == "rule":
        result = classify_by_rules(ticket_text)
        print(result)
    elif method == "llm":
        result = classify_by_llm(ticket_text)
        print(result)
    elif method == "decision":
        result = classify_with_decision_model(ticket_text)
        print(result)

    else:
        raise NotImplementedError(f"Unknown classification method {method!r}")


def run_sample(ticket_id: int, runs: int) -> None:
    prompt = build_drafting_prompt(ticket_text(ticket_id))
    slot, all_drafts = sample_drafts(prompt, runs)
    print(f"model: {slot.model_name}")
    for setting, drafts in all_drafts.items():
        print(f"{setting}: {len(set(drafts))} different replies in {len(drafts)} calls")
        print(f"  first: {drafts[0][:100]!r}")
        print(f"  last:  {drafts[-1][:100]!r}")


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "classify":
        run_classify(args.ticket_text, args.method)
    elif args.command == "bakeoff":
        print(format_scoreboard(run_bakeoff()))
    elif args.command == "calculate_tokens":
        print(report())
    elif args.command == "sample":
        run_sample(args.ticket_id, args.runs)

    return 0
