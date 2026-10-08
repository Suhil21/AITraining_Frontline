import argparse
from collections.abc import Sequence

from app.analysis.context_report import report
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


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "classify":
        run_classify(args.ticket_text, args.method)
    elif args.command == "bakeoff":
        print(format_scoreboard(run_bakeoff()))
    elif args.command == "calculate_tokens":
        print(report())

    return 0
