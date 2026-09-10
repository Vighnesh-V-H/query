import argparse
import json
import sys
from pathlib import Path

from query import config, english, interactions, llm, source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="query", description="Query - AI support agent for SpotifyCares")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke-llm", help="make one LLM call via a configured role")
    smoke.add_argument("--prompt", default="Reply with the single word: ok")
    smoke.add_argument("--role", default="generator", choices=list(config.ROLES))

    read_source = sub.add_parser(
        "read-source", help="report cached CSV paths, schemas, and row counts"
    )
    read_source.add_argument(
        "--archive",
        type=Path,
        default=source.DEFAULT_ARCHIVE_PATH,
        help=f"source ZIP archive (default: {source.DEFAULT_ARCHIVE_PATH})",
    )
    read_source.add_argument(
        "--extract-dir",
        type=Path,
        default=source.DEFAULT_EXTRACT_DIR,
        help=f"raw-data cache directory (default: {source.DEFAULT_EXTRACT_DIR})",
    )

    build_interactions_parser = sub.add_parser(
        "build-interactions",
        help="stitch SpotifyCares chains into Interactions and report counts",
    )
    build_interactions_parser.add_argument(
        "--twcs",
        type=Path,
        default=interactions.DEFAULT_TWCS_PATH,
        help=f"twcs CSV path (default: {interactions.DEFAULT_TWCS_PATH})",
    )
    build_interactions_parser.add_argument(
        "--brand",
        default=interactions.DEFAULT_BRAND,
        help=f"brand author id to stitch (default: {interactions.DEFAULT_BRAND})",
    )
    build_interactions_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the counts report as JSON",
    )
    build_interactions_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path to write Interactions as JSON Lines",
    )

    filter_english_parser = sub.add_parser(
        "filter-english",
        help="drop non-English Interactions and report retained vs filtered volume",
    )
    filter_english_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=interactions.DEFAULT_INTERACTIONS_PATH,
        help=f"Interactions JSONL path (default: {interactions.DEFAULT_INTERACTIONS_PATH})",
    )
    filter_english_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path to write retained Interactions as JSON Lines",
    )
    filter_english_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the filter report as JSON",
    )

    args = parser.parse_args(argv)

    if args.command == "smoke-llm":
        try:
            reply = llm.call_llm(args.prompt, role=args.role)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"model: {reply.model}")
        print(f"reply: {reply.content.strip()}")
        return 0

    if args.command == "read-source":
        try:
            report = source.read_source_archive(args.archive, args.extract_dir)
        except source.SourceArchiveError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        status = "extracted" if report.extracted else "cached"
        print(f"archive: {report.archive_path}")
        print(f"extraction: {report.extraction_dir} ({status})")
        for csv_file in report.files:
            print(f"schema: {csv_file.path} -> {', '.join(csv_file.columns)}")
            print(f"rows: {csv_file.path} -> {csv_file.rows}")
        print(f"total rows across CSV files: {report.total_rows}")
        return 0

    if args.command == "build-interactions":
        try:
            found, report = interactions.build_interactions(args.twcs, args.brand)
            output_path = (
                interactions.write_interactions_jsonl(found, args.out)
                if args.out
                else None
            )
            if args.report:
                payload = {
                    "brand_id": report.brand_id,
                    "total_rows": report.total_rows,
                    "inbound_rows": report.inbound_rows,
                    "brand_rows": report.brand_rows,
                    "seed_count": report.seed_count,
                    "opening_count": report.opening_count,
                    "absorbed_openings": report.absorbed_openings,
                    "interactions": report.interactions,
                    "unanswered_openings": report.unanswered_openings,
                    "turns_total": report.turns_total,
                    "turns_customer": report.turns_customer,
                    "turns_brand": report.turns_brand,
                }
                _write_json_report(args.report, payload)
        except (interactions.InteractionsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        lines = [
            f"brand: {report.brand_id}",
            f"source rows: {report.total_rows} (inbound {report.inbound_rows}, brand {report.brand_rows})",
            f"seeds: {report.seed_count}",
            f"openings: {report.opening_count} (absorbed {report.absorbed_openings})",
            f"interactions: {report.interactions}",
            f"unanswered openings: {report.unanswered_openings}",
            f"turns: {report.turns_total} (customer {report.turns_customer}, brand {report.turns_brand})",
        ]
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        if output_path is not None:
            print(f"interactions written: {output_path}")
        return 0

    if args.command == "filter-english":
        try:
            found = interactions.read_interactions_jsonl(args.input)
            retained, report = english.filter_english(found)
            output_path = (
                interactions.write_interactions_jsonl(retained, args.out)
                if args.out
                else None
            )
            if args.report:
                payload = {
                    "total": report.total,
                    "retained": report.retained,
                    "filtered": report.filtered,
                    "filter_rate": report.filter_rate,
                    "no_signal": report.no_signal,
                    "filtered_by_language": dict(report.filtered_by_language),
                }
                _write_json_report(args.report, payload)
        except (interactions.InteractionsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        lines = [
            f"input: {args.input} ({report.total} interactions)",
            f"retained: {report.retained} "
            f"(english {report.retained - report.no_signal}, "
            f"no confident signal {report.no_signal})",
            f"filtered: {report.filtered} ({report.filter_rate:.2%})",
        ]
        if report.filtered_by_language:
            breakdown = ", ".join(
                f"{language} {count}" for language, count in report.filtered_by_language.items()
            )
            lines.append(f"filtered by language: {breakdown}")
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        if output_path is not None:
            print(f"interactions written: {output_path}")
        return 0

    raise SystemExit(f"unknown command: {args.command}")


def _write_json_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
