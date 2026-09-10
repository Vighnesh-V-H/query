import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from query import closure, config, english, interactions, llm, sampling, source


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

    sample_split_parser = sub.add_parser(
        "sample-split",
        help="draw a deterministic sample and split it into RAG pool and holdout",
    )
    sample_split_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=sampling.DEFAULT_INPUT_PATH,
        help=f"Interactions JSONL path (default: {sampling.DEFAULT_INPUT_PATH})",
    )
    sample_split_parser.add_argument(
        "--seed",
        type=int,
        default=sampling.DEFAULT_SEED,
        help=f"sampling seed (default: {sampling.DEFAULT_SEED})",
    )
    sample_split_parser.add_argument(
        "--sample-size",
        type=int,
        default=sampling.DEFAULT_SAMPLE_SIZE,
        help=f"Interactions to sample (default: {sampling.DEFAULT_SAMPLE_SIZE})",
    )
    sample_split_parser.add_argument(
        "--holdout-size",
        type=int,
        default=sampling.DEFAULT_HOLDOUT_SIZE,
        help=f"Interactions reserved as holdout (default: {sampling.DEFAULT_HOLDOUT_SIZE})",
    )
    sample_split_parser.add_argument(
        "--rag-out",
        type=Path,
        default=None,
        help="optional path to write the RAG pool as JSON Lines",
    )
    sample_split_parser.add_argument(
        "--holdout-out",
        type=Path,
        default=None,
        help="optional path to write the holdout as JSON Lines",
    )
    sample_split_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the sample report as JSON",
    )

    label_closure_parser = sub.add_parser(
        "label-closure",
        help="label obvious Resolved/Uncertain/Unresolved closures with keyword heuristics",
    )
    label_closure_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=closure.DEFAULT_INPUT_PATH,
        help=f"Interactions JSONL path (default: {closure.DEFAULT_INPUT_PATH})",
    )
    label_closure_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path to write closure labels as JSON Lines",
    )
    label_closure_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the closure report as JSON",
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

    if args.command == "sample-split":
        problem = _sample_split_path_error(
            args.input, args.rag_out, args.holdout_out, args.report
        )
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        staged: list[tuple[Path, Path]] = []
        try:
            found = interactions.read_interactions_jsonl(args.input)
            rag_pool, holdout, report = sampling.sample_and_split(
                found,
                seed=args.seed,
                sample_size=args.sample_size,
                holdout_size=args.holdout_size,
            )
            if args.rag_out:
                staged.append(
                    (
                        args.rag_out,
                        interactions.stage_interactions_jsonl(rag_pool, args.rag_out),
                    )
                )
            if args.holdout_out:
                staged.append(
                    (
                        args.holdout_out,
                        interactions.stage_interactions_jsonl(holdout, args.holdout_out),
                    )
                )
            if args.report:
                payload = {
                    "seed": report.seed,
                    "input_total": report.input_total,
                    "sample_size": report.sample_size,
                    "sampled": report.sampled,
                    "sample_rate": report.sample_rate,
                    "rag_pool": report.rag_pool,
                    "holdout": report.holdout,
                }
                staged.append((args.report, _stage_json_report(args.report, payload)))
            _commit_staged_outputs(staged)
        except (interactions.InteractionsError, sampling.SamplingError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            for _, temporary in staged:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        lines = [
            f"input: {args.input} ({report.input_total} interactions)",
            f"seed: {report.seed}",
            (
                f"sampled: {report.sampled} of {report.input_total} "
                f"({report.sample_rate:.2%})"
            ),
            f"rag pool: {report.rag_pool} interactions",
            f"holdout: {report.holdout} interactions",
        ]
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        if args.rag_out:
            print(f"rag pool written: {args.rag_out}")
        if args.holdout_out:
            print(f"holdout written: {args.holdout_out}")
        return 0

    if args.command == "label-closure":
        problem = _output_paths_error(args.input, args.out, args.report)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        staged: list[tuple[Path, Path]] = []
        try:
            found = interactions.read_interactions_jsonl(args.input)
            labels, report = closure.label_closures(found)
            if args.out:
                staged.append(
                    (args.out, closure.stage_closure_labels_jsonl(labels, args.out))
                )
            if args.report:
                payload = {
                    "total": report.total,
                    "resolved": report.resolved,
                    "uncertain": report.uncertain,
                    "unresolved": report.unresolved,
                    "needs_adjudication": report.needs_adjudication,
                    "labeled": report.labeled,
                    "label_rate": report.label_rate,
                    "flagged_by_reason": dict(report.flagged_by_reason),
                }
                staged.append((args.report, _stage_json_report(args.report, payload)))
            _commit_staged_outputs(staged)
        except (interactions.InteractionsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            for _, temporary in staged:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        lines = [
            f"input: {args.input} ({report.total} interactions)",
            f"resolved: {report.resolved}",
            f"uncertain: {report.uncertain}",
            f"unresolved: {report.unresolved}",
            (
                f"needs adjudication: {report.needs_adjudication} "
                f"({report.adjudication_rate:.2%})"
            ),
        ]
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        if args.out:
            print(f"labels written: {args.out}")
        return 0

    raise SystemExit(f"unknown command: {args.command}")


def _sample_split_path_error(
    input_path: Path,
    rag_out: Path | None,
    holdout_out: Path | None,
    report: Path | None,
) -> str | None:
    """Reject sample-split outputs that would clobber an input or each other."""
    if (rag_out is None) != (holdout_out is None):
        return "provide both --rag-out and --holdout-out, or neither"
    return _output_paths_error(input_path, rag_out, holdout_out, report)


def _output_paths_error(input_path: Path, *outputs: Path | None) -> str | None:
    """Reject outputs that would clobber the input or each other."""
    provided = [path for path in outputs if path is not None]
    resolved = [path.resolve() for path in provided] + [input_path.resolve()]
    if len(set(resolved)) != len(resolved):
        return "outputs must be distinct from each other and from --in"
    return None


def _write_json_report(path: Path, payload: dict) -> None:
    """Write a JSON report, swapping the file in atomically.

    Mirrors the JSONL writer: a failed or interrupted write must not truncate
    the previous report.
    """
    temporary_path = _stage_json_report(path, payload)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _stage_json_report(path: Path, payload: dict) -> Path:
    """Write a JSON report to a temporary sibling, ready to be swapped in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def _commit_staged_outputs(staged: list[tuple[Path, Path]]) -> None:
    """Swap staged outputs into place, rolling back if a later swap fails.

    Every output is staged before this runs, so a write failure leaves every
    destination untouched. If a swap fails, destinations already swapped are
    restored from the bytes captured just before their replace (or removed when
    they did not exist); rollback is best-effort and never masks the original
    error.
    """
    replaced: list[tuple[Path, bytes | None]] = []
    try:
        for destination, temporary in staged:
            previous = destination.read_bytes() if destination.is_file() else None
            os.replace(temporary, destination)
            replaced.append((destination, previous))
    except BaseException:
        for destination, previous in reversed(replaced):
            try:
                if previous is None:
                    destination.unlink(missing_ok=True)
                else:
                    destination.write_bytes(previous)
            except OSError:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
