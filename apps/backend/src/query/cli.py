import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from query import (
    adjudication,
    closure,
    config,
    discovery,
    embedding,
    english,
    golden,
    intent_labels,
    interactions,
    llm,
    minilm,
    reconciliation,
    resolution,
    sampling,
    source,
    taxonomy,
)


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

    adjudicate_parser = sub.add_parser(
        "adjudicate-closures",
        help="adjudicate flagged closures with the labeler role and write the full labeled sample",
    )
    adjudicate_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=adjudication.DEFAULT_INTERACTIONS_PATH,
        help=f"Interactions JSONL path (default: {adjudication.DEFAULT_INTERACTIONS_PATH})",
    )
    adjudicate_parser.add_argument(
        "--labels",
        type=Path,
        default=adjudication.DEFAULT_LABELS_PATH,
        help=f"heuristic closure labels JSONL path (default: {adjudication.DEFAULT_LABELS_PATH})",
    )
    adjudicate_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path to write the full labeled sample as JSON Lines",
    )
    adjudicate_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the adjudication report as JSON",
    )
    adjudicate_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent labeler calls (default: 1)",
    )
    adjudicate_parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="optional JSONL cache of labeler verdicts; matching entries are reused until the file is deleted",
    )

    resolution_parser = sub.add_parser(
        "build-resolution-dataset",
        help="join final labels onto the RAG pool and mark Resolved Cases retrieval-eligible",
    )
    resolution_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=resolution.DEFAULT_INTERACTIONS_PATH,
        help=f"Interactions JSONL path (default: {resolution.DEFAULT_INTERACTIONS_PATH})",
    )
    resolution_parser.add_argument(
        "--labels",
        type=Path,
        default=resolution.DEFAULT_LABELS_PATH,
        help=f"final closure labels JSONL path (default: {resolution.DEFAULT_LABELS_PATH})",
    )
    resolution_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"optional path to write the resolution dataset as JSON Lines (e.g. {resolution.DEFAULT_DATASET_PATH})",
    )
    resolution_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the resolution report as JSON",
    )

    minilm_build_parser = sub.add_parser(
        "build-minilm-index",
        help="index Resolved Cases with local MiniLM embeddings and record the build time",
    )
    minilm_build_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=minilm.DEFAULT_DATASET_PATH,
        help=f"resolution dataset JSONL path (default: {minilm.DEFAULT_DATASET_PATH})",
    )
    minilm_build_parser.add_argument(
        "--index-out",
        type=Path,
        default=None,
        help=f"optional path to write the MiniLM index as JSON (e.g. {minilm.DEFAULT_INDEX_PATH})",
    )
    minilm_build_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the build report as JSON",
    )

    minilm_retrieve_parser = sub.add_parser(
        "retrieve-minilm",
        help="rank Historical Cases for a query with a MiniLM index",
    )
    minilm_retrieve_parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help=f"persisted MiniLM index JSON path (e.g. {minilm.DEFAULT_INDEX_PATH})",
    )
    minilm_retrieve_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=None,
        help="resolution dataset JSONL path: builds an ephemeral index instead of --index",
    )
    minilm_retrieve_parser.add_argument(
        "--query",
        required=True,
        help="customer message to rank Historical Cases against",
    )
    minilm_retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=minilm.DEFAULT_TOP_K,
        help=f"cases to return (default: {minilm.DEFAULT_TOP_K})",
    )
    minilm_retrieve_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the ranked cases as JSON",
    )

    discover_parser = sub.add_parser(
        "discover-intents",
        help="cluster RAG-pool Customer Messages and map each cluster to the seed taxonomy",
    )
    discover_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=discovery.DEFAULT_INPUT_PATH,
        help=f"Interactions JSONL path (default: {discovery.DEFAULT_INPUT_PATH})",
    )
    discover_parser.add_argument(
        "--taxonomy",
        type=Path,
        default=taxonomy.DEFAULT_SEED_TAXONOMY_PATH,
        help=f"seed taxonomy Markdown path (default: {taxonomy.DEFAULT_SEED_TAXONOMY_PATH})",
    )
    discover_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"optional path to write the clusters as JSON Lines (e.g. {discovery.DEFAULT_CLUSTERS_PATH})",
    )
    discover_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the discovery report as JSON",
    )
    discover_parser.add_argument(
        "--review",
        type=Path,
        default=None,
        help=f"optional path to write the Markdown review (e.g. {discovery.DEFAULT_REVIEW_PATH})",
    )
    discover_parser.add_argument(
        "--clusters",
        type=int,
        default=discovery.DEFAULT_CLUSTERS,
        help=f"number of KMeans clusters (default: {discovery.DEFAULT_CLUSTERS})",
    )
    discover_parser.add_argument(
        "--seed",
        type=int,
        default=discovery.DEFAULT_SEED,
        help=f"KMeans random seed (default: {discovery.DEFAULT_SEED})",
    )
    discover_parser.add_argument(
        "--examples",
        type=int,
        default=discovery.DEFAULT_EXAMPLES,
        help=f"example messages per cluster (default: {discovery.DEFAULT_EXAMPLES})",
    )
    discover_parser.add_argument(
        "--candidates",
        type=int,
        default=discovery.DEFAULT_CANDIDATES,
        help=f"candidate seed intents per cluster (default: {discovery.DEFAULT_CANDIDATES})",
    )
    discover_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent labeler calls (default: 1)",
    )
    discover_parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="optional JSONL cache of mapping verdicts; matching entries are reused until the file is deleted",
    )

    reconcile_parser = sub.add_parser(
        "reconcile-intents",
        help="check the final intent taxonomy's decisions against the discovery clusters",
    )
    reconcile_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=discovery.DEFAULT_CLUSTERS_PATH,
        help=f"intent clusters JSONL path (default: {discovery.DEFAULT_CLUSTERS_PATH})",
    )
    reconcile_parser.add_argument(
        "--seed",
        type=Path,
        default=taxonomy.DEFAULT_SEED_TAXONOMY_PATH,
        help=f"seed taxonomy Markdown path (default: {taxonomy.DEFAULT_SEED_TAXONOMY_PATH})",
    )
    reconcile_parser.add_argument(
        "--taxonomy",
        type=Path,
        default=taxonomy.DEFAULT_FINAL_TAXONOMY_PATH,
        help=f"final taxonomy Markdown path (default: {taxonomy.DEFAULT_FINAL_TAXONOMY_PATH})",
    )
    reconcile_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"optional path to write the coverage report as JSON (e.g. {reconciliation.DEFAULT_REPORT_PATH})",
    )

    sample_golden_parser = sub.add_parser(
        "sample-golden",
        help="stratify holdout Interactions into a Golden Set labeling queue",
    )
    sample_golden_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=golden.DEFAULT_HOLDOUT_PATH,
        help=f"holdout JSONL path (default: {golden.DEFAULT_HOLDOUT_PATH}; never the RAG pool)",
    )
    sample_golden_parser.add_argument(
        "--taxonomy",
        type=Path,
        default=taxonomy.DEFAULT_FINAL_TAXONOMY_PATH,
        help=f"final taxonomy Markdown path (default: {taxonomy.DEFAULT_FINAL_TAXONOMY_PATH})",
    )
    sample_golden_parser.add_argument(
        "--hints",
        type=Path,
        default=golden.DEFAULT_HINTS_PATH,
        help=(
            "intent/decision hint cache, read and extended with fresh labeler "
            f"verdicts (default: {golden.DEFAULT_HINTS_PATH})"
        ),
    )
    sample_golden_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"optional path to write the queue as JSON Lines (e.g. {golden.DEFAULT_QUEUE_PATH})",
    )
    sample_golden_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"optional path to write the sampling report as JSON (e.g. {golden.DEFAULT_SAMPLING_REPORT_PATH})",
    )
    sample_golden_parser.add_argument(
        "--target",
        type=int,
        default=golden.DEFAULT_TARGET_SIZE,
        help=f"Golden Set size to plan (default: {golden.DEFAULT_TARGET_SIZE})",
    )
    sample_golden_parser.add_argument(
        "--floor",
        type=int,
        default=golden.DEFAULT_FLOOR,
        help=f"minimum examples per intent (default: {golden.DEFAULT_FLOOR})",
    )
    sample_golden_parser.add_argument(
        "--auto-share",
        type=float,
        default=golden.DEFAULT_AUTO_SHARE,
        help=f"target share of predicted-auto examples (default: {golden.DEFAULT_AUTO_SHARE})",
    )
    sample_golden_parser.add_argument(
        "--seed",
        type=int,
        default=golden.DEFAULT_SEED,
        help=f"selection seed (default: {golden.DEFAULT_SEED})",
    )
    sample_golden_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent labeler calls (default: 1)",
    )

    label_golden_parser = sub.add_parser(
        "label-golden",
        help="serve a Golden Set queue with the full Interaction for hand-labeling",
    )
    label_golden_parser.add_argument(
        "--queue",
        type=Path,
        default=golden.DEFAULT_QUEUE_PATH,
        help=f"queue JSONL path (default: {golden.DEFAULT_QUEUE_PATH})",
    )
    label_golden_parser.add_argument(
        "--holdout",
        type=Path,
        default=golden.DEFAULT_HOLDOUT_PATH,
        help=f"holdout JSONL path with the source Interactions (default: {golden.DEFAULT_HOLDOUT_PATH}; never the RAG pool)",
    )
    label_golden_parser.add_argument(
        "--taxonomy",
        type=Path,
        default=taxonomy.DEFAULT_FINAL_TAXONOMY_PATH,
        help=f"final taxonomy Markdown path (default: {taxonomy.DEFAULT_FINAL_TAXONOMY_PATH})",
    )
    label_golden_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=f"path to write the Golden Set as JSON Lines (e.g. {golden.DEFAULT_GOLDEN_PATH})",
    )
    label_golden_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"optional path to write the labeling report as JSON (e.g. {golden.DEFAULT_LABELING_REPORT_PATH})",
    )

    label_intents_parser = sub.add_parser(
        "label-intents",
        help="label a deterministic dev slice of the RAG pool with final-taxonomy intents",
    )
    label_intents_parser.add_argument(
        "--in",
        dest="input",
        type=Path,
        default=intent_labels.DEFAULT_POOL_PATH,
        help=(
            "Interactions JSONL path; must be the RAG pool or a subset of it "
            f"(default: {intent_labels.DEFAULT_POOL_PATH})"
        ),
    )
    label_intents_parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help=(
            "optional JSONL of hand corrections "
            '({"interaction_id", "intent", "justification"}) applied as '
            "source: human over the labeler verdicts"
        ),
    )
    label_intents_parser.add_argument(
        "--taxonomy",
        type=Path,
        default=taxonomy.DEFAULT_FINAL_TAXONOMY_PATH,
        help=f"final taxonomy Markdown path (default: {taxonomy.DEFAULT_FINAL_TAXONOMY_PATH})",
    )
    label_intents_parser.add_argument(
        "--dev-size",
        type=int,
        default=intent_labels.DEFAULT_DEV_SIZE,
        help=f"Interactions to label (default: {intent_labels.DEFAULT_DEV_SIZE})",
    )
    label_intents_parser.add_argument(
        "--seed",
        type=int,
        default=intent_labels.DEFAULT_SEED,
        help=f"dev-slice ranking seed (default: {intent_labels.DEFAULT_SEED})",
    )
    label_intents_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"optional path to write the dev labels as JSON Lines (e.g. {intent_labels.DEFAULT_LABELS_PATH})",
    )
    label_intents_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the label distribution report as JSON",
    )
    label_intents_parser.add_argument(
        "--review",
        type=Path,
        default=None,
        help=f"optional path to write the Markdown review of labels per intent (e.g. {intent_labels.DEFAULT_REVIEW_PATH})",
    )
    label_intents_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent labeler calls (default: 1)",
    )
    label_intents_parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="optional JSONL cache of labeler verdicts; matching entries are reused until the file is deleted",
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
            (
                f"retained: {report.retained} "
                f"(english {report.retained - report.no_signal}, "
                f"no confident signal {report.no_signal})"
            ),
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

    if args.command == "adjudicate-closures":
        problem = _output_paths_error(
            args.input,
            args.labels,
            args.out,
            args.report,
            args.cache,
            message="paths must be distinct from each other",
        )
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        staged: list[tuple[Path, Path]] = []
        try:
            found = interactions.read_interactions_jsonl(args.input)
            heuristic = closure.read_closure_labels_jsonl(args.labels)
            cache = adjudication.AdjudicationCache(args.cache) if args.cache else None
            labels, report = adjudication.adjudicate_closures(
                found, heuristic, workers=args.workers, cache=cache
            )
            if args.out:
                staged.append(
                    (
                        args.out,
                        adjudication.stage_adjudicated_labels_jsonl(labels, args.out),
                    )
                )
            if args.report:
                payload = {
                    "total": report.total,
                    "resolved": report.resolved,
                    "uncertain": report.uncertain,
                    "unresolved": report.unresolved,
                    "adjudicated": report.adjudicated,
                    "heuristic": {
                        "resolved": report.heuristic_resolved,
                        "uncertain": report.heuristic_uncertain,
                        "unresolved": report.heuristic_unresolved,
                    },
                    "adjudicated_labels": {
                        "resolved": report.adjudicated_resolved,
                        "uncertain": report.adjudicated_uncertain,
                        "unresolved": report.adjudicated_unresolved,
                    },
                    "flagged_by_reason": dict(report.flagged_by_reason),
                    "models": list(report.models),
                }
                staged.append((args.report, _stage_json_report(args.report, payload)))
            _commit_staged_outputs(staged)
        except (
            interactions.InteractionsError,
            closure.ClosureLabelsError,
            adjudication.AdjudicationError,
            OSError,
        ) as exc:
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
            f"adjudicated: {report.adjudicated} flagged",
            f"resolved: {report.resolved}",
            f"uncertain: {report.uncertain}",
            f"unresolved: {report.unresolved}",
        ]
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        if args.out:
            print(f"labels written: {args.out}")
        return 0

    if args.command == "build-resolution-dataset":
        problem = _output_paths_error(args.input, args.labels, args.out, args.report)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        staged: list[tuple[Path, Path]] = []
        try:
            found = interactions.read_interactions_jsonl(args.input)
            labels = adjudication.read_adjudicated_labels_jsonl(args.labels)
            records, report = resolution.build_resolution_dataset(found, labels)
            if args.out:
                staged.append(
                    (
                        args.out,
                        resolution.stage_resolution_dataset_jsonl(records, args.out),
                    )
                )
            if args.report:
                payload = {
                    "dataset_version": report.dataset_version,
                    "total": report.total,
                    "resolved": report.resolved,
                    "uncertain": report.uncertain,
                    "unresolved": report.unresolved,
                    "retrieval_eligible": report.retrieval_eligible,
                    "retrieval_share": report.retrieval_share,
                    "by_source": {
                        "heuristic": _source_split_json(report.heuristic),
                        "labeler": _source_split_json(report.labeler),
                    },
                    "models": list(report.models),
                }
                staged.append((args.report, _stage_json_report(args.report, payload)))
            _commit_staged_outputs(staged)
        except (
            interactions.InteractionsError,
            adjudication.AdjudicationError,
            resolution.ResolutionDatasetError,
            OSError,
        ) as exc:
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
            f"labels: {args.labels} ({report.total} labeled)",
            f"resolved: {report.resolved}",
            f"uncertain: {report.uncertain}",
            f"unresolved: {report.unresolved}",
            (
                f"retrieval-eligible: {report.retrieval_eligible} "
                f"({report.retrieval_share:.2%})"
            ),
            f"heuristic: {_source_split_summary(report.heuristic)}",
            f"labeler: {_source_split_summary(report.labeler)}",
        ]
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        if args.out:
            print(f"dataset written: {args.out}")
        return 0

    if args.command == "build-minilm-index":
        problem = _output_paths_error(args.input, args.index_out, args.report)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        staged: list[tuple[Path, Path]] = []
        try:
            records = resolution.read_resolution_dataset_jsonl(args.input)
            index, report = minilm.build_minilm_index(
                records, embedding.MiniLMEmbedder()
            )
            if args.index_out:
                staged.append(
                    (
                        args.index_out,
                        minilm.stage_minilm_index_json(index, args.index_out),
                    )
                )
            if args.report:
                payload = {
                    "index_version": report.index_version,
                    "total_records": report.total_records,
                    "indexed": report.indexed,
                    "skipped": report.skipped,
                    "embedding_model": report.embedding_model,
                    "embedding_dim": report.embedding_dim,
                    "build_time_s": report.build_time_s,
                }
                staged.append((args.report, _stage_json_report(args.report, payload)))
            _commit_staged_outputs(staged)
        except (
            resolution.ResolutionDatasetError,
            minilm.MiniLMError,
            embedding.EmbeddingError,
            OSError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            for _, temporary in staged:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        lines = [
            f"input: {args.input} ({report.total_records} records)",
            f"indexed: {report.indexed} resolved cases (skipped {report.skipped})",
            f"model: {report.embedding_model} (dim {report.embedding_dim})",
            f"build time: {report.build_time_s:.2f}s",
        ]
        for line in lines:
            print(line)
        if args.index_out:
            print(f"index written: {args.index_out}")
        if args.report:
            print(f"report written: {args.report}")
        return 0

    if args.command == "retrieve-minilm":
        if (args.index is None) == (args.input is None):
            print("error: provide exactly one of --index or --in", file=sys.stderr)
            return 1
        if args.top_k < 1:
            print("error: --top-k must be >= 1", file=sys.stderr)
            return 1
        problem = _output_paths_error(args.index, args.input, args.report)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        try:
            embedder = embedding.MiniLMEmbedder()
            if args.index is not None:
                index = minilm.read_minilm_index_json(args.index)
            else:
                records = resolution.read_resolution_dataset_jsonl(args.input)
                index, _ = minilm.build_minilm_index(records, embedder)
            cases = minilm.retrieve_minilm(
                index, args.query, embedder, top_k=args.top_k
            )
            if args.report:
                payload = {
                    "query": args.query,
                    "top_k": args.top_k,
                    "results": [
                        {
                            "interaction_id": case.interaction_id,
                            "score": case.score,
                        }
                        for case in cases
                    ],
                }
                _write_json_report(args.report, payload)
        except (
            resolution.ResolutionDatasetError,
            minilm.MiniLMError,
            embedding.EmbeddingError,
            OSError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        origin = args.index if args.index is not None else args.input
        print(f"index: {origin} ({len(index.doc_ids)} cases)")
        print(f'query: "{args.query}"')
        if not cases:
            print("no matching cases")
        for rank, case in enumerate(cases, start=1):
            print(
                f"{rank}. interaction {case.interaction_id} "
                f"(score {case.score:.3f})"
            )
        if args.report:
            print(f"report written: {args.report}")
        return 0

    if args.command == "discover-intents":
        problem = _output_paths_error(
            args.input,
            args.taxonomy,
            args.out,
            args.report,
            args.review,
            args.cache,
            message="paths must be distinct from each other",
        )
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        staged: list[tuple[Path, Path]] = []
        try:
            found = interactions.read_interactions_jsonl(args.input)
            seeds = taxonomy.read_seed_taxonomy(args.taxonomy)
            cache = discovery.DiscoveryCache(args.cache) if args.cache else None
            clusters, report = discovery.discover_intents(
                found,
                seeds,
                embedding.MiniLMEmbedder(),
                clusters=args.clusters,
                seed=args.seed,
                examples=args.examples,
                candidates=args.candidates,
                workers=args.workers,
                cache=cache,
            )
            if args.out:
                staged.append(
                    (
                        args.out,
                        discovery.stage_intent_clusters_jsonl(clusters, args.out),
                    )
                )
            if args.report:
                payload = {
                    "clusters": report.clusters,
                    "messages": report.messages,
                    "requested_clusters": report.requested_clusters,
                    "seed": report.seed,
                    "embedding_model": report.embedding_model,
                    "mapped_clusters": report.mapped_clusters,
                    "new_clusters": report.new_clusters,
                    "junk_clusters": report.junk_clusters,
                    "mapped_messages": report.mapped_messages,
                    "new_messages": report.new_messages,
                    "junk_messages": report.junk_messages,
                    "mapped_share": report.mapped_share,
                    "per_intent": {
                        intent_id: {
                            "clusters": counts.clusters,
                            "messages": counts.messages,
                        }
                        for intent_id, counts in report.per_intent.items()
                    },
                    "absent_intents": list(report.absent_intents),
                    "models": list(report.models),
                }
                staged.append((args.report, _stage_json_report(args.report, payload)))
            if args.review:
                staged.append(
                    (
                        args.review,
                        discovery.stage_review_markdown(
                            clusters, report, seeds, args.review
                        ),
                    )
                )
            _commit_staged_outputs(staged)
        except (
            interactions.InteractionsError,
            taxonomy.TaxonomyError,
            embedding.EmbeddingError,
            discovery.DiscoveryError,
            OSError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            for _, temporary in staged:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        lines = [
            f"input: {args.input} ({report.messages} interactions)",
            (
                f"clusters: {report.clusters} "
                f"(k={report.requested_clusters}, seed={report.seed})"
            ),
            (
                f"mapped: {report.mapped_clusters} clusters, "
                f"{report.mapped_messages} messages ({report.mapped_share:.2%})"
            ),
            f"new: {report.new_clusters} clusters, {report.new_messages} messages",
            f"junk: {report.junk_clusters} clusters, {report.junk_messages} messages",
        ]
        if report.absent_intents:
            lines.append(
                "seed intents with no mapped cluster: "
                + ", ".join(report.absent_intents)
            )
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        if args.out:
            print(f"clusters written: {args.out}")
        if args.review:
            print(f"review written: {args.review}")
        return 0

    if args.command == "reconcile-intents":
        problem = _output_paths_error(
            args.input,
            args.seed,
            args.taxonomy,
            args.report,
            message="paths must be distinct from each other",
        )
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        try:
            clusters = discovery.read_intent_clusters_jsonl(args.input)
            seeds = taxonomy.read_seed_taxonomy(args.seed)
            final = taxonomy.read_final_taxonomy(args.taxonomy)
            report = reconciliation.reconcile_intents(clusters, seeds, final)
            if args.report:
                payload = {
                    "taxonomy_version": report.taxonomy_version,
                    "clusters": report.clusters,
                    "messages": report.messages,
                    "covered_messages": report.covered_messages,
                    "covered_share": report.covered_share,
                    "per_intent": {
                        intent_id: {
                            "clusters": counts.clusters,
                            "messages": counts.messages,
                        }
                        for intent_id, counts in report.per_intent.items()
                    },
                }
                _write_json_report(args.report, payload)
        except (
            discovery.DiscoveryError,
            taxonomy.TaxonomyError,
            reconciliation.ReconciliationError,
            OSError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        lines = [
            (
                f"clusters: {args.input} "
                f"({report.clusters} clusters, {report.messages} messages)"
            ),
            f"taxonomy: {args.taxonomy} (v{report.taxonomy_version})",
        ]
        for intent_id, counts in report.per_intent.items():
            lines.append(
                f"{intent_id}: {counts.clusters} clusters, {counts.messages} messages"
            )
        lines.append(
            f"covered: {report.covered_messages} messages "
            f"({report.covered_share:.2%})"
        )
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        return 0

    if args.command == "sample-golden":
        problem = _output_paths_error(
            args.input,
            args.taxonomy,
            args.hints,
            args.out,
            args.report,
            message="golden sampling paths must be distinct from each other",
        )
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        staged: list[tuple[Path, Path]] = []
        try:
            found = interactions.read_interactions_jsonl(args.input)
            labelable = tuple(
                interaction
                for interaction in found
                if interactions.has_customer_message(interaction)
            )
            excluded = len(found) - len(labelable)
            final = taxonomy.read_final_taxonomy(args.taxonomy)
            cache = golden.HintCache(args.hints)
            hints, models = golden.classify_hints(
                labelable, final, workers=args.workers, cache=cache
            )
            items, report = golden.plan_queue(
                hints,
                final.intent_ids,
                target_size=args.target,
                floor=args.floor,
                auto_share=args.auto_share,
                seed=args.seed,
            )
            if args.out:
                staged.append((args.out, golden.stage_queue_jsonl(items, args.out)))
            if args.report:
                payload = {
                    "seed": report.seed,
                    "hinted": report.hinted,
                    "target_size": report.target_size,
                    "floor": report.floor,
                    "auto_share_target": report.auto_share_target,
                    "selected": report.selected,
                    "selected_auto": report.selected_auto,
                    "selected_escalate": report.selected_escalate,
                    "predicted_auto_share": report.predicted_auto_share,
                    "balance_target_met": report.balance_target_met,
                    "models": list(models),
                    "per_intent": {
                        intent_id: {
                            "available": allocation.available,
                            "available_auto": allocation.available_auto,
                            "available_escalate": allocation.available_escalate,
                            "selected": allocation.selected,
                            "selected_auto": allocation.selected_auto,
                            "selected_escalate": allocation.selected_escalate,
                        }
                        for intent_id, allocation in report.per_intent.items()
                    },
                }
                staged.append((args.report, _stage_json_report(args.report, payload)))
            _commit_staged_outputs(staged)
        except (
            interactions.InteractionsError,
            taxonomy.TaxonomyError,
            golden.GoldenError,
            OSError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            for _, temporary in staged:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        lines = [
            f"input: {args.input} ({len(found)} interactions)",
            (
                f"hints: {args.hints} ({report.hinted} hinted, "
                f"models: {', '.join(models) or 'none'})"
            ),
            (
                f"target: {report.target_size} "
                f"(floor {report.floor}, auto share {report.auto_share_target:.2%})"
            ),
            (
                f"selected: {report.selected} "
                f"(predicted auto {report.selected_auto}, "
                f"escalate {report.selected_escalate}, "
                f"share {report.predicted_auto_share:.2%}; "
                f"balance target {'met' if report.balance_target_met else 'NOT met'})"
            ),
        ]
        for intent_id, allocation in report.per_intent.items():
            lines.append(
                f"{intent_id}: {allocation.selected}/{allocation.available} "
                f"(auto {allocation.selected_auto}/{allocation.available_auto}, "
                f"escalate {allocation.selected_escalate}/{allocation.available_escalate})"
            )
        if excluded:
            lines.append(
                f"excluded: {excluded} interactions with a blank opening message"
            )
        if not golden.MIN_GOLDEN_SIZE <= report.selected <= golden.MAX_GOLDEN_SIZE:
            lines.append(
                f"warning: selected {report.selected} is outside the spec's "
                f"{golden.MIN_GOLDEN_SIZE}-{golden.MAX_GOLDEN_SIZE} Golden Set range"
            )
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        if args.out:
            print(f"queue written: {args.out}")
        return 0

    if args.command == "label-golden":
        problem = _output_paths_error(
            args.queue,
            args.holdout,
            args.taxonomy,
            args.out,
            args.report,
            message="golden labeling paths must be distinct from each other",
        )
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        try:
            final = taxonomy.read_final_taxonomy(args.taxonomy)
            queue = golden.read_queue_jsonl(args.queue, final.intent_ids)
            found = interactions.read_interactions_jsonl(args.holdout)
            _, report = golden.label_golden(
                queue, found, final, args.out, ask=input, tell=print
            )
            if args.report:
                payload = {
                    "golden_set_version": report.golden_set_version,
                    "taxonomy_version": report.taxonomy_version,
                    "queue_total": report.queue_total,
                    "labeled": report.labeled,
                    "skipped": report.skipped,
                    "remaining": report.remaining,
                    "complete": report.complete,
                    "auto_share": report.auto_share,
                    "notes_share": report.notes_share,
                    "by_decision": dict(report.by_decision),
                    "by_intent": {
                        intent_id: {
                            "total": counts.total,
                            "auto": counts.auto,
                            "escalate": counts.escalate,
                        }
                        for intent_id, counts in report.by_intent.items()
                    },
                    "hint_agreement": {
                        "intent": {
                            "agree": report.hint_intent_agreement,
                            "share": report.hint_intent_agreement_share,
                        },
                        "decision": {
                            "agree": report.hint_decision_agreement,
                            "share": report.hint_decision_agreement_share,
                        },
                    },
                }
                _write_json_report(args.report, payload)
        except (
            golden.GoldenError,
            interactions.InteractionsError,
            taxonomy.TaxonomyError,
            OSError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        lines = [
            f"queue: {args.queue} ({report.queue_total} queued)",
            f"holdout: {args.holdout} ({len(found)} interactions)",
            (
                f"labeled: {report.labeled} "
                f"(auto {report.by_decision[golden.AUTO]}, "
                f"escalate {report.by_decision[golden.ESCALATE]})"
            ),
            f"remaining: {report.remaining}",
            f"notes filled: {report.notes_filled}",
        ]
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        if Path(args.out).is_file():
            print(f"golden set written: {args.out}")
        return 0

    if args.command == "label-intents":
        pool_guard = (
            intent_labels.DEFAULT_POOL_PATH
            if intent_labels.DEFAULT_POOL_PATH.resolve() != args.input.resolve()
            else None
        )
        problem = _output_paths_error(
            args.input,
            pool_guard,
            args.taxonomy,
            args.corrections,
            args.out,
            args.report,
            args.review,
            args.cache,
            message="paths must be distinct from each other",
        )
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 1
        staged: list[tuple[Path, Path]] = []
        try:
            found = interactions.read_interactions_jsonl(args.input)
            excluded = sum(
                1
                for interaction in found
                if not interactions.has_customer_message(interaction)
            )
            if pool_guard is None:
                pool = found
            else:
                try:
                    pool = interactions.read_interactions_jsonl(
                        intent_labels.DEFAULT_POOL_PATH
                    )
                except interactions.InteractionsError as exc:
                    raise intent_labels.IntentLabelError(
                        f"cannot read the RAG pool "
                        f"{intent_labels.DEFAULT_POOL_PATH}: {exc}"
                    ) from exc
            intent_labels.require_pool_membership(found, pool)
            final = taxonomy.read_final_taxonomy(args.taxonomy)
            corrections = (
                intent_labels.read_corrections_jsonl(
                    args.corrections, final.intent_ids
                )
                if args.corrections
                else None
            )
            cache = intent_labels.IntentLabelCache(args.cache) if args.cache else None
            labels, report = intent_labels.label_dev_slice(
                found,
                final,
                dev_size=args.dev_size,
                seed=args.seed,
                workers=args.workers,
                cache=cache,
                corrections=corrections,
            )
            if args.out:
                staged.append(
                    (
                        args.out,
                        intent_labels.stage_intent_labels_jsonl(labels, args.out),
                    )
                )
            if args.report:
                payload = {
                    "taxonomy_version": report.taxonomy_version,
                    "seed": report.seed,
                    "input_total": report.input_total,
                    "requested": report.requested,
                    "total": report.total,
                    "per_intent": dict(report.per_intent),
                    "sources": {
                        "labeler": report.labeler,
                        "human": report.human,
                    },
                    "models": list(report.models),
                }
                staged.append((args.report, _stage_json_report(args.report, payload)))
            if args.review:
                staged.append(
                    (
                        args.review,
                        intent_labels.stage_review_markdown(
                            labels, report, final, args.review
                        ),
                    )
                )
            _commit_staged_outputs(staged)
        except (
            interactions.InteractionsError,
            taxonomy.TaxonomyError,
            intent_labels.IntentLabelError,
            OSError,
        ) as exc:
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
            f"taxonomy: {args.taxonomy} (v{report.taxonomy_version})",
            (
                f"dev slice: {report.total} of {report.input_total} "
                f"(requested {report.requested}, seed {report.seed})"
            ),
        ]
        for intent_id in final.intent_ids:
            lines.append(f"{intent_id}: {report.per_intent.get(intent_id, 0)}")
        lines.append(f"sources: labeler {report.labeler}, human {report.human}")
        if excluded:
            lines.append(
                f"excluded: {excluded} interactions with a blank opening message"
            )
        for line in lines:
            print(line)
        if args.report:
            print(f"report written: {args.report}")
        if args.out:
            print(f"labels written: {args.out}")
        if args.review:
            print(f"review written: {args.review}")
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
    return _output_paths_error(
        input_path,
        rag_out,
        holdout_out,
        report,
        message="sample outputs must be distinct from each other and from --in",
    )


def _output_paths_error(
    *paths: Path | None,
    message: str = "outputs must be distinct from each other and from --in",
) -> str | None:
    """Reject a set of paths that would clobber each other or an input."""
    provided = [path for path in paths if path is not None]
    resolved = [path.resolve() for path in provided]
    if len(set(resolved)) != len(resolved):
        return message
    return None


def _source_split_json(counts: resolution.SourceCounts) -> dict:
    """Serialize one provenance split's category counts."""
    return {
        "total": counts.total,
        "resolved": counts.resolved,
        "uncertain": counts.uncertain,
        "unresolved": counts.unresolved,
    }


def _source_split_summary(counts: resolution.SourceCounts) -> str:
    """Render one provenance split's category counts for the console."""
    return (
        f"{counts.total} (resolved {counts.resolved}, "
        f"uncertain {counts.uncertain}, unresolved {counts.unresolved})"
    )


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
