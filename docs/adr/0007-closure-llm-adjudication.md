# Flagged closures are adjudicated with the labeler role

Ticket 6's heuristics labeled 1,694 of the 3,000 RAG-pool Interactions and flagged 1,306 (43.5%) as ambiguous — 1,195 brand DM deflections, 81 unclear customer closings, 20 customer acknowledgements after a DM move, and 10 customer DM announcements — with `label: null` (ADR-0006). Ticket 7 resolves every flagged case with one call to the configured `labeler` role, producing the full labeled sample ADR-0001 requires.

Each call sends the labeler the three category definitions, the heuristic flag reason, and the Interaction's visible transcript, and asks for raw JSON:

```json
{"label": "resolved | uncertain | unresolved", "justification": "one sentence naming the decisive evidence"}
```

A move to DMs hides the outcome, so the prompt says to judge only what the transcript shows: a friendly hand-off with no visible evidence of a fix is not Resolved. Replies are validated (`label` in the taxonomy, non-empty one-line `justification`); an invalid reply is retried once with a repair prompt, then the run fails rather than guessing. Heuristic labels pass through unchanged, so only the 1,306 ambiguous cases ever cost a call.

`data/closure-labels-final.jsonl` is the full labeled sample: `{interaction_id, label, source, reason, flag_reason, model}` where `source` is `heuristic` or `labeler`, `reason` is the fixed heuristic reason or the labeler's one-line justification, `flag_reason` records why the case needed adjudication, and `model` records which labeler produced the verdict. The report counts the full sample's distribution and the heuristic vs adjudicated split. `--workers` bounds concurrent calls, and the optional `--cache` file records each verdict as it completes, keyed by prompt hash, so a slow or interrupted run resumes without paying twice. The cache pins verdicts rather than just caching work: any entry whose prompt hash matches is reused, so delete the cache file to relabel with a different model or prompt.

## Considered Options

- **Label flagged cases Uncertain directly**: rejected. A DM hand-off may hide a real resolution; ADR-0001 exists so the ambiguous middle is adjudicated rather than baked into a label. The labeler recovered 25 Resolved Cases the heuristics could not.
- **Send all 3,000 Interactions to the labeler**: rejected. 56.5% are already labeled deterministically and free; paying for them adds cost and non-determinism with no accuracy gain.
- **Batch many Interactions per call**: rejected. Error isolation and prompt size suffer, and one malformed reply would taint a whole batch.
- **No cache**: rejected for the full run. A single labeler call takes tens of seconds, so a 1,306-call run lasts hours; without a cache an interruption re-pays for every finished call.

## Consequences

- The full run labels all 3,000 RAG-pool Interactions: 215 Resolved, 2,569 Uncertain, 216 Unresolved. The labeler recovered 25 Resolved and 29 Unresolved from the flagged middle; the other 1,252 flagged cases stayed Uncertain — the taxonomy's answer when the visible thread shows no resolution.
- Only Resolved Cases (190 heuristic plus 25 adjudicated) are retrieval-eligible Historical Cases for reply generation.
- The labeler is the pipeline's only non-deterministic stage; rerunning the command can move labels in the ambiguous middle, and the distribution above describes one snapshot, not a guaranteed output.
- The recorded snapshot (under the gitignored `data/`, like the other pipeline artifacts) came from an external run of the exported prompts, so every record's `model` is `external`. The default configuration routes `adjudicate-closures` to the configured `labeler` role instead; rerunning it may change the ambiguous middle.
- `data/closure-labels-final.jsonl` replaces the flagged interim file as ticket 8's input: no record has `label: null`, and `source` carries the heuristic vs LLM provenance ticket 8 must preserve.
