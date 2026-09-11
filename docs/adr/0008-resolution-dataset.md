# One versioned resolution dataset embeds its content; eligibility is derived from the label

Ticket 8 turns the full labeled sample (ADR-0007) into the dataset everything downstream draws from. Each RAG-pool Interaction gets one record carrying the final label, its provenance, a derived retrieval-eligibility flag, and the normalized Interaction content itself:

```json
{
  "interaction_id": 118851,
  "dataset_version": 1,
  "label": "resolved",
  "retrieval_eligible": true,
  "source": "heuristic",
  "reason": "...",
  "flag_reason": null,
  "model": null,
  "customer_id": "...",
  "brand_id": "SpotifyCares",
  "turns": ["..."]
}
```

Only Resolved Cases are retrieval-eligible Historical Cases (ADR-0001, decision 4). `retrieval_eligible` is computed from the label at build time, so an Uncertain or Unresolved case cannot be hand-marked eligible; the dataset reader re-checks the flag against the label. The dataset keeps the whole labeled sample rather than only the Resolved Cases because downstream evaluation needs the negatives — retrieval indexes only the eligible records, but the golden-set and failure-analysis work can sample any of them.

`dataset_version` is the record schema's contract version, bumped when a field changes meaning; the label provenance stays per record: `source` is `heuristic` or `labeler` (the heuristic vs LLM split), `reason` is the fixed heuristic reason or the labeler's one-line justification, `flag_reason` records what sent the case to adjudication, and `model` names the labeler that decided it. The build validates the label contract and the join at its own boundary — every label carries the provenance its source requires (heuristics none, labelers both a flag and a model), every input Interaction has a unique id and exactly one label, and no label references an unknown Interaction — and fails rather than emitting a partial dataset. The recorded report counts each category and each source split, plus the retrieval-eligible volume and share.

## Considered Options

- **Emit only the Resolved Cases**: rejected. The labeled negatives are what make evaluation (intent, reply, escalation, failure analysis) possible; filtering at index time instead of build time also keeps one dataset file as the single downstream contract.
- **Keep labels separate from content and reference Interactions by id**: rejected. Downstream would have to re-join two files and could drift; the content is already normalized and small (~2.5 MB), so embedding it makes the dataset the one thing a consumer reads.
- **Store `retrieval_eligible` as an input field**: rejected. Deriving it from the label removes the failure mode where a bad join or a hand edit marks an Uncertain case as evidence.
- **Version the dataset only in the report sidecar**: rejected. A record that does not declare its own schema version cannot be trusted once files are concatenated or copied, and the reader needs the version to reject incompatible snapshots.
- **Skip validation and trust the label file**: rejected. A partial join would silently shrink the RAG index; the build fails loudly on missing, unknown, duplicate, or provenance-inconsistent labels, and on duplicate input Interactions, instead.

## Consequences

- The recorded snapshot holds all 3,000 RAG-pool Interactions: 215 retrieval-eligible Resolved Cases (190 heuristic, 25 labeler), 2,569 Uncertain, and 216 Unresolved. The report's source split is 1,694 heuristic (190 Resolved / 1,317 Uncertain / 187 Unresolved) and 1,306 labeler (25 / 1,252 / 29).
- `data/resolution-dataset.jsonl` is the input contract for retrieval (tickets 15–17) and the evaluation stages that follow; only records with `retrieval_eligible: true` may enter a retrieval index.
- The dataset is deterministic given the inputs; rerunning adjudication can move labels in the ambiguous middle and therefore the file's content, but the schema version only changes when the contract changes — the version describes the shape, not the snapshot.
- The reader parses every record end to end, including the embedded Interaction, so a truncated or hand-edited dataset fails at read time rather than inside retrieval.
