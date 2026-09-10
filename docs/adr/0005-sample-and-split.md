# Rank-sample by hashed seed; the holdout lives in its own file

Ticket 5 draws the development sample from the 29,130 English Interactions: 4,000 sampled, split into a 3,000-Interaction RAG pool and a 1,000-Interaction holdout reserved for the Golden Set (decisions 5 and 6). Every Interaction is ranked by `sha256("{seed}:{interaction_id}")` and the top 4,000 are kept; the first 1,000 of that ranking become the holdout, the remaining 3,000 the RAG pool. Because the ranking key depends only on the seed and the Interaction's own id, the same seed selects the same sample on any machine, under any Python version, and regardless of the order the input JSONL arrives in — the "deterministic across machines" criterion is a property of the construction, not of Python's random-number implementation. The seed (default 42), requested sizes, and actual sizes are reported in `data/sample-report.json`.

The two pools are written as separate JSONL files (`data/rag-pool.jsonl`, `data/holdout.jsonl`). The separation is structural on purpose: retrieval indexes only resolved RAG-pool cases, and the Golden Set is drawn only from the holdout, so a golden example's own Interaction can never be retrieved as evidence for itself (decision 6). The stage is re-runnable: byte-identical output for a given input and seed.

## Considered Options

- **`random.Random(seed).sample`**: rejected. The selection consumes the population in input order, so reordering or regenerating the input silently changes the sample, and stability relies on Python's internal sampling algorithm rather than on a stated contract.
- **Take the first N by interaction id**: rejected. Interaction ids are the opening tweets' ids, which are time-ordered, so the sample would skew to the dataset's earliest months and the holdout would not be representative.
- **Reservoir sampling**: rejected. It streams well but carries RNG state; hash ranking is stateless and re-derivable from the seed alone.

## Consequences

- `data/rag-pool.jsonl` is the input contract for resolution labeling and the retrieval index (tickets 6–8, 15–17); `data/holdout.jsonl` is the input contract for the Golden Set (tickets 18–19). Both are the same normalized JSONL format as the English filter's output.
- Input Interactions must have unique `interaction_id`s: the JSONL reader rejects duplicates, so the ranking cannot place one logical Interaction in both pools and the selection stays independent of input order even for concatenated or malformed files.
- Selection is prefix-stable: increasing `--sample-size` with the same seed only adds Interactions, so the sample can grow without disturbing earlier picks.
- When the input holds fewer Interactions than requested, the sample covers everything and the holdout scales down with the requested ratio (floored); the report states the actual sizes rather than pretending the target was met.
