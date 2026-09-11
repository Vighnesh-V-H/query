# Intent discovery embeds and clusters the RAG pool, then maps each cluster with the labeler

Decision 12 says the intent taxonomy must reconcile data-driven discovery against domain knowledge; ticket 9 wrote that knowledge down (`docs/intent-seed-taxonomy.md`) and ticket 10 is the data's answer. Every opening Customer Message of the RAG pool is embedded with all-MiniLM-L6-v2 and clustered with KMeans; each cluster is then mapped to one seed intent id, or flagged `new` (a coherent support theme the seed misses) or `junk` (not a support issue), with a one-line justification stored as audit data.

Embeddings are local and free (decision 9): the model runs through fastembed/ONNX (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, ~90 MB downloaded once), chosen so the venv does not drag PyTorch in for a 90 MB model. Vectors are L2-normalized, so cosine similarity is a dot product and KMeans' Euclidean distance ranks like cosine. Clustering is seeded KMeans (`--clusters`, default 30; `--seed`, default 42; `n_init=10`, Lloyd) over unit vectors; over-clustering is deliberate, because ticket 11 merges and splits while naming, and 8–15 final intents cannot be derived from one cluster partition alone.

Mapping is a labeler call per cluster, not a similarity lookup. The prompt holds the 13 support definitions, the top `--candidates` seed similarities as evidence, the `--examples` messages closest to the cluster centroid, and the seed taxonomy's boundary rules (payment vs plan vs family, ads on the Free tier, playback vs app technical). The reply is `{"mapping": <seed id> | "new" | "junk", "justification": ...}`, validated against the seed ids and retried once with a repair prompt. The flag is semantic and cannot come from a threshold: the recorded run's bare-mention clusters ("@SpotifyCares", "help") sit closer to `playback` (0.82) than any seed sits to its true themes, so similarity alone would file 476 noise messages under playback. Verdicts are cached by prompt hash in a crash-safe append-only file, like closure adjudication (ADR-0007).

The stage emits three artifacts from the same records: `data/intent-clusters.jsonl` (per cluster: size, mapping, justification, candidates, examples) is ticket 11's input, `data/intent-discovery-report.json` counts clusters and messages per mapping and lists the seed intents no cluster maps to, and `docs/intent-discovery.md` is the review artifact with examples per cluster. A shared contract validator checks mapping, justification, candidates, and examples in both the builder and the reader, so the stage cannot stage a cluster its own reader rejects. Clusters cover the RAG pool only; the holdout stays reserved for the Golden Set.

## Considered Options

- **Map clusters by embedding similarity**: rejected. Similarity ranks topic proximity, not whether messages state an issue; the recorded noise clusters are the proof. An LLM verdict on examples can see the difference.
- **Choose clusters with a silhouette sweep**: rejected. The optimum is not guaranteed meaningful, and ticket 11 reconciles names anyway; one configurable k keeps the review small and the run fast.
- **Label every message with an LLM and cluster the labels**: rejected. Thirty cluster calls cost far less than 3,000 message calls, and the cluster is the unit ticket 11 reviews.
- **sentence-transformers for MiniLM**: rejected. It pulls PyTorch (gigabytes) into the venv and CI; fastembed runs the same weights through ONNX with a fraction of the footprint.
- **No cache**: rejected at 30 slow labeler calls for the same reason as ADR-0007: an interrupted run should not re-pay for finished clusters.

## Consequences

- The recorded snapshot over all 3,000 RAG-pool messages yields 30 clusters: 26 mapped (2,501 messages), 1 `new` (23 messages, concert presale codes — the first concrete candidate for ticket 11), and 3 `junk` (476 one-word replies and bare mentions). `account_admin` and `devices_connectivity` receive no cluster; that absence is evidence for ticket 11's merge/drop decisions.
- Clustering is deterministic for a fixed model, seed, and library versions, but the labeler is not: reruns can move a cluster's verdict, and the committed review is one snapshot. Delete the cache to relabel from scratch; prompts embed the seed definitions, so a taxonomy edit invalidates stale verdicts by hash.
- The first run downloads the model into the user's fastembed cache; later runs, and the whole discovery stage after that, work offline.
- `apps/backend` gains fastembed, numpy, and scikit-learn; ticket 13's TF-IDF baseline and ticket 16's MiniLM retrieval reuse them.
