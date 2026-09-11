# Merge discovery and seed into one versioned taxonomy with checked reconciliation decisions

Decision 12 makes the intent taxonomy a reconciliation: ticket 9 wrote the seed as domain knowledge (`docs/intent-seed-taxonomy.md`), discovery (ADR-0009) clustered the RAG pool and mapped every cluster, and ticket 11 owns the final names and version. The final taxonomy is `docs/intent-taxonomy.md` — version 1, **12 support intents plus an `other` fallback**, inside the 8–15 target of the specification. Every intent carries a one-line definition, an origin, and two verbatim examples — drawn from the recorded discovery clusters, except `other`, which draws on the junk clusters and the seed document. Later stages import it through `query.taxonomy.read_final_taxonomy()`, which enforces the contract (one version line, 8–15 unique intents, a definition and at least one example each, well-formed decisions); the seed remains as input evidence only.

The reconciliation decisions live in the document's two tables and are re-checked by `query reconcile-intents` against `data/intent-clusters.jsonl`:

- `account_access` and `account_admin` **merge** into `account`: three clusters (13, 19, 23; 332 messages) mapped to access and none to admin, and the seed's boundary note already anticipated the merge. One account intent covers recovery and management together.
- `devices_connectivity` is **dropped** and its device/platform scope **folds** into `app_technical`: no cluster mapped to it, and the labeler filed the device and platform clusters (4, 29) under `app_technical`. No data-thin intent reaches the classifier.
- The one cluster flagged `new` — 23 concert-presale-code requests — is **promoted** as `presale_codes`; `other` must not absorb a new support theme just to keep the count tidy.
- The other ten support intents are **kept**; no split had cluster evidence. The three `junk` clusters reconcile to `other`.

## Considered Options

- **Keep `account_access` and `account_admin` separate**: rejected. Discovery mapped no cluster to admin, and the seed's own boundary note allowed the merge; a class with zero training examples cannot be learned or evaluated.
- **Keep `devices_connectivity` despite zero clusters**: rejected. Its scope is real but spread across clusters the labeler filed under `app_technical` and `playback`; keeping the id would weaken both the classifier and the stratified Golden Set for no gain.
- **Absorb the presale-code cluster into `other`**: rejected. It is a coherent, actionable support theme the seed misses, and the seed explicitly forbids `other` absorbing a new theme.
- **A JSON decisions file next to the taxonomy**: rejected. The decisions and the intents they produce must be read together; two files would drift, and the review surface is one document.
- **Store the taxonomy as Python constants**: rejected. The taxonomy is a review artifact first — definitions, examples, evidence — and the strict Markdown reader gives later stages structured data without making the document code.
- **Trust the recorded decisions without checking them**: rejected. The labeler is non-deterministic and discovery can be rerun; `query reconcile-intents` fails when a cluster mapping no longer routes through the decisions, so drift is caught instead of silently disagreeing with the document.

## Consequences

- The recorded reconciliation covers all 30 clusters and 3,000 messages: the 12 support intents receive 27 clusters and 2,524 messages (84.13%), and `other` receives the 3 junk clusters and 476 messages.
- `presale_codes` is deliberately thin (23 messages, 0.77%): its share of the holdout may fall below the Golden Set's ~10-example floor, which is a recorded evaluation caveat, not a reason to hide the theme.
- The classifier stages (tickets 13–14) and the dev-data labeling (ticket 12) import the final taxonomy only; the seed is frozen as evidence.
- Rerunning `discover-intents` can move a cluster's mapping because the labeler is non-deterministic; `reconcile-intents` then fails until the decisions are reviewed, which is the intended drift signal. A taxonomy change — new intent, renamed id, changed definition — bumps the document version.
