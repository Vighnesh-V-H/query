# Intent dev data labels a deterministic RAG-pool slice with the labeler

Ticket 13's TF-IDF baseline needs labeled examples to train on and ticket 14's
LLM classifier needs them to sanity-check against; ticket 12 is that data.
`query label-intents` draws a deterministic slice of the RAG pool (default
500, `--dev-size`, ranked by SHA-256 of the seed and the interaction id, the
same scheme as the sample-and-split stage) and labels each opening Customer
Message with the configured `labeler` role at temperature 0.0, constrained to
the final taxonomy's intent ids. Ranking is by hash, so a smaller trial slice
is the rank-prefix of a larger one under the same seed. Only the RAG pool is
ever labeled: the
holdout stays reserved for the Golden Set, so a dev slice drawn from it would
leak evaluation data into training.

Each label records its provenance: fresh labels carry `source: labeler` with
the model's one-line justification and model id, and labels corrected by hand
during spot-checking carry `source: human` with no model, so the source always
says who decided. Every label also carries the taxonomy version, and the
report counts the label distribution per intent — including intents that
receive zero labels, so thin intents stay visible instead of silently
vanishing — plus the source split and the models used. A shared contract
validator checks labels in both the builder and the reader, and an optional
Markdown review lists the labels per intent for the human audit recorded in
`docs/intent-dev-spot-check.md`.

The labeler is the non-deterministic step, so verdicts are cached by prompt
hash in a crash-safe append-only file, like closure adjudication (ADR-0007)
and discovery (ADR-0009): an interrupted run resumes without paying for
finished messages, and a taxonomy edit invalidates stale verdicts because the
prompt embeds the intent definitions. The cache pins verdicts — delete it when
changing the labeler model and fresh verdicts are wanted.

## Considered Options

- **Label the whole RAG pool**: rejected. Three thousand labeler calls cost
  far more than a dev slice needs: a few hundred examples train the baseline
  and check the LLM classifier, and the distribution report says honestly
  which intents the slice under-covers.
- **Stratify the slice by intent**: rejected. Stratification needs intent
  labels first — the chicken-and-egg the Golden Set (ticket 18) solves with
  its own sampling pass. The dev slice is an honest random rank with its
  distribution reported, thin intents included; stratification belongs to
  evaluation data, not training data.
- **Human-first hand labeling**: rejected. Five hundred hand labels are too
  slow for a dev slice; LLM-assisted labeling with a documented spot-check of
  a 30-label subsample (28/30 agreement) is the proportionate evidence at
  this stage.
- **Draw the slice from the holdout**: rejected. The holdout is reserved for
  the Golden Set (decision 6); labeling it for training would leak
  evaluation data into the baseline.
- **Pseudo-label from discovery cluster mappings**: rejected. A cluster
  mapping is one verdict for dozens of messages and inherits the clustering's
  errors; the classifiers need per-message verdicts with justifications.
- **No cache**: rejected for the same reason as ADR-0007 and ADR-0009: an
  interrupted run of hundreds of slow labeler calls must not re-pay for
  finished messages.

## Consequences

- The recorded 500-label run is committed (`data/intent-dev-labels.jsonl`,
  `data/intent-dev-report.json`, `docs/intent-dev-review.md`) so downstream
  stages run on a fresh checkout, like the cluster artifact; regenerating it
  needs the provider credential.
- The default 500-label slice costs hundreds of labeler calls; `--workers`
  bounds concurrency and `--cache` makes reruns resume. The committed
  spot-check audited a 30-label trial slice (seed 42): 28/30 agreement, with
  the two misses both on documented taxonomy boundaries
  (vague device-trouble → `other` vs `app_technical`; paid-but-inactive
  upgrade → `subscription_plans` vs `billing_payment`).
- Thin intents can vanish from small slices: the trial 30-slice held zero
  `library_playlists` and zero `presale_codes` labels. The report keeps zeros
  visible, and the committed 500-label slice covers all 13 intents with
  `presale_codes` thinnest at 8 (1.6%) — a recorded training caveat for ticket
  13, not a reason to hide the theme.
- Rerunning with the same seed labels the same slice, but the labeler can
  move labels: a fresh 500 run moved three of the audited 30 trial verdicts,
  so the committed `data/intent-dev-labels.jsonl` (with its report and
  Markdown review) is the pinned training record. The cache replays a run's
  verdicts by prompt hash; a taxonomy edit changes the prompt and therefore
  the hash, so stale verdicts miss automatically.
- Human corrections are first-class records (`source: human`), so the
  spot-check can promote a fix without losing the label contract downstream
  stages validate.
