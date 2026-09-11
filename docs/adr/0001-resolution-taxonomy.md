# Three-category resolution taxonomy instead of a binary resolved/unresolved split

The Twitter dataset has no ground-truth resolution label. The obvious proxy — "thread ended on a brand reply, customer went silent" — misclassifies DM-deflection endings ("Please DM us your order number" followed by silence) as resolved, which would poison the RAG knowledge base with bad support behavior. We instead label each Interaction as **Resolved** (evidence of natural closure: explicit customer acknowledgement, or a brand reply indicating the issue/action was completed), **Uncertain** (the visible ending shows no evidence the issue was solved — typically the customer went silent after the brand's reply, but also an unclear customer closing that neither confirms resolution nor continues the issue), or **Unresolved** (customer keeps asking, or brand said it couldn't help). Only Resolved Cases enter the Historical Case index used for reply generation.

Detection is a hybrid: keyword heuristics prefilter obvious cases, and a cheap LLM adjudicates the ambiguous middle (DM-deflection, silence, and unclear customer endings), emitting a one-line justification per label that doubles as audit data for the report.

## Considered Options

- Binary "brand's final reply + ≥72h silence = resolved": rejected — accepts DM deflection and unanswered issues as evidence of good resolutions.
- Pure LLM labeling of all threads: rejected — most cases are obvious; hybrid cuts cost with no quality loss on the clear tail.
- Pure heuristics: rejected — the ambiguous middle is exactly where deflection lives; keyword matching can't adjudicate it.

## Consequences

- Preprocessing must label resolution before anything downstream runs; the RAG index and golden set draw only from the labeled sample.
- A customer message arriving after closure starts a new Interaction (see CONTEXT.md) — multi-issue threads are split rather than treated as one.
