# Decision Log

Non-obvious decisions made while building Query v0, and why. (Assignment deliverable: 10–15 decisions.)

1. **Brand: SpotifyCares.** Highest data quality among candidates after volume/response-rate/resolved-thread analysis (40% response rate, ~4K resolved threads, tight subscription domain). AmazonHelp had 5× the volume but multilingual noise we chose not to pay for.

2. **Single-turn scope.** The agent sees only the first customer message of an Interaction. Multi-turn context handling is an explicit non-build; it keeps the case format, golden set, and baselines simple and honestly scoped.

3. **Three-category resolution taxonomy, detected by hybrid heuristics.** Resolved / Uncertain / Unresolved instead of "thread ended on brand reply = resolved"; DM-deflection endings would otherwise poison the RAG index with bad support behavior. Closure detection is hybrid: keyword heuristics prefilter obvious cases, and a cheap LLM adjudicates the ambiguous middle (DM deflection, silence endings) with one-line justifications that double as audit data. See `docs/adr/0001-resolution-taxonomy.md`.

4. **Only Resolved Cases enter the RAG index.** Uncertain Cases are excluded from reply-generation evidence by construction, not by filtering at query time.

5. **Sample-then-process.** ~4K threads sampled up front; resolution labeling, RAG index, and golden set all draw from that sample. Makes the <15-minute reproduction promise trivial.

6. **Holdout split to prevent leakage.** ~3K of the sample → RAG index pool, ~1K → holdout. The golden set is drawn only from the holdout, so headline numbers are not inflated by the generator retrieving a golden example's own thread.

7. **Headline metric: false-auto rate.** "X% of auto-handled messages shouldn't have been" is the number that proves trustworthiness, which is what the assignment actually tests. Intent Macro-F1, reply judge score, and auto-handling rate are supporting numbers.

8. **Free-tier OpenRouter models, all roles swappable via config.** Llama-3.3-70B (generator), Gemini-2.0-Flash (judge — deliberately a different family than the generator to avoid self-preference bias), Llama-3.1-8B (closure labeler / classifier). Cost: $0 reproduction; price paid: retries/backoff for rate limits.

9. **Local embeddings (all-MiniLM-L6-v2) for clustering and retrieval.** Free, offline, deterministic — safer for the 15-minute reproducibility story than API embeddings.

10. **Minimal tweet cleaning.** Mentions and truncated URLs stripped; emoji, casing, and tone kept — they are signal for brand-consistency scoring. Aggressive normalization is scoped to the TF-IDF baseline only.

11. **English-only filter for v0.** One LLM classifying Spanish messages against an English taxonomy fails quietly; documented as a scope limit with filtered volume reported.

12. **Intent taxonomy: clustering discovered, domain knowledge reconciled.** Embedding clusters name candidate intents; a seeded domain list (login, billing, playback, premium…) acts as a sanity checklist. Neither alone is defensible.

13. **Golden set: stratified with floors.** Proportional across intents with a ~10-example per-intent floor, ~60/40 auto/escalate balance, so rare intents are visible and false-auto rate has enough positives to be measurable.

14. **Judge validated, not trusted.** ~50 golden examples double-rated by human and judge to produce the judge–human agreement evidence the assignment demands.

15. **Deploy anyway.** The assignment doesn't require it, but a live demo (FastAPI + Docker on Railway/Fly with bundled local embeddings, Next.js chat page) makes the system verifiable end-to-end by anyone. ADR to follow when that work starts.
