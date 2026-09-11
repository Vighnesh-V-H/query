# Decision Log

Non-obvious decisions made while building Query v0, and why. (Assignment deliverable: 10–15 decisions.)

1. **Brand: SpotifyCares.** Highest data quality among candidates after volume/response-rate/resolved-thread analysis (40% response rate, ~4K resolved threads, tight subscription domain). AmazonHelp had 5× the volume but multilingual noise we chose not to pay for.

2. **Single-turn scope.** The agent sees only the first customer message of an Interaction. Multi-turn context handling is an explicit non-build; it keeps the case format, golden set, and baselines simple and honestly scoped.

3. **Three-category resolution taxonomy, detected by hybrid heuristics.** Resolved / Uncertain / Unresolved instead of "thread ended on brand reply = resolved"; DM-deflection endings would otherwise poison the RAG index with bad support behavior. Closure detection is hybrid: keyword heuristics prefilter obvious cases, and a cheap LLM adjudicates the ambiguous middle (DM deflection, silence endings) with one-line justifications that double as audit data. See `docs/adr/0001-resolution-taxonomy.md`.

4. **Only Resolved Cases enter the RAG index.** Uncertain Cases are excluded from reply-generation evidence by construction, not by filtering at query time.

5. **Sample-then-process.** ~4K threads sampled up front; resolution labeling, RAG index, and golden set all draw from that sample. Makes the <15-minute reproduction promise trivial. Selection is hash-ranked by seed, so the sample is reproducible across machines (see `docs/adr/0005-sample-and-split.md`).

6. **Holdout split to prevent leakage.** ~3K of the sample → RAG index pool, ~1K → holdout. The golden set is drawn only from the holdout, so headline numbers are not inflated by the generator retrieving a golden example's own thread.

7. **Headline metric: false-auto rate.** "X% of auto-handled messages shouldn't have been" is the number that proves trustworthiness, which is what the assignment actually tests. Intent Macro-F1, reply judge score, and auto-handling rate are supporting numbers.

8. **NVIDIA NIM models, all roles swappable via config.** DeepSeek-V4-Pro (generator), Nemotron-3-Super-120B (judge — deliberately a different family than the generator to avoid self-preference bias), Nemotron-3.5-Lightning (closure labeler / classifier), chosen at the user's request from the NVIDIA API catalog. Cost: development is within NVIDIA's free trial credits; per-token pricing applies only if credits are exhausted; all three roles were verified as invocable against this account's catalog (see `apps/backend/configs/models.yaml`, which owns the exact model ids and OpenAI-compatible base URL). Override a role with `QUERY_<ROLE>_MODEL` and authenticate with `NVIDIA_API_KEY` or `OPENROUTER_API_KEY` (first set wins).

9. **Local embeddings (all-MiniLM-L6-v2) for clustering and retrieval.** Free, offline, deterministic — safer for the 15-minute reproducibility story than API embeddings.

10. **Minimal tweet cleaning.** Mentions and truncated URLs stripped; emoji, casing, and tone kept — they are signal for brand-consistency scoring. Aggressive normalization is scoped to the TF-IDF baseline only.

11. **English-only filter for v0.** One LLM classifying Spanish messages against an English taxonomy fails quietly; the filter drops only confidently non-English opening messages and reports the 2.24% it removed by language (plus the ambiguous short messages it keeps). See `docs/adr/0004-english-filter.md`.

12. **Intent taxonomy: clustering discovered, domain knowledge reconciled.** Embedding clusters name candidate intents; a seeded domain list (login, billing, playback, premium…) acts as a sanity checklist (see `docs/intent-seed-taxonomy.md`). Neither alone is defensible. Ticket 11 merged them into the versioned final taxonomy (`docs/intent-taxonomy.md`, see `docs/adr/0010-final-intent-taxonomy.md`): `account_access` + `account_admin` become one `account` intent, `devices_connectivity` folds into `app_technical`, and the discovered presale-code theme is promoted, for 12 support intents plus `other`. Later stages import the final taxonomy; the seed stays as input evidence.

13. **Golden set: stratified with floors.** Proportional across intents with a ~10-example per-intent floor, ~60/40 auto/escalate balance, so rare intents are visible and false-auto rate has enough positives to be measurable.

14. **Judge validated, not trusted.** ~50 golden examples double-rated by human and judge to produce the judge–human agreement evidence the assignment demands.

15. **Deploy anyway.** The assignment doesn't require it, but a live demo (FastAPI + Docker on Railway/Fly with bundled local embeddings, Next.js chat page) makes the system verifiable end-to-end by anyone. ADR to follow when that work starts.
