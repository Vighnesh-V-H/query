# Keyword closure heuristics label obvious endings; DM deflections go to adjudication

Ticket 6 gives every RAG-pool Interaction a closure verdict (ADR-0001). The keyword heuristics read only the closing turns — the final turn, the final customer turn, and the final brand turn — and apply rules in priority order so contradictory signals cannot leak into a label: a customer continuation ("still not working", a question) is Unresolved even when it thanks the brand; a customer fix report ("it works now") or a pure acknowledgement ("Thanks!", "sorted") is Resolved; a brand completion claim ("we've fixed it", "you're welcome") is Resolved; a brand refusal ("unfortunately we can't") is Unresolved; a brand question or a promise to investigate followed by silence is Uncertain — the ADR-0001 definition; and any other brand-last silence is Uncertain too. Every verdict carries a fixed reason string, written per Interaction to `data/closure-labels.jsonl` as `{interaction_id, label, reason, needs_adjudication}`.

The ambiguous middle is flagged, not guessed. Brand replies that move the conversation to DMs, customer messages that only announce a DM, and customer closings that neither acknowledge nor continue get `label: null`, `needs_adjudication: true`, and a reason explaining the ambiguity, ready for the labeler-role adjudication in ticket 7. Two rules exist to keep false Resolved labels out of the RAG index: a thank-you only counts as acknowledgement when it is essentially the whole message (long messages usually carry a new request with the courtesy attached), and the opening customer message can never acknowledge help it has not received. Apostrophes are normalized before matching because the dataset mixes straight and typographic ones.

## Considered Options

- **Label DM deflections Uncertain directly**: rejected as the primary treatment. The private conversation is exactly where resolution may have happened; keywords cannot see it, which is what the LLM adjudication stage (ticket 7) exists for. The flag keeps that uncertainty explicit instead of baking it into a label.
- **Pure heuristics over all endings**: rejected. DM deflections and ambiguous customer closings cannot be told apart from genuine closure by keywords; guessing there would poison the RAG pool with bad support behaviour (ADR-0001).
- **Pure LLM labeling**: rejected. Most endings are clear, keyword labels are deterministic and free, and each carries an auditable fixed reason.

## Consequences

- The run labels 1,712 of 3,000 Interactions (57.1%) definitively — 211 Resolved, 1,231 Uncertain, 270 Unresolved — and flags 1,288 (42.9%), of which 94% are DM deflections.
- `data/closure-labels.jsonl` is the input contract for ticket 7 (adjudication) and ticket 8 (resolution dataset). Flagged records have `label: null`: a consumer that only understands labels must skip them, never default them.
- Only Resolved Cases (the 211 heuristic ones plus whatever ticket 7 recovers) become retrieval-eligible Historical Cases; Uncertain and Unresolved never enter the index (decision 4).
