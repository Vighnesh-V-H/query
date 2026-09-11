# Keyword closure heuristics label obvious endings; DM deflections go to adjudication

Ticket 6 gives every RAG-pool Interaction a closure verdict (ADR-0001). The keyword heuristics read only the closing turns — the final turn, the final customer turn, and the final brand turn — and apply rules in priority order so contradictory signals cannot leak into a label:

1. a customer continuation ("still not working", a question) is Unresolved even when it also thanks the brand;
2. a customer fix report ("it works now") or a pure acknowledgement ("Thanks!", "sorted") is Resolved, unless the brand's preceding reply moved the conversation to DMs;
3. a brand reply that claims completion ("we've fixed it", "you're all set") is Resolved even when it points at a DM for the details;
4. a brand reply that moves the conversation to DMs is flagged;
5. a brand refusal ("unfortunately we can't") is Unresolved, checked after the DM rule so channel refusals followed by a DM request ("we don't offer phone support, can you DM us?") are flagged too;
6. a customer acknowledgement followed by a closing courtesy ("you're welcome") or a plain sign-off is Resolved;
7. a brand question or a promise to investigate followed by silence is Uncertain — the ADR-0001 definition;
8. any other brand-last ending is Uncertain too.

Every verdict carries a fixed reason string, written per Interaction to `data/closure-labels.jsonl` as `{interaction_id, label, reason, needs_adjudication}`.

Closing courtesies are not completion claims: "you're welcome" or "glad to hear" only count next to a customer acknowledgement, so a courtesy with no acknowledgement is Uncertain rather than a guessed Resolved; a courtesy or sign-off next to a request for information or a promise to investigate is still Uncertain, because the issue is open. Two more rules keep false Resolved labels out of the RAG index: a thank-you only counts as acknowledgement when it is essentially the whole message (long messages usually carry a new request with the courtesy attached), and the opening customer message can never acknowledge help it has not received. A negated fix ("it hasn't been fixed yet") is a continuation, not an acknowledgement, and apostrophes are normalized before matching because the dataset mixes straight and typographic ones.

The ambiguous middle is flagged, not guessed: brand replies that move the conversation to DMs, customer messages that only announce a DM, and customer closings that neither acknowledge nor continue get `label: null`, `needs_adjudication: true`, and a reason. This is how ADR-0001's "silence endings" are read here: a plain answer-then-silence has no hidden branch and is Uncertain by the taxonomy's definition, while a DM move can hide the outcome in a private conversation no later stage can read — that is the ambiguity ticket 7's labeler adjudicates.

## Considered Options

- **Label DM deflections Uncertain directly**: rejected as the primary treatment. The private conversation is exactly where resolution may have happened; keywords cannot see it, which is what the LLM adjudication stage (ticket 7) exists for. The flag keeps that uncertainty explicit instead of baking it into a label.
- **Flag every brand-last silence ending**: rejected. It would flag ~80% of the sample and leave the heuristics labeling only the tail, failing ticket 6's "clear majority" criterion; silence after a non-deflecting brand reply is already the Uncertain definition, so no adjudication is needed to say so.
- **Pure heuristics over all endings**: rejected. DM deflections and ambiguous customer closings cannot be told apart from genuine closure by keywords; guessing there would poison the RAG pool with bad support behaviour (ADR-0001).
- **Pure LLM labeling**: rejected. Most endings are clear, keyword labels are deterministic and free, and each carries an auditable fixed reason.

## Consequences

- The run labels 1,694 of 3,000 Interactions (56.5%) definitively — 190 Resolved, 1,317 Uncertain, 187 Unresolved — and flags 1,306 (43.5%), of which 1,225 (94%) involve a move to DMs. Among endings that never touch DMs, 95.4% get a definitive label.
- `data/closure-labels.jsonl` is the input contract for ticket 7 (adjudication). Flagged records have `label: null`: a consumer that only understands labels must skip them, never default them. Ticket 7's adjudicated output, `data/closure-labels-final.jsonl` (ADR-0007), is what feeds ticket 8's resolution dataset.
- Only Resolved Cases (the 190 heuristic ones plus whatever ticket 7 recovers) become retrieval-eligible Historical Cases; Uncertain and Unresolved never enter the index (decision 4).
