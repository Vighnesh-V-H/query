# Stitch Interactions from in_response_to edges, ordered by created_at

The twcs dataset records each tweet's reply target (`in_response_to_tweet_id`, single and reliable) alongside a `response_tweet_id` list that is incomplete and can include indirect mentions; we reconstruct SpotifyCares Interactions by walking `in_response_to_tweet_id` only and ordering turns by parsed `created_at` timestamps, because tweet ids are not monotonic with reply time (55% of reply edges invert id order) while timestamps never do.

Seeds are inbound tweets that engage the brand: they mention the handle (prefix match, so `@SpotifyCaresHelp` does not match), were replied to by the brand, or reply to the brand. From each seed we climb to the opening Customer Message — passing *through* intervening brand turns (a customer answering a brand question continues the same Interaction) and through same-author ancestors that also engage the brand (customer self-reply chains). The opening then grows downward into a dyad restricted to the customer author and the brand, so third-party interlopers start their own Interactions instead of polluting the customer's. An Interaction requires at least one brand turn: without a brand reply, no closure verdict (per ADR-0001) can ever be assigned, so such chains are reported as unanswered openings rather than emitted.

## Consequences

- Counts reported per stage (seeds → openings → Interactions → turns) keep the report honest about what was dropped and why.
- The JSONL output (`data/interactions.jsonl`) is the input contract for the English filter (ticket 4) and sampling (ticket 5); those stages must not re-parse twcs.csv.
