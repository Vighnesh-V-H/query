# Filter English with Lingua, keep when ambiguous

The English filter (ticket 4) keeps an Interaction when its opening Customer Message reads as English — the only text the intent classifier sees. Detection uses Lingua, restricted to English plus the languages actually present in the dataset and the major world languages, with handles and links stripped first and Lingua's minimum relative distance (0.25) left to answer "no confident signal" for ambiguous short tweets. An Interaction is dropped only on positive non-English evidence; ambiguous messages (roughly 10% of the set, mostly "How?", "help pls", link- or emoji-only openings) are kept, because the classifier misclassifying a non-English message is the failure this stage exists to prevent, not the only one it must avoid. On the full set the filter retains 29,130 of 29,796 Interactions (2.24% filtered, dominated by Indonesian, Tagalog, Dutch, and French).

## Considered Options

- **`langdetect` / `py3langid`**: rejected. Both n-gram detectors misread short informal English as Nordic or African languages with maximal confidence (`"i am still having payment issues"` → Norwegian, `"did you get my tweet re billing?"` → Afrikaans), which would have dropped ~1,000 genuine English Interactions.
- **LLM labeler over all Interactions**: rejected. ~30K calls is slow, costs trial credits, and is non-deterministic across reruns — the stage must be a documented, re-runnable step.
- **Hand-rolled stopword/script heuristic**: rejected. Overlapping function words (`no`, `do`, `me`) made it unreliable on Taglish and Spanish, and it would be custom logic to defend rather than a library with known behavior.

## Consequences

- Lingua 2.2 requires Python >= 3.12, so the workspace `requires-python` floor moved from 3.11 to 3.12.
- `data/interactions-en.jsonl` is the input contract for sampling (ticket 5); the report (`data/english-filter-report.json`) records retained, no-signal, and filtered-by-language counts so the report can state the retained volume honestly.
- The filter is deterministic: a pinned language set, offline models, and no randomness, and only the opening message is classified — brand turns in another language never drop an otherwise English Interaction.
