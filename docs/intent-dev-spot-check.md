# Intent dev-label spot-check (ticket 12)

Human audit of a 30-label trial slice from `query label-intents`, against the
final taxonomy v1 (`docs/intent-taxonomy.md`). The slice is deterministic
(`--dev-size 30 --seed 42` over `data/rag-pool.jsonl`, 3,000 Interactions);
the labeler is `nvidia/nemotron-3.5-lightning-30b-a3b` at temperature 0.0, one
call per opening Customer Message. Method: read each message, assign the
intent independently from the taxonomy definitions and boundary rules, then
compare with the labeler verdict and its justification. Ranking is by hash of
seed and interaction id, so this 30-slice is the rank-prefix of the default
500-slice under the same seed and its findings transfer.

## Result: 28/30 agree

| Interaction | Predicted | Verdict | Note |
|-------------|-----------|---------|------|
| 64359 | `account` | agree | login/password failure; "app" mention is not the issue |
| 116621 | `billing_payment` | agree | payment not working |
| 130415 | `app_technical` | agree* | two issues (vague account difficulty + help site down); site outage is the specific, actionable one |
| 189914 | `billing_payment` | agree | price-rise question is a money matter |
| 637652 | `family_plan` | agree | spouse invite error |
| 645099 | `family_plan` | agree | address-verification removal + reinvite error |
| 708733 | `content_availability` | agree | US-vs-UK catalog gap is region-blocking, not market availability |
| 723755 | `other` | **flag** | "trouble with Spotify on Galaxy S7" names a device problem; see finding 1 |
| 820213 | `feature_feedback` | agree | artist-block feature request |
| 853079 | `feature_feedback` | agree | UI/UX criticism of playlist import, not a load/edit failure |
| 853096 | `other` | agree | casual share, no issue |
| 1161837 | `app_technical` | agree* | update dropped songs; cause-attributed-to-update beats library symptom |
| 1280145 | `app_technical` | agree | Roku-upgrade breakage; account symptom, device cause |
| 1307344 | `subscription_plans` | agree | ad-free promo failure is a promo issue per boundary rules |
| 1611670 | `playback` | agree | songs vanishing mid-stream |
| 1682030 | `other` | agree | non-actionable fragment + link |
| 1799578 | `account` | agree | email-change blocked as "taken" |
| 1851669 | `market_availability` | agree | continent-availability complaint |
| 1895275 | `other` | agree | names no problem |
| 1915249 | `market_availability` | agree | Venezuela launch question |
| 1948220 | `downloads_offline` | agree | sarcastic thanks correctly read as undownload report |
| 2118797 | `content_availability` | agree | album track-list change |
| 2158585 | `feature_feedback` | agree* | explicit-lyrics auto-skip setting is a feature ask, not a playback fault |
| 2239773 | `content_availability` | agree | album request (Lemonade) |
| 2311102 | `feature_feedback` | agree* | ad-language complaint is UX criticism, not plan choice |
| 2317485 | `market_availability` | agree | South Africa launch question |
| 2391990 | `app_technical` | agree | iPhone X update status |
| 2555541 | `content_availability` | agree | song suddenly missing |
| 2826954 | `app_technical` | agree | Connect/device behavior; justification even cites the devices fold |
| 2902781 | `subscription_plans` | **flag** | paid upgrade not applied; see finding 2 |

Asterisks are borderline-accepts: defensible calls on documented boundaries,
kept as agreements.

## Committed 500-label run

The full dev slice was later labeled end to end with the same seed, taxonomy,
and labeler (`--dev-size 500 --cache data/intent-dev-cache.jsonl`); its
artifacts are committed at `data/intent-dev-labels.jsonl`,
`data/intent-dev-report.json`, and `docs/intent-dev-review.md`. The 30 trial
interactions above are the rank-prefix of that slice, so the same human
judgments audit the committed labels. The labeler is non-deterministic even at
temperature 0.0, and the fresh run moved three of the trial verdicts:

| Interaction | Trial | Committed | Human audit |
|-------------|-------|-----------|-------------|
| 189914 | `billing_payment` | `subscription_plans` | `billing_payment` (money matter) |
| 723755 | `other` | `app_technical` | `app_technical` (finding 1) |
| 1161837 | `app_technical` | `library_playlists` | `app_technical` (update cause beats library symptom) |

Under the audit's independent judgments the committed 30-prefix agrees 27/30
(723755 now matches; 189914, 1161837, and 2902781 disagree). The committed
labels are therefore the pinned record: the prompt-hash cache replays them
exactly, and a fresh run may move verdicts again. All 13 intents have labels
in the committed 500, thinnest `presale_codes` at 8 (1.6%).

## Findings

1. **Vague device-trouble filed as `other` (723755).** "Anyone else having
   trouble with Spotify on Galaxy S7?" states a device problem a support team
   would act on, but its chatter phrasing ("anyone else") pulled the verdict
   to `other`. Hypothesis: the prompt's junk guidance overfires on short
   crowd-asking messages. Fix candidate: one few-shot example mapping short
   device-trouble ("trouble on <device>", no specifics) to `app_technical`.
2. **Paid-but-inactive upgrade filed as `subscription_plans` (2902781).** "I
   upgraded to premium but it hasn't updated in app" sits on the documented
   billing/subscription boundary; the discovery-era rule ("a completed
   payment that did not unlock the plan is billing") favors
   `billing_payment`. Fix candidate: carry that rule explicitly into the
   labeling prompt with one example.
3. **Single-label pressure is visible but contained.** 130415 carries two
   issues; the labeler picked the specific one and named it. No change: v0 is
   single-turn, single-label by scope (decisions 2, 12), and multi-issue
   messages are rare in the slice.
4. **Thin intents vanish at n=30.** The slice holds zero `library_playlists`
   and zero `presale_codes` labels. Expected at this size (presale is 0.77%
   of the RAG pool) and honestly reported by the per-intent zeros — not a
   labeling failure, but the full dev slice still under-covers presale (~4
   expected at n=500), which ticket 13's training split must handle.

## Follow-ups

- Ticket 13 (TF-IDF baseline): train on the full dev slice; consider
  class weights for `presale_codes` and other thin intents.
- Ticket 14 (LLM classifier): adopt findings 1–2 as prompt examples; reuse
  this 30-label slice as the first sanity-check set.
- Re-audit after any prompt change; the committed labels and the prompt-hash
  cache pin this run's verdicts, so a re-run with new examples recomputes
  cleanly.
