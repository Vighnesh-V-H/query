# Intent seed taxonomy

A domain-knowledge candidate taxonomy for SpotifyCares support, to reconcile data-driven discovery against (ticket 10) and merge into the final versioned taxonomy (ticket 11). Discovery says what is in the data; this seed says what we expect to find; neither decides the final names alone (`docs/decisions.md`, decision 12).

The seed has **13 support intents plus an `other` fallback**; the final target is 8–15 intents (`docs/specs-v0.md` §4). Nothing here is final: discovery may find a seed intent absent, split one into two, or surface a new cluster; ticket 11 records every merge and split. Intent ids are snake_case and serve as candidates; ticket 11 owns the final ids and version.

## Seed intents

| # | Intent id | One-line definition | Rough signal |
|---|-----------|---------------------|--------------|
| 1 | `account_access` | The customer cannot get into their account: wrong or forgotten credentials, password resets that never arrive, Facebook-linked sign-in, locked or compromised accounts. | ~11% |
| 2 | `account_admin` | Managing the account itself: changing email, username, or password; cancelling, closing, or deleting the account; verifying ownership before a change. | ~2% |
| 3 | `billing_payment` | Money matters: charges and receipts, failed or duplicate payments, updating card or PayPal details, refunds, and gift cards. | ~12% |
| 4 | `subscription_plans` | Choosing or changing what the customer pays for: Free vs Premium, trials and promos, Student and Duo plans, upgrades and downgrades, Hulu bundles, and ads on the Free tier. | ~20% |
| 5 | `family_plan` | Running an existing Family plan: sending or accepting invites, adding or removing members, address verification, and members losing access. | ~5% |
| 6 | `playback` | Getting music to play: songs that will not play, playback errors mid-stream, and shuffle, queue, repeat, skip, or volume behavior. | ~11% |
| 7 | `library_playlists` | The customer's own collection: playlists that cannot be loaded, edited, or found, liked and saved songs, and library organization or limits. | ~11% |
| 8 | `downloads_offline` | Offline listening: downloads that fail, vanish, or use cellular data, offline mode, and storage. | ~4% |
| 9 | `devices_connectivity` | Getting Spotify onto and across devices: Spotify Connect, speakers, TV, car, and watch integrations, Bluetooth and AirPlay, and platform app support. | ~4% |
| 10 | `app_technical` | The app or site misbehaving outside normal playback: crashes, freezes, loading and error screens, outages, and bugs introduced by an update. | ~7% |
| 11 | `content_availability` | The catalog: missing, removed, or region-blocked songs and albums, release and add-this-artist requests, lyrics, and explicit or clean versions. | ~7% |
| 12 | `market_availability` | Where Spotify itself is available: country-launch questions, using the service abroad, and moving an account between countries. | ~2% |
| 13 | `feature_feedback` | Opinions rather than faults: feature requests, UI/UX criticism, and recommendation or algorithm complaints. | ~4% |
| 14 | `other` | Fallback class for messages that fit no support intent: praise, jokes, bare mentions, support-channel chatter, and other non-actionable noise. | ~39% unmatched |

**Rough signal** is an ad hoc keyword scan over all 3,000 opening Customer Messages in `data/rag-pool.jsonl` (not a committed command), not labels. Patterns overlap (a family billing problem hits three rows), and 39% of messages match no pattern at all, so the numbers are directional: they show a theme is present, not how it should be counted. Discovery clusters are the real measurement.

## Boundary calls and deliberate folds

- **`account_access` vs `account_admin`** — recovery from being blocked out goes to access; changing or ending the account goes to admin. A hacked account that turns into an ownership fight sits on the line; the two may merge into one `account` intent at finalization.
- **`playback` vs `app_technical`** — a fault in the music stream or player is `playback`; the app or site failing on any screen is `app_technical`.
- **`billing_payment` vs `subscription_plans` and `family_plan`** — any charge, receipt, or payment-method problem is `billing_payment`; choosing or changing a plan is `subscription_plans`; managing Family members after purchase is `family_plan`. A completed payment that did not unlock the plan ("paid but still on Free") is `billing_payment`.
- **`library_playlists` vs `downloads_offline`** — an offline-listening failure is `downloads_offline`; a saved album or playlist disappearing without a download context is `library_playlists`.
- **`content_availability` vs `market_availability`** — a specific track or album is content; the service in a country is market.
- **Deliberate folds** — ads on the Free tier → `subscription_plans`; lyrics and explicit/clean versions → `content_availability`; gift cards → `billing_payment`; podcast-specific problems (20 messages) → `playback` or `app_technical`, too thin to seed alone.
- **`other` is a fallback, not a discovery target** — discovery maps clusters to the 13 support ids, flags a support theme that matches none as new, and flags non-support clusters as junk; `other` exists so the classifier has a home for messages that belong to no intent (including junk-like chatter), and must not absorb a new support theme just to keep the count tidy.

## Representative messages

One verbatim opening Customer Message per intent, from `data/rag-pool.jsonl` (interaction id in brackets):

- `account_access` [1975169]: @SpotifyCares My account got logged out and I can’t get back in. I’ve tried just resetting the password but I get nothing in my email. Help
- `account_admin` [1051992]: @SpotifyCares I need to delete my account
- `billing_payment` [1698944]: @SpotifyCares hi! My monthly payment was declined because my credit card was locked, it is unlocked now can you process that for me
- `subscription_plans` [2135081]: Well I was told 30 days then I go on Spotify and it tells me 7 days? @115888 ??? https://t.co/JqXLoeinat
- `family_plan` [855247]: @117153 I sent an "invitation to join Family Premium" to my husband but the link and the code provided in the email didn't work.
- `playback` [115249]: @117153 @SpotifyCares Still not playing music
- `library_playlists` [211133]: @SpotifyCares Hello here to ask you why did you delete the Flume's House Party playlist from spotify I used to listen to it and its gone :((
- `downloads_offline` [1825251]: Serious Qs tho @115888 Why does the app keep forgetting I've saved albums and frequently removes downloaded music?
- `devices_connectivity` [2569857]: @spotifycares Why oh why can't you disable spotify connect?! It causes so many issues!
- `app_technical` [1864818]: @115888 any chance you could sort out the glitch that means we have no control on the iPhone lock screen?
- `content_availability` [466070]: Why was "Too Much To Ask" by Niall Horan removed from Spotify Sessions Hits playlist ??  @SpotifyCares
- `market_availability` [1839742]: @SpotifyCares Hello Spotify! When are you going to launch Spotify in Romania? Thanks have a great one!
- `feature_feedback` [2032051]: @115888 I have a fantastic idea for your platform. Who can I talk to with regard to it?
- `other` [2220683]: @SpotifyCares that's dope. You're awesome! https://t.co/qi3UN271zq

## What this does not decide

- **High-risk categories get no seed intent.** `CONTEXT.md`'s High-Risk Categories cut across these ids (credentials and security → `account_access`/`account_admin`, payment disputes → `billing_payment`, account deletion → `account_admin`, abuse or legal threats → `other`); escalation policy keys off them separately (ticket 21). A threat or abuse message is `other` by topic but is still always escalated — risk is a separate axis from intent, and this taxonomy does not encode it.
- **Classifier choice and training data.** The seed is input to discovery and finalization, not the classifier; TF-IDF and LLM classifier work (tickets 13–14) consume the final taxonomy only.
- **Multi-intent messages.** In v0 the classifier sees only the opening Customer Message (`docs/decisions.md`, decision 2), so the taxonomy labels the message as a whole, not every issue it mentions.

## Next

- Ticket 10 maps each embedding cluster to an id above, or flags it as new or junk, with example messages per cluster for review.
- Ticket 11 merges seed and clusters into the final 8–15-intent taxonomy with definitions and examples, versioned and importable by later stages.
