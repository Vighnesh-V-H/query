# Intent taxonomy

**Version:** 1

The final v0 intent taxonomy for SpotifyCares support, reconciled by ticket 11 between data-driven discovery (ticket 10: `docs/intent-discovery.md` and the cluster artifact `data/intent-clusters.jsonl`) and the seeded domain list (ticket 9: `docs/intent-seed-taxonomy.md`). Discovery says what is in the data, the seed says what we expect to find (`docs/decisions.md`, decision 12); this document is the merge, and it supersedes the seed as the single source of truth. Later stages import it with `query.taxonomy.read_final_taxonomy()` and never read the seed.

The taxonomy holds **12 support intents plus an `other` fallback**, inside the 8–15 target of `docs/specs-v0.md` §4. Every merge, fold, and promotion behind the list is recorded under Reconciliation, and `query reconcile-intents` re-checks those decisions against the cluster artifact: it fails when a cluster mapping no longer routes through the recorded decisions, so this document cannot silently drift from the data.

## Intents

| # | Intent id | Definition | Origin |
|---|-----------|------------|--------|
| 1 | `account` | Getting into and managing the account: login failures, wrong or forgotten credentials, password resets, Facebook-linked sign-in, hacked or compromised accounts, and changing or deleting account details. | merged `account_access` + `account_admin` |
| 2 | `billing_payment` | Money matters: charges and receipts, failed or duplicate payments, updating card or PayPal details, refunds, and gift cards. | seed, kept |
| 3 | `subscription_plans` | Choosing or changing what the customer pays for: Free vs Premium, trials and promos, Student and Duo plans, upgrades and downgrades, Hulu bundles, and ads on the Free tier. | seed, kept |
| 4 | `family_plan` | Running an existing Family plan: sending or accepting invites, adding or removing members, address verification, and members losing access. | seed, kept |
| 5 | `playback` | Getting music to play: songs that will not play, playback errors mid-stream, and shuffle, queue, repeat, skip, or volume behavior. | seed, kept |
| 6 | `library_playlists` | The customer's own collection: playlists that cannot be loaded, edited, or found, liked and saved songs, and library organization or limits. | seed, kept |
| 7 | `downloads_offline` | Offline listening: downloads that fail, vanish, or use cellular data, offline mode, and storage. | seed, kept |
| 8 | `app_technical` | The app or site misbehaving outside normal playback — crashes, freezes, loading and error screens, outages, and update bugs — plus getting Spotify onto and across devices (speakers, TV, car, watch, Bluetooth, and AirPlay). | seed, kept; absorbs `devices_connectivity` |
| 9 | `content_availability` | The catalog: missing, removed, or region-blocked songs and albums, release and add-this-artist requests, lyrics, and explicit or clean versions. | seed, kept |
| 10 | `market_availability` | Where Spotify itself is available: country-launch questions, using the service abroad, and moving an account between countries. | seed, kept |
| 11 | `feature_feedback` | Opinions rather than faults: feature requests, UI/UX criticism, and recommendation or algorithm complaints. | seed, kept |
| 12 | `presale_codes` | Artist presale codes: requests for a code, codes that never arrived, and how Spotify presale access works. | promoted from discovery cluster 1 |
| 13 | `other` | Fallback class for messages that fit no support intent: praise, jokes, bare mentions, support-channel chatter, and other non-actionable noise. | seed, fallback |

## Representative messages

Two verbatim opening Customer Messages per intent, drawn from the recorded discovery clusters (interaction id in brackets); `other` draws on the junk clusters and the seed document.

- `account` [2779503]: @SpotifyCares I cant log into my account , I need this fixed like now
- `account` [1819027]: @SpotifyCares my account was just hacked. Someone else is logged in and changed the email address. Help!
- `billing_payment` [367624]: @SpotifyCares hi, I got charged for premium although I had cancelled my subscription. Please make it right. Thanks
- `billing_payment` [2647464]: Trying to pay for premium but won’t accept my card? @SpotifyCares
- `subscription_plans` [1916783]: @115888 stop making us see ads no one wants to pay
- `subscription_plans` [2429632]: @115888 why are your ads so loud??
- `family_plan` [2978154]: @SpotifyCares Can't join to Family Premium for Family with my invitation. 3 - Oops something went wrong, please try again. In family redeem form.
- `family_plan` [1846413]: @SpotifyCares I can’t send and invite to a family member to join my family premium plan, keeps sayin try again, please help https://t.co/Ts21uwbF4w
- `playback` [2920760]: @SpotifyCares Getting error message "Can't play song" for everything. Tried all the step in the troubleshooting website. Any advice?
- `playback` [1840273]: @SpotifyCares my music isn't playing
- `library_playlists` [1622794]: @SpotifyCares @497162 all my playlists disappeared ... but no help in the last 2 days..
- `library_playlists` [1717248]: @115888 thanks for deleting all my downloaded songs....
- `downloads_offline` [890659]: @115888 WHY DO YOU UNSAVE MY MUSIC AT LEAST ONCE A WEEK I LITERALLY PAY FOR PREMIUM SO I DONT WASTE MY DATA
- `downloads_offline` [2021446]: I’m so tired of @115888 constantly un downloading songs and playlists. LET ME LIVE
- `app_technical` [1516306]: .@115888 needs to get their shit together for the nth time. The app crashed on me like 3 times in the past hour and a half
- `app_technical` [2395935]: Are you kidding me @SpotifyCares? 3 app updates since the iPhone X launch and still no support for it? WTF
- `content_availability` [1086999]: @SpotifyCares why can’t i stream some artists songs? Like @1721 @236757 it’s says that the songs are not available https://t.co/sAxwqqVvHJ
- `content_availability` [831863]: @115888 please reupload @17971's debut album thank you in advance.
- `market_availability` [2641123]: @115888 But when are you gonna be available in India?
- `market_availability` [2317485]: @115888 when are you coming to south africa? we've been waiting patiently for too long now. do the thing please, thanks.
- `feature_feedback` [2466912]: @SpotifyCares I wish you’d let us see who follows our playlists☹️
- `feature_feedback` [2346347]: @115888 Why are you always sneaking songs that aren't on my playlist into the mix? I know you have a browse, and I go there. Let me play my playlist with MY songs. Getting tired of it.
- `presale_codes` [2150438]: @115888 How do we get presale codes for @37032 tour??
- `presale_codes` [2203840]: @115888 I DDNT GET MY PRESALE CODE FOR @37032 CONCERT!!! WHERE IT AT?!?? 😭
- `other` [2220683]: @SpotifyCares that's dope. You're awesome! https://t.co/qi3UN271zq
- `other` [1386925]: @115888 @SpotifyCares

## Reconciliation

The recorded discovery run put all 3,000 RAG-pool messages into 30 clusters: 26 mapped to seed intents (2,501 messages), 1 flagged `new` (23), and 3 `junk` (476). The decisions below move from that snapshot to the final list. Cluster ids refer to `data/intent-clusters.jsonl`.

### Seed decisions

| Seed intent | Decision | Final intent | Evidence |
|-------------|----------|--------------|----------|
| `account_access` | merged | `account` | 3 clusters (13, 19, 23), 332 messages — login failures, compromised accounts, and generic account help; governs the merged intent. |
| `account_admin` | merged | `account` | 0 clusters; its themes (changing email or password, deleting the account) belong with recovery in one `account` intent, as the seed's boundary note already anticipated. |
| `billing_payment` | kept | `billing_payment` | 5 clusters (2, 5, 10, 16, 28), 490 messages — the largest mapped theme. |
| `subscription_plans` | kept | `subscription_plans` | 1 cluster (3), 42 messages — ads, trials, and promo complaints; distinct from a charge problem, so it is not folded into `billing_payment`. |
| `family_plan` | kept | `family_plan` | 1 cluster (6), 86 messages — invite and join failures. |
| `playback` | kept | `playback` | 2 clusters (17, 27), 252 messages. |
| `library_playlists` | kept | `library_playlists` | 2 clusters (8, 14), 184 messages. |
| `downloads_offline` | kept | `downloads_offline` | 1 cluster (26), 100 messages. |
| `devices_connectivity` | dropped | - | 0 clusters; the labeler mapped the device and platform-support clusters (4, 29) to `app_technical`, so the scope folds into that definition instead of staying a data-thin intent. |
| `app_technical` | kept | `app_technical` | 3 clusters (4, 12, 29), 282 messages — crashes and update bugs, including iPhone X support; absorbs `devices_connectivity`. |
| `content_availability` | kept | `content_availability` | 3 clusters (15, 18, 25), 281 messages. |
| `market_availability` | kept | `market_availability` | 2 clusters (7, 24), 183 messages. |
| `feature_feedback` | kept | `feature_feedback` | 3 clusters (0, 9, 20), 269 messages. |

### New themes

| Cluster | Decision | Final intent | Evidence |
|---------|----------|--------------|----------|
| 1 | promoted | `presale_codes` | 23 messages that uniformly ask for or miss artist presale codes; a coherent support theme no seed intent covers, and `other` must not absorb a new support theme. |

The three `junk` clusters (11, 21, 22; 476 messages) carry no support issue — bare mentions, one-word replies, support-channel chatter — and reconcile to `other`, the fallback every non-support message shares.

### Not decided here

- No seed intent split: no second intent had cluster support, so the reconciliation is ten keeps, two merges, one drop, and one promotion.
- Boundary calls within the kept intents stay as the seed wrote them (`docs/intent-seed-taxonomy.md`).
- High-risk categories, classifier choice and training data, and multi-intent messages remain out of scope; ticket 21 owns risk as a separate axis.
