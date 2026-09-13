# Query frontend (minimal chat UI)

Minimal Next.js single page for issue #30. A message form renders a **mocked**
pipeline response — intent + confidence, reply, evidence case ids, and the
auto/escalate decision + reason — so UI work proceeds while the backend is built.

> Mock lives in `lib/mock.ts` and is clearly marked. Issue #31 replaces
> `getMockedPipelineResult` with a real backend fetch. The `PipelineResult`
> shape (`intent`, `confidence`, `evidence`, `reply`, `decision`, `reason`) is
> already compatible with the future backend JSON (specs-v0.md §§4–7).

## Run locally

Requires Node 18+ (tested on Node 20/24).

```sh
cd apps/frontend
npm install
npm run dev
```

Open http://localhost:3000.

Other commands (from `apps/frontend`):

```sh
npm run build   # production build
npm start       # serve the production build
npm run typecheck  # tsc --noEmit
```

## What to try

- Submit a normal message (e.g. "My music isn't playing") → mocked `AUTO` result.
- Submit "I was charged twice" or "I can't log in" → mocked `ESCALATE` result
  (payment dispute / security credentials always escalate per the decision policy).
- Submit empty text → validation error; over-long text → length error.
