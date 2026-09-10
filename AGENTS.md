# AGENTS.md

Guidelines for AI agents working on this repository ("Query" — a customer support agent).

## 1. Branch discipline — never work directly on an existing branch

Before starting any issue, fix, or feature:

1. Check the current branch: `git branch --show-current`
2. Check the working tree: `git status --short`
3. If the working tree is dirty, **commit the pending changes first** (or stash them only if the user asks) before doing anything else.
4. Create a new branch from the current state and do all work there.

Never make changes directly on the current working branch. This protects the branch from corruption.

## 2. Review and manual verification before moving on

- After completing any code change, **review it with multiple sub-agents** to check correctness, edge cases, and consistency with the codebase.
- End-to-end correctness must be verified (and, where possible, tested manually) before proceeding to the next feature or fix.
- The user reviews each change manually — do not start the next task until the current one is confirmed.

## 3. Branch and commit conventions

Use standard naming and commit message styles:

- **Branch names:** `<type>/<short-description>` — e.g., `feat/customer-intake`, `fix/retry-logic`, `chore/update-docs`.
- **Commit messages:** Conventional Commits format — `<type>: <short summary in lowercase>`, e.g., `feat: add ticket triage flow`, `fix: handle empty webhook payload`. Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`.
- Commit messages should be concise, imperative, and describe what the change does.
- **Keep commits small.** One logical change per commit — never bundle an unrelated fix, refactor, or docs change into a feature commit. Big commits slow down gate review and make regressions hard to bisect.

## 4. Push through the no-mistakes gate

- Never push directly to `origin`. All pushes go through the local validation gate: `git push no-mistakes <branch>`.
- The gate runs the pipeline `intent → rebase → review → test → document → lint → push → PR → CI-watch` before the branch goes public.
- Review gate findings and resolve them (approve fixes, apply changes) before moving on.
- If the gate reports "Checks passed", stop and ask the user to review and merge the PR.

## 5. Change report after every completed task

After completing any task, fix, or change, report to the user in this standard format:

```text
## Change report — <task/issue title>

### Modified files
| File | What was done | Impact on other files |
|------|---------------|----------------------|
| path/to/file | what changed there and why | which files/behavior it affects |

### New files (if any)
| File | Purpose | Consumed by |
|------|---------|-------------|

### Removed files (if any)
| File | Why removed | What replaced it |
|------|-------------|------------------|

### Verification
How the change was verified (tests run, commands executed, manual checks).

### Follow-ups
Tickets, blockers, or downstream effects this change creates.
```

Keep it concise: one row per file, no diff dumps. Every completed task gets this report before moving on.

## Additional guidelines

Will be added as the project evolves.
