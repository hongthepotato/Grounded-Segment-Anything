# Decisions

Long-form record of completed work that surfaced patterns, lessons, or design
decisions worth keeping. Each file captures *why* the work happened, *what*
shipped, and the *patterns* established — context that doesn't fit in a commit
message and is too long-lived to live in `TODOS.md`.

## Layout

One file per completed TODO item, named `NN-<short-slug>.md` where `NN` matches
the original item number from `TODOS.md`. Numbers are stable (commit messages
and PR descriptions reference them as "item N" / "TODO #N") so existing
backlinks resolve.

## Why these moved out of TODOS.md

`TODOS.md` was originally append-only with a `## Completed` section at the
bottom. As the project grew, completed entries (especially the multi-week
mypy baseline cleanup, item #6) ballooned the file past the point where
scanning for active items was easy. Moving long-form completed entries here
keeps `TODOS.md` focused on what's still open, while preserving the full
historical context that explains *why* certain patterns exist in the codebase.

## Reading order

If you're new to the project and want the design rationale for current
patterns, read in chronological order:

1. [04-pre-commit-hooks.md](04-pre-commit-hooks.md) — local + CI lint parity
2. [10-augmentation-validator-tests.md](10-augmentation-validator-tests.md) — error-path test coverage approach
3. [13-merger-py-mypy-fix.md](13-merger-py-mypy-fix.md) — `Any`-at-the-boundary precedent for PEFT
4. [06-mypy-baseline-cleanup.md](06-mypy-baseline-cleanup.md) — the big one; established the patterns most subsequent type work follows
5. [16-job-id-lineage-plumbing.md](16-job-id-lineage-plumbing.md) — first follow-up that surfaced from #6
