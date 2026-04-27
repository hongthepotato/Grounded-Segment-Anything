# 9. Clean up ruff baseline (3593 findings at ci-and-tests merge time)

**Status:** Complete 2026-04-25 across 5 PRs merged into `agentic`.

**Final state:** Ruff `check` and `format --check` both gating in CI
(no `continue-on-error`). 0 findings across `core ml_engine api
augmentation`. Set the precedent later followed by mypy (TODO #6 / PR #40).

## PR chain (5 PRs total)

- PR #29 (`chore/ruff-baseline-core`) — auto-fix + format
- PR #30 (`chore/ruff-baseline-api`) — auto-fix + format + 4 manual
- PR #31 (`chore/ruff-baseline-augmentation`) — auto-fix + format + 2 manual + dead code removal
- PR #32 (`chore/ruff-baseline-ml-engine`) — auto-fix + format + bumped `line-length` to 110 (was 88)
- PR #33 (`ci/gate-ruff-checks`) — flipped `continue-on-error: true` → false on the ruff steps; cleaned up `tests/cli` along the way + deleted dead CLIs

## Initial baseline (3593 findings)

Measured at `ci-and-tests` merge time:

| Code | Count | Description |
|---|---|---|
| `W293` | 1486 | blank-line-whitespace |
| `E501` | 1027 | line-too-long |
| `I001` | 367 | unsorted-imports |
| `F401` | 291 | unused-import |
| `W291` | 183 | trailing-whitespace |
| `F841` | 34 | unused-variable |
| `F821` | 31 | undefined-name |

`F821` was the one to watch — undefined-name flags can be real bugs. The
manual-cleanup PRs investigated each one; most were stale references in
unused branches that came out as part of the dead-code removal in
PR #31 (augmentation).

## Per-directory pattern

`uv run ruff check . --fix` auto-fixed ~1732 of the 3593 in one pass
(W293/W291/F401 are mechanical). The remaining ~1860 needed:

- `ruff format` for line-length and style
- Manual review for `F821` (undefined-name) and `F841` (unused-variable)
- Per-directory line-length policy: bumped to 110 in PR #32 to absorb
  long signature lines in ml_engine/ without aggressive line-wrapping
  that would have hurt readability

## Why per-directory PRs (not one big PR)

3000-line auto-fix diffs are unreviewable as a single unit. Splitting by
top-level dir kept each PR around 500-700 lines, reviewable in 15
minutes, and let the `--fix` blast be reviewed alongside the manual
fixes for that same area (so the reviewer could see "auto-fix removed
this import" next to "manual fix renamed this variable").

## Why this matters

Set the precedent that a clean baseline + gate flip is achievable as a
sequence of manageable PRs. The same playbook then ran for mypy
(TODO #6 / PR #34-40). Both gates now block merges, matching the
behavior contributors expect from `pyproject.toml`-configured tools.

## Pre-commit relationship

PR #26 (TODO #4) wired ruff into pre-commit BEFORE this cleanup, with
`--fix` deliberately disabled — the ~3500-finding baseline would have
produced a destructive first-run diff for any contributor running
`pre-commit run --all-files`. After this cleanup, per-file pre-commit
runs flag only NEW violations as files are touched, which is the
ongoing-maintenance model the project relies on.
