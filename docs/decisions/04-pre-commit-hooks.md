# 4. Pre-commit hooks

**Completed:** 2026-04-24 via PR #26 merged into `agentic`.

**What shipped:** `.pre-commit-config.yaml` with file-hygiene hooks (trailing
whitespace, EOF newline, YAML/TOML validity, large-file cap, merge conflict
markers, private-key detection, debug-statements, Python AST validity) plus
`ruff check`, `ruff format --check`, and `mypy` via `uv run --no-sync` so the
hook binaries match CI exactly. Added `pre-commit>=3.6.0` to the `dev` extra.
Onboarding: `uv sync --extra dev && uv run pre-commit install`.

**Design notes worth remembering:** `--fix` deliberately NOT enabled for ruff
because the ~3500-finding baseline would produce a destructive first-run diff.
Per-file cleanup happens naturally as pre-commit flags new violations when
you touch a file — see item 9.
