# 13. Type-annotate `ml_engine/export/merger.py`

**Completed:** 2026-04-25 on `chore/mypy-merger-hygiene`, stacked on `ci/gate-ruff-checks`.

**What shipped:** Three pre-existing mypy errors in `ml_engine/export/merger.py`
fixed without `# type: ignore` band-aids:

- Lines 46 + 54: `peft_model.merge_and_unload()` was flagged `"Tensor" not callable`
  because `nn.Module.__getattr__` is typed `Tensor | Module` and mypy can't see
  through PEFT's runtime delegation. Fix: rebind via `peft_model: Any = model.model`
  (and `direct_peft: Any = model` on the direct branch) — local `Any` annotation
  signals "trust the duck-typed runtime contract here, mypy can't help."
- Line 105: `checkpoint["metadata"].update(extra_metadata)` was flagged because
  mypy widened the dict literal to `Collection[Any]`. Fix: explicit
  `checkpoint: Dict[str, Any] = {...}` annotation prevents the inference widening.

## Why `Any` and NOT a Protocol

The original sketch in this TODO proposed a `@runtime_checkable Protocol`.
Investigation showed PEFT ships `py.typed` but `PeftModel.merge_and_unload`
is exposed via `__getattr__` delegation to `self.base_model` (a `LoraModel`).
mypy fundamentally cannot follow `__getattr__` for arbitrary attribute names —
so even `from peft import PeftModel` and using `isinstance(x, PeftModel)`
would NOT have helped. The Protocol approach would have worked but added 15
lines of new abstraction for a 3-line type fix. `Any` is honest about what's
actually true: mypy can't help here, the runtime hasattr() checks ARE the
contract.

## The SKIP=mypy precedent

This TODO existed because `SKIP=mypy git commit` became habitual across 5
PRs. With merger.py now mypy-clean, the bypass should no longer be needed
for `ml_engine/export/`. (Other dirs still had baseline mypy errors covered
by item 6's broader cleanup, which subsequently completed.)

## Design notes worth remembering

Before assuming a 3rd-party library's typing will solve a static-checking
problem, verify it's not using `__getattr__`-based delegation. Many ML
libraries (PEFT, transformers, accelerate) lean on dynamic composition and
have similar gaps. `Any` at the boundary is a legitimate fix when the
dynamic surface can't be statically modeled.
