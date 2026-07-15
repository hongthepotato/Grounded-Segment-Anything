# 12. Priority-2 unit test roster — 6 remaining files from ci-and-tests design doc

**Status:** Complete 2026-04-27 across 4 PRs merged into `agentic`.

**Final state:** All 6 P2 files exist. Combined coverage gain across the
roster: ~470 new tests (375 passing + 47 deliberate xfail markers
documenting source-level gaps, two of which were closed inline; the
remaining 38 xfails are tracked in TODOs #18 and #19).

## PR chain (4 PRs)

| File | PR / branch | Tests | Notes |
|---|---|---|---|
| `export/test_merger.py` + `export/test_packager.py` + `distillation/test_pseudo_label.py` | `test/p2-roster-export-distillation` | 63 pass + 8 xfail | Earlier PR (already merged) |
| `api/test_schemas.py` | `test/p2-api-schemas` (PR #44) | 128 pass + 37 xfail | Sub-PR 1/3 of the second wave |
| `augmentation/test_parameter_system.py` | `test/p2-augmentation-parameter-system` (PR #43) | 142 pass + 0 xfail | Sub-PR 2/3 — all 5 source-level gaps fixed inline |
| `augmentation/test_characteristic_translator.py` | `test/p2-augmentation-characteristic-translator` | 92 pass + 1 xfail | Sub-PR 3/3 — capped at "factory dispatch + 5 most-used characteristics" per scope cap |

## What shipped

Six new unit-test files covering the six modules listed in the
ci-and-tests design doc's Priority-2 roster (the seventh,
`test_augmentation_factory.py`, was covered by item #10 separately).

The tests follow a single reference pattern from
`tests/unit/augmentation/test_augmentation_factory.py`: class-per-area,
parametrize variants with descriptive ids, separate happy paths from
error paths, and probe the gap between docstring promises and what the
source actually enforces.

## Test design — the harsh-test approach

Per "be harsh on the unit test, the test should be designed to catch
bugs not just let it pass easily," every Field / function / dataclass
in scope was checked against its docstring contract. Where the
docstring promised a constraint that the source code didn't enforce,
an `xfail(strict=True)` test was added pinning the gap with a
`reason=` line pointing at the exact source line + the recommended fix.

This means the test file IS a structured to-do list:
- Today: gap is real → xfail PASSES (test "expected to fail" did fail)
- Source tightened: xfail UNEXPECTEDLY PASSES → strict mode trips → CI
  fails → developer flips `@pytest.mark.xfail` off → test becomes a
  permanent regression guard

## Real bugs surfaced and FIXED inline

**`augmentation/parameter_system.py` (5 categories, all fixed in the
same PR after caller audit):**

- `convert_to_numeric` accepted `True`/`False` because
  `isinstance(True, int)` is True in Python. Downstream
  `RangeParameter(False, True)` would silently build a `[0.0, 1.0]`
  range from booleans. Fixed by adding an explicit
  `isinstance(value, bool)` check BEFORE the int/float pass-through.
- `convert_to_numeric` rejected `'inf'` / `'-inf'` / `'nan'` strings
  via the int-branch fallback (confusing wrapped TypeError). Per
  product call ("inf isn't useful in albumentations parameter ranges"),
  added explicit rejection with a clear "Non-finite float string not
  supported" message + the same finite-check on direct `float('inf')`
  values.
- `RangeParameter(0.5, 0.9, is_integer=True)` silently truncated to
  `[0, 0]` via `int()` floor-toward-zero. Fixed by raising
  `ValueError` on non-integer bounds in `__post_init__`. Use
  `RangeParameter.integer_range(...)` for explicit float→int.
- `integer_range(float('inf'), 10)` leaked raw `OverflowError`.
  Defensive fix: added `OverflowError` to the except tuple.
- `is_scalar()` used `==` on floats, so `RangeParameter(0.1 + 0.2, 0.3)`
  reported `is_scalar()=False`. Fixed by switching to `math.isclose`.
  Same tolerance also applied to the `min_val > max_val` validation.

**`api/schemas.py` (8 SAFE categories closed inline; 7 truly-breaking
deferred to TODO #18):**

The 8 safe ones — server-controlled fields, mathematical invariants,
or COCO-spec conformance third-party data already meets:
- `JobProgressSchema.overall_progress` range `(0.0-1.0)` + `ge=0` on
  `current_epoch/total_epochs/current_step/total_steps` ints.
- `JobProgressSchema` epoch/step invariant `(current <= total)` via
  `@model_validator`.
- `DistillationRequest.split_config` sum-and-non-negative via
  `@model_validator` using `math.isclose`.
- `COCOImageSchema.width/height > 0`.
- `COCOAnnotationSchema.bbox` length=4 (COCO `[x, y, w, h]`).
- `COCOAnnotationSchema.iscrowd` `Literal[0, 1]` (COCO binary).
- `COCOAnnotationSchema.score` 0.0-1.0 (confidence semantic).
- `VisualizationInfo.annotation_count >= 0`.

## Real bugs surfaced and DEFERRED (filed as separate TODOs)

- **TODO #18:** 7 truly-breaking `api/schemas.py` validator gaps
  (37 xfails) — 4 enums, HTTP code range, non-empty list, paired flag.
  All require frontend audit before flipping (clients today may send
  capitalization variants or legacy values).
- **TODO #19:** `_keep_higher_p` `KeyError` on missing `p` key
  (1 xfail) — single defensive fix; deferred because no live rule
  violates the constraint today.

## Patterns established

- **`xfail(strict=True)` as embedded to-do list.** Each gap becomes a
  test with `reason=` pointing at the source line + recommended fix.
  When the fix lands, strict mode trips and the test self-promotes to
  a regression guard. Same pattern then used in TODOs #18 and #19.
- **Audit before deferring.** "Be harsh" surfaced the gaps; auditing
  caller risk separated the safe-to-fix-now subset (parameter_system
  fully fixed; api/schemas 8 of 15 fixed) from the "needs client
  coordination" subset (the rest of api/schemas, kept as TODO #18).
- **Class-name `Gap` suffix as grep target.** Test classes that
  document a gap (`TestApiResponseStatusEnumGap`,
  `TestKeepHigherPMissingProbabilityKey`) are findable via `grep -r
  "Gap" tests/`. When TODO #18 / #19 ship, that grep should return
  empty.

## Coverage outcome

The roster moved coverage well above the 52% ratchet baseline
established at the original ci-and-tests merge. Each PR included a
COVERAGE_MIN bump if the new tests crossed a threshold (handled
per-PR in the diff, not consolidated).
