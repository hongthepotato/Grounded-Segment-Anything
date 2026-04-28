# 17. Restore `text_threshold` token-level filtering in `GroundingDINODetector`

**Status:** Complete 2026-04-28.

## What shipped

`text_threshold` is now wired through the inference path so the knob actually
does something. Previously the parameter was accepted by `detect()` and
immediately discarded (`_ = text_threshold`), a silent regression from the
`#6` cleanup.

**Three files changed:**

- `ml_engine/inference/detectors/grounding_dino.py` — `logits_to_class_scores`
  gained a `text_threshold: float = 0.0` parameter. When `text_threshold > 0`,
  each token logit is multiplied by `(logit > threshold).float()`, zeroing
  sub-threshold tokens before the per-class mean is computed. `detect()` now
  passes `text_threshold` through instead of silently dropping it.

- `tests/unit/ml_engine/inference/__init__.py` — new empty package marker.

- `tests/unit/ml_engine/inference/test_grounding_dino_detector.py` — 32
  adversarial tests across two suites (detailed below).

## Design decisions

### Strict `>` comparison (not `>=`)

Tokens whose sigmoid score equals exactly `text_threshold` are zeroed.
This matches the conventional interpretation — "above threshold" means
strictly above — and is verifiable with a single boundary test.

### Mean denominator includes zeroed tokens (dilution)

After masking, the mean is taken over the full set of token indices for
that class, not just the surviving ones. This intentionally dilutes the
class score when some tokens are zeroed: a class supported by one strong
token and two sub-threshold tokens receives a lower score than one where
all three tokens exceed the threshold.

The alternative (mean over surviving tokens only) would amplify a single
surviving token and can produce counter-intuitive behaviour — a class
supported by two mediocre tokens both just above threshold could score
higher than a class with one excellent token and one sub-threshold token.
Dilution is the conservative choice: it decreases confidence as evidence
weakens rather than holding it constant.

### Zero threshold is identity

When `text_threshold == 0.0` the masking branch is skipped entirely
(`if text_threshold > 0.0`), producing bit-identical output to the
pre-fix code path. This is verified explicitly in the test suite.

### Training path is unaffected

`logits_to_class_scores` is only called from `GroundingDINODetector.detect()`.
The training path (`GroundedSAM.detect()` → raw `GroundingDINO` forward →
DETR-style loss) never calls this function. Changing its behaviour has zero
impact on loss computation or gradient flow.

## Choosing `text_threshold` at runtime

The default is `0.0` (no filtering). A practical starting point is to run
inference on a held-out validation split with `text_threshold=0.0`, collect
the distribution of per-token sigmoid scores for TP vs FP detections, and
pick a threshold that cuts false positives without material recall loss.
Values in the range `0.1–0.25` are a reasonable first sweep.

## Test coverage (32 adversarial tests)

`TestLogitsToClassScoresAdversarial` (18 tests):
- threshold is not silently dropped (count changes when applied)
- strict boundary: score == threshold is zeroed, score > threshold survives
- input tensor not mutated
- class-winner flip: lower-scoring class wins after high scorer is zeroed
- score magnitude preserved for tokens above threshold
- multi-token dilution: exact arithmetic verified with known logits
- zero threshold produces bit-identical output to default call
- threshold=1.0 zeros all tokens (sigmoid never reaches 1.0)
- device stays on CPU throughout
- output shape matches (nq, num_classes)
- unsorted `tok_indices` handled correctly
- per-query independence (different rows use their own logits)
- missing class stays zero regardless of threshold

`TestDetectTextThresholdAdversarial` (14 tests):
- old bug reproduced: with original code, count does not change
- boundary exactness at `detect()` integration level
- threshold=1.0 drops all detections
- threshold=0.0 is a no-op
- confidence magnitudes correct after filtering
- class ids correct when one class is zeroed
- NMS still fires with threshold active
- NMS sees post-mask scores (not original logits)
- multi-token dilution pushes class below `box_threshold`
- dilution + lower `box_threshold` keeps detection
- monotone: count never increases as threshold rises across a sweep
