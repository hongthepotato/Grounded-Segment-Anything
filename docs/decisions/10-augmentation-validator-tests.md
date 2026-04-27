# 10. Error-path coverage for `augmentation_factory._validate_bboxes`

**Completed:** 2026-04-24 in the same PR as the original Completed-section addition.

**What shipped:** `tests/unit/augmentation/test_augmentation_factory.py` — 44
parametrized tests across 7 test classes, one per error class:
`TestValidInputs` (5), `TestInvalidContainer` (5), `TestInvalidBboxElement`
(8), `TestInvalidCoordinateTypes` (8), `TestInvalidDimensions` (6),
`TestOutOfBounds` (8), `TestErrorIndexing` (1). Every branch of the COCO
validator exits under test: type mismatches raise `TypeError`, value/bounds
mismatches raise `ValueError`, and the reported bbox index is 1-indexed.

**Design notes worth remembering:** validator is called with `self=None`
because its body uses no instance state — `self` is only there because it's
an instance method. Avoids constructing a full augmentation pipeline for
every test. Fast (~8s cold for all 44 tests), isolated from albumentations
state.
