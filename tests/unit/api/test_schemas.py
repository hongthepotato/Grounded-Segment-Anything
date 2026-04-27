"""
Unit tests for api/schemas.py.

These tests are deliberately HARSH: they don't just hit lines for coverage,
they probe the gap between what the schemas' docstrings promise and what
the validators actually enforce. Many schemas declare semantic constraints
in docstrings (`status: 'succeed' or 'failed'`, `output_mode: 'boxes' or
'masks' or 'both'`, COCO bbox is `[x, y, w, h]`) but the field type is
just `str` / `List[float]` with no enum / no length check / no range
check. Wire data that violates the docstring is silently accepted.

Such gaps are filed as `xfail(strict=True)` tests below — they document
the exact contract the schema SHOULD enforce, and when someone fixes
the validator, the strict-xfail will alert (because the test starts
passing). Tests that the schema DOES enforce today are plain passes
that protect against regressions.

Reference pattern: tests/unit/augmentation/test_augmentation_factory.py
(class-per-area, parametrize variants, descriptive ids).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from api.schemas import (
    ApiResponse,
    AutoLabelRequest,
    COCOAnnotationSchema,
    COCOCategorySchema,
    COCOImageSchema,
    DistillationRequest,
    JobCreate,
    JobListResponse,
    JobProgressSchema,
    JobResponse,
    QueueStatusResponse,
    VisualizationInfo,
    VisualizationListResponse,
    WebSocketEvent,
    WorkerResponse,
    error_response,
    success_response,
)

# ---------------------------------------------------------------------------
# Section 1: ApiResponse — wrapper used by every endpoint
# ---------------------------------------------------------------------------


class TestApiResponseRequiredFields:
    """code + status are required; data + error are optional."""

    def test_minimal_succeed(self):
        r = ApiResponse(code=200, status="succeed")
        assert r.data is None
        assert r.error is None

    def test_minimal_failed(self):
        r = ApiResponse(code=400, status="failed", error="bad request")
        assert r.error == "bad request"

    def test_missing_code_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            ApiResponse(status="succeed")
        assert "code" in str(exc_info.value)

    def test_missing_status_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            ApiResponse(code=200)
        assert "status" in str(exc_info.value)

    def test_data_can_be_arbitrary(self):
        """data: Optional[T] is generic — accepts dicts, lists, primitives, None."""
        for payload in [{"x": 1}, [1, 2], "text", 42, None]:
            r = ApiResponse(code=200, status="succeed", data=payload)
            assert r.data == payload


class TestApiResponseStatusEnumGap:
    """
    Docstring at api/schemas.py:45 says status is 'succeed' or 'failed'.
    Field is plain `str` with no validator. Anything is accepted today.
    Filed as xfail so when someone tightens it to a Literal/enum, the
    test trips and the constraint becomes load-bearing.
    """

    @pytest.mark.parametrize(
        "bad_status",
        ["", "OK", "ok", "Succeed", "true", "yes", "passed", "ERROR", " succeed "],
        ids=lambda v: f"status={v!r}",
    )
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:45 — status field is `str` with no enum constraint; "
        "docstring promises 'succeed'|'failed'. Tighten to Literal['succeed', 'failed'].",
    )
    def test_invalid_status_should_reject(self, bad_status):
        with pytest.raises(ValidationError):
            ApiResponse(code=200, status=bad_status)

    def test_valid_status_succeed(self):
        ApiResponse(code=200, status="succeed")  # should not raise

    def test_valid_status_failed(self):
        ApiResponse(code=400, status="failed", error="x")


class TestApiResponseCodeRangeGap:
    """
    Docstring at api/schemas.py:44 says code is 'HTTP status code'. Valid
    HTTP codes are 100-599. Field is plain `int` with no `ge`/`le`. Out-
    of-range integers (negatives, zero, four-digit) silently accepted.
    """

    @pytest.mark.parametrize(
        "bad_code",
        [-1, 0, 99, 600, 999, 10000],
        ids=lambda v: f"code={v}",
    )
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:44 — code field is `int` with no `ge=100, le=599`. "
        "Out-of-range HTTP codes silently accepted.",
    )
    def test_out_of_range_code_should_reject(self, bad_code):
        with pytest.raises(ValidationError):
            ApiResponse(code=bad_code, status="succeed")

    @pytest.mark.parametrize("ok_code", [100, 200, 201, 400, 422, 500, 599])
    def test_in_range_code_accepted(self, ok_code):
        ApiResponse(code=ok_code, status="succeed")


class TestSuccessAndErrorHelpers:
    """The plain-dict helpers — these are the actual code paths used by routes."""

    def test_success_response_default_code(self):
        r = success_response(data={"id": "abc"})
        assert r == {"code": 200, "status": "succeed", "data": {"id": "abc"}}

    def test_success_response_custom_code(self):
        r = success_response(data=None, code=201)
        assert r == {"code": 201, "status": "succeed", "data": None}

    def test_error_response_default_code(self):
        r = error_response("nope")
        assert r == {"code": 400, "status": "failed", "error": "nope"}

    def test_error_response_custom_code(self):
        r = error_response("nope", code=422)
        assert r == {"code": 422, "status": "failed", "error": "nope"}

    @pytest.mark.parametrize("empty_error", ["", " "])
    def test_error_response_accepts_empty_string_today(self, empty_error):
        """
        Documenting current behavior: empty/whitespace error strings are
        not rejected. If product wants to enforce non-empty error
        messages, the helper would need a guard. Keeping this as a plain
        assertion of current behavior; flip to xfail if the product calls
        it a bug.
        """
        r = error_response(empty_error)
        assert r["error"] == empty_error


# ---------------------------------------------------------------------------
# Section 2: JobProgressSchema
# ---------------------------------------------------------------------------


class TestJobProgressSchema:
    def test_all_defaults(self):
        p = JobProgressSchema()
        assert p.current_epoch == 0
        assert p.total_epochs == 0
        assert p.current_step == 0
        assert p.total_steps == 0
        assert p.metrics == {}
        assert p.message == ""
        assert p.overall_progress == 0.0

    def test_metrics_default_factory_is_per_instance(self):
        """default_factory=dict must yield a fresh dict per instance — otherwise
        the classic Python mutable-default footgun bites: mutating one
        instance's metrics would leak into another."""
        a = JobProgressSchema()
        b = JobProgressSchema()
        a.metrics["x"] = 1.0
        assert b.metrics == {}, (
            "JobProgressSchema.metrics is leaking between instances — the "
            "default_factory pattern at api/schemas.py:96 is broken."
        )


class TestJobProgressOverallProgressRangeGap:
    """
    Docstring at api/schemas.py:98 says overall_progress is "0.0 to 1.0".
    Field has no `ge=0.0, le=1.0`. Accepts anything.
    """

    @pytest.mark.parametrize(
        "bad_progress",
        [-0.01, -1.0, 1.01, 2.0, 100.0, float("inf")],
        ids=lambda v: f"progress={v}",
    )
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:98 — overall_progress is `float` with no range. "
        "Docstring promises 0.0-1.0. Tighten to `Field(ge=0.0, le=1.0)`.",
    )
    def test_out_of_range_progress_should_reject(self, bad_progress):
        with pytest.raises(ValidationError):
            JobProgressSchema(overall_progress=bad_progress)

    @pytest.mark.parametrize("ok_progress", [0.0, 0.5, 1.0])
    def test_in_range_progress_accepted(self, ok_progress):
        JobProgressSchema(overall_progress=ok_progress)


class TestJobProgressEpochInvariantGap:
    """
    Conceptual invariant: current_epoch <= total_epochs. Not enforced
    (no model_validator). A worker that sends current_epoch=99 with
    total_epochs=10 is accepted as valid progress. This is a clear
    sign the worker is misreporting state.
    """

    @pytest.mark.parametrize(
        "current,total",
        [(5, 3), (100, 1), (1, 0)],
        ids=lambda v: f"v={v}",
    )
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:89-98 — no model_validator enforcing "
        "current_epoch <= total_epochs (or current_step <= total_steps).",
    )
    def test_current_exceeds_total_should_reject(self, current, total):
        with pytest.raises(ValidationError):
            JobProgressSchema(current_epoch=current, total_epochs=total)


# ---------------------------------------------------------------------------
# Section 3: JobCreate — the actual POST body for /api/jobs
# ---------------------------------------------------------------------------


class TestJobCreateRequiredFields:
    def test_minimal_valid(self):
        j = JobCreate(job_type="teacher_training", config={"data_path": "x.json"})
        assert j.priority == 0
        assert j.tags == []
        assert j.output_dir is None

    def test_missing_job_type_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            JobCreate(config={"data_path": "x.json"})
        assert "job_type" in str(exc_info.value)

    def test_missing_config_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            JobCreate(job_type="teacher_training")
        assert "config" in str(exc_info.value)

    def test_tags_default_factory_is_per_instance(self):
        """Same per-instance check as JobProgressSchema.metrics — guards
        against `tags=[]` being shared mutable default."""
        a = JobCreate(job_type="teacher_training", config={})
        b = JobCreate(job_type="teacher_training", config={})
        a.tags.append("x")
        assert b.tags == []

    def test_empty_config_dict_accepted_today(self):
        """Documenting current behavior: config={} is accepted. The handler
        will fail later when trying to read data_path. Worth flipping to
        a model_validator if early validation matters."""
        JobCreate(job_type="teacher_training", config={})


class TestJobCreateJobTypeEnumGap:
    """
    Docstring at api/schemas.py:124 says job_type is one of
    (teacher_training, student_distillation). Field is plain `str` —
    typos like "teacher_train" or "TEACHER_TRAINING" silently accepted
    (handler fails later with a confusing dispatch error).
    """

    @pytest.mark.parametrize(
        "bad_type",
        ["", "teacher_train", "TEACHER_TRAINING", "training", "auto_label", " teacher_training"],
    )
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:124 — job_type is `str`; docstring lists allowed "
        "values. Tighten to Literal/enum so bad values reject at the boundary "
        "instead of failing with a cryptic handler-dispatch error.",
    )
    def test_unknown_job_type_should_reject(self, bad_type):
        with pytest.raises(ValidationError):
            JobCreate(job_type=bad_type, config={})


# ---------------------------------------------------------------------------
# Section 4: AutoLabelRequest — has actual `Field(ge=0.0, le=1.0)` validation
# ---------------------------------------------------------------------------


class TestAutoLabelRequestThresholds:
    """The threshold fields ARE validated. Test boundaries + out-of-range."""

    @pytest.mark.parametrize(
        "field_name",
        ["box_threshold", "text_threshold", "nms_threshold"],
    )
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_thresholds_accept_in_range(self, field_name, value):
        kwargs = {"image_paths": ["a.jpg"], "classes": ["cat"], field_name: value}
        AutoLabelRequest(**kwargs)

    @pytest.mark.parametrize(
        "field_name",
        ["box_threshold", "text_threshold", "nms_threshold"],
    )
    @pytest.mark.parametrize("bad_value", [-0.01, -1.0, 1.01, 2.0])
    def test_thresholds_reject_out_of_range(self, field_name, bad_value):
        kwargs = {"image_paths": ["a.jpg"], "classes": ["cat"], field_name: bad_value}
        with pytest.raises(ValidationError):
            AutoLabelRequest(**kwargs)


class TestAutoLabelRequestOutputModeEnumGap:
    """Docstring at api/schemas.py:245 says 'boxes', 'masks', or 'both'."""

    @pytest.mark.parametrize(
        "bad_mode",
        ["", "Boxes", "BOXES", "box", "all", "boxes, masks", "none"],
    )
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:245 — output_mode is `str`; tighten to "
        "Literal['boxes', 'masks', 'both'] so unknown values reject at "
        "the boundary.",
    )
    def test_unknown_output_mode_should_reject(self, bad_mode):
        with pytest.raises(ValidationError):
            AutoLabelRequest(image_paths=["a.jpg"], classes=["cat"], output_mode=bad_mode)

    @pytest.mark.parametrize("ok_mode", ["boxes", "masks", "both"])
    def test_valid_output_modes(self, ok_mode):
        AutoLabelRequest(image_paths=["a.jpg"], classes=["cat"], output_mode=ok_mode)


class TestAutoLabelRequestNonEmptyListGap:
    """Empty image_paths or classes is meaningless — autolabeler has nothing
    to do. Schema accepts both today; handler hits a noop or worse."""

    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:243 — image_paths is `List[str]` with no "
        "min_length; empty list accepted. Tighten to `Field(min_length=1)`.",
    )
    def test_empty_image_paths_should_reject(self):
        with pytest.raises(ValidationError):
            AutoLabelRequest(image_paths=[], classes=["cat"])

    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:244 — classes is `List[str]` with no "
        "min_length; empty list accepted. Tighten to `Field(min_length=1)`.",
    )
    def test_empty_classes_should_reject(self):
        with pytest.raises(ValidationError):
            AutoLabelRequest(image_paths=["a.jpg"], classes=[])


# ---------------------------------------------------------------------------
# Section 5: DistillationRequest
# ---------------------------------------------------------------------------


class TestDistillationRequestRequiredFields:
    def test_minimal_valid(self):
        DistillationRequest(data_path="x.json", image_paths=["a.jpg"])

    def test_missing_data_path_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DistillationRequest(image_paths=["a.jpg"])
        assert "data_path" in str(exc_info.value)

    def test_missing_image_paths_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DistillationRequest(data_path="x.json")
        assert "image_paths" in str(exc_info.value)


class TestDistillationRequestSplitConfigGap:
    """Docstring at api/schemas.py:277 says 'sum=1.0'. No model_validator."""

    @pytest.mark.parametrize(
        "bad_split",
        [
            {"train": 0.5},  # missing val/test, sums to 0.5
            {"train": 0.5, "val": 0.5, "test": 0.5},  # sums to 1.5
            {"train": 99.0},  # nonsense ratio
            {"train": 0.0, "val": 0.0, "test": 0.0},  # sums to 0
            {"train": -0.1, "val": 0.6, "test": 0.5},  # negative
        ],
        ids=lambda v: f"split={v}",
    )
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:276 — split_config has no validator enforcing "
        "the docstring promise 'sum=1.0'. Add a model_validator that checks "
        "sum and rejects negative ratios.",
    )
    def test_invalid_split_should_reject(self, bad_split):
        with pytest.raises(ValidationError):
            DistillationRequest(
                data_path="x.json",
                image_paths=["a.jpg"],
                split_config=bad_split,
            )

    def test_valid_split_accepted(self):
        DistillationRequest(
            data_path="x.json",
            image_paths=["a.jpg"],
            split_config={"train": 0.7, "val": 0.15, "test": 0.15},
        )


class TestDistillationPairedFieldGap:
    """
    Docstring at api/schemas.py:266-271: teacher_dir is "required with
    unlabeled_image_paths" — they're a paired flag. Setting one without
    the other is a misconfiguration; should be caught at schema layer.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:266-271 — teacher_dir + unlabeled_image_paths "
        "form a paired flag per the docstring; passing only one is a misconfig "
        "but no model_validator enforces the pairing.",
    )
    def test_teacher_dir_without_unlabeled_should_reject(self):
        with pytest.raises(ValidationError):
            DistillationRequest(
                data_path="x.json",
                image_paths=["a.jpg"],
                teacher_dir="/some/teacher",
                unlabeled_image_paths=None,
            )

    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:266-271 — same paired-flag invariant, other side.",
    )
    def test_unlabeled_without_teacher_dir_should_reject(self):
        with pytest.raises(ValidationError):
            DistillationRequest(
                data_path="x.json",
                image_paths=["a.jpg"],
                teacher_dir=None,
                unlabeled_image_paths=["b.jpg"],
            )


# ---------------------------------------------------------------------------
# Section 6: COCO* schemas — well-defined external format, gaps are bugs
# ---------------------------------------------------------------------------


class TestCOCOImageSchemaDimensionGap:
    """COCO image dimensions must be positive — width=0 or height=-1 are not
    valid images. Schema is `int` with no `gt=0`."""

    @pytest.mark.parametrize(
        "bad_w,bad_h",
        [(0, 100), (-1, 100), (100, 0), (100, -50), (-1, -1)],
    )
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:292-293 — width/height are `int` with no `gt=0`. "
        "COCO format requires positive dimensions.",
    )
    def test_non_positive_dims_should_reject(self, bad_w, bad_h):
        with pytest.raises(ValidationError):
            COCOImageSchema(id=1, file_name="a.jpg", width=bad_w, height=bad_h)

    def test_valid_image(self):
        img = COCOImageSchema(id=1, file_name="a.jpg", width=640, height=480)
        assert img.width == 640


class TestCOCOAnnotationSchema:
    def test_minimal_valid(self):
        a = COCOAnnotationSchema(id=1, image_id=2, category_id=3)
        assert a.bbox is None
        assert a.iscrowd == 0

    def test_valid_bbox_4_floats(self):
        COCOAnnotationSchema(id=1, image_id=2, category_id=3, bbox=[10.0, 20.0, 50.0, 50.0])


class TestCOCOAnnotationBboxLengthGap:
    """COCO bbox is exactly [x, y, w, h] — 4 elements. Schema is
    `Optional[List[float]]` with no length constraint. A bbox of length 3
    or 5 is structurally invalid but accepted today."""

    @pytest.mark.parametrize(
        "bad_bbox",
        [[], [1.0], [1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
        ids=lambda v: f"len={len(v)}",
    )
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:302 — bbox is `Optional[List[float]]` with no "
        "length constraint. COCO format requires exactly 4 elements [x,y,w,h]. "
        "Tighten to `min_length=4, max_length=4` or use a Tuple[float, float, float, float].",
    )
    def test_wrong_length_bbox_should_reject(self, bad_bbox):
        with pytest.raises(ValidationError):
            COCOAnnotationSchema(id=1, image_id=2, category_id=3, bbox=bad_bbox)


class TestCOCOAnnotationIsCrowdGap:
    """COCO iscrowd is binary (0 = polygon, 1 = RLE). Schema accepts any
    int. -1 or 99 are nonsense but pass."""

    @pytest.mark.parametrize("bad_iscrowd", [-1, 2, 5, 100])
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:306 — iscrowd is `int` with no enum. COCO "
        "spec is binary 0/1. Tighten to Literal[0, 1].",
    )
    def test_non_binary_iscrowd_should_reject(self, bad_iscrowd):
        with pytest.raises(ValidationError):
            COCOAnnotationSchema(id=1, image_id=2, category_id=3, iscrowd=bad_iscrowd)


class TestCOCOAnnotationScoreRangeGap:
    """score is detection confidence — should be 0..1. Schema is plain
    `Optional[float]`."""

    @pytest.mark.parametrize("bad_score", [-0.1, 1.5, 100.0])
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:305 — score is `Optional[float]` with no "
        "range. Detection confidence must be 0.0-1.0.",
    )
    def test_out_of_range_score_should_reject(self, bad_score):
        with pytest.raises(ValidationError):
            COCOAnnotationSchema(id=1, image_id=2, category_id=3, score=bad_score)


class TestCOCOCategorySchema:
    def test_required_fields(self):
        c = COCOCategorySchema(id=1, name="cat")
        assert c.id == 1
        assert c.name == "cat"

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            COCOCategorySchema(id=1)


# ---------------------------------------------------------------------------
# Section 7: Round-trip serialization — model_dump + model_validate must
# preserve all fields. Catches drift between Field defaults and the actual
# JSON shape the frontend sees.
# ---------------------------------------------------------------------------


class TestRoundTripSerialization:
    def test_api_response_roundtrip(self):
        original = ApiResponse(code=200, status="succeed", data={"id": "x"})
        dumped = original.model_dump()
        restored = ApiResponse.model_validate(dumped)
        assert restored == original

    def test_job_create_roundtrip(self):
        original = JobCreate(
            job_type="teacher_training",
            config={"data_path": "x.json", "epochs": 5},
            priority=10,
            output_dir="/out",
            tags=["a", "b"],
        )
        restored = JobCreate.model_validate(original.model_dump())
        assert restored == original

    def test_job_response_roundtrip_with_datetimes(self):
        now = datetime.now(timezone.utc)
        original = JobResponse(
            id="abc",
            type="teacher_training",
            status="running",
            created_at=now,
            started_at=now,
            duration_seconds=3.5,
            accuracy=0.87,
        )
        dumped = original.model_dump()
        restored = JobResponse.model_validate(dumped)
        assert restored == original
        assert restored.created_at == now

    def test_distillation_request_roundtrip_full(self):
        original = DistillationRequest(
            data_path="x.json",
            image_paths=["a.jpg"],
            teacher_dir="/teacher",
            unlabeled_image_paths=["u.jpg"],
            student_model="yolov8n-seg",
            student_size="n",
            split_config={"train": 0.7, "val": 0.15, "test": 0.15},
            training={"epochs": 10},
            priority=5,
            tags=["x"],
        )
        restored = DistillationRequest.model_validate(original.model_dump())
        assert restored == original

    def test_websocket_event_roundtrip(self):
        original = WebSocketEvent(
            type="progress",
            job_id="abc",
            timestamp="2026-04-27T12:00:00Z",
            progress=JobProgressSchema(current_epoch=5, total_epochs=10),
        )
        restored = WebSocketEvent.model_validate(original.model_dump())
        assert restored == original
        assert restored.progress is not None
        assert restored.progress.current_epoch == 5


# ---------------------------------------------------------------------------
# Section 8: Type coercion — pydantic v2 quirks worth pinning
# ---------------------------------------------------------------------------


class TestTypeCoercion:
    def test_int_field_accepts_string_int(self):
        """Pydantic v2 default mode coerces string→int. Documenting so a
        future strict-mode flip doesn't silently break callers."""
        j = JobCreate(job_type="x", config={}, priority="5")  # type: ignore[arg-type]
        assert j.priority == 5
        assert isinstance(j.priority, int)

    def test_int_field_rejects_non_numeric_string(self):
        with pytest.raises(ValidationError):
            JobCreate(job_type="x", config={}, priority="five")  # type: ignore[arg-type]

    def test_float_field_accepts_int(self):
        """0..1 thresholds at int 0 and 1 should work — documents that
        passing the boundary as int (not float) is fine."""
        AutoLabelRequest(image_paths=["a"], classes=["c"], box_threshold=1)
        AutoLabelRequest(image_paths=["a"], classes=["c"], box_threshold=0)

    def test_bool_to_int_coercion(self):
        """Pydantic v2 coerces bool → int (True→1, False→0). Documenting:
        a caller passing `iscrowd=True` would silently store iscrowd=1.
        If iscrowd ever gets tightened to Literal[0, 1] (per the gap test
        above), this coercion may need explicit handling."""
        a = COCOAnnotationSchema(id=1, image_id=1, category_id=1, iscrowd=True)  # type: ignore[arg-type]
        assert a.iscrowd == 1

    def test_none_for_optional_field(self):
        j = JobCreate(job_type="x", config={}, output_dir=None)
        assert j.output_dir is None

    def test_extra_field_silently_dropped(self):
        """Pydantic v2 default behavior: extra fields ignored, not rejected.
        Documenting because some routes may want strict mode (rejection)
        to catch typo'd field names (e.g., `tag` vs `tags`)."""
        j = JobCreate(
            job_type="x",
            config={},
            tag=["typo"],  # type: ignore[call-arg]  -- typo of `tags`
        )
        assert j.tags == []  # silently lost the typo'd value


# ---------------------------------------------------------------------------
# Section 9: List/Dict response containers
# ---------------------------------------------------------------------------


class TestJobListResponse:
    def test_minimal_valid_empty_list(self):
        r = JobListResponse(jobs=[], total=0, limit=10, offset=0)
        assert r.jobs == []

    def test_total_and_limit_can_be_inconsistent(self):
        """No model_validator enforces e.g. `len(jobs) <= limit` or
        `total >= len(jobs)`. Documenting current behavior — flip to xfail
        if this becomes a contract."""
        r = JobListResponse(jobs=[], total=100, limit=5, offset=0)
        assert r.total == 100  # accepted despite empty jobs list


class TestQueueStatusResponse:
    def test_required_fields(self):
        r = QueueStatusResponse(queue_length=0, workers=[], job_counts={})
        assert r.queue_length == 0


class TestVisualizationListResponse:
    def test_total_field_independent_of_images_length(self):
        """Same caveat as JobListResponse — no consistency check between
        total and len(images)."""
        r = VisualizationListResponse(job_id="abc", total=99, images=[])
        assert r.total == 99


class TestVisualizationInfo:
    def test_required_fields(self):
        v = VisualizationInfo(filename="a.png", original="a.jpg", annotation_count=3)
        assert v.annotation_count == 3

    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:333 — annotation_count is `int` with no "
        "`ge=0`. A negative count is nonsense but accepted.",
    )
    def test_negative_annotation_count_should_reject(self):
        with pytest.raises(ValidationError):
            VisualizationInfo(filename="a.png", original="a.jpg", annotation_count=-1)


class TestWorkerResponseStatusEnumGap:
    """Docstring at api/schemas.py:188 says 'idle, busy, offline'."""

    @pytest.mark.parametrize("bad_status", ["", "Idle", "BUSY", "running", "down"])
    @pytest.mark.xfail(
        strict=True,
        reason="api/schemas.py:188 — status is `str`; tighten to "
        "Literal['idle', 'busy', 'offline'] per the docstring.",
    )
    def test_unknown_worker_status_should_reject(self, bad_status):
        with pytest.raises(ValidationError):
            WorkerResponse(id="w1", gpu_id=0, hostname="h", status=bad_status)

    @pytest.mark.parametrize("ok_status", ["idle", "busy", "offline"])
    def test_valid_worker_status(self, ok_status):
        WorkerResponse(id="w1", gpu_id=0, hostname="h", status=ok_status)
