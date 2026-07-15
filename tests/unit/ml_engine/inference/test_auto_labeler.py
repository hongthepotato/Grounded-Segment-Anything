"""Unit tests for ml_engine.inference.auto_labeler.AutoLabeler.

These are bug-hunting tests, not coverage padding: the detector/segmenter are
injected fakes and cv2 / transform_image_path are patched so the ORCHESTRATION
contract is what's under test. Several assertions are written to fail if the
implementation is wrong, e.g.:
  - file_name must be the ORIGINAL full path in BOTH the loaded and the
    load-failure paths (regression trap: an earlier version basename'd the
    transformed path on failure, so a failed image got a different file_name
    than a loaded one).
  - segment() must receive the RGB image and the EXACT detection boxes (the
    detect->segment hand-off), not the BGR image or a copy.
  - xyxy boxes must convert to COCO xywh.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from ml_engine.inference import auto_labeler as al
from ml_engine.inference.auto_labeler import AutoLabeler
from ml_engine.inference.config import (
    OUTPUT_BOTH,
    OUTPUT_BOXES_ONLY,
    OUTPUT_MASKS_ONLY,
    AutoLabelerConfig,
    DetectionThresholds,
)

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _Detection:
    def __init__(self, boxes=None, class_ids=None, confidences=None):
        boxes = boxes or []
        self.boxes_xyxy = np.array(boxes, dtype=float).reshape(-1, 4) if boxes else np.empty((0, 4))
        self.class_ids = np.array(class_ids or [], dtype=int)
        self.confidences = np.array(confidences or [], dtype=float)


class _FakeDetector:
    def __init__(self, detection: _Detection):
        self.detection = detection
        self.calls = []

    def detect(self, image, prompts, box_threshold, text_threshold, nms_threshold):
        self.calls.append(
            {"prompts": prompts, "box": box_threshold, "text": text_threshold, "nms": nms_threshold}
        )
        return self.detection


class _FakeSegmenter:
    def __init__(self, masks=None):
        self.masks = masks if masks is not None else []
        self.calls = []

    def segment(self, image_rgb, boxes):
        self.calls.append((image_rgb, boxes))
        return self.masks


def _make_labeler(output_mode=OUTPUT_BOTH, detector=None, segmenter=None, thresholds=None):
    cfg = AutoLabelerConfig(output_mode=output_mode, device="cpu")
    if thresholds is not None:
        cfg.thresholds = thresholds
    labeler = AutoLabeler(cfg)
    labeler._detector = detector  # inject; _get_detector returns it (not None)
    labeler._segmenter = segmenter
    return labeler


@pytest.fixture
def patched_io():
    """transform_image_path -> identity; cv2.imread -> valid 100x200 BGR image by
    default; cv2.cvtColor -> a distinct RGB array. Tests may override imread."""
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)  # shape[:2] -> h=100, w=200
    rgb = np.ones((100, 200, 3), dtype=np.uint8)  # distinct object, so hand-off is checkable
    with (
        patch.object(al, "transform_image_path", side_effect=lambda p: p),
        patch.object(al.cv2, "imread", return_value=bgr) as imread,
        patch.object(al.cv2, "cvtColor", return_value=rgb) as cvt,
    ):
        yield imread, cvt


# --------------------------------------------------------------------------- #
# file_name consistency — the regression trap
# --------------------------------------------------------------------------- #


class TestFileNameContract:
    def test_loaded_image_keeps_full_original_path(self, patched_io):
        det = _FakeDetector(_Detection(boxes=[[1, 1, 2, 2]], class_ids=[0], confidences=[0.9]))
        labeler = _make_labeler(OUTPUT_BOXES_ONLY, detector=det)
        r = labeler.label_images(["data/imgs/cat.jpg"], ["x"])[0]
        # full path, NOT basename "cat.jpg"
        assert r["image_info"]["file_name"] == "data/imgs/cat.jpg"

    def test_loaded_and_failed_images_report_the_same_full_path(self, patched_io):
        # image 0 loads, image 1 fails to decode (imread -> None).
        imread, _ = patched_io
        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        imread.side_effect = [bgr, None]
        det = _FakeDetector(_Detection(boxes=[[1, 1, 2, 2]], class_ids=[0], confidences=[0.9]))
        labeler = _make_labeler(OUTPUT_BOTH, detector=det)

        results = labeler.label_images(["data/a.jpg", "data/b.jpg"], ["x"])

        # BUG TRAP: pre-fix, the failed image's file_name was basename("data/b.jpg")
        # == "b.jpg", differing from the loaded image's full path. Both must be full.
        assert results[0]["image_info"]["file_name"] == "data/a.jpg"
        assert results[1]["image_info"]["file_name"] == "data/b.jpg"


# --------------------------------------------------------------------------- #
# Orchestration correctness
# --------------------------------------------------------------------------- #


class TestOrchestration:
    def test_both_mode_full_result(self, patched_io):
        det = _FakeDetector(_Detection(boxes=[[10, 20, 30, 50]], class_ids=[0], confidences=[0.9]))
        seg = _FakeSegmenter(masks=["MASK"])
        labeler = _make_labeler(OUTPUT_BOTH, detector=det, segmenter=seg)
        r = labeler.label_images(["d/img.jpg"], ["cat"])[0]
        assert r["class_ids"] == [0]
        assert r["scores"] == [0.9]
        assert r["image_info"] == {"file_name": "d/img.jpg", "width": 200, "height": 100}
        assert r["boxes"] == [[10.0, 20.0, 20.0, 30.0]]  # xyxy -> xywh
        assert r["masks"] == ["MASK"]

    def test_box_to_coco_conversion(self, patched_io):
        det = _FakeDetector(_Detection(boxes=[[15, 25, 40, 100]], class_ids=[1], confidences=[0.5]))
        labeler = _make_labeler(OUTPUT_BOXES_ONLY, detector=det)
        r = labeler.label_images(["d/a.png"], ["x"])[0]
        assert r["boxes"] == [[15.0, 25.0, 25.0, 75.0]]  # w=40-15=25, h=100-25=75

    def test_multiple_boxes_all_converted(self, patched_io):
        det = _FakeDetector(
            _Detection(boxes=[[0, 0, 2, 2], [5, 5, 6, 9]], class_ids=[0, 1], confidences=[0.8, 0.7])
        )
        labeler = _make_labeler(OUTPUT_BOXES_ONLY, detector=det)
        r = labeler.label_images(["d/a.png"], ["x", "y"])[0]
        assert r["boxes"] == [[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 1.0, 4.0]]

    def test_multiple_images_produce_ordered_results(self, patched_io):
        det = _FakeDetector(_Detection(boxes=[[1, 1, 2, 2]], class_ids=[0], confidences=[0.9]))
        labeler = _make_labeler(OUTPUT_BOXES_ONLY, detector=det)
        results = labeler.label_images(["d/a.png", "d/b.png", "d/c.png"], ["x"])
        assert [r["image_info"]["file_name"] for r in results] == ["d/a.png", "d/b.png", "d/c.png"]


class TestDetectSegmentHandoff:
    def test_detect_receives_config_thresholds_and_prompts(self, patched_io):
        det = _FakeDetector(_Detection(boxes=[[1, 1, 2, 2]], class_ids=[0], confidences=[0.9]))
        labeler = _make_labeler(
            OUTPUT_BOXES_ONLY, detector=det, thresholds=DetectionThresholds(box=0.3, text=0.4, nms=0.6)
        )
        labeler.label_images(["d/a.png"], ["cat", "dog"])
        assert det.calls[0] == {"prompts": ["cat", "dog"], "box": 0.3, "text": 0.4, "nms": 0.6}

    def test_segmenter_receives_rgb_and_exact_detection_boxes(self, patched_io):
        _, cvt = patched_io
        detection = _Detection(boxes=[[1, 1, 2, 2]], class_ids=[0], confidences=[0.9])
        seg = _FakeSegmenter(masks=["MASK"])
        labeler = _make_labeler(OUTPUT_BOTH, detector=_FakeDetector(detection), segmenter=seg)
        labeler.label_images(["d/a.png"], ["x"])
        assert len(seg.calls) == 1
        image_rgb, boxes = seg.calls[0]
        assert image_rgb is cvt.return_value  # RGB (cvtColor output), not the BGR image
        assert boxes is detection.boxes_xyxy  # exact hand-off, not a copy/transform

    def test_no_detections_skips_segment_and_yields_empty(self, patched_io):
        seg = _FakeSegmenter(masks=["SHOULD_NOT_APPEAR"])
        labeler = _make_labeler(OUTPUT_BOTH, detector=_FakeDetector(_Detection()), segmenter=seg)
        r = labeler.label_images(["d/a.png"], ["x"])[0]
        assert r["class_ids"] == [] and r["scores"] == []
        assert r["boxes"] == [] and r["masks"] == []
        assert seg.calls == []  # len(boxes)==0 -> segment must be skipped


class TestOutputModeGating:
    def test_boxes_only_has_boxes_no_masks_and_never_segments(self, patched_io):
        det = _FakeDetector(_Detection(boxes=[[1, 1, 2, 2]], class_ids=[0], confidences=[0.9]))
        seg = _FakeSegmenter(masks=["MASK"])
        labeler = _make_labeler(OUTPUT_BOXES_ONLY, detector=det, segmenter=seg)
        r = labeler.label_images(["d/a.png"], ["x"])[0]
        assert "boxes" in r and "masks" not in r
        assert seg.calls == []

    def test_masks_only_has_masks_no_boxes(self, patched_io):
        det = _FakeDetector(_Detection(boxes=[[1, 1, 2, 2]], class_ids=[0], confidences=[0.9]))
        seg = _FakeSegmenter(masks=["MASK"])
        labeler = _make_labeler(OUTPUT_MASKS_ONLY, detector=det, segmenter=seg)
        r = labeler.label_images(["d/a.png"], ["x"])[0]
        assert "masks" in r and "boxes" not in r
        assert r["masks"] == ["MASK"]


class TestGuardsAndCallbacks:
    def test_empty_image_paths_raises(self):
        labeler = _make_labeler(detector=_FakeDetector(_Detection()))
        with pytest.raises(ValueError, match="No image"):
            labeler.label_images([], ["cat"])

    def test_unreadable_image_warns_and_skips_detection(self, patched_io, caplog):
        imread, _ = patched_io
        imread.return_value = None
        det = _FakeDetector(_Detection(boxes=[[1, 1, 2, 2]], class_ids=[0], confidences=[0.9]))
        labeler = _make_labeler(OUTPUT_BOTH, detector=det)
        with caplog.at_level("WARNING"):
            r = labeler.label_images(["d/missing.jpg"], ["x"])[0]
        assert any("Could not load image" in rec.message for rec in caplog.records)
        assert r["class_ids"] == [] and r["boxes"] == [] and r["masks"] == []
        assert det.calls == []  # detection must be skipped for an unreadable image

    def test_progress_callback_invoked_per_image(self, patched_io):
        det = _FakeDetector(_Detection(boxes=[[1, 1, 2, 2]], class_ids=[0], confidences=[0.9]))
        labeler = _make_labeler(OUTPUT_BOXES_ONLY, detector=det)
        seen = []
        labeler.label_images(
            ["d/a.png", "d/b.png"], ["x"], progress_callback=lambda c, t, m: seen.append((c, t, m))
        )
        assert [(c, t) for c, t, _ in seen] == [(1, 2), (2, 2)]
        assert "a.png" in seen[0][2] and "b.png" in seen[1][2]

    @pytest.mark.parametrize(
        "mode,has_boxes,has_masks",
        [(OUTPUT_BOTH, True, True), (OUTPUT_BOXES_ONLY, True, False), (OUTPUT_MASKS_ONLY, False, True)],
    )
    def test_empty_result_shape_and_full_path(self, mode, has_boxes, has_masks):
        labeler = _make_labeler(mode, detector=_FakeDetector(_Detection()))
        r = labeler._empty_result("some/dir/pic.jpg")
        assert r["class_ids"] == [] and r["scores"] == []
        # full path preserved (not basename) so it matches the success path
        assert r["image_info"] == {"file_name": "some/dir/pic.jpg", "width": 0, "height": 0}
        assert ("boxes" in r) is has_boxes
        assert ("masks" in r) is has_masks
