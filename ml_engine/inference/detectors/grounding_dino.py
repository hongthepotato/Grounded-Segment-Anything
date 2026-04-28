"""
GroundingDINO detector implementation.

Text-prompted object detection using Grounding DINO with token-level
class assignment. Uses direct token-to-class
score aggregation.

pred_logits (nq, max_text_len) is converted to per-class scores by
averaging over each class's token positions.
"""

import logging
from typing import Any, Dict, List

import cv2
import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.util.inference import load_model, preprocess_caption
from PIL import Image
from torchvision.ops import box_convert, nms

from ml_engine.inference.detectors.base import DetectionResult

logger = logging.getLogger(__name__)


def build_positive_map(
    tokenizer,
    caption: str,
    num_classes: int,
) -> Dict[int, List[int]]:
    """Map each class index to its token positions in the tokenized caption.

    The caption is formatted as ``"class0. class1. class2."`` with ``.``
    as the delimiter.  The BERT tokenizer produces special tokens
    ([CLS], [SEP]) and punctuation (``.``, ``?``) that act as boundaries.
    Tokens between consecutive boundaries belong to one class.

    Equivalent to MMDetection's ``create_positive_map_label_to_token``.

    Returns:
        ``{class_idx: [token_pos, ...], ...}``
    """
    special_ids = set(tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"]))
    input_ids = tokenizer(caption)["input_ids"]

    positive_map: Dict[int, List[int]] = {}
    class_idx = 0
    prev = 0

    for pos, tid in enumerate(input_ids):
        if tid not in special_ids:
            continue
        span = list(range(prev + 1, pos))
        if span and class_idx < num_classes:
            positive_map[class_idx] = span
            class_idx += 1
        prev = pos

    return positive_map


def logits_to_class_scores(
    logits: torch.Tensor,
    positive_map: Dict[int, List[int]],
    num_classes: int,
    text_threshold: float = 0.0,
) -> torch.Tensor:
    """Convert per-token logits to per-class scores via mean aggregation.

    Equivalent to MMDetection's ``convert_grounding_to_cls_scores``.

    Args:
        logits: Sigmoided token-level logits, shape ``(nq, max_text_len)``.
        positive_map: ``{class_idx: [token_positions]}``.
        num_classes: Total number of classes.
        text_threshold: Tokens whose sigmoided score is <= this value are
            zeroed before the per-class mean is computed. 0.0 (default)
            keeps all tokens (preserves original behaviour).

    Returns:
        Per-class scores, shape ``(nq, num_classes)``.
    """
    scores = torch.zeros(logits.shape[0], num_classes, device=logits.device)
    for cls_idx, tok_indices in positive_map.items():
        tok_logits = logits[:, tok_indices]
        if text_threshold > 0.0:
            tok_logits = tok_logits * (tok_logits > text_threshold).float()
        scores[:, cls_idx] = tok_logits.mean(dim=-1)
    return scores


_IMAGE_TRANSFORM = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR ndarray → normalised tensor ready for GroundingDINO."""
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    tensor, _ = _IMAGE_TRANSFORM(pil, None)
    return tensor


class GroundingDINODetector:
    """Object detector using Grounding DINO with token-level class mapping.

    Example::

        detector = GroundingDINODetector(
            config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            checkpoint_path="data/models/pretrained/groundingdino_swint_ogc.pth",
            device="cuda"
        )
        result = detector.detect(image, ["cat", "dog"])
    """

    def __init__(
        self,
        config_path: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path: str = "data/models/pretrained/groundingdino_swint_ogc.pth",
        device: str = "cuda",
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        # Lazy-loaded GroundingDINO instance. Annotated Any (not Optional[Module])
        # because (a) it's set in _load_model and every consumer is gated by
        # _load_model(), so the None state is a transient init detail; (b) the
        # PEFT-wrapped model exposes attributes (.tokenizer, .to, .eval) via
        # nn.Module.__getattr__ which mypy can't see through. Same boundary
        # pattern as model_trainers/base.py self.model: Any.
        self._model: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading Grounding DINO model...")
        self._model = load_model(self.config_path, self.checkpoint_path, device=str(self.device))
        self._model.to(self.device)
        self._model.eval()
        logger.info("Grounding DINO loaded successfully")

    def detect(
        self,
        image: np.ndarray,
        prompts: List[str],
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
        nms_threshold: float = 0.7,
    ) -> DetectionResult:
        """Detect objects in a single BGR image.

        Args:
            image: BGR image (OpenCV format).
            prompts: Class names to detect.
            box_threshold: Minimum per-class score to keep a detection.
            text_threshold: Token-level confidence gate. Tokens whose
                sigmoided score is <= this value are zeroed before the
                per-class mean is computed. Queries where all tokens for
                every class are zeroed will not pass box_threshold.
            nms_threshold: IoU threshold for NMS.

        Returns:
            DetectionResult with boxes, confidences, and class_ids.
        """
        self._load_model()

        caption = preprocess_caption(".".join(prompts))
        positive_map = build_positive_map(self._model.tokenizer, caption, len(prompts))
        if not positive_map:
            logger.warning("Could not build token map for prompts %s", prompts)
            return DetectionResult(
                boxes_xyxy=np.empty((0, 4)),
                confidences=np.empty(0),
                class_ids=np.empty(0, dtype=int),
            )

        img_tensor = preprocess_image(image).to(self.device)
        h, w = image.shape[:2]

        with torch.no_grad():
            outputs = self._model(img_tensor[None], captions=[caption])

        pred_logits = outputs["pred_logits"].sigmoid()[0]  # (nq, max_text_len)
        pred_boxes = outputs["pred_boxes"][0]  # (nq, 4) cxcywh 0-1

        # cls_scores is of shape (nq, num_classes)
        cls_scores = logits_to_class_scores(  # (nq, num_classes)
            pred_logits, positive_map, len(prompts), text_threshold
        )

        # pick the class with the highest score for each query
        # if 'dim' is specified, max will return (values, indices)
        max_scores, class_ids = cls_scores.max(dim=-1)  # (nq,), (nq,)
        keep = max_scores > box_threshold
        if not keep.any():
            return DetectionResult(
                boxes_xyxy=np.empty((0, 4)),
                confidences=np.empty(0),
                class_ids=np.empty(0, dtype=int),
            )

        scores_kept = max_scores[keep]  # filter by masking
        classes_kept = class_ids[keep]
        boxes_kept = pred_boxes[keep]

        boxes_pixel = boxes_kept * torch.tensor([w, h, w, h], device=boxes_kept.device)
        boxes_xyxy = box_convert(boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy")

        # remove potential boxes on the same object
        nms_idx = nms(boxes_xyxy, scores_kept, nms_threshold)

        return DetectionResult(
            boxes_xyxy=boxes_xyxy[nms_idx].cpu().numpy(),
            confidences=scores_kept[nms_idx].cpu().numpy(),
            class_ids=classes_kept[nms_idx].cpu().numpy().astype(int),
        )
