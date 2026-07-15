"""
SAM-HQ segmenter with LoRA support for fine-tuned inference.

Box-prompted segmentation using SAM-HQ with optional LoRA adapters.
Implements SegmenterProtocol so it can be swapped with MobileSAMSegmenter.
"""

import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deps" / "segment_anything"))

# E402 suppressed: segment_anything lives in deps/, not on sys.path until the
# insert above runs. The path manipulation has to come before this import.
from segment_anything.utils.transforms import ResizeLongestSide  # noqa: E402

logger = logging.getLogger(__name__)

SAM_INPUT_SIZE = 1024
SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


class SAMHQSegmenter:
    """
    Segmenter using fine-tuned SAM-HQ with optional LoRA adapters.

    Implements the same interface as MobileSAMSegmenter (SegmenterProtocol).

    Example:
        segmenter = SAMHQSegmenter(
            base_checkpoint="data/models/pretrained/sam_hq_vit_b.pth",
            lora_adapter_path="experiments/exp1/sam/lora_adapters",
            device="cuda"
        )
        masks = segmenter.segment(image_rgb, boxes_xyxy)
    """

    def __init__(
        self,
        base_checkpoint: str,
        lora_adapter_path: Optional[str] = None,
        model_type: str = "vit_b",
        device: str = "cuda",
    ):
        self.base_checkpoint = base_checkpoint
        self.lora_adapter_path = lora_adapter_path
        self.model_type = model_type
        self.device = torch.device(device)

        # Lazy-loaded SAM-HQ instance. Annotated Any (not Optional[Module])
        # for the same reason as grounding_dino.GroundingDINODetector._model:
        # the None state is a transient init detail (every consumer is gated
        # by _load_model), and PEFT-wrapped attributes route through
        # nn.Module.__getattr__ which mypy can't see through.
        self._model: Any = None
        self._transform = ResizeLongestSide(SAM_INPUT_SIZE)

    def _load_model(self) -> None:
        """Load SAM-HQ model lazily on first use."""
        if self._model is not None:
            return

        from ml_engine.models.teacher.sam_lora import load_sam_hq_with_lora

        logger.info(
            "Loading SAM-HQ segmenter (type=%s, lora=%s)...",
            self.model_type,
            self.lora_adapter_path is not None,
        )

        self._model = load_sam_hq_with_lora(
            base_checkpoint=self.base_checkpoint,
            lora_adapter_path=self.lora_adapter_path,
            model_type=self.model_type,
            merge=True,
        )
        self._model.to(self.device)
        self._model.eval()
        logger.info("SAM-HQ segmenter loaded successfully")

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Resize, normalize, and pad an RGB numpy image to a 1024x1024 tensor."""
        resized = self._transform.apply_image(image)
        # [H, W, 3](numpy) -> [3, H, W](torch)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
        tensor = (tensor - SAM_PIXEL_MEAN) / (SAM_PIXEL_STD + 1e-8)

        _, h, w = tensor.shape
        padded = torch.zeros(3, SAM_INPUT_SIZE, SAM_INPUT_SIZE)
        padded[:, :h, :w] = tensor
        return padded

    def _transform_boxes(self, boxes: np.ndarray, orig_h: int, orig_w: int) -> torch.Tensor:
        """Transform xyxy boxes from original image space to 1024x1024 space."""
        transformed = self._transform.apply_boxes(boxes.astype(np.float64), (orig_h, orig_w))
        return torch.from_numpy(transformed).float()

    @torch.no_grad()
    def segment(self, image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """
        Generate segmentation masks for detected boxes.

        Args:
            image: RGB image (H, W, 3)
            boxes: Array of boxes in xyxy format, shape (N, 4)

        Returns:
            List of binary masks, one per box, each (H, W) uint8
        """
        if len(boxes) == 0:
            return []

        self._load_model()

        orig_h, orig_w = image.shape[:2]
        image_tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
        box_tensor = self._transform_boxes(boxes, orig_h, orig_w)
        box_tensor = box_tensor.unsqueeze(0).to(self.device)  # build a batch of 1

        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        outputs = self._model(image_tensor, box_prompts=box_tensor)
        pred_masks = outputs["pred_masks"]

        full_masks = SAMHQLoRA.upscale_masks(pred_masks, (SAM_INPUT_SIZE, SAM_INPUT_SIZE))
        full_masks = full_masks.squeeze(0)

        new_h, new_w = ResizeLongestSide.get_preprocess_shape(orig_h, orig_w, SAM_INPUT_SIZE)

        import cv2

        masks = []
        for i in range(full_masks.shape[0]):
            mask_1024 = full_masks[i].cpu().numpy()
            mask_cropped = mask_1024[:new_h, :new_w]

            mask_original = cv2.resize(
                mask_cropped.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
            masks.append((mask_original > 0).astype(np.uint8))

        return masks
