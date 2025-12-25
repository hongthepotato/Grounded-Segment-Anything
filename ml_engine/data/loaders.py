"""
COCO dataset loaders with multi-model support.

This module provides PyTorch Dataset classes for loading COCO format
datasets with support for multiple model types (Grounding DINO, SAM, YOLO).
"""

from typing import Dict, List, Any, Callable
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
from augmentation import ConfigurableAugmentationPipeline
from ml_engine.data.preprocessing import MultiModelPreprocessor
from groundingdino.util.misc import NestedTensor


class COCODataset(Dataset):
    """
    COCO format dataset with multi-model support.
    
    This dataset class:
    - Receives pre-loaded COCO data
    - Loads images using a path resolver function
    - Supports boxes, masks
    - Returns data in a format suitable for multiple models
    - Handles missing annotations gracefully
    
    Args:
        coco_data: Pre-loaded COCO format dictionary (from DataManager)
        image_path_resolver: Function that takes file_name and returns actual filesystem path
        return_boxes: Whether to return bounding boxes
        return_masks: Whether to return segmentation masks
    
    Example:
        >>> from ml_engine.data.manager import DataManager
        >>> manager = DataManager('train.json', image_paths=[...])
        >>> train_data = manager.get_split('train')
        >>> dataset = COCODataset(
        >>>     coco_data=train_data,
        >>>     image_path_resolver=manager.get_image_path,
        >>>     return_boxes=True,
        >>>     return_masks=True
        >>> )
    """

    def __init__(
        self,
        coco_data: Dict,
        image_path_resolver: Callable[[str], str],
        return_boxes: bool = True,
        return_masks: bool = True,
    ):
        self.coco_data = coco_data
        self.image_path_resolver = image_path_resolver
        self.return_boxes = return_boxes
        self.return_masks = return_masks

        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']

        # Create lookup tables
        self._create_lookup_tables()

        sorted_categories = sorted(self.categories, key=lambda x: x['id'])
        self.class_names = [cat['name'] for cat in sorted_categories]
        # Example: ['ear', 'defect', 'label'] at indices [0, 1, 2]

    def _create_lookup_tables(self):
        """Create efficient lookup tables for images and annotations."""
        # Image ID to image metadata
        self.image_id_to_info = {img['id']: img for img in self.images}

        # Image ID to list of annotations
        self.image_id_to_anns = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_anns:
                self.image_id_to_anns[image_id] = []
            self.image_id_to_anns[image_id].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a dataset sample.
        
        Returns RAW list formats for flexibility
        Returns:
            Dictionary containing:
                - image: PIL Image
                - boxes: List of bounding boxes [[x,y,w,h], ...] COCO format (if return_boxes=True)
                - masks: List of binary masks [mask1(H,W), mask2(H,W), ...] (if return_masks=True)
                - labels: List of category IDs [0, 1, 2, ...]
                - image_id: Original image ID
                - file_name: Image file name
                - image_size: (width, height)
        """
        # Get image metadata
        img_info = self.images[idx]
        image_id, file_name = img_info['id'], img_info['file_name']

        # Load image using path resolver
        image_path = Path(self.image_path_resolver(file_name))
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size

        # Get annotations for this image
        anns = self.image_id_to_anns[image_id]

        # Prepare output dictionary
        sample = {
            'image': image,
            'image_id': image_id,
            'file_name': file_name,
            'image_size': (orig_width, orig_height),
            'labels': []
        }

        # Process annotations
        if self.return_boxes:
            sample['boxes'] = []
        if self.return_masks:
            sample['masks'] = []

        for ann in anns:
            cat_id = ann['category_id']
            sample['labels'].append(cat_id)

            # Bounding box
            if self.return_boxes and 'bbox' in ann:
                bbox = ann['bbox']  # [x, y, width, height]
                sample['boxes'].append(bbox)

            # Segmentation mask
            # Data is normalized to compressed RLE by normalize_coco_annotations
            if self.return_masks and 'segmentation' in ann:
                segmentation = ann['segmentation']

                # Debug assertion to catch non-normalized data
                assert isinstance(segmentation, dict) and isinstance(segmentation.get('counts'), bytes), \
                    f"Segmentation not normalized - must be compressed RLE, got {type(segmentation)}"

                mask = mask_utils.decode(segmentation)
                sample['masks'].append(mask)

        return sample


class TeacherDataset(COCODataset):
    """
    Dataset specifically for teacher model training.
    
    This variant includes additional preprocessing and formatting
    specific to teacher models
    
    Data Flow:
        1. COCODataset returns raw lists:
           - boxes: [[x,y,w,h], ...] (COCO format)
           - masks: [mask1(H,W), mask2(H,W), ...] (list of arrays)
           - labels: [0, 1, 2, ...]
           
        2. If augmentation enabled:
           - Apply augmentation in COCO format
           - Annotations remain in augmented image space
           
        3. For each model:
           - Apply model-specific preprocessing (resize + pad)
           - Transform annotations using metadata
           - Store per-model: {image_tensor, boxes, masks, labels, metadata}
           
        4. Result structure:
           sample['preprocessed'] = {
               'grounding_dino': {
                   'image': tensor (3, H1, W1),
                   'boxes': array (N, 4) in dino space,
                   'masks': array (N, H1, W1),
                   'labels': array (N,),
                   'metadata': {...}
               },
               'sam': {
                   'image': tensor (3, H2, W2),
                   'boxes': array (N, 4) in sam space,
                   'masks': array (N, H2, W2),
                   'labels': array (N,),
                   'metadata': {...}
               }
           }
    
    Args:
        coco_data: Pre-loaded COCO format dictionary
        image_path_resolver: Function that takes file_name and returns actual filesystem path
        preprocessor: MultiModelPreprocessor instance
        augmentation_pipeline: ConfigurableAugmentationPipeline instance
        return_boxes: Whether to return bounding boxes
        return_masks: Whether to return segmentation masks
    
    Example:
        >>> # Use DataManager to create this dataset
        >>> manager = DataManager('train.json', image_paths=[...])
        >>> dataset = DatasetFactory.create_dataset(
        >>>     coco_data=manager.get_split('train'),
        >>>     image_path_resolver=manager.get_image_path,
        >>>     ...
        >>> )
    """

    def __init__(
        self,
        coco_data: Dict,
        image_path_resolver: Callable[[str], str],
        preprocessor: MultiModelPreprocessor=None,
        augmentation_pipeline: ConfigurableAugmentationPipeline=None,
        return_boxes: bool = True,
        return_masks: bool = True,
        sam_single_object_sampling: bool = False
    ):
        super().__init__(
            coco_data=coco_data,
            image_path_resolver=image_path_resolver,
            return_boxes=return_boxes,
            return_masks=return_masks
        )

        self.preprocessor = preprocessor
        self.augmentation_pipeline = augmentation_pipeline
        self.sam_single_object_sampling = sam_single_object_sampling
        # class_names inherited from parent (self.class_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample with teacher-specific preprocessing.
        
        1. Get base sample (COCO format)
        2. Apply augmentation (optional)
        3. Preprocess for each model using OFFICIAL implementations
        4. Return ONLY preprocessed data per model
        
        Returns:
            Dict with:
                - image_id: int
                - file_name: str
                - image_size: tuple (original size)
                - preprocessed: Dict[model_name, Dict] where each model has:
                    {'image': tensor, 'boxes': array, 'masks': array, 'labels': array}
        """
        # Get base sample (lists: boxes, masks, labels)
        sample = super().__getitem__(idx)

        # Apply augmentation if enabled
        if self.augmentation_pipeline is not None:
            augmented = self.augmentation_pipeline(
                image=np.array(sample['image']),
                masks=sample.get('masks'),
                bboxes=sample.get('boxes')
            )

            # Update sample with augmented data
            sample['image'] = Image.fromarray(augmented['image'])
            sample['masks'] = augmented.get('masks')  # Still list format
            sample['boxes'] = augmented.get('bboxes')

        # Prepare annotations for preprocessing
        sample_boxes = sample.get('boxes')
        sample_masks = sample.get('masks')
        sample_labels = sample.get('labels')

        # Convert to numpy arrays for preprocessing
        boxes_np = np.array(sample_boxes, dtype=np.float32) if sample_boxes else None
        masks_np = np.stack(sample_masks, axis=0) if sample_masks else None

        # Apply model-specific preprocessing using OFFICIAL implementations
        preprocessed_dict = self.preprocessor.preprocess_batch(
            sample['image'],
            boxes=boxes_np,
            masks=masks_np
        )

        # Transform annotations for each model using official methods
        preprocessed_data = {}
        for model_name, (image_tensor, metadata) in preprocessed_dict.items():
            model_preprocessor = self.preprocessor.get_preprocessor(model_name)

            # Determine which annotations to use for this model
            if model_name == 'sam' and self.sam_single_object_sampling:
                # SAM single object sampling: pick 1 random object
                # This follows original SAM-HQ training strategy for memory efficiency
                if masks_np is not None and len(masks_np) > 0:
                    idx = random.randint(0, len(masks_np) - 1)
                    use_boxes = boxes_np[idx:idx+1] if boxes_np is not None else None
                    use_masks = masks_np[idx:idx+1]
                    use_labels = [sample_labels[idx]] if sample_labels else []
                else:
                    use_boxes, use_masks, use_labels = None, None, []
            else:
                # Default: use all objects (DINO and SAM without sampling)
                use_boxes = boxes_np
                use_masks = masks_np
                use_labels = sample_labels

            if self.return_boxes and use_boxes is not None and len(use_boxes) > 0:
                model_boxes = model_preprocessor.transform_boxes(use_boxes, metadata)
            else:
                model_boxes = np.zeros((0, 4), dtype=np.float32)

            if self.return_masks and use_masks is not None and len(use_masks) > 0:
                model_masks = model_preprocessor.transform_masks(use_masks, metadata)
            else:
                h, w = metadata['final_size']
                model_masks = np.zeros((0, h, w), dtype=np.float32)

            # Store per-model preprocessed data
            preprocessed_data[model_name] = {
                'image': image_tensor,
                'boxes': model_boxes,
                'masks': model_masks,
                'labels': np.array(use_labels, dtype=np.int64),
                'metadata': metadata
            }

        return {
            'image_id': sample['image_id'],
            'file_name': sample['file_name'],
            'image_size': sample['image_size'],
            'preprocessed': preprocessed_data
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for preprocessed teacher data.
    
    Handles per-model data with different characteristics:
    - DINO: Variable image sizes → create NestedTensor with padding masks
    - SAM: Fixed 1024×1024 → stack directly
    - Both: Variable objects per image → pad to max_objs
    
    Args:
        batch: List of samples from dataset, each with:
            - image_id, file_name, image_size (metadata)
            - preprocessed: Dict[model_name, Dict[str, Any]]
    
    Returns:
        Dict with:
            - image_ids, file_names, image_sizes (metadata lists)
            - preprocessed: Dict[model_name, Dict] with batched tensors:
                - images: NestedTensor for DINO (with .tensor and .mask), [B, C, H, W] for others
                - boxes: [B, max_objs, 4] (padded, already transformed)
                - masks: [B, max_objs, H, W] (padded, already transformed)
                - labels: [B, max_objs] (padded with -1)
    """
    # Extract metadata
    image_ids = [item['image_id'] for item in batch]
    file_names = [item['file_name'] for item in batch]
    image_sizes = [item['image_size'] for item in batch]

    # Get list of models present in this batch
    available_models = list(batch[0]['preprocessed'].keys())

    # Process each model's data separately
    preprocessed_batch = {}

    for model_name in available_models:
        # Extract this model's data from all items
        model_data = [item['preprocessed'][model_name] for item in batch]

        # Get max number of objects for this model's data
        max_objs = max(len(d['boxes']) for d in model_data)
        if max_objs == 0:
            max_objs = 1  # Avoid empty tensors

        # Pad boxes, masks, labels for this model
        padded_boxes = []
        padded_masks = []
        padded_labels = []

        for i, data in enumerate(model_data):
            boxes = data['boxes']
            masks = data['masks']
            labels = data['labels']

            # Data consistency validation
            num_boxes = len(boxes)
            num_labels = len(labels)
            num_masks = len(masks) if len(masks) > 0 else 0

            assert num_boxes == num_labels, (
                f"Data inconsistency in {file_names[i]} ({model_name}): "
                f"{num_boxes} boxes but {num_labels} labels"
            )

            if num_masks > 0:
                assert num_boxes == num_masks, (
                    f"Data inconsistency in {file_names[i]} ({model_name}): "
                    f"{num_boxes} boxes but {num_masks} masks"
                )

            num_objs = num_boxes

            # Pad boxes [N, 4] → [max_objs, 4]
            if num_objs > 0:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                if num_objs < max_objs:
                    padding = torch.zeros((max_objs - num_objs, 4), dtype=torch.float32)
                    boxes_tensor = torch.cat([boxes_tensor, padding], dim=0)
            else:
                boxes_tensor = torch.zeros((max_objs, 4), dtype=torch.float32)
            padded_boxes.append(boxes_tensor)

            # Pad masks [N, H, W] → [max_objs, H, W]
            if num_objs > 0 and num_masks > 0:
                masks_tensor = torch.tensor(masks, dtype=torch.float32)
                if num_objs < max_objs:
                    mask_h, mask_w = masks_tensor.shape[-2:]
                    padding = torch.zeros((max_objs - num_objs, mask_h, mask_w), dtype=torch.float32)
                    masks_tensor = torch.cat([masks_tensor, padding], dim=0)
            else:
                # Default size based on model
                # SAM uses 256x256 (native decoder output resolution for training)
                if model_name == 'sam':
                    masks_tensor = torch.zeros((max_objs, 256, 256), dtype=torch.float32)
                else:
                    # For DINO or others, use a reasonable default
                    masks_tensor = torch.zeros((max_objs, 256, 256), dtype=torch.float32)
            padded_masks.append(masks_tensor)

            # Pad labels [N] → [max_objs] with -1 (ignore_index)
            if num_objs > 0:
                labels_tensor = torch.tensor(labels, dtype=torch.long)
                if num_objs < max_objs:
                    padding = torch.full((max_objs - num_objs,), -1, dtype=torch.long)
                    labels_tensor = torch.cat([labels_tensor, padding], dim=0)
            else:
                labels_tensor = torch.full((max_objs,), -1, dtype=torch.long)
            padded_labels.append(labels_tensor)

        # Handle images: create NestedTensor for DINO, stack SAM
        if model_name == 'grounding_dino':
            # DINO: Variable-sized images - create NestedTensor with padding masks
            images = [d['image'] for d in model_data]
            metadata = [d['metadata'] for d in model_data]
            batched_images = _create_dino_nested_tensor(images, metadata)
        elif model_name == 'sam':
            # SAM: Fixed 1024×1024 - just stack
            images = [d['image'] for d in model_data]
            batched_images = torch.stack(images)
        else:
            # Default: just stack
            images = [d['image'] for d in model_data]
            batched_images = torch.stack(images)

        # Store batched data for this model
        preprocessed_batch[model_name] = {
            'images': batched_images,                  # NestedTensor for DINO, [B, C, H, W] for others
            'boxes': torch.stack(padded_boxes),        # [B, max_objs, 4]
            'masks': torch.stack(padded_masks),        # [B, max_objs, H, W]
            'labels': torch.stack(padded_labels),      # [B, max_objs]
            'metadata': [d['metadata'] for d in model_data]  # List of metadata dicts per image
        }

    # Return metadata + per-model batched data
    return {
        'image_ids': image_ids,
        'file_names': file_names,
        'image_sizes': image_sizes,
        'preprocessed': preprocessed_batch
    }


def _create_dino_nested_tensor(images: List[torch.Tensor], metadata_list: List[Dict]) -> 'NestedTensor':
    """
    Create NestedTensor for Grounding DINO with accurate padding masks.
    
    NestedTensor consists of:
    - tensor: batched images [B, 3, H, W] padded to max size
    - mask: binary mask [B, H, W] where True/1 = padded pixels, False/0 = valid pixels
    
    The mask tells the model's attention mechanism which pixels to ignore.
    
    Args:
        images: List of [C, H, W] tensors (already normalized, variable H/W)
        metadata_list: List of metadata dicts, each containing 'final_size' (h, w)
    
    Returns:
        NestedTensor with padded images and corresponding masks
    """
    # Find max dimensions
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    # Per-channel padding values (normalized black)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    pad_values = (0 - mean) / std

    padded_images = []
    padding_masks = []

    for img, metadata in zip(images, metadata_list):
        c, h, w = img.shape

        # Get actual valid size from metadata (should match h, w)
        final_h, final_w = metadata['final_size']
        assert h == final_h and w == final_w, f"Size mismatch: image {h}x{w} vs metadata {final_h}x{final_w}"

        # Pad image if needed
        if h < max_h or w < max_w:
            # Create padded image
            padded_img = torch.zeros((c, max_h, max_w), dtype=img.dtype, device=img.device)
            for ch in range(c):
                padded_img[ch, :, :] = pad_values[ch]
            padded_img[:, :h, :w] = img
            # Create padding mask: False for valid pixels, True for padded pixels
            mask = torch.ones((max_h, max_w), dtype=torch.bool, device=img.device)
            mask[:h, :w] = False  # Valid region
        else:
            padded_img = img
            # No padding needed, all pixels are valid
            mask = torch.zeros((max_h, max_w), dtype=torch.bool, device=img.device)

        padded_images.append(padded_img)
        padding_masks.append(mask)

    # Stack into batch tensors
    batched_images = torch.stack(padded_images)  # [B, C, H, W]
    batched_masks = torch.stack(padding_masks)    # [B, H, W]

    return NestedTensor(batched_images, batched_masks)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with appropriate settings.
    
    CRITICAL: Uses 'spawn' multiprocessing context to ensure worker processes
    correctly inherit CUDA_VISIBLE_DEVICES. With 'fork' (default on Linux),
    workers inherit the parent's CUDA context, which causes GPU conflicts.
    
    Args:
        dataset: PyTorch Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory
    
    Returns:
        DataLoader instance
    """
    import torch.multiprocessing as mp
    
    # Use 'spawn' to ensure clean CUDA context in workers
    # This is critical when CUDA_VISIBLE_DEVICES is set programmatically
    mp_context = mp.get_context('spawn') if num_workers > 0 else None
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        multiprocessing_context=mp_context  # Use spawn, not fork
    )
