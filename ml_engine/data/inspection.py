"""
Dataset inspection utilities for COCO format datasets.

This module provides functions to inspect COCO datasets and determine
what annotation types are available (boxes, masks, keypoints, etc.).
"""

from typing import Any, Dict, List

from core.constants import (
    ANNOTATION_MODE_COMBINED,
    # POSE_MODEL,  # Uncomment when pose estimation is implemented
    ANNOTATION_MODE_DETECTION,
    ANNOTATION_MODE_SEGMENTATION,
    GROUNDING_DINO,
    MODE_COMBINED,
    MODE_DETECTION,
    MODE_SEGMENTATION,
    SAM,
)


def inspect_dataset(coco_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect COCO dataset to determine available annotations and metadata.

    This is the core function that drives data-driven pipeline behavior.
    The data structure itself tells us what annotations are available.

    Args:
        coco_data: COCO format dictionary with keys:
            - 'images': List of image metadata
            - 'annotations': List of annotations
            - 'categories': List of category definitions

    Returns:
        Dictionary containing:
            - has_boxes (bool): Whether bounding boxes are present
            - has_masks (bool): Whether segmentation masks are present
            - num_classes (int): Number of classes
            - class_mapping (Dict[int, str]): Mapping from category ID to name
            - category_id_to_index (Dict[int, int]): Mapping from category ID to 0-based index
            - index_to_category_id (Dict[int, int]): Mapping from 0-based index to category ID
            - num_images (int): Total number of images
            - num_annotations (int): Total number of annotations
            - annotation_mode (str): Detected mode (for reporting only)

    Example:
        >>> coco_data = load_json('train.json')
        >>> info = inspect_dataset(coco_data)
        >>> print(info['has_boxes'])  # True
        >>> print(info['class_mapping'])  # {0: 'ear', 1: 'defect', ...}
    """
    annotations = coco_data.get("annotations")
    categories = coco_data.get("categories")
    images = coco_data.get("images")

    has_boxes = any("bbox" in ann and ann["bbox"] is not None and ann["bbox"] != [] for ann in annotations)
    has_masks = any(
        "segmentation" in ann and ann["segmentation"] is not None and ann["segmentation"] != []
        for ann in annotations
    )

    # Extract class information
    num_classes = len(categories)
    # class_mapping: category_id -> category_name (for display/logging)
    class_mapping = {cat["id"]: cat["name"] for cat in categories}
    # category_id_to_index: category_id -> 0-based index (for model training)
    category_id_to_index = {cat["id"]: idx for idx, cat in enumerate(categories)}
    # index_to_category_id: 0-based index -> category_id (for prediction decoding)
    index_to_category_id = {idx: cat["id"] for idx, cat in enumerate(categories)}

    # Determine annotation mode (for reporting purposes only)
    if has_boxes and has_masks:
        annotation_mode = ANNOTATION_MODE_COMBINED
    elif has_boxes:
        annotation_mode = ANNOTATION_MODE_DETECTION
    elif has_masks:
        annotation_mode = ANNOTATION_MODE_SEGMENTATION
    else:
        raise KeyError("No valid annotations found in dataset")

    # Compute statistics
    num_images = len(images)
    num_annotations = len(annotations)

    # Count annotations per class
    class_counts = {}
    for ann in annotations:
        cat_id = ann.get("category_id")
        if cat_id is not None:
            class_counts[cat_id] = class_counts.get(cat_id, 0) + 1

    return {
        "has_boxes": has_boxes,
        "has_masks": has_masks,
        "num_classes": num_classes,
        "class_mapping": class_mapping,  # category_id -> name
        "category_id_to_index": category_id_to_index,  # category_id -> 0-based index
        "index_to_category_id": index_to_category_id,  # 0-based index -> category_id
        "num_images": num_images,
        "num_annotations": num_annotations,
        "annotation_mode": annotation_mode,
        "class_counts": class_counts,
    }


def detect_annotation_mode(coco_data: Dict[str, Any]) -> str:
    """
    Detect original annotation mode BEFORE normalization.

    This function captures the ORIGINAL user intent - what type of annotations
    were provided. It is used for MODEL SELECTION (which teachers to load).

    IMPORTANT: Call this BEFORE normalize_coco_annotations() to capture
    the original state. After normalization, boxes may be auto-generated
    from masks, which would change the detected mode.

    Args:
        coco_data: COCO format dictionary (before normalization)

    Returns:
        str: One of 'detection', 'segmentation', or 'combined'
        - 'detection': Only bounding boxes provided
        - 'segmentation': Only segmentation masks provided
        - 'combined': Both boxes and masks provided

    Raises:
        ValueError: If no valid annotations found

    Example:
        >>> mode = detect_annotation_mode(coco_data)
        >>> if mode == 'segmentation':
        >>>     # User provided masks only, load SAM
        >>>     models = ['sam']
    """
    annotations = coco_data.get("annotations", [])

    has_boxes = any("bbox" in ann and ann["bbox"] is not None and ann["bbox"] != [] for ann in annotations)
    has_masks = any(
        "segmentation" in ann and ann["segmentation"] is not None and ann["segmentation"] != []
        for ann in annotations
    )

    if has_boxes and has_masks:
        return MODE_COMBINED
    if has_boxes:
        return MODE_DETECTION
    if has_masks:
        return MODE_SEGMENTATION
    raise ValueError("No valid annotations found in dataset")


def get_required_models_from_mode(annotation_mode: str) -> List[str]:
    """
    Determine which models to load based on ORIGINAL annotation mode.

    This function maps the annotation mode to the required teacher models.
    It should be used with the mode returned by detect_annotation_mode().

    For segmentation-only data, GroundingDINO is always co-trained alongside
    SAM. The normalization step auto-generates bounding boxes from mask
    contours, giving DINO valid training targets. This guarantees a detector
    is available for pseudo-labeling during knowledge distillation.

    Args:
        annotation_mode: One of 'detection', 'segmentation', or 'combined'

    Returns:
        List of model names to load:
        - 'detection'    -> ['grounding_dino']
        - 'segmentation' -> ['grounding_dino', 'sam']
        - 'combined'     -> ['grounding_dino', 'sam']

    Raises:
        ValueError: If unknown annotation mode

    Example:
        >>> mode = detect_annotation_mode(coco_data)
        >>> models = get_required_models_from_mode(mode)
        >>> for model_name in models:
        >>>     load_model(model_name)
    """
    if annotation_mode == MODE_COMBINED:
        return [GROUNDING_DINO, SAM]
    if annotation_mode == MODE_DETECTION:
        return [GROUNDING_DINO]
    if annotation_mode == MODE_SEGMENTATION:
        return [GROUNDING_DINO, SAM]
    raise ValueError(f"Unknown annotation mode: {annotation_mode}")


def get_recommended_student_model(annotation_mode: str, size: str = "s") -> str:
    """
    Select the appropriate student model based on annotation mode and size.

    Args:
        annotation_mode: One of 'detection', 'segmentation', or 'combined'
        size: Model size variant - 'n', 's', 'm', 'l', or 'x'

    Returns:
        Ultralytics model name string, e.g. 'yolov8s-seg'

    Raises:
        ValueError: If unknown annotation mode or invalid size
    """
    valid_sizes = ("n", "s", "m", "l", "x")
    if size not in valid_sizes:
        raise ValueError(f"Invalid size '{size}'. Must be one of {valid_sizes}")

    if annotation_mode in (MODE_COMBINED, MODE_SEGMENTATION):
        return f"yolov8{size}-seg"
    if annotation_mode == MODE_DETECTION:
        return f"yolov8{size}"
    raise ValueError(f"Unknown annotation mode: {annotation_mode}")
