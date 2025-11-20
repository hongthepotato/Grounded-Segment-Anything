"""
Dataset inspection utilities for COCO format datasets.

This module provides functions to inspect COCO datasets and determine
what annotation types are available (boxes, masks, keypoints, etc.).
"""

import json
from typing import Dict, Any, List
from pathlib import Path


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
            - num_images (int): Total number of images
            - num_annotations (int): Total number of annotations
            - annotation_mode (str): Detected mode (for reporting only)
    
    Example:
        >>> coco_data = load_json('train.json')
        >>> info = inspect_dataset(coco_data)
        >>> print(info['has_boxes'])  # True
        >>> print(info['class_mapping'])  # {0: 'ear', 1: 'defect', ...}
    """
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    images = coco_data.get('images', [])

    # Validate minimum required fields
    if not annotations:
        raise ValueError("Dataset has no annotations!")
    if not categories:
        raise ValueError("Dataset has no categories defined!")
    if not images:
        raise ValueError("Dataset has no images defined!")

    # Check for annotation types by inspecting actual data
    has_boxes = any('bbox' in ann for ann in annotations)
    has_masks = any('segmentation' in ann for ann in annotations)
    # has_keypoints = any('keypoints' in ann for ann in annotations)

    # Extract class information
    num_classes = len(categories)
    class_mapping = {cat['id']: cat['name'] for cat in categories}

    # Determine annotation mode (for reporting purposes only)
    if has_boxes and has_masks:
        annotation_mode = "DETECTION_AND_SEGMENTATION"
    elif has_boxes:
        annotation_mode = "DETECTION_ONLY"
    elif has_masks:
        annotation_mode = "SEGMENTATION_ONLY"
    else:
        raise KeyError("No valid annotations found in dataset")

    # Compute statistics
    num_images = len(images)
    num_annotations = len(annotations)

    # Count annotations per class
    class_counts = {}
    for ann in annotations:
        cat_id = ann.get('category_id')
        if cat_id is not None:
            class_counts[cat_id] = class_counts.get(cat_id, 0) + 1

    return {
        'has_boxes': has_boxes,
        'has_masks': has_masks,
        'num_classes': num_classes,
        'class_mapping': class_mapping,
        'num_images': num_images,
        'num_annotations': num_annotations,
        'annotation_mode': annotation_mode,
        'class_counts': class_counts
    }


def load_and_inspect_dataset(json_path: str) -> Dict[str, Any]:
    """
    Load COCO JSON file and inspect it.
    
    Args:
        json_path: Path to COCO format JSON file
    
    Returns:
        Dataset inspection dictionary (same as inspect_dataset)
    
    Example:
        >>> info = load_and_inspect_dataset('data/raw/train.json')
        >>> if info['has_boxes']:
        >>>     print("Dataset has bounding boxes, will train Grounding DINO")
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    return inspect_dataset(coco_data)


def print_dataset_report(dataset_info: Dict[str, Any]) -> None:
    """
    Print a formatted report of dataset inspection results.
    
    Args:
        dataset_info: Output from inspect_dataset()
    
    Example:
        >>> info = inspect_dataset(coco_data)
        >>> print_dataset_report(info)
    """
    print("━" * 60)
    print(" Dataset Inspection Report")
    print("━" * 60)

    print(f"\n Images: {dataset_info['num_images']}")
    print(f" Annotations: {dataset_info['num_annotations']}")
    print(f" Classes: {dataset_info['num_classes']}")

    print("\n Annotations Available:")
    print(f"   ├─ Bounding boxes: {'✓' if dataset_info['has_boxes'] else '✗'}")
    print(f"   ├─ Segmentation masks: {'✓' if dataset_info['has_masks'] else '✗'}")
    # print(f"   └─ Keypoints: {'✓' if dataset_info['has_keypoints'] else '✗'}")

    print(f"\n Detected Mode: {dataset_info['annotation_mode']}")

    print("\n  Classes:")
    class_counts = dataset_info.get('class_counts', {})
    for class_id, class_name in sorted(dataset_info['class_mapping'].items()):
        count = class_counts.get(class_id, 0)
        print(f"   ├─ {class_id}: {class_name} ({count} instances)")

    print("\n Recommended Pipeline:")
    if dataset_info['has_boxes'] and dataset_info['has_masks']:
        print("   ├─ Teacher models: grounding_dino + sam")
        print("   ├─ Student model: yolov8_seg (detection + segmentation)")
        print("   └─ Command: python cli/train_teacher.py --data train.json")
    elif dataset_info['has_boxes']:
        print("   ├─ Teacher model: grounding_dino")
        print("   ├─ Student model: yolov8 (detection only)")
        print("   └─ Command: python cli/train_teacher.py --data train.json")
    elif dataset_info['has_masks']:
        print("   ├─ Teacher model: sam")
        print("   ├─ Student model: yolov8_seg or fastsam")
        print("   └─ Command: python cli/train_teacher.py --data train.json")

    print("━" * 60)


def get_required_models(dataset_info: Dict[str, Any]) -> List[str]:
    """
    Determine which models need to be loaded based on dataset inspection.
    
    This is a key function for data-driven model loading.
    
    Args:
        dataset_info: Output from inspect_dataset()
    
    Returns:
        List of model names to load (e.g., ['grounding_dino', 'sam'])
    
    Example:
        >>> info = inspect_dataset(coco_data)
        >>> models = get_required_models(info)
        >>> for model_name in models:
        >>>     load_model(model_name)  # Load only what's needed
    """
    required_models = []

    if dataset_info['has_boxes']:
        required_models.append('grounding_dino')

    if dataset_info['has_masks']:
        required_models.append('sam')

    # if dataset_info['has_keypoints']:
    #     required_models.append('pose_model')

    return required_models


def get_recommended_student_model(dataset_info: Dict[str, Any], size: str = 's') -> str:
    """
    Recommend appropriate student model based on dataset annotations.
    
    Args:
        dataset_info: Output from inspect_dataset()
        size: Model size variant ('n', 's', 'm', 'l', 'x')
    
    Returns:
        Student model name (e.g., 'yolov8s-seg', 'yolov8s', 'fastsam-s')
    
    Example:
        >>> info = inspect_dataset(coco_data)
        >>> student = get_recommended_student_model(info, size='s')
        >>> print(student)  # 'yolov8s-seg' if both boxes and masks
    """
    if dataset_info['has_boxes'] and dataset_info['has_masks']:
        return f'yolov8{size}-seg'  # Detection + Segmentation
    elif dataset_info['has_boxes']:
        return f'yolov8{size}'  # Detection only
    elif dataset_info['has_masks']:
        return f'fastsam-{size}'  # Segmentation only
    elif dataset_info['has_keypoints']:
        return f'yolov8{size}-pose'  # Pose estimation
    else:
        raise ValueError("No valid annotations found in dataset")
