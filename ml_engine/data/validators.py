"""
Data validation and auto-preprocessing utilities for COCO datasets.

This module handles:
- COCO format validation
- Auto-generation of missing bbox from segmentation masks
- Auto-computation of area from masks
- Data quality checks

ID Design Rules (for consistency and simplicity):
- All IDs are non-negative integers (>= 0)
- Image IDs: Can be any non-negative int (e.g., 0, 1, 2, 5, 100, ...)
- Annotation IDs: Can be any non-negative int (e.g., 0, 1, 2, 5, 100, ...)
- Category IDs: MUST satisfy categories[i]['id'] == i
  
Why categories are special:
  Categories must be ordered: categories[0].id=0, categories[1].id=1, etc.
  This enables direct indexing:
    predicted_class = 2
    category_name = categories[predicted_class]['name']  # No lookup/mapping needed!
  
  This eliminates ID→Index remapping throughout the ML pipeline.
"""

import copy
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from pycocotools import mask as mask_utils
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def validate_coco_format(coco_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that data conforms to COCO format.
    
    ID Requirements:
        - All IDs must be non-negative integers (>= 0)
        - Categories must satisfy: categories[i]['id'] == i (order matters!)
        - Image/Annotation IDs can be arbitrary non-negative integers
    
    Args:
        coco_data: Dictionary with COCO format data
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    
    Example:
        >>> is_valid, errors = validate_coco_format(coco_data)
        >>> if not is_valid:
        >>>     for error in errors:
        >>>         print(f"Error: {error}")
    """
    errors = []

    errors.extend(_validate_top_level_keys(coco_data))
    if errors:
        return False, errors

    # Extract data
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    # Validate each section
    errors.extend(_validate_images(images))
    errors.extend(_validate_annotations(annotations))
    errors.extend(_validate_categories(categories))
    errors.extend(_validate_referential_integrity(images, annotations, categories))

    is_valid = len(errors) == 0
    return is_valid, errors

def _validate_top_level_keys(coco_data: Dict[str, Any]) -> List[str]:
    """Validate that all required top-level keys are present."""
    errors = []
    required_keys = ['images', 'annotations', 'categories']

    for key in required_keys:
        if key not in coco_data:
            errors.append(f"Missing required key: '{key}'")
    return errors

def _validate_images(images: List[Dict]) -> List[str]:
    """Validate images section."""
    errors = []

    if not images:
        errors.append("No images found in dataset")
        return errors

    required_keys = ['id', 'width', 'height', 'file_name']
    for idx, img in enumerate(images):
        # Check required keys exist
        missing_keys = [key for key in required_keys if key not in img]
        if missing_keys:
            errors.append(f"Image {idx}: missing required keys: {missing_keys}")
            continue

        # ID: must be non-negative int
        img_id = img['id']
        if not isinstance(img_id, int):
            errors.append(f"Image {idx}: `id` must be an integer, got {type(img_id).__name__}")
        elif img_id < 0:
            errors.append(f"Image {idx}: `id` must be >= 0, got {img_id}")

        # Width: must be positive integer number
        width = img['width']
        if not isinstance(width, int):
            errors.append(f"Image {idx}: `width` must be an integer, got {type(width).__name__}")
        elif width <= 0:
            errors.append(f"Image {idx}: `width` must be > 0, got {width}")

        # Height: must be positive integer number
        height = img['height']
        if not isinstance(height, int):
            errors.append(f"Image {idx}: `height` must be an integer, got {type(height).__name__}")
        elif height <= 0:
            errors.append(f"Image {idx}: `height` must be > 0, got {height}")

        # File name: must be non-empty string
        file_name = img['file_name']
        if not isinstance(file_name, str):
            errors.append(f"Image {idx}: `file_name` must be a string, got {type(file_name).__name__}")
        elif not file_name.strip():
            errors.append(f"Image {idx}: `file_name` cannot be empty")

    return errors

def _validate_annotations(annotations: List[Dict]) -> List[str]:
    """Validate annotations section."""
    errors = []

    if not annotations:
        errors.append("No annotations found in dataset")
        return errors

    required_keys = ['id', 'image_id', 'category_id']

    for idx, ann in enumerate(annotations):
        missing_keys = [key for key in required_keys if key not in ann]
        if missing_keys:
            errors.append(f"Annotation {idx}: missing required keys: {missing_keys}")
            continue

        # ID: must be non-negative int
        ann_id = ann['id']
        if not isinstance(ann_id, int):
            errors.append(f"Annotation {idx}: `id` must be an integer, got {type(ann_id).__name__}")
        elif ann_id < 0:
            errors.append(f"Annotation {idx}: `id` must be >= 0, got {ann_id}")

        # Image ID: must be non-negative int (references image.id)
        image_id = ann['image_id']
        if not isinstance(image_id, int):
            errors.append(f"Annotation {idx}: `image_id` must be an integer, got {type(image_id).__name__}")
        elif image_id < 0:
            errors.append(f"Annotation {idx}: `image_id` must be >= 0, got {image_id}")

        # Category ID: must be non-negative int
        category_id = ann['category_id']
        if not isinstance(category_id, int):
            errors.append(f"Annotation {idx}: `category_id` must be an integer, got {type(category_id).__name__}")
        elif category_id < 0:
            errors.append(f"Annotation {idx}: `category_id` must be >= 0, got {category_id}")

        # Check that at least one annotation type exists
        if not any(k in ann for k in ['bbox', 'segmentation']):
            errors.append(f"Annotation {idx}: no bbox or segmentation found")

        # Validate bbox if present
        if 'bbox' in ann:
            bbox_errors = _validate_bbox(ann['bbox'], ann, idx)
            errors.extend(bbox_errors)

        # Validate iscrowd if present
        if 'iscrowd' in ann:
            if ann['iscrowd'] not in [0, 1]:
                errors.append(f"Annotation {idx}: `iscrowd` must be 0 or 1")

    return errors

def _validate_bbox(bbox: Any, annotation: Dict, ann_idx: int) -> List[str]:
    """
    Validate bounding box format.

    Design: Allow None/empty list ONLY if valid segmentation exists.
    All other cases must be strictly valid: [x, y, width, height] with positive w/h.

    Args:
        bbox: Bounding box value to validate
        annotation: Full annotation dict (to check for segmentation)
        ann_idx: Annotation index (for error messages)

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Allow None or empty list ONLY if valid segmentation exists
    if bbox is None or bbox == []:
        # Check if this annotation has segmentation
        if 'segmentation' not in annotation or not annotation['segmentation']:
            errors.append(
                f"Annotation {ann_idx}: `bbox` is None/empty but no valid segmentation found. "
                "Either bbox or segmentation must be provided."
            )
        return errors

    # All other cases must be strictly valid
    if not isinstance(bbox, list):
        errors.append(f"Annotation {ann_idx}: `bbox` must be a list, got {type(bbox).__name__}")
        return errors

    if len(bbox) != 4:
        errors.append(f"Annotation {ann_idx}: `bbox` must have 4 elements [x, y, width, height], got {len(bbox)}")
        return errors

    # Check all elements are numbers
    if not all(isinstance(x, int) for x in bbox):
        errors.append(f"Annotation {ann_idx}: `bbox` elements must be integer")
        return errors

    # Check width and height are positive
    x, y, width, height = bbox
    if width <= 0:
        errors.append(f"Annotation {ann_idx}: `bbox` width must be > 0, got {width}")
    if height <= 0:
        errors.append(f"Annotation {ann_idx}: `bbox` height must be > 0, got {height}")

    # Optional: warn about suspicious values
    if x < 0 or y < 0:
        errors.append(f"Annotation {ann_idx}: `bbox` has negative x/y coordinates: [{x}, {y}, {width}, {height}]")

    return errors


def _validate_categories(categories: List[Dict]) -> List[str]:
    """
    Validate categories section.
    
    CRITICAL: Enforces categories[i]['id'] == i (order and IDs must match).
    
    Valid example:
        categories[0] = {'id': 0, 'name': 'person'}   ✓
        categories[1] = {'id': 1, 'name': 'car'}      ✓
        categories[2] = {'id': 2, 'name': 'dog'}      ✓
    
    Invalid examples:
        categories[0] = {'id': 1, ...}  ✗ (id should be 0, not 1)
        categories[2] = {'id': 0, ...}  ✗ (id should be 2, not 0)
    
    Why enforce order?
    - Direct indexing: categories[predicted_class_id] gives correct category
    - No ID→Index mapping needed anywhere in the codebase
    - No sorting required before use
    - Eliminates entire class of bugs from index mismatches
    """
    errors = []

    if not categories:
        errors.append("No categories found in dataset")
        return errors

    required_keys = ['id', 'name']
    for idx, cat in enumerate(categories):
        # Check required keys exist
        missing_keys = [key for key in required_keys if key not in cat]
        if missing_keys:
            errors.append(f"Category {idx}: missing required keys: {missing_keys}")
            continue
        
        # ID: must be int
        cat_id = cat['id']
        if not isinstance(cat_id, int):
            errors.append(f"Category {idx}: `id` must be int, got {type(cat_id).__name__}")
        
        # Name: must be non-empty string
        name = cat['name']
        if not isinstance(name, str):
            errors.append(f"Category {idx}: `name` must be a string, got {type(name).__name__}")
        elif not name.strip():
            errors.append(f"Category {idx}: `name` cannot be empty")
    
    # Enforce: categories[i]['id'] == i
    # This ensures direct array indexing: categories[predicted_class] gives correct category
    for idx, cat in enumerate(categories):
        if 'id' not in cat:
            continue
        
        cat_id = cat['id']
        if not isinstance(cat_id, int):
            continue
        
        if cat_id != idx:
            errors.append(
                f"Category at index {idx} has id={cat_id}, but must have id={idx}. "
                f"Categories must be ordered sequentially: categories[0].id=0, categories[1].id=1, etc. "
                f"Please reorder your categories in the annotation file."
            )

    return errors

def _validate_referential_integrity(
    images: List[Dict],
    annotations: List[Dict],
    categories: List[Dict]
) -> List[str]:
    """Validate referential integrity between images, annotations, and categories."""
    errors = []

    # Check for unique image IDs
    image_ids = [img['id'] for img in images]
    if len(image_ids) != len(set(image_ids)):
        errors.append("Image IDs are not unique")

    # Create lookup sets
    image_id_set = set(image_ids)
    category_id_set = {cat['id'] for cat in categories}

    # Check annotations reference valid images and categories
    for idx, ann in enumerate(annotations):
        if ann['image_id'] not in image_id_set:
            errors.append(f"Annotation {idx}: references non-existent image_id {ann['image_id']}")

        if ann['category_id'] not in category_id_set:
            errors.append(f"Annotation {idx}: references non-existent category_id {ann['category_id']}")

    return errors

def compute_bbox_from_mask(segmentation: Any, height: int, width: int) -> List[float]:
    """
    Compute tight bounding box from segmentation mask.
    
    This generates ONE tight box per mask annotation.
    
    Args:
        segmentation: Polygon list or RLE dict
        height: Image height (used to clip/validate bbox)
        width: Image width (used to clip/validate bbox)
    
    Returns:
        Bounding box [x_min, y_min, width, height] in COCO format
    
    Example:
        >>> # Polygon format
        >>> seg = [[x1,y1,x2,y2,x3,y3,x4,y4]]
        >>> bbox = compute_bbox_from_mask(seg, height=480, width=640)
        >>> print(bbox)  # [x, y, w, h]
        
        >>> # RLE format
        >>> seg = {'counts': [...], 'size': [480, 640]}
        >>> bbox = compute_bbox_from_mask(seg)
    """
    bbox = None

    if isinstance(segmentation, list):
        # Polygon format - list of [x1,y1,x2,y2,...]
        if not segmentation or not segmentation[0]:
            return [0, 0, 0, 0]

        # Flatten all polygons
        all_x = []
        all_y = []
        for poly in segmentation:
            poly_array = np.array(poly).reshape(-1, 2)
            all_x.extend(poly_array[:, 0])
            all_y.extend(poly_array[:, 1])

        if not all_x or not all_y:
            return [0, 0, 0, 0]

        x_min = float(np.min(all_x))
        x_max = float(np.max(all_x))
        y_min = float(np.min(all_y))
        y_max = float(np.max(all_y))

        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    elif isinstance(segmentation, dict):
        # RLE format
        if 'counts' in segmentation:
            # Decode RLE
            mask = mask_utils.decode(segmentation)

            # Find bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if not rows.any() or not cols.any():
                return [0, 0, 0, 0]

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]

    if bbox is None:
        return [0, 0, 0, 0]

    x, y, w, h = bbox

    # Clip coordinates to [0, image_dimension]
    x = max(0, min(x, width))
    y = max(0, min(y, height))

    # Adjust width and height to stay within bounds
    w = min(w, width - x)
    h = min(h, height - y)

    bbox = [x, y, w, h]
    return bbox


def compute_area_from_mask(segmentation: Any, height: int, width: int) -> float:
    """
    Compute area from segmentation mask.
    
    Args:
        segmentation: Polygon list or RLE dict
        height: Image height (used to compute area)
        width: Image width (used to compute area)
    
    Returns:
        Area in pixels
    
    Example:
        >>> seg = [[x1,y1,x2,y2,x3,y3,x4,y4]]
        >>> area = compute_area_from_mask(seg, height=480, width=640)
    """
    if isinstance(segmentation, list):
        # Polygon format - convert to RLE first
        if not segmentation:
            return 0.0

        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        area = float(mask_utils.area(rle))
        return area

    if isinstance(segmentation, dict):
        # RLE format
        if 'counts' in segmentation:
            area = float(mask_utils.area(segmentation))
            return area

    return 0.0


def preprocess_coco_dataset(coco_data: Dict[str, Any], in_place: bool = True) -> Dict[str, Any]:
    """
    Auto-generate missing bbox and area fields from segmentation masks.
    
    This runs during dataset validation (python cli/validate_dataset.py).
    Frontend annotation tools don't need to compute boxes - the platform
    handles this automatically.
    
    CRITICAL: Each mask annotation generates ONE tight bounding box.
    Multiple disconnected objects should be separate annotations.
    
    Args:
        coco_data: COCO format dictionary
        in_place: Whether to modify coco_data in place (default: True)
    
    Returns:
        Preprocessed COCO data (same reference if in_place=True)
    
    Example:
        >>> coco_data = load_json('annotations.json')
        >>> coco_data = preprocess_coco_dataset(coco_data)
        >>> # Now all annotations have bbox and area fields
    """
    if not in_place:
        coco_data = copy.deepcopy(coco_data)

    # Create image lookup for dimensions
    image_lookup = {img['id']: img for img in coco_data['images']}

    annotations = coco_data['annotations']
    bbox_generated = 0
    area_generated = 0

    for ann in annotations:
        image_id = ann['image_id']
        img_info = image_lookup.get(image_id)

        if img_info is None:
            logger.warning("Annotation %s: image_id %s not found", ann['id'], image_id)
            continue

        height = img_info.get('height')
        width = img_info.get('width')

        # Auto-generate bbox from segmentation if missing
        # Check: (1) segmentation exists, (2) is valid (not None/empty), (3) bbox missing or invalid
        has_valid_seg = ('segmentation' in ann and
                        ann['segmentation'] is not None and
                        ann['segmentation'] != [])
        has_valid_bbox = ('bbox' in ann and
                         ann['bbox'] is not None and
                         ann['bbox'] != [])

        if has_valid_seg and not has_valid_bbox:
            bbox = compute_bbox_from_mask(ann['segmentation'], height, width)
            ann['bbox'] = bbox
            bbox_generated += 1
            logger.debug("Generated bbox for annotation %s", ann['id'])

        # Auto-compute area from segmentation if missing
        if has_valid_seg and 'area' not in ann:
            area = compute_area_from_mask(ann['segmentation'], height, width)
            ann['area'] = area
            area_generated += 1
            logger.debug("Generated area for annotation %s", ann['id'])

        # Compute area from bbox if still missing
        elif has_valid_bbox and 'area' not in ann:
            bbox = ann['bbox']
            ann['area'] = bbox[2] * bbox[3]  # width * height
            area_generated += 1

    if bbox_generated > 0:
        logger.info(" Auto-generated %d bounding boxes from masks", bbox_generated)
    if area_generated > 0:
        logger.info(" Auto-computed %d areas", area_generated)

    return coco_data


def check_data_quality(coco_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform QUALITY checks on COCO dataset.
    
    IMPORTANT: This runs AFTER validate_coco_format(), so we assume structure is valid.
    This method checks for QUALITY issues that won't break training but might affect performance.
    
    Quality checks (warnings, not errors):
    - Images without annotations (wasteful but ok)
    - Class imbalance (might affect training)
    - Small objects (might be hard to detect)
    - Low samples per class (might overfit)
    
    Args:
        coco_data: COCO format dictionary (already validated)
    
    Returns:
        Dictionary with quality metrics and warnings
    """
    results = {
        'total_images': len(coco_data['images']),
        'total_annotations': len(coco_data['annotations']),
        'images_without_annotations': 0,
        'small_objects': 0,  # Bbox area < threshold
        'large_objects': 0,   # Bbox area > 80% of image
        'samples_per_class': {},
        'class_distribution': {},
        'warnings': []
    }

    # Build image to annotations mapping
    image_to_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_anns:
            image_to_anns[image_id] = []
        image_to_anns[image_id].append(ann)

    # Create image lookup for dimensions
    image_lookup = {img['id']: img for img in coco_data['images']}

    # Check images without annotations
    for img in coco_data['images']:
        if img['id'] not in image_to_anns:
            results['images_without_annotations'] += 1

    # Analyze annotations for quality issues
    for ann in coco_data['annotations']:
        # Class distribution
        cat_id = ann['category_id']
        results['class_distribution'][cat_id] = results['class_distribution'].get(cat_id, 0) + 1

        # Check for small/large objects (quality concern, not validity issue)
        if 'bbox' in ann and ann['bbox'] is not None and ann['bbox'] != []:
            bbox = ann['bbox']
            bbox_area = bbox[2] * bbox[3]  # width * height

            # Get image dimensions
            img_info = image_lookup.get(ann['image_id'])
            if img_info and 'width' in img_info and 'height' in img_info:
                img_area = img_info['width'] * img_info['height']

                # Small objects (< 1% of image) might be hard to detect
                if bbox_area < img_area * 0.01:
                    results['small_objects'] += 1

                # Very large objects (> 80% of image) might indicate annotation errors
                if bbox_area > img_area * 0.8:
                    results['large_objects'] += 1

    # Generate warnings
    if results['images_without_annotations'] > 0:
        pct = 100 * results['images_without_annotations'] / results['total_images']
        results['warnings'].append(
            f"{results['images_without_annotations']} images ({pct:.1f}%) have no annotations (will be ignored)"
        )

    if results['small_objects'] > 0:
        pct = 100 * results['small_objects'] / results['total_annotations']
        results['warnings'].append(
            f"{results['small_objects']} annotations ({pct:.1f}%) have very small objects (<1% of image area)"
        )

    if results['large_objects'] > 0:
        pct = 100 * results['large_objects'] / results['total_annotations']
        results['warnings'].append(
            f"{results['large_objects']} annotations ({pct:.1f}%) have very large objects (>80% of image area)"
        )

    # Check class imbalance
    if results['class_distribution']:
        counts = list(results['class_distribution'].values())
        max_count = max(counts)
        min_count = min(counts)
        if max_count / min_count > 10:
            results['warnings'].append(
                f"High class imbalance detected (ratio {max_count/min_count:.1f}:1)"
            )

        # Check for classes with very few samples
        results['samples_per_class'] = results['class_distribution']
        low_sample_classes = [cat_id for cat_id, count in results['samples_per_class'].items() if count < 10]
        if low_sample_classes:
            results['warnings'].append(
                f"{len(low_sample_classes)} classes have fewer than 10 samples (might overfit)"
            )

    return results


def split_dataset(
    coco_data: Dict[str, Any],
    splits: Dict[str, float] = None,
    stratify: bool = True,
    random_seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Split COCO dataset into train/val/test with stratification.
    
    Args:
        coco_data: COCO format dictionary
        splits: Dictionary with split ratios, e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15}
        stratify: Whether to maintain class distribution across splits
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing split datasets
    
    Example:
        >>> splits = split_dataset(
        >>>     coco_data,
        >>>     splits={'train': 0.7, 'val': 0.15, 'test': 0.15},
        >>>     stratify=True,
        >>>     random_seed=42
        >>> )
        >>> train_data = splits['train']
        >>> val_data = splits['val']
    """
    # Set default splits if not provided
    if splits is None:
        splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    # Step 1: Validate inputs
    _validate_split_ratios(splits)

    # Step 2: Prepare data structures
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    image_to_anns = _build_image_to_annotations_mapping(annotations)
    image_with_ann_ids = _get_images_with_annotations(images, image_to_anns)

    # Step 3: Perform splitting
    if stratify:
        split_ids = _stratified_split(image_with_ann_ids, image_to_anns, splits, random_seed)
    else:
        split_ids = _random_split(image_with_ann_ids, splits, random_seed)

    # Step 4: Create split datasets with validation
    result = _create_split_datasets(
        split_ids, images, annotations, categories
    )

    return result


def _validate_split_ratios(splits: Dict[str, float]) -> None:
    """Validate that split ratios sum to 1.0."""
    total_ratio = sum(splits.values())
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")


def _build_image_to_annotations_mapping(annotations: List[Dict]) -> Dict[int, List[Dict]]:
    """Build mapping from image_id to list of annotations."""
    image_to_anns = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_to_anns:
            image_to_anns[image_id] = []
        image_to_anns[image_id].append(ann)
    return image_to_anns


def _get_images_with_annotations(
    images: List[Dict],
    image_to_anns: Dict[int, List[Dict]]
) -> List[int]:
    """Get list of image IDs that have at least one annotation."""
    return [img['id'] for img in images if img['id'] in image_to_anns]


def _stratified_split(
    image_with_ann_ids: List[int],
    image_to_anns: Dict[int, List[Dict]],
    splits: Dict[str, float],
    random_seed: int
) -> Dict[str, set]:
    """
    Perform stratified split to maintain class distribution.
    
    Falls back to random split if stratification fails (too few samples per class).
    """
    # Map each image to its primary class
    image_to_class = {}
    for img_id in image_with_ann_ids:
        anns = image_to_anns[img_id]
        cat_ids = [ann['category_id'] for ann in anns]
        primary_class = max(set(cat_ids), key=cat_ids.count)
        image_to_class[img_id] = primary_class

    # Check if stratification is possible
    class_counts = {}
    for class_id in image_to_class.values():
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    num_classes = len(class_counts)
    min_samples = min(class_counts.values())

    # Need at least 2 different classes for stratification
    if num_classes < 2:
        logger.warning(
            "Cannot stratify: dataset has only %d class(es). "
            "Stratification requires at least 2 classes. Falling back to random split.",
            num_classes
        )
        logger.warning("Class distribution: %s", class_counts)
        return _random_split(image_with_ann_ids, splits, random_seed)

    # Check 2: Need at least 2 samples per class
    if min_samples < 2:
        logger.warning(
            "Cannot stratify: some classes have fewer than 2 samples. "
            "Falling back to random split."
        )
        logger.warning("Class distribution: %s", class_counts)
        return _random_split(image_with_ann_ids, splits, random_seed)

    # Perform stratified splitting
    try:
        image_ids_array = np.array(image_with_ann_ids)
        classes = np.array([image_to_class[img_id] for img_id in image_with_ann_ids])

        # First split: train vs (val + test)
        train_size = splits.get('train', 0.7)
        train_ids, temp_ids = train_test_split(
            image_ids_array,
            train_size=train_size,
            stratify=classes,
            random_state=random_seed
        )

        # Second split: val vs test
        if 'val' in splits and 'test' in splits:
            val_ratio = splits['val'] / (splits['val'] + splits['test'])
            temp_classes = np.array([image_to_class[img_id] for img_id in temp_ids])

            val_ids, test_ids = train_test_split(
                temp_ids,
                train_size=val_ratio,
                stratify=temp_classes,
                random_state=random_seed
            )
        else:
            val_ids = temp_ids
            test_ids = []

        return {
            'train': set(train_ids),
            'val': set(val_ids),
            'test': set(test_ids) if len(test_ids) > 0 else set()
        }

    except ValueError as e:
        # Stratification failed
        logger.warning("Stratification failed: %s. Falling back to random split.", str(e))
        return _random_split(image_with_ann_ids, splits, random_seed)


def _random_split(
    image_ids: List[int],
    splits: Dict[str, float],
    random_seed: int
) -> Dict[str, set]:
    """Perform random split without stratification."""
    np.random.seed(random_seed)

    shuffled_ids = np.array(image_ids)
    np.random.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    split_ids = {}
    current_idx = 0

    for split_name, ratio in splits.items():
        n_split = int(n_total * ratio)
        split_ids[split_name] = set(shuffled_ids[current_idx:current_idx + n_split])
        current_idx += n_split

    return split_ids


def _create_split_datasets(
    split_ids: Dict[str, set],
    images: List[Dict],
    annotations: List[Dict],
    categories: List[Dict]
) -> Dict[str, Dict[str, Any]]:
    """
    Create split datasets and validate each split has all classes.
    
    Args:
        split_ids: Dict mapping split_name → set of image IDs
        images: List of all images
        annotations: List of all annotations
        categories: List of all categories
    
    Returns:
        Dict mapping split_name → COCO format dict for that split
    """
    result = {}
    all_category_ids = {cat['id'] for cat in categories}

    for split_name, img_ids in split_ids.items():
        if not img_ids:
            continue

        # Filter images and annotations for this split
        split_images = [img for img in images if img['id'] in img_ids]
        split_annotations = [ann for ann in annotations if ann['image_id'] in img_ids]

        # Validate: Check if this split has all classes
        split_category_ids = {ann['category_id'] for ann in split_annotations}
        missing_classes = all_category_ids - split_category_ids

        if missing_classes:
            _warn_missing_classes(split_name, missing_classes, categories)

        result[split_name] = {
            'images': split_images,
            'annotations': split_annotations,
            'categories': categories
        }

    return result


def _warn_missing_classes(
    split_name: str,
    missing_class_ids: set,
    categories: List[Dict]
) -> None:
    """Warn user that some classes are missing from a split."""
    missing_names = [cat['name'] for cat in categories if cat['id'] in missing_class_ids]

    logger.warning(
        "Split '%s' is missing %d classes: %s",
        split_name, len(missing_class_ids), missing_names
    )
    logger.warning(
        "This happens with small datasets or very imbalanced classes. Consider:"
    )
    logger.warning("  1. Collecting more samples for underrepresented classes")
    logger.warning("  2. Using a different split ratio")
    logger.warning("  3. Disabling stratification (set stratify=False)")
    logger.warning("  4. Using cross-validation instead of fixed splits")
