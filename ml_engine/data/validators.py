"""
Data validation and auto-preprocessing utilities for COCO datasets.

This module handles:
- COCO format validation
- Auto-generation of missing bbox from segmentation masks
- Auto-computation of area from masks
- Data quality checks

ID Design Rules:
- All IDs are non-negative integers (>= 0)
- Image IDs: Can be any non-negative int (e.g., 0, 1, 2, 5, 100, ...)
- Annotation IDs: Can be any non-negative int (e.g., 0, 1, 2, 5, 100, ...)
- Category IDs: Can be any non-negative int, but must be UNIQUE
  
Category ID Handling:
  The training pipeline builds a cat_id → index mapping internally:
    class_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}

"""

import copy
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from pycocotools import mask as mask_utils
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from .utils import has_valid_list_field, has_valid_numeric_field

logger = logging.getLogger(__name__)


def validate_coco_format(coco_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that data conforms to COCO format.
    
    ID Requirements:
        - All IDs must be non-negative integers (>= 0)
        - Category IDs must be unique
        - Image/Annotation IDs must be unique non-negative integers
    
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
        
        # Validate segmentation if present
        if 'segmentation' in ann:
            seg_errors = _validate_segmentation(ann['segmentation'], idx)
            errors.extend(seg_errors)

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

    # Check all elements are numbers (int or float, per COCO spec)
    if not all(isinstance(x, (int, float)) for x in bbox):
        errors.append(f"Annotation {ann_idx}: `bbox` elements must be numbers (int or float)")
        return errors

    # Check width and height are positive
    x, y, width, height = bbox
    if width < 0:
        errors.append(f"Annotation {ann_idx}: `bbox` width must be >= 0, got {width}")
    if height < 0:
        errors.append(f"Annotation {ann_idx}: `bbox` height must be >= 0, got {height}")

    # Optional: warn about suspicious values
    if x < 0 or y < 0:
        errors.append(f"Annotation {ann_idx}: `bbox` has negative x/y coordinates: [{x}, {y}, {width}, {height}]")

    return errors


def _validate_segmentation(segmentation: Any, ann_idx: int) -> List[str]:
    """
    Validate segmentation format.
    
    Frontend produces polygon format only, but backend accepts both polygon and RLE
    for compatibility with external COCO datasets.
    
    Accepted formats:
    1. None or [] - bbox-only annotation (valid)
    2. Polygon format: [[x1,y1,x2,y2,...], [x1,y1,x2,y2,...], ...]
       - Frontend produces this
       - Multiple polygons = parts of ONE object (e.g., person with spread arms)
    3. RLE format: {'counts': [...], 'size': [h, w]}
       - External datasets may use this
       - Frontend never produces this
    
    Args:
        segmentation: Segmentation value to validate
        ann_idx: Annotation index (for error messages)
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if segmentation is None or segmentation == []:
        return errors

    # Check if RLE format (dict with 'counts' and 'size')
    if isinstance(segmentation, dict):
        if 'counts' not in segmentation:
            errors.append(f"Annotation {ann_idx}: RLE `segmentation` must have 'counts' field")
        if 'size' not in segmentation:
            errors.append(f"Annotation {ann_idx}: RLE `segmentation` must have 'size' field")
        return errors

    if not isinstance(segmentation, list):
        errors.append(
            f"Annotation {ann_idx}: `segmentation` must be list (polygons) or dict (RLE), "
            f"got {type(segmentation).__name__}"
        )
        return errors

    # Validate each polygon
    for poly_idx, polygon in enumerate(segmentation):
        if not isinstance(polygon, list):
            errors.append(
                f"Annotation {ann_idx}, polygon {poly_idx}: "
                f"polygon must be a list, got {type(polygon).__name__}"
            )
            continue

        # Empty polygon is invalid
        if len(polygon) == 0:
            errors.append(
                f"Annotation {ann_idx}, polygon {poly_idx}: "
                f"polygon cannot be empty"
            )
            continue

        if len(polygon) < 6:
            errors.append(
                f"Annotation {ann_idx}, polygon {poly_idx}: "
                f"polygon must have at least 6 coordinates (3 points), got {len(polygon)}"
            )
            continue

        # Coordinates must be even (pairs of x,y)
        if len(polygon) % 2 != 0:
            errors.append(
                f"Annotation {ann_idx}, polygon {poly_idx}: "
                f"polygon must have even number of coordinates (x,y pairs), got {len(polygon)}"
            )
            continue

        # All coordinates must be numbers (int or float, per COCO spec)
        if not all(isinstance(coord, (int, float)) for coord in polygon):
            errors.append(
                f"Annotation {ann_idx}, polygon {poly_idx}: "
                f"all coordinates must be numbers (int or float)"
            )

    return errors


def _validate_categories(categories: List[Dict]) -> List[str]:
    """
    Validate categories section.
    
    Requirements:
    - Each category must have 'id' (non-negative int) and 'name' (non-empty string)
    - Category IDs must be unique
    
    Valid examples:
        categories = [
            {'id': 0, 'name': 'cat'}, 
            {'id': 1, 'name': 'dog'},
            {'id': 2, 'name': 'bird'}
        ]  ✓ Sequential
        categories = [
            {'id': 1, 'name': 'person'},
            {'id': 90, 'name': 'toothbrush'},
            {'id': 91, 'name': 'toothpaste'}
        ]  ✓ Sparse
    
    Invalid examples:
        categories = [{'id': 1, 'name': 'cat'}, {'id': 1, 'name': 'dog'}]  ✗ Duplicate IDs
        categories = [{'id': -1, 'name': 'cat'}]  ✗ Negative ID
    
    Note: The training pipeline will build a cat_id → index mapping internally
    """
    errors = []

    if not categories:
        errors.append("No categories found in dataset")
        return errors

    required_keys = ['id', 'name']
    seen_ids = set()

    for idx, cat in enumerate(categories):
        missing_keys = [key for key in required_keys if key not in cat]
        if missing_keys:
            errors.append(f"Category {idx}: missing required keys: {missing_keys}")
            continue

        cat_id = cat['id']
        if not isinstance(cat_id, int):
            errors.append(f"Category {idx}: `id` must be int, got {type(cat_id).__name__}")
        elif cat_id < 0:
            errors.append(f"Category {idx}: `id` must be >= 0, got {cat_id}")
        else:
            if cat_id in seen_ids:
                errors.append(f"Category {idx}: duplicate id={cat_id} (already used by another category)")
            seen_ids.add(cat_id)

        name = cat['name']
        if not isinstance(name, str):
            errors.append(f"Category {idx}: `name` must be a string, got {type(name).__name__}")
        elif not name.strip():
            errors.append(f"Category {idx}: `name` cannot be empty")

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


def _normalize_to_rle(segmentation: Any, height: int, width: int) -> dict:
    """
    Normalize any segmentation format to compressed RLE.
    
    COCO segmentation can be in three formats:
    1. Polygon: [[x1,y1,x2,y2,...], ...]
    2. Uncompressed RLE: {'counts': [int, int, ...], 'size': [h, w]}
    3. Compressed RLE: {'counts': b'...', 'size': [h, w]}
    
    pycocotools functions (decode, area, etc.) require compressed RLE.
    This function normalizes all formats to compressed RLE.
    
    Args:
        segmentation: Polygon list or RLE dict (compressed or uncompressed)
        height: Image height
        width: Image width
    
    Returns:
        Compressed RLE dict with bytes counts
    """
    if isinstance(segmentation, list):
        # Polygon format -> convert to compressed RLE
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.merge(rles)

    if isinstance(segmentation, dict):
        counts = segmentation.get('counts')

        # Already compressed (bytes or str) -> return as-is
        if isinstance(counts, (bytes, str)):
            return segmentation

        # Uncompressed (list of ints) -> convert to compressed RLE
        # frPyObjects handles uncompressed RLE dicts
        return mask_utils.frPyObjects(segmentation, height, width)

    raise TypeError(
        f"Segmentation must be list (polygon) or dict (RLE), "
        f"got {type(segmentation).__name__}"
    )


def compute_bbox_from_mask(segmentation: Any, height: int, width: int) -> List[float]:
    """
    Compute tight bounding box from segmentation mask.
    
    This generates ONE tight box per mask annotation.
    
    Args:
        segmentation: Polygon list or RLE dict (compressed or uncompressed)
                     If already compressed RLE (bytes), no conversion needed.
        height: Image height (used to clip/validate bbox)
        width: Image width (used to clip/validate bbox)
    
    Returns:
        Bounding box [x_min, y_min, width, height] in COCO format
    
    Example:
        >>> # Polygon format
        >>> seg = [[x1,y1,x2,y2,x3,y3,x4,y4]]
        >>> bbox = compute_bbox_from_mask(seg, height=480, width=640)
        >>> print(bbox)  # [x, y, w, h]
        
        >>> # RLE format (compressed or uncompressed)
        >>> seg = {'counts': [...], 'size': [480, 640]}
        >>> bbox = compute_bbox_from_mask(seg, height=480, width=640)
        
        >>> # Already compressed RLE (most efficient)
        >>> seg = {'counts': b'...', 'size': [480, 640]}
        >>> bbox = compute_bbox_from_mask(seg, height=480, width=640)
    """
    # If already compressed RLE, use directly
    if isinstance(segmentation, dict) and isinstance(segmentation.get('counts'), bytes):
        rle = segmentation
    else:
        rle = _normalize_to_rle(segmentation, height, width)

    # Decode to binary mask
    mask = mask_utils.decode(rle)

    # Find bounding box from binary mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        raise ValueError(
            "Mask is completely empty (all zeros). "
            "Cannot generate bounding box from empty mask."
        )

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    x, y = float(x_min), float(y_min)
    w, h = float(x_max - x_min + 1), float(y_max - y_min + 1)

    # Clip coordinates to image bounds
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = min(w, width - x)
    h = min(h, height - y)

    return [x, y, w, h]


def compute_area_from_mask(segmentation: Any, height: int, width: int) -> float:
    """
    Compute area from segmentation mask.
    
    Args:
        segmentation: Polygon list or RLE dict (compressed or uncompressed)
                     If already compressed RLE (bytes), no conversion needed.
        height: Image height (used to compute area)
        width: Image width (used to compute area)
    
    Returns:
        Area in pixels
    
    Example:
        >>> seg = [[x1,y1,x2,y2,x3,y3,x4,y4]]
        >>> area = compute_area_from_mask(seg, height=480, width=640)
        
        >>> # Already compressed RLE (most efficient)
        >>> seg = {'counts': b'...', 'size': [480, 640]}
        >>> area = compute_area_from_mask(seg, height=480, width=640)
    """
    # If already compressed RLE, use directly
    if isinstance(segmentation, dict) and isinstance(segmentation.get('counts'), bytes):
        rle = segmentation
    else:
        rle = _normalize_to_rle(segmentation, height, width)

    return float(mask_utils.area(rle))


def normalize_coco_annotations(coco_data: Dict[str, Any], in_place: bool = True) -> Dict[str, Any]:
    """
    Normalize COCO annotations to canonical form.
    
    This is the CANONICAL FORM function that ensures all annotations are in a
    consistent, normalized state after loading. It performs:
    
    1. Auto-generate missing bbox from segmentation masks
    2. Auto-compute missing area from masks or bboxes
    3. Normalize all segmentation formats to compressed RLE
    
    After this function, ALL segmentations are guaranteed to be in compressed RLE
    format: {'counts': b'...', 'size': [h, w]}. This eliminates format handling
    complexity in downstream consumers (loaders, preprocessors, etc.).
    
    Args:
        coco_data: COCO format dictionary
        in_place: Whether to modify coco_data in place (default: True)
    
    Returns:
        Normalized COCO data with:
        - All annotations have valid bbox [x, y, w, h]
        - All annotations have valid area (float)
        - All segmentations are compressed RLE format
    
    Example:
        >>> coco_data = load_json('annotations.json')
        >>> coco_data = normalize_coco_annotations(coco_data)
        >>> # Now all annotations are in canonical form
        >>> # Segmentations are all compressed RLE (bytes counts)
    """
    if not in_place:
        coco_data = copy.deepcopy(coco_data)

    # Create image lookup (map: image_id -> single_image dictionary)
    image_lookup = {img['id']: img for img in coco_data['images']}

    annotations = coco_data['annotations']
    bbox_generated = 0
    area_generated = 0
    seg_normalized = 0

    for ann in annotations:
        image_id = ann['image_id']
        img_info = image_lookup.get(image_id)

        if img_info is None:
            logger.warning("Annotation %s: image_id %s not found", ann['id'], image_id)
            continue

        height = img_info.get('height')
        width = img_info.get('width')

        # Check for valid segmentation (polygon list or RLE dict)
        seg = ann.get('segmentation')
        has_valid_seg = seg is not None and seg != [] and isinstance(seg, (list, dict))

        # STEP 1: Normalize segmentation to compressed RLE FIRST
        # This avoids duplicate normalization in bbox/area computation
        if has_valid_seg:
            # Check if already compressed RLE (bytes counts)
            if isinstance(seg, dict) and isinstance(seg.get('counts'), bytes):
                pass
            else:
                seg = _normalize_to_rle(seg, height, width)
                ann['segmentation'] = seg
                seg_normalized += 1
                logger.debug("Normalized segmentation for annotation %s", ann['id'])

        # STEP 2: Generate bbox from normalized segmentation (no duplicate normalization)
        has_valid_bbox = has_valid_list_field(ann, 'bbox')
        if has_valid_seg and not has_valid_bbox:
            bbox = compute_bbox_from_mask(seg, height, width)
            ann['bbox'] = bbox
            bbox_generated += 1
            logger.debug("Generated bbox for annotation %s", ann['id'])

        # STEP 3: Generate area from normalized segmentation
        has_valid_area = has_valid_numeric_field(ann, 'area', 0)
        if has_valid_seg and not has_valid_area:
            area = compute_area_from_mask(seg, height, width)
            ann['area'] = area
            area_generated += 1
            logger.debug("Generated area for annotation %s", ann['id'])

        # Compute area from bbox if still missing or invalid
        elif has_valid_bbox and not has_valid_area:
            bbox = ann['bbox']
            ann['area'] = bbox[2] * bbox[3]  # width * height
            area_generated += 1

    if bbox_generated > 0:
        logger.info("  Auto-generated %d bounding boxes from masks", bbox_generated)
    if area_generated > 0:
        logger.info("  Auto-computed %d areas", area_generated)
    if seg_normalized > 0:
        logger.info("  Normalized %d segmentations to compressed RLE", seg_normalized)
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
    - Suspicious bboxes (bbox area >> mask area, likely multiple objects in one annotation)

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
        'suspicious_bboxes': 0,  # Bbox area >> mask area (likely multiple objects in one annotation)
        'samples_per_class': {},
        'class_distribution': {},
        'warnings': []
    }

    # Build image_id to annotations(list) mapping
    image_to_anns = _build_image_to_annotations_mapping(coco_data['annotations'])

    # Create image_id to image dictionary lookup
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
        if has_valid_list_field(ann, 'bbox'):
            bbox = ann['bbox']
            bbox_area = bbox[2] * bbox[3]  # width * height

            # Get image dimensions
            img_info = image_lookup.get(ann['image_id'])
            img_area = img_info['width'] * img_info['height']

            # Small objects (< 1% of image) might be hard to detect
            if bbox_area < img_area * 0.01:
                results['small_objects'] += 1

            # Very large objects (> 80% of image) might indicate annotation errors
            if bbox_area > img_area * 0.8:
                results['large_objects'] += 1

        # Check for suspicious bboxes (bbox area >> actual mask area)
        # This likely means multiple disconnected objects in one annotation
        if has_valid_list_field(ann, 'bbox') and has_valid_list_field(ann, 'segmentation'):

            bbox = ann['bbox']
            bbox_area = bbox[2] * bbox[3]  # width * height

            # Only check if bbox is non-degenerate
            if bbox_area > 0:
                img_info = image_lookup.get(ann['image_id'])
                try:
                    mask_area = compute_area_from_mask(
                        ann['segmentation'],
                        img_info['height'],
                        img_info['width']
                    )
                    if mask_area > 0 and bbox_area > mask_area * 2.5:
                        results['suspicious_bboxes'] += 1
                except Exception:
                    # If mask area computation fails, skip this check
                    pass

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

    if results['suspicious_bboxes'] > 0:
        pct = 100 * results['suspicious_bboxes'] / results['total_annotations']
        results['warnings'].append(
            f"{results['suspicious_bboxes']} annotations ({pct:.1f}%) have suspicious bboxes "
            f"(bbox area >> mask area). This likely means multiple disconnected objects in one annotation. "
            f"COCO format requires: one object = one annotation. "
            f"Please split these into separate annotations in your labeling tool."
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
    Split COCO dataset into train/val/test with optional multi-label stratification.
    
    Stratification (if enabled):
    - Uses multi-label stratification (iterstrat library)
    - Maintains distribution of all label combinations across splits
    - Proper for object detection where images can have multiple classes
    - Falls back to random split if stratification fails (small dataset, etc.)
    
    Args:
        coco_data: COCO format dictionary
        splits: Dictionary with split ratios, e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15}
        stratify: Whether to use multi-label stratification
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing split datasets
    
    Example:
        >>> splits = split_dataset(
        >>>     coco_data,
        >>>     splits={'train': 0.7, 'val': 0.15, 'test': 0.15},
        >>>     stratify=True,  # Multi-label stratification
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
    Multi-label stratified split using iterative stratification.
    
    Each image can have multiple classes, and stratification maintains the distribution
    of all label combinations across splits.
    
    Falls back to random split if:
    - Dataset too small (< 10 images)
    - Not enough label diversity
    - Stratification algorithm fails
    """
    # Early validation: need reasonable dataset size
    n_images = len(image_with_ann_ids)
    if n_images < 10:
        logger.warning(
            "Dataset too small for stratification (%d images). "
            "Need at least 10 images. Falling back to random split.",
            n_images
        )
        return _random_split(image_with_ann_ids, splits, random_seed)

    # Get number of classes from annotations
    all_classes = set()
    for img_id in image_with_ann_ids:
        anns = image_to_anns[img_id]
        for ann in anns:
            all_classes.add(ann['category_id'])

    # num_classes = len(all_classes)
    max_class_id = max(all_classes)

    # Build multi-label indicator matrix
    # Each row = image, each column = class, value = 1 if class present in image
    label_matrix = np.zeros((n_images, max_class_id + 1), dtype=int)

    for idx, img_id in enumerate(image_with_ann_ids):
        anns = image_to_anns[img_id]
        for ann in anns:
            label_matrix[idx, ann['category_id']] = 1

    # Check for label diversity (need different label combinations)
    unique_patterns = np.unique(label_matrix, axis=0)
    if len(unique_patterns) < 2:
        logger.warning(
            "Insufficient label diversity: all images have identical label combinations. "
            "Falling back to random split."
        )
        return _random_split(image_with_ann_ids, splits, random_seed)

    # Perform multi-label stratified splitting
    try:
        X = np.arange(n_images).reshape(-1, 1)
        y = label_matrix

        # First split: train vs (val + test)
        train_size = splits.get('train')
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - train_size,
            random_state=random_seed
        )

        train_idx, temp_idx = next(msss.split(X, y))
        train_ids = {image_with_ann_ids[i] for i in train_idx}

        # Second split: val vs test
        if 'val' in splits and 'test' in splits:
            val_ratio = splits['val'] / (splits['val'] + splits['test'])
            X_temp = X[temp_idx]
            y_temp = y[temp_idx]

            msss_val = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=1 - val_ratio,
                random_state=random_seed
            )

            val_idx_local, test_idx_local = next(msss_val.split(X_temp, y_temp))

            # Map back to original image IDs
            temp_img_ids = [image_with_ann_ids[i] for i in temp_idx]
            val_ids = {temp_img_ids[i] for i in val_idx_local}
            test_ids = {temp_img_ids[i] for i in test_idx_local}
        else:
            val_ids = {image_with_ann_ids[i] for i in temp_idx}
            test_ids = set()

        logger.info(
            "Multi-label stratified split successful "
            f"(train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)})"
        )

        return {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }

    except Exception as e:
        logger.warning(
            "Multi-label stratification failed: %s. "
            "This can happen with very small datasets or unusual label distributions. "
            "Falling back to random split.",
            str(e)
        )
        return _random_split(image_with_ann_ids, splits, random_seed)


def _random_split(
    image_ids: List[int],
    splits: Dict[str, float],
    random_seed: int
) -> Dict[str, set]:
    """Perform random split without stratification.
    
    Uses round() instead of int() truncation to fairly allocate images.
    Example: 4 images with 70/15/15 split
      - int():   train=2, val=0, test=0  (loses 2 images, val/test empty!)
      - round(): train=2, val=1, test=1  (correct distribution)
    """
    np.random.seed(random_seed)

    shuffled_ids = np.array(image_ids)
    np.random.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)

    # Use round() for fair allocation instead of int() truncation
    split_sizes = {name: round(n_total * ratio) for name, ratio in splits.items()}

    # Adjust for rounding errors - modify largest split to absorb difference
    total_allocated = sum(split_sizes.values())
    if total_allocated != n_total:
        largest_split = max(split_sizes.keys(), key=lambda k: split_sizes[k])
        split_sizes[largest_split] += n_total - total_allocated

    # Create split IDs
    split_ids = {}
    current_idx = 0
    for split_name, n_split in split_sizes.items():
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
    Create split datasets and validate class distribution quality.
    
    Validates that each split:
    1. Contains all classes (warns if missing)
    2. Has reasonable class distribution (logs for debugging)
    
    This validation is important even with multi-label stratification because:
    - Very rare classes might still be missing from small splits
    - Stratification might fail and fall back to random split
    - Helps detect data quality issues early
    
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

    # Calculate overall class distribution for comparison
    total_class_counts = {}
    for ann in annotations:
        cat_id = ann['category_id']
        total_class_counts[cat_id] = total_class_counts.get(cat_id, 0) + 1

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

        # Optional: Log class distribution for this split (debugging)
        if logger.isEnabledFor(logging.DEBUG):
            split_class_counts = {}
            for ann in split_annotations:
                cat_id = ann['category_id']
                split_class_counts[cat_id] = split_class_counts.get(cat_id, 0) + 1

            logger.debug("Split '%s' class distribution:", split_name)
            for cat_id in sorted(all_category_ids):
                split_count = split_class_counts.get(cat_id, 0)
                total_count = total_class_counts.get(cat_id, 0)
                percentage = (split_count / total_count * 100) if total_count > 0 else 0
                cat_name = next((c['name'] for c in categories if c['id'] == cat_id), f"class_{cat_id}")
                logger.debug("  %s: %d (%.1f%% of total)", cat_name, split_count, percentage)

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
