"""
Utility functions for COCO data processing.
"""
from typing import Dict, Any, List

def has_valid_list_field(data: Dict[str, Any], field: str) -> bool:
    r"""
    Check if a field exists and has a valid list value.
    Valid means: key exists, value is not None, and value is not empty.

    Args:
        data: Dictionary containing the data to check.
        field: Field name to validate

    Returns:
        True if field has valid list value, False otherwise.

    Example:
        >>> ann = {'bbox': [10, 10, 50, 50]}
        >>> has_valid_list_field(ann, 'bbox') # True
        >>>
        >>> ann = {'bbox': None}
        >>> has_valid_list_field(ann, 'bbox') # False
        >>>
        >>> ann = {'bbox': []}
        >>> has_valid_list_field(ann, 'bbox') # False
        >>>
    """
    return (field in data and
            data[field] is not None and
            data[field] != [])

def has_valid_numeric_field(data: Dict, field: str, min_value: float = 0) -> bool:
    """
    Check if a field exists and has a valid numeric value.
    
    Valid means: key exists, value is not None, and value > min_value.
    
    Args:
        data: Dictionary to check
        field: Field name to validate
        min_value: Minimum valid value (exclusive)
    
    Returns:
        True if field has valid numeric value, False otherwise
    
    Example:
        >>> ann = {'area': 150.5}
        >>> has_valid_numeric_field(ann, 'area')  # True
        >>> 
        >>> ann = {'area': 0}
        >>> has_valid_numeric_field(ann, 'area')  # False
        >>> 
        >>> ann = {'area': None}
        >>> has_valid_numeric_field(ann, 'area')  # False
    """
    return (field in data and
            data[field] is not None and
            data[field] > min_value)

# def build_id_lookup(items: List[Dict], id_field: str = 'id') -> Dict[Any, Dict]:
#     """
#     Build lookup dictionary from list of items by ID field.

#     Args:
#         items: List of dictionaries
#         id_field: Name of the ID field (default: 'id')

#     Returns:
#         Dictionary mapping ID → item

#     Example:
#         >>> images = [{'id': 1, 'width': 640}, {'id': 2, 'width': 800}]
#         >>> lookup = build_id_lookup(images)
#         >>> lookup[1]  # {'id': 1, 'width': 640}
#     """
#     return {item[id_field]: item for item in items}


# def build_image_to_annotations_mapping(annotations: List[Dict]) -> Dict[int, List[Dict]]:
#     """
#     Build mapping from image_id to list of annotations.

#     Args:
#         annotations: List of annotation dictionaries

#     Returns:
#         Dictionary mapping image_id → list of annotations

#     Example:
#         >>> annotations = [
#         ...     {'id': 1, 'image_id': 0, 'bbox': [...]},
#         ...     {'id': 2, 'image_id': 0, 'bbox': [...]},
#         ...     {'id': 3, 'image_id': 1, 'bbox': [...]}
#         ... ]
#         >>> mapping = build_image_to_annotations_mapping(annotations)
#         >>> len(mapping[0])  # 2 (two annotations for image 0)
#     """
#     image_to_anns = {}
#     for ann in annotations:
#         image_id = ann['image_id']
#         if image_id not in image_to_anns:
#             image_to_anns[image_id] = []
#         image_to_anns[image_id].append(ann)
#     return image_to_anns
