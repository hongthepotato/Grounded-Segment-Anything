import logging
from typing import Dict, Any, Set, List, Optional
from pathlib import Path
import os
import random
from collections import defaultdict
import yaml

logger = logging.getLogger(__name__)


"""
COCO dataset merger for combining GT labels with teacher pseudo-labels.
"""

def merge_coco_datasets(
    labeled_coco: Dict[str, Any],
    pseudo_coco: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge a ground-truth COCO dataset with teacher-generated pseudo-labels.

    Images that appear in both datasets (matched by file_name) keep only GT
    annotations. Pseudo-labels are added only for images not in the GT set.
    All IDs are re-indexed to avoid collisions.

    Args:
        labeled_coco: Original ground-truth COCO dict
        pseudo_coco: Teacher-generated COCO dict

    Returns:
        Merged COCO dict with unified images, annotations, and categories
    """
    gt_filenames: Set[str] = {
        img['file_name'] for img in labeled_coco.get('images', [])
    }

    categories = labeled_coco.get('categories', [])
    cat_names_gt = {c['name'] for c in categories}
    for cat in pseudo_coco.get('categories', []):
        if cat['name'] not in cat_names_gt:
            categories.append(cat)
            cat_names_gt.add(cat['name'])

    merged_images = []
    merged_annotations = []
    next_image_id = 1
    next_ann_id = 1

    old_to_new_image = {}

    for img in labeled_coco.get('images', []):
        new_id = next_image_id
        old_to_new_image[(img['id'], 'gt')] = new_id
        merged_images.append({**img, 'id': new_id})
        next_image_id += 1

    for ann in labeled_coco.get('annotations', []):
        new_img_id = old_to_new_image.get((ann['image_id'], 'gt'))
        if new_img_id is None:
            continue
        merged_annotations.append({
            **ann,
            'id': next_ann_id,
            'image_id': new_img_id,
        })
        next_ann_id += 1

    pseudo_added = 0
    for img in pseudo_coco.get('images', []):
        if img['file_name'] in gt_filenames:
            continue
        new_id = next_image_id
        old_to_new_image[(img['id'], 'pseudo')] = new_id
        merged_images.append({**img, 'id': new_id})
        next_image_id += 1
        pseudo_added += 1

    for ann in pseudo_coco.get('annotations', []):
        new_img_id = old_to_new_image.get((ann['image_id'], 'pseudo'))
        if new_img_id is None:
            continue
        merged_annotations.append({
            **ann,
            'id': next_ann_id,
            'image_id': new_img_id,
        })
        next_ann_id += 1

    logger.info(
        "Merged datasets: %d GT images + %d pseudo images = %d total, %d annotations",
        len(gt_filenames), pseudo_added, len(merged_images), len(merged_annotations)
    )

    return {
        'images': merged_images,
        'annotations': merged_annotations,
        'categories': categories,
    }


"""
COCO to YOLO segmentation format converter.

Converts COCO JSON annotations to the YOLO-seg label format that
ultralytics expects for instance segmentation training.
"""

def _find_image_file(file_name: str, source_dirs: List[str]) -> Optional[str]:
    """Locate an image across multiple source directories."""
    for src in source_dirs:
        candidate = Path(src) / file_name
        if candidate.exists():
            return str(candidate)
        for child in Path(src).rglob(os.path.basename(file_name)):
            if child.is_file():
                return str(child)
    return None


def _polygon_to_yolo_seg(
    polygon: List[float],
    img_width: int,
    img_height: int,
) -> Optional[List[float]]:
    """Normalize a COCO polygon [x1,y1,x2,y2,...] to YOLO 0-1 coords."""
    if len(polygon) < 6:
        return None
    normalized = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / img_width
        y = polygon[i + 1] / img_height
        normalized.extend([round(x, 6), round(y, 6)])
    return normalized


def convert_coco_to_yolo_seg(
    coco_data: Dict[str, Any],
    image_source_dirs: List[str],
    output_dir: str,
    split_ratios: Optional[Dict[str, float]] = None,
    class_names: Optional[List[str]] = None,
    seed: int = 42,
) -> str:
    """
    Convert COCO annotations to YOLO segmentation format.

    Creates the directory structure ultralytics expects:
        output_dir/
            images/train/  images/val/  [images/test/]
            labels/train/  labels/val/  [labels/test/]
            data.yaml

    Args:
        coco_data: Merged COCO dict (images, annotations, categories)
        image_source_dirs: Directories to search for image files
        output_dir: Root output directory
        split_ratios: e.g. {'train': 0.7, 'val': 0.15, 'test': 0.15}
        class_names: Override category names (use COCO categories if None)
        seed: Random seed for splitting

    Returns:
        Absolute path to the generated data.yaml
    """
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

    out = Path(output_dir)
    categories = coco_data.get('categories', [])
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])

    if class_names is None:
        class_names = [c['name'] for c in sorted(categories, key=lambda c: c['id'])]

    cat_id_to_idx = {c['id']: idx for idx, c in enumerate(sorted(categories, key=lambda c: c['id']))}

    anns_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann['image_id']].append(ann)

    random.seed(seed)
    shuffled = list(images)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * split_ratios.get('train', 0.7))
    n_val = int(n * split_ratios.get('val', 0.15))

    splits = {}
    for img in shuffled[:n_train]:
        splits[img['id']] = 'train'
    for img in shuffled[n_train:n_train + n_val]:
        splits[img['id']] = 'val'
    for img in shuffled[n_train + n_val:]:
        splits[img['id']] = 'test'

    split_names = sorted(set(splits.values()))
    for split in split_names:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for img in images:
        img_id = img['id']
        split = splits.get(img_id, 'train')
        file_name = img['file_name']
        w, h = img['width'], img['height']

        src_path = _find_image_file(file_name, image_source_dirs)
        if src_path is None:
            logger.warning("Image not found: %s", file_name)
            skipped += 1
            continue

        dst_image = out / 'images' / split / os.path.basename(file_name)
        if not dst_image.exists():
            try:
                os.symlink(os.path.abspath(src_path), str(dst_image))
            except OSError:
                import shutil
                shutil.copy2(src_path, str(dst_image))

        img_anns = anns_by_image.get(img_id, [])
        label_name = Path(os.path.basename(file_name)).stem + '.txt'
        label_path = out / 'labels' / split / label_name

        lines = []
        for ann in img_anns:
            cat_idx = cat_id_to_idx.get(ann.get('category_id'))
            if cat_idx is None:
                continue

            segs = ann.get('segmentation', [])
            if segs and isinstance(segs, list) and isinstance(segs[0], list):
                for polygon in segs:
                    yolo_poly = _polygon_to_yolo_seg(polygon, w, h)
                    if yolo_poly:
                        coords_str = ' '.join(str(c) for c in yolo_poly)
                        lines.append(f"{cat_idx} {coords_str}")

        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        converted += 1

    data_yaml = {
        'path': str(out.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)},
    }
    if 'test' in split_names:
        data_yaml['test'] = 'images/test'

    yaml_path = out / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info(
        "COCO->YOLO: %d images converted, %d skipped, %d classes, saved to %s",
        converted, skipped, len(class_names), yaml_path
    )

    return str(yaml_path.resolve())
