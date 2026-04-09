"""
ZIP export for Ultralytics YOLO student weights (distillation / yolo_seg_labeled).

Mirrors the teacher ``exports/*_package.zip`` layout so GET /api/jobs/{id}/export
with ``format=merged_pth`` returns a real ZIP for student runs.
"""

import logging
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_student_yolo_export_zip(output_dir: Path) -> Optional[Path]:
    """
    Write ``{output_dir}/exports/student_model_package.zip`` containing:

    - ``best.pt`` (from ``student_model/best.pt``)
    - ``data.yaml`` if ``yolo_dataset/data.yaml`` exists (class names / data config)

    Returns:
        Path to the ZIP, or None if ``student_model/best.pt`` is missing.
    """
    output_dir = Path(output_dir)
    best_pt = output_dir / "student_model" / "best.pt"
    if not best_pt.exists():
        logger.warning("Skipping student export zip: missing %s", best_pt)
        return None

    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    zip_path = exports_dir / "student_model_package.zip"

    data_yaml = output_dir / "yolo_dataset" / "data.yaml"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(best_pt, "best.pt")
        if data_yaml.exists():
            zf.write(data_yaml, "data.yaml")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info("Student YOLO export package: %s (%.1f MB)", zip_path, size_mb)
    return zip_path
