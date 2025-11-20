"""
Quick import test to verify all modules are loadable.

Run this after installation to ensure everything is working.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test importing all major modules."""
    
    print("Testing imports...")
    print("-" * 60)
    
    tests = []
    
    # Core modules
    try:
        from core.config import load_config, generate_config
        from core.logger import setup_logger
        from core.constants import PROJECT_ROOT
        tests.append(("core", True))
        print("[PASS] core modules")
    except Exception as e:
        tests.append(("core", False))
        print(f"[FAIL] core modules: {e}")
    
    # Data modules
    try:
        from ml_engine.data import (
            inspect_dataset,
            COCODataset,
            MultiModelPreprocessor,
            validate_coco_format
        )
        tests.append(("ml_engine.data", True))
        print("[PASS] ml_engine.data")
    except Exception as e:
        tests.append(("ml_engine.data", False))
        print(f"[FAIL] ml_engine.data: {e}")
    
    # Training modules
    try:
        from ml_engine.training import (
            TeacherTrainer,
            TrainingManager,
            CheckpointManager,
            apply_lora,
            verify_freezing
        )
        tests.append(("ml_engine.training", True))
        print("[PASS] ml_engine.training")
    except Exception as e:
        tests.append(("ml_engine.training", False))
        print(f"[FAIL] ml_engine.training: {e}")
    
    # Model modules
    try:
        from ml_engine.models.teacher import (
            GroundingDINOLoRA,
            SAMLoRA,
            load_grounding_dino_with_lora,
            load_sam_with_lora
        )
        tests.append(("ml_engine.models.teacher", True))
        print("[PASS] ml_engine.models.teacher")
    except Exception as e:
        tests.append(("ml_engine.models.teacher", False))
        print(f"[FAIL] ml_engine.models.teacher: {e}")
    
    # Augmentation
    try:
        from augmentation import get_augmentation_registry
        tests.append(("augmentation", True))
        print("[PASS] augmentation")
    except Exception as e:
        tests.append(("augmentation", False))
        print(f"[FAIL] augmentation: {e}")
    
    print("-" * 60)
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    print(f"\nResults: {passed}/{total} modules imported successfully")
    
    if passed == total:
        print("[SUCCESS] All imports successful!")
        return True
    else:
        print("[FAILED] Some imports failed")
        return False


if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)

