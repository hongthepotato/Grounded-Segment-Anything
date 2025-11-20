"""
Verification script for platform setup.

Checks:
- Python version
- Required packages installed
- Directory structure
- Config files present
- CUDA availability
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False


def check_packages():
    """Check required packages."""
    print("\n📦 Checking required packages...")
    
    required_packages = [
        'torch',
        'torchvision',
        'peft',
        'transformers',
        'albumentations',
        'tensorboard',
        'pycocotools',
        'PIL',
        'numpy',
        'yaml',
        'tqdm',
        'sklearn'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'yaml':
                import yaml
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (not installed)")
            all_installed = False
    
    return all_installed


def check_cuda():
    """Check CUDA availability."""
    print("\n🎮 Checking CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU memory: {memory:.1f} GB")
            return True
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
            return False
    except ImportError:
        print(f"  ✗ PyTorch not installed")
        return False


def check_directory_structure():
    """Check directory structure."""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        'ml_engine/data',
        'ml_engine/models/teacher',
        'ml_engine/training',
        'augmentation',
        'cli',
        'core',
        'configs/defaults',
        'tests/unit',
        'tests/integration',
        'data/raw',
        'data/models/pretrained',
        'experiments'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = PROJECT_ROOT / dir_path
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def check_config_files():
    """Check config files."""
    print("\n⚙️  Checking config files...")
    
    required_configs = [
        'configs/defaults/preprocessing.yaml',
        'configs/defaults/training_dynamics.yaml',
        'configs/defaults/checkpoint_config.yaml',
        'configs/defaults/teacher_grounding_dino_lora.yaml',
        'configs/defaults/teacher_sam_lora.yaml'
    ]
    
    all_exist = True
    for config_path in required_configs:
        path = PROJECT_ROOT / config_path
        if path.exists():
            print(f"  ✓ {config_path}")
        else:
            print(f"  ✗ {config_path} (missing)")
            all_exist = False
    
    return all_exist


def check_cli_scripts():
    """Check CLI scripts."""
    print("\n🖥️  Checking CLI scripts...")
    
    required_scripts = [
        'cli/validate_dataset.py',
        'cli/train_teacher.py',
        'cli/utils.py'
    ]
    
    all_exist = True
    for script_path in required_scripts:
        path = PROJECT_ROOT / script_path
        if path.exists():
            print(f"  ✓ {script_path}")
        else:
            print(f"  ✗ {script_path} (missing)")
            all_exist = False
    
    return all_exist


def check_imports():
    """Check that modules can be imported."""
    print("\n🔍 Checking module imports...")
    
    modules_to_import = [
        ('ml_engine.data', 'inspect_dataset'),
        ('ml_engine.data', 'COCODataset'),
        ('ml_engine.training', 'TeacherTrainer'),
        ('ml_engine.training', 'apply_lora'),
        ('core.config', 'load_config'),
        ('augmentation', 'get_augmentation_registry')
    ]
    
    all_imported = True
    for module, attr in modules_to_import:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            print(f"  ✓ {module}.{attr}")
        except Exception as e:
            print(f"  ✗ {module}.{attr} ({e})")
            all_imported = False
    
    return all_imported


def main():
    """Main verification."""
    print("=" * 60)
    print("Platform Setup Verification")
    print("=" * 60)
    
    results = {
        'Python version': check_python_version(),
        'Packages': check_packages(),
        'CUDA': check_cuda(),
        'Directory structure': check_directory_structure(),
        'Config files': check_config_files(),
        'CLI scripts': check_cli_scripts(),
        'Module imports': check_imports()
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All checks passed! Platform is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your COCO dataset in data/raw/")
        print("2. Run: python cli/validate_dataset.py --data data/raw/annotations.json")
        print("3. Run: python cli/train_teacher.py --data train.json --val val.json --output exp1")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())


