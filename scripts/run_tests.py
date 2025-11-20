"""
Test runner script for the platform.

Runs all unit and integration tests with coverage reporting.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_tests(test_type='all', verbose=True):
    """
    Run tests.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'all')
        verbose: Verbose output
    """
    print("=" * 60)
    print("Running Tests")
    print("=" * 60)
    
    if test_type in ['unit', 'all']:
        print("\n🧪 Running unit tests...")
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/unit/',
            '-v' if verbose else '',
            '--tb=short'
        ]
        result = subprocess.run([c for c in cmd if c], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print("❌ Unit tests failed")
            return False
    
    if test_type in ['integration', 'all']:
        print("\n🔗 Running integration tests...")
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/integration/',
            '-v' if verbose else '',
            '--tb=short'
        ]
        result = subprocess.run([c for c in cmd if c], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print("❌ Integration tests failed")
            return False
    
    print("\n✅ All tests passed!")
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run tests')
    parser.add_argument('--type', choices=['unit', 'integration', 'all'],
                        default='all', help='Type of tests to run')
    parser.add_argument('--quiet', action='store_true',
                        help='Less verbose output')
    
    args = parser.parse_args()
    
    success = run_tests(test_type=args.type, verbose=not args.quiet)
    sys.exit(0 if success else 1)


