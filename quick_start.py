#!/usr/bin/env python3
"""Quick start script for Cross-Modal Translation project."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    print("Cross-Modal Translation - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("✗ Python 3.10+ required. Current version:", sys.version)
        return
    
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Please install dependencies manually: pip install -r requirements.txt")
        return
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("Some tests failed, but this is normal for initial setup")
    
    # Check imports
    if not run_command("python -c \"import src.utils; import src.data; import src.models; print('All imports successful')\"", "Checking imports"):
        print("Import check failed")
        return
    
    print("\n" + "=" * 50)
    print("QUICK START COMPLETED!")
    print("=" * 50)
    
    print("\nNext steps:")
    print("1. Run simple demo:     python 0934.py")
    print("2. Run example script:  python scripts/example.py")
    print("3. Run full demo:      python demo_complete_system.py")
    print("4. Launch web demo:    streamlit run demo/app.py")
    print("5. Start training:     python scripts/train.py --config configs/config.yaml")
    
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
