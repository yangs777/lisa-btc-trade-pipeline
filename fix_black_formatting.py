#!/usr/bin/env python3
"""Apply Black formatting fixes to all Python files."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run black formatter on all Python files."""
    # Find all Python files
    python_files = []
    for pattern in ["src/**/*.py", "tests/**/*.py"]:
        python_files.extend(Path(".").glob(pattern))
    
    if not python_files:
        print("No Python files found!")
        return 1
    
    print(f"Found {len(python_files)} Python files to format")
    
    # Try to install black if not available
    try:
        subprocess.run([sys.executable, "-m", "black", "--version"], 
                      check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Black not found, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "black"],
                      check=True)
    
    # Run black
    cmd = [sys.executable, "-m", "black", "src", "tests"]
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())