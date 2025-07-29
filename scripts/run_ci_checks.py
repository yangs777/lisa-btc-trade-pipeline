#!/usr/bin/env python3
"""Run CI checks for the project."""

import subprocess
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} passed!")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} failed!")
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    """Run all CI checks."""
    print("🚀 Running CI checks for BTC/USDT τ-SAC Trading System")
    
    # List of checks to run
    checks = [
        (["python3", "-m", "ruff", "check", "src/", "tests/"], "Linting (ruff)"),
        (["python3", "-m", "black", "--check", "src/", "tests/"], "Code formatting (black)"),
        (["python3", "-m", "isort", "--check-only", "src/", "tests/"], "Import sorting (isort)"),
        (["python3", "-m", "mypy", "src/", "tests/"], "Type checking (mypy)"),
        (["python3", "-m", "pytest", "tests/", "-v"], "Unit tests"),
        (["python3", "-m", "bandit", "-r", "src/", "-ll"], "Security scan (bandit)"),
    ]
    
    failed_checks = []
    
    for cmd, description in checks:
        if not run_command(cmd, description):
            failed_checks.append(description)
    
    print("\n" + "="*60)
    print("CI CHECK SUMMARY")
    print("="*60)
    
    if failed_checks:
        print(f"❌ {len(failed_checks)} check(s) failed:")
        for check in failed_checks:
            print(f"  - {check}")
        sys.exit(1)
    else:
        print("✅ All CI checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()