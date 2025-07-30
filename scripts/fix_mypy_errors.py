#!/usr/bin/env python3
"""Fix common mypy errors automatically."""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple


def get_mypy_errors() -> List[str]:
    """Run mypy and capture errors."""
    result = subprocess.run(
        ["mypy", "src", "--no-error-summary"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().split("\n") if result.stdout else []


def fix_import_errors(file_path: str, content: str) -> str:
    """Fix import type errors."""
    # Add type: ignore for common untyped imports
    untyped_imports = {
        "yaml": "import yaml  # type: ignore[import-untyped]",
        "ccxt": "import ccxt  # type: ignore[import-untyped]",
        "ta": "import ta  # type: ignore[import-untyped]",
        "pandas_ta": "import pandas_ta  # type: ignore[import-untyped]",
        "binance": "from binance import Client  # type: ignore[import-untyped]",
        "gymnasium": "import gymnasium  # type: ignore[import-untyped]",
        "stable_baselines3": "from stable_baselines3 import SAC  # type: ignore[import-untyped]",
        "shap": "import shap  # type: ignore[import-untyped]",
        "numba": "from numba import jit  # type: ignore[import-untyped]",
        "prometheus_client": "import prometheus_client  # type: ignore[import-untyped]",
        "grafana_api": "from grafana_api import GrafanaFace  # type: ignore[import-untyped]",
    }
    
    for module, replacement in untyped_imports.items():
        # Simple import
        content = re.sub(
            rf"^import {module}$",
            f"import {module}  # type: ignore[import-untyped]",
            content,
            flags=re.MULTILINE
        )
        # From import
        content = re.sub(
            rf"^from {module} import",
            f"from {module} import  # type: ignore[import-untyped]",
            content,
            flags=re.MULTILINE
        )
    
    return content


def fix_type_annotations(file_path: str, content: str) -> str:
    """Fix common type annotation issues."""
    # Fix dict -> Dict[str, Any]
    if "from typing import" not in content:
        content = "from typing import Dict, List, Any, Optional, Union, Tuple\n\n" + content
    
    # Replace untyped dict/list
    content = re.sub(r"\bdict\b(?!\[)", "Dict[str, Any]", content)
    content = re.sub(r"\blist\b(?!\[)", "List[Any]", content)
    
    # Fix asyncio.Task without type parameter
    content = re.sub(r"asyncio\.Task(?!\[)", "asyncio.Task[Any]", content)
    
    return content


def fix_unused_ignore(file_path: str, content: str) -> str:
    """Fix unused type: ignore comments."""
    # Replace broad [import] with specific [import-untyped]
    content = re.sub(
        r"# type: ignore\[import\]",
        "# type: ignore[import-untyped]",
        content
    )
    
    return content


def process_file(file_path: Path) -> bool:
    """Process a single Python file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content
        
        # Apply fixes
        content = fix_import_errors(str(file_path), content)
        content = fix_type_annotations(str(file_path), content)
        content = fix_unused_ignore(str(file_path), content)
        
        # Write back if changed
        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            print(f"Fixed: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function."""
    src_dir = Path("src")
    if not src_dir.exists():
        print("Error: src directory not found")
        return
    
    # Get all Python files
    python_files = list(src_dir.rglob("*.py"))
    
    # Process files
    fixed_count = 0
    for file_path in python_files:
        if process_file(file_path):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")
    
    # Run mypy again to check
    print("\nRunning mypy to verify fixes...")
    result = subprocess.run(
        ["mypy", "src", "--no-error-summary"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ All mypy errors fixed!")
    else:
        print("⚠️ Some errors remain:")
        print(result.stdout)


if __name__ == "__main__":
    main()