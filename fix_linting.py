#!/usr/bin/env python3
"""Fix linting issues in the codebase."""

import re
from pathlib import Path


def fix_whitespace_in_docstrings(content: str) -> str:
    """Remove trailing whitespace from blank lines in docstrings."""
    lines = content.split('\n')
    in_docstring = False
    quote_style = None
    fixed_lines = []
    
    for line in lines:
        # Check for docstring start/end
        if '"""' in line:
            if not in_docstring:
                in_docstring = True
                quote_style = '"""'
            elif quote_style == '"""':
                in_docstring = False
        elif "'''" in line:
            if not in_docstring:
                in_docstring = True
                quote_style = "'''"
            elif quote_style == "'''":
                in_docstring = False
        
        # Fix whitespace on blank lines inside docstrings
        if in_docstring and line.strip() == '' and len(line) > 0:
            fixed_lines.append('')
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_file(file_path: Path) -> bool:
    """Fix linting issues in a single file."""
    try:
        content = file_path.read_text()
        original = content
        
        # Fix whitespace in docstrings
        content = fix_whitespace_in_docstrings(content)
        
        if content != original:
            file_path.write_text(content)
            print(f"Fixed: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Fix all Python files in src directory."""
    src_dir = Path("src")
    if not src_dir.exists():
        print("src directory not found!")
        return
    
    fixed_count = 0
    for py_file in src_dir.rglob("*.py"):
        if fix_file(py_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()