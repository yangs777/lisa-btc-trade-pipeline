#!/usr/bin/env python3
"""
Generate regression test from a test failure.

This script analyzes test failure logs and creates a regression test
that reproduces the issue to prevent future regressions.
"""
import argparse
import ast
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def parse_pytest_traceback(traceback: str) -> Dict[str, Any]:
    """Parse pytest traceback to extract failure information."""
    info = {
        "test_file": None,
        "test_function": None,
        "line_number": None,
        "error_type": None,
        "error_message": None,
        "assertion": None,
        "locals": {}
    }
    
    # Extract test file and function
    test_pattern = r"(test_\w+\.py):(\d+): in (\w+)"
    match = re.search(test_pattern, traceback)
    if match:
        info["test_file"] = match.group(1)
        info["line_number"] = int(match.group(2))
        info["test_function"] = match.group(3)
    
    # Extract error type and message
    error_pattern = r"(\w+Error): (.+?)(?:\n|$)"
    match = re.search(error_pattern, traceback)
    if match:
        info["error_type"] = match.group(1)
        info["error_message"] = match.group(2)
    
    # Extract assertion failure
    assert_pattern = r"assert (.+?)(?:\n|$)"
    match = re.search(assert_pattern, traceback)
    if match:
        info["assertion"] = match.group(1)
    
    # Extract local variables (simplified)
    locals_pattern = r"(\w+) = (.+?)(?:\n|$)"
    for match in re.finditer(locals_pattern, traceback):
        var_name = match.group(1)
        var_value = match.group(2)
        info["locals"][var_name] = var_value
    
    return info


def generate_regression_test(info: Dict[str, Any], test_name: Optional[str] = None) -> str:
    """Generate regression test code from failure information."""
    if not test_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = f"test_regression_{timestamp}"
    
    # Generate imports
    imports = [
        "import pytest",
        "from unittest.mock import Mock, patch",
    ]
    
    if info["test_file"]:
        # Try to infer module imports from test file
        module_path = Path(info["test_file"]).stem
        if module_path.startswith("test_"):
            module_name = module_path[5:]  # Remove 'test_' prefix
            imports.append(f"# TODO: Import actual module under test")
            imports.append(f"# from src.{module_name} import ...")
    
    # Generate test function
    test_code = f"""
def {test_name}():
    \"\"\"
    Regression test for {info.get('error_type', 'failure')} in {info.get('test_function', 'unknown')}.
    
    Original error: {info.get('error_message', 'Unknown error')}
    Generated from failure on: {datetime.now().isoformat()}
    \"\"\"
    # Setup - reproduce conditions that caused the failure"""
    
    # Add local variables if available
    if info["locals"]:
        test_code += "\n    # Local variables from failure:"
        for var, value in info["locals"].items():
            # Try to safely represent the value
            try:
                # Check if it's a valid Python literal
                ast.literal_eval(value)
                test_code += f"\n    {var} = {value}"
            except:
                test_code += f"\n    {var} = ...  # Original value: {value}"
    
    # Add the assertion that failed
    if info["assertion"]:
        test_code += f"\n    \n    # This assertion failed:"
        test_code += f"\n    # assert {info['assertion']}"
        test_code += f"\n    \n    # TODO: Fix the test data or logic to make this pass"
        test_code += f"\n    with pytest.raises({info.get('error_type', 'Exception')}):"
        test_code += f"\n        # Reproduce the failure"
        test_code += f"\n        pass  # TODO: Add actual test code"
    else:
        test_code += f"\n    \n    # TODO: Add test implementation"
        test_code += f"\n    # The original test failed with: {info.get('error_type', 'Unknown')}"
        test_code += f"\n    pass"
    
    # Combine everything
    full_code = "\n".join(imports) + "\n\n" + test_code + "\n"
    
    return full_code


def create_regression_file(
    traceback: str,
    output_dir: Path = Path("tests/regression"),
    test_name: Optional[str] = None
) -> Path:
    """Create a regression test file from a traceback."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse the traceback
    info = parse_pytest_traceback(traceback)
    
    # Generate test code
    test_code = generate_regression_test(info, test_name)
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_regression_{timestamp}.py"
    if info["test_function"]:
        filename = f"test_regression_{info['test_function']}_{timestamp}.py"
    
    output_path = output_dir / filename
    
    # Write the file
    output_path.write_text(test_code)
    
    print(f"✅ Generated regression test: {output_path}")
    print(f"   Error type: {info.get('error_type', 'Unknown')}")
    print(f"   Original test: {info.get('test_function', 'Unknown')}")
    
    return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate regression test from pytest failure"
    )
    parser.add_argument(
        "traceback_file",
        nargs="?",
        help="File containing pytest traceback (or stdin)"
    )
    parser.add_argument(
        "--output-dir",
        default="tests/regression",
        help="Output directory for regression tests"
    )
    parser.add_argument(
        "--test-name",
        help="Custom name for the test function"
    )
    parser.add_argument(
        "--from-ci",
        action="store_true",
        help="Parse from CI log format"
    )
    
    args = parser.parse_args()
    
    # Read traceback
    if args.traceback_file and args.traceback_file != "-":
        traceback = Path(args.traceback_file).read_text()
    else:
        traceback = sys.stdin.read()
    
    if not traceback.strip():
        print("❌ No traceback provided")
        sys.exit(1)
    
    # Create regression test
    output_path = create_regression_file(
        traceback,
        Path(args.output_dir),
        args.test_name
    )
    
    print(f"\n💡 Next steps:")
    print(f"1. Review and edit: {output_path}")
    print(f"2. Add proper imports and test implementation")
    print(f"3. Run: pytest {output_path}")
    print(f"4. Commit when test passes")


if __name__ == "__main__":
    main()