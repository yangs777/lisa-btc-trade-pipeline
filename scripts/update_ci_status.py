#!/usr/bin/env python3
"""Update CI status report with current coverage and test metrics."""

import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


def get_coverage_stats():
    """Extract coverage statistics from coverage.xml."""
    coverage_file = Path("coverage.xml")
    if not coverage_file.exists():
        return None
    
    tree = ET.parse(coverage_file)
    root = tree.getroot()
    
    return {
        "line_rate": float(root.get("line-rate", 0)),
        "lines_valid": int(root.get("lines-valid", 0)),
        "lines_covered": int(root.get("lines-covered", 0)),
        "timestamp": datetime.fromtimestamp(int(root.get("timestamp", 0)) / 1000).isoformat()
    }


def get_test_stats():
    """Get test statistics by running pytest."""
    try:
        result = subprocess.run(
            ["pytest", "--tb=short", "-q"],
            capture_output=True,
            text=True
        )
        
        # Parse pytest output
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if "passed" in line or "failed" in line:
                parts = line.split()
                passed = 0
                failed = 0
                for part in parts:
                    if "passed" in part:
                        passed = int(part.replace("passed", "").strip())
                    elif "failed" in part:
                        failed = int(part.replace("failed", "").strip())
                
                return {
                    "total": passed + failed,
                    "passed": passed,
                    "failed": failed,
                    "success_rate": passed / (passed + failed) if (passed + failed) > 0 else 0
                }
    except Exception as e:
        print(f"Error running pytest: {e}")
    
    return None


def update_status_report():
    """Update the CI status report."""
    coverage_stats = get_coverage_stats()
    test_stats = get_test_stats()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "coverage": {
            "percentage": coverage_stats["line_rate"] * 100 if coverage_stats else 0,
            "lines_covered": coverage_stats.get("lines_covered", 0),
            "lines_total": coverage_stats.get("lines_valid", 0)
        },
        "tests": test_stats or {"total": 0, "passed": 0, "failed": 0, "success_rate": 0},
        "status": "PASSING" if coverage_stats and coverage_stats["line_rate"] >= 0.20 else "FAILING"
    }
    
    # Write report
    report_path = Path("CI_STATUS_REPORT.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Also create markdown report
    md_report = f"""# CI Status Report

Generated: {report['timestamp']}

## Coverage Status
- **Coverage**: {report['coverage']['percentage']:.2f}%
- **Lines Covered**: {report['coverage']['lines_covered']:,} / {report['coverage']['lines_total']:,}
- **Status**: {report['status']}

## Test Status
- **Total Tests**: {report['tests']['total']}
- **Passed**: {report['tests']['passed']}
- **Failed**: {report['tests']['failed']}
- **Success Rate**: {report['tests']['success_rate']:.1%}

## Thresholds
- Coverage Threshold: 20% (Current: {report['coverage']['percentage']:.2f}%)
- Next Target: 35%
"""
    
    with open("CI_STATUS_REPORT.md", "w") as f:
        f.write(md_report)
    
    print(f"CI Status: {report['status']}")
    print(f"Coverage: {report['coverage']['percentage']:.2f}%")
    

if __name__ == "__main__":
    update_status_report()