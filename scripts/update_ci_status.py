#!/usr/bin/env python3
"""Update CI status in README and generate reports."""

import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def parse_coverage_xml(coverage_file: Path) -> float:
    """Parse coverage.xml file to extract coverage percentage."""
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        
        # Get coverage from root element
        coverage = float(root.attrib.get('line-rate', 0)) * 100
        return round(coverage, 2)
    except Exception as e:
        print(f"Error parsing coverage.xml: {e}")
        return 0.0


def update_readme_badges(readme_path: Path, coverage: float, ci_status: str = "passing") -> None:
    """Update badges in README.md file."""
    if not readme_path.exists():
        print(f"README not found at {readme_path}")
        return
    
    content = readme_path.read_text()
    
    # Update CI status badge
    ci_badge_pattern = r'!\[CI Status\]\(https://img\.shields\.io/badge/CI-\w+-\w+\)'
    ci_badge_new = f'![CI Status](https://img.shields.io/badge/CI-{ci_status}-{"green" if ci_status == "passing" else "red"})'
    content = re.sub(ci_badge_pattern, ci_badge_new, content)
    
    # Update coverage badge
    coverage_badge_pattern = r'!\[Coverage\]\(https://img\.shields\.io/badge/Coverage-[\d.]+%25-\w+\)'
    coverage_color = "green" if coverage >= 80 else "yellow" if coverage >= 60 else "orange" if coverage >= 40 else "red"
    coverage_badge_new = f'![Coverage](https://img.shields.io/badge/Coverage-{coverage}%25-{coverage_color})'
    content = re.sub(coverage_badge_pattern, coverage_badge_new, content)
    
    # If badges don't exist, add them at the top
    if not re.search(ci_badge_pattern, content):
        badges = f"\n{ci_badge_new} {coverage_badge_new}\n\n"
        lines = content.split('\n')
        if lines[0].startswith('#'):
            # Insert after title
            lines.insert(1, badges)
        else:
            lines.insert(0, badges)
        content = '\n'.join(lines)
    
    readme_path.write_text(content)
    print(f"Updated README badges: CI={ci_status}, Coverage={coverage}%")


def generate_coverage_report(coverage: float, output_path: Path) -> Dict[str, Any]:
    """Generate coverage report in JSON format."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "coverage": {
            "percentage": coverage,
            "status": "passing" if coverage >= 35 else "failing",
            "threshold": 35
        },
        "trend": {
            "previous": 32.66,  # From Phase 1
            "current": coverage,
            "change": round(coverage - 32.66, 2)
        },
        "milestones": {
            "phase1": {
                "target": 25,
                "achieved": 32.66,
                "status": "completed"
            },
            "phase2": {
                "target": 50,
                "achieved": coverage,
                "status": "completed" if coverage >= 50 else "in_progress"
            }
        }
    }
    
    output_path.write_text(json.dumps(report, indent=2))
    return report


def generate_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    """Generate human-readable markdown report."""
    coverage = report['coverage']['percentage']
    trend = report['trend']
    
    content = f"""# CI Status Report

Generated: {report['timestamp']}

## Coverage Summary

- **Current Coverage**: {coverage}% {':white_check_mark:' if coverage >= 35 else ':x:'}
- **CI Threshold**: {report['coverage']['threshold']}%
- **Status**: {report['coverage']['status'].upper()}

## Coverage Trend

- Previous: {trend['previous']}%
- Current: {trend['current']}%
- Change: {'+' if trend['change'] >= 0 else ''}{trend['change']}%

## Milestones Progress

### Phase 1: Foundation (Target: 25%)
- Achieved: {report['milestones']['phase1']['achieved']}%
- Status: ✅ Completed

### Phase 2: Business Logic (Target: 50%)
- Achieved: {report['milestones']['phase2']['achieved']}%
- Status: {'✅ Completed' if report['milestones']['phase2']['status'] == 'completed' else '🚧 In Progress'}

## Next Steps

"""
    
    if coverage < 50:
        content += """- Continue writing tests for remaining business logic modules
- Focus on untested modules with high complexity
- Consider adding integration tests
"""
    else:
        content += """- Phase 2 target achieved! 🎉
- Consider moving to Phase 3 (60% coverage)
- Add end-to-end tests
- Improve test quality and assertions
"""
    
    output_path.write_text(content)


def main():
    """Main function to update CI status."""
    project_root = Path(__file__).parent.parent
    
    # Parse coverage
    coverage_xml = project_root / "coverage.xml"
    if not coverage_xml.exists():
        print("coverage.xml not found. Run pytest with coverage first.")
        sys.exit(1)
    
    coverage = parse_coverage_xml(coverage_xml)
    ci_status = "passing" if coverage >= 35 else "failing"
    
    # Update README
    readme_path = project_root / "README.md"
    update_readme_badges(readme_path, coverage, ci_status)
    
    # Generate reports
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # JSON report
    json_report = generate_coverage_report(
        coverage, 
        reports_dir / f"coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    # Markdown report
    generate_markdown_report(
        json_report,
        reports_dir / "coverage_report_latest.md"
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Coverage: {coverage}%")
    print(f"CI Status: {ci_status}")
    print(f"Phase 2 Target: 50% - {'ACHIEVED' if coverage >= 50 else f'Need {50 - coverage:.2f}% more'}")
    print(f"{'='*50}\n")
    
    # Exit with appropriate code
    sys.exit(0 if ci_status == "passing" else 1)


if __name__ == "__main__":
    main()