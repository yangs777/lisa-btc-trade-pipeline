#!/usr/bin/env python3
"""Update CI status in README and generate reports."""

import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


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


def get_previous_coverage(reports_dir: Path) -> float:
    """Get previous coverage from latest JSON report."""
    if not reports_dir.exists():
        return 0.0
    
    # Find all coverage report JSON files
    json_files = list(reports_dir.glob("coverage_report_*.json"))
    if not json_files:
        return 0.0
    
    # Sort by modification time to get the latest
    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_json, 'r') as f:
            data = json.load(f)
            return data.get('coverage', {}).get('percentage', 0.0)
    except Exception as e:
        print(f"Error reading previous report: {e}")
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


def get_phase_info(coverage: float) -> Dict[str, Any]:
    """Determine phase information based on coverage."""
    phases = {
        "phase1": {
            "name": "Foundation",
            "target": 25,
            "achieved": 32.66,  # Historical value
            "status": "completed"
        },
        "phase2": {
            "name": "Business Logic", 
            "target": 50,
            "achieved": coverage if coverage >= 32.66 else 32.66,
            "status": "completed" if coverage >= 50 else "in_progress" if coverage >= 32.66 else "planned"
        },
        "phase3": {
            "name": "Integration & E2E",
            "target": 60,
            "achieved": coverage if coverage >= 50 else 0,
            "status": "completed" if coverage >= 60 else "in_progress" if coverage >= 50 else "planned"
        },
        "phase4": {
            "name": "Advanced Testing",
            "target": 75,
            "achieved": coverage if coverage >= 60 else 0,
            "status": "completed" if coverage >= 75 else "in_progress" if coverage >= 60 else "planned"
        }
    }
    return phases


def generate_coverage_report(coverage: float, output_path: Path, previous_coverage: float) -> Dict[str, Any]:
    """Generate coverage report in JSON format."""
    ci_threshold = 35  # From CI configuration
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "coverage": {
            "percentage": coverage,
            "status": "passing" if coverage >= ci_threshold else "failing",
            "threshold": ci_threshold
        },
        "trend": {
            "previous": previous_coverage,
            "current": coverage,
            "change": round(coverage - previous_coverage, 2)
        },
        "milestones": get_phase_info(coverage)
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

- **Current Coverage**: {coverage}% {':white_check_mark:' if coverage >= report['coverage']['threshold'] else ':x:'}
- **CI Threshold**: {report['coverage']['threshold']}%
- **Status**: {report['coverage']['status'].upper()}

## Coverage Trend

- Previous: {trend['previous']}%
- Current: {trend['current']}%
- Change: {'+' if trend['change'] >= 0 else ''}{trend['change']}%

## Milestones Progress
"""
    
    # Add phase information
    for phase_key, phase_data in report['milestones'].items():
        phase_num = phase_key.replace('phase', '')
        status_emoji = '✅' if phase_data['status'] == 'completed' else '🚧' if phase_data['status'] == 'in_progress' else '🔜'
        
        content += f"""
### Phase {phase_num}: {phase_data['name']} (Target: {phase_data['target']}%)
- Achieved: {phase_data['achieved']}%
- Status: {status_emoji} {phase_data['status'].replace('_', ' ').title()}
"""
    
    content += "\n## Next Steps\n\n"
    
    current_phase = None
    for phase_key, phase_data in report['milestones'].items():
        if phase_data['status'] == 'in_progress':
            current_phase = phase_data
            break
    
    if current_phase:
        if current_phase['target'] == 50:
            content += """- Continue writing tests for remaining business logic modules
- Focus on untested modules with high complexity
- Consider adding integration tests
"""
        elif current_phase['target'] == 60:
            content += """- Add integration tests for API endpoints
- Create end-to-end test scenarios
- Test risk manager and trading loop integration
"""
        elif current_phase['target'] == 75:
            content += """- Add comprehensive E2E tests
- Performance and stress testing
- Mock external dependencies properly
"""
    else:
        # All phases completed or none started
        if coverage >= 75:
            content += """- Excellent coverage achieved! 🎉
- Focus on maintaining coverage levels
- Add tests for new features
- Consider property-based testing
"""
        else:
            content += """- Begin with Phase 1 foundation tests
- Focus on critical path coverage
- Set up test infrastructure
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
    
    # Get previous coverage
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    previous_coverage = get_previous_coverage(reports_dir)
    
    # Determine CI status
    ci_threshold = 35  # From CI configuration
    ci_status = "passing" if coverage >= ci_threshold else "failing"
    
    # Update README
    readme_path = project_root / "README.md"
    update_readme_badges(readme_path, coverage, ci_status)
    
    # Generate reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON report
    json_report = generate_coverage_report(
        coverage, 
        reports_dir / f"coverage_report_{timestamp}.json",
        previous_coverage
    )
    
    # Markdown report
    generate_markdown_report(
        json_report,
        reports_dir / "coverage_report_latest.md"
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Coverage: {coverage}%")
    print(f"Previous: {previous_coverage}%")
    print(f"Change: {'+' if coverage >= previous_coverage else ''}{coverage - previous_coverage:.2f}%")
    print(f"CI Status: {ci_status}")
    
    # Show current phase progress
    phases = get_phase_info(coverage)
    for phase_key, phase_data in phases.items():
        if phase_data['status'] == 'in_progress':
            remaining = phase_data['target'] - coverage
            print(f"\nPhase {phase_key[-1]} Target: {phase_data['target']}% - " + 
                  (f"Need {remaining:.2f}% more" if remaining > 0 else "ACHIEVED"))
            break
    
    print(f"{'='*50}\n")
    
    # Exit with appropriate code
    sys.exit(0 if ci_status == "passing" else 1)


if __name__ == "__main__":
    main()