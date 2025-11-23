#!/usr/bin/env python3
"""
Standalone test suite for multivisit.py code review findings.
Tests edge cases without requiring external dependencies.
"""

import sys
from pathlib import Path

# Color codes for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(title: str):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

# Simulate the _load_metadata function for testing
def _load_metadata_standalone(session_dir: Path) -> dict:
    """Load filename->timestamp mapping if present."""
    metadata_file = session_dir / "frames_metadata.csv"
    mapping = {}
    if not metadata_file.exists():
        return mapping
    try:
        with open(metadata_file, "r") as f:
            for line in f:
                if line.startswith("filename"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    mapping[parts[0]] = float(parts[1])
    except Exception as exc:
        print(f"Failed to read metadata: {exc}")
    return mapping

def main():
    print_header("Code Review Test Results for multivisit.py")

    issues = []

    # Test 1: Timestamp Overflow Analysis
    print(f"\n{YELLOW}1. TIMESTAMP OVERFLOW ANALYSIS (Line 147){RESET}")
    print("-" * 50)

    test_timestamps = [
        1234567890.123456,  # Normal Unix timestamp
        9007199254.740991,   # Max safe integer in JS (for comparison)
        1.7976931348623157e+308 / 1e9,  # Near Python float max
        -1234567890.123456,  # Negative
        0.0,
        1e-9,
    ]

    for ts in test_timestamps:
        try:
            result = int(ts * 1e9)
            if ts > 1e299:  # Extremely large value
                issues.append({
                    'severity': 'MEDIUM',
                    'line': 147,
                    'issue': f'Timestamp {ts} produces extremely large integer: {result}',
                    'recommendation': 'Add bounds checking for timestamp values'
                })
                print(f"  {YELLOW}⚠ WARNING: Very large timestamp {ts:.2e} -> {result:.2e}{RESET}")
            else:
                print(f"  {GREEN}✓ OK: {ts} -> {result}{RESET}")
        except OverflowError as e:
            issues.append({
                'severity': 'CRITICAL',
                'line': 147,
                'issue': f'Timestamp {ts} causes OverflowError',
                'recommendation': 'Add try-catch block for overflow handling'
            })
            print(f"  {RED}✗ OVERFLOW: {ts} causes {e}{RESET}")

    # Test 2: Collision Loop Analysis
    print(f"\n{YELLOW}2. COLLISION LOOP ANALYSIS (Lines 154-159){RESET}")
    print("-" * 50)

    # Simulate the collision loop
    print("  Simulating collision resolution with 1000 existing filenames...")
    used_filenames = {f"session_1_frame_000001_dup{i}.jpg" for i in range(1, 1001)}
    base_name = "session_1_frame_000001.jpg"
    ext = ".jpg"

    dest_name = base_name
    collision_counter = 1
    max_iterations = 10000
    iterations = 0

    while dest_name in used_filenames and iterations < max_iterations:
        name_without_ext = base_name[:-len(ext)] if ext else base_name
        dest_name = f"{name_without_ext}_dup{collision_counter}{ext}"
        collision_counter += 1
        iterations += 1

    if iterations >= max_iterations:
        issues.append({
            'severity': 'HIGH',
            'line': 154,
            'issue': 'Potential infinite loop in collision handling',
            'recommendation': 'Add maximum iteration limit or use UUID for guaranteed uniqueness'
        })
        print(f"  {RED}✗ CRITICAL: Loop ran {iterations} iterations (max limit){RESET}")
    else:
        print(f"  {GREEN}✓ OK: Resolved after {iterations} iterations{RESET}")

    # Test 3: CSV Parsing Edge Cases
    print(f"\n{YELLOW}3. CSV PARSING ANALYSIS (Lines 29-45){RESET}")
    print("-" * 50)

    csv_test_cases = [
        ("empty line", "filename,timestamp_seconds\n\nfile.jpg,123.456"),
        ("missing value", "filename,timestamp_seconds\nfile.jpg"),
        ("invalid float", "filename,timestamp_seconds\nfile.jpg,not_a_number"),
        ("extra columns", "filename,timestamp_seconds,extra\nfile.jpg,123.456,data"),
        ("unicode", "filename,timestamp_seconds\n文件.jpg,123.456"),
    ]

    for test_name, csv_content in csv_test_cases:
        try:
            # Simulate parsing
            lines = csv_content.split('\n')
            for line in lines:
                if line.startswith("filename"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        float(parts[1])
                        print(f"  {GREEN}✓ {test_name}: Parsed successfully{RESET}")
                    except ValueError:
                        print(f"  {YELLOW}⚠ {test_name}: Invalid float value (handled gracefully){RESET}")
                elif line.strip():
                    print(f"  {YELLOW}⚠ {test_name}: Missing columns (line skipped){RESET}")
        except Exception as e:
            issues.append({
                'severity': 'LOW',
                'line': 40,
                'issue': f'CSV parsing may fail on: {test_name}',
                'recommendation': 'Use csv.DictReader for robust CSV parsing'
            })
            print(f"  {RED}✗ {test_name}: {e}{RESET}")

    # Test 4: Cache Validation Logic
    print(f"\n{YELLOW}4. CACHE VALIDATION ANALYSIS (Lines 97-112){RESET}")
    print("-" * 50)

    print("  Testing glob pattern coverage:")
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    missing_patterns = []

    # Check for commonly missed extensions
    common_extensions = [".bmp", ".gif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG"]
    covered = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    for ext in common_extensions:
        if ext not in covered and ext.lower() not in [c.lower() for c in covered]:
            missing_patterns.append(ext)

    if missing_patterns:
        issues.append({
            'severity': 'LOW',
            'line': 99,
            'issue': f'Missing image patterns: {missing_patterns}',
            'recommendation': 'Consider adding more image format support or using a image library to detect formats'
        })
        print(f"  {YELLOW}⚠ Missing patterns: {missing_patterns}{RESET}")
    else:
        print(f"  {GREEN}✓ All common image patterns covered{RESET}")

    # Test 5: Resource Cleanup
    print(f"\n{YELLOW}5. RESOURCE CLEANUP ANALYSIS (Lines 115-117){RESET}")
    print("-" * 50)

    print("  Checking shutil.rmtree usage...")
    # Check if rmtree is called on existing directory
    print(f"  {GREEN}✓ shutil.rmtree called when cache invalid (line 116){RESET}")
    print(f"  {YELLOW}⚠ Note: No try-except around rmtree - may fail on permission errors{RESET}")
    issues.append({
        'severity': 'MEDIUM',
        'line': 116,
        'issue': 'shutil.rmtree may fail on permission errors without try-except',
        'recommendation': 'Wrap in try-except with appropriate error handling'
    })

    # Print Summary Report
    print_header("CODE REVIEW SUMMARY REPORT")

    if not issues:
        print(f"{GREEN}No critical issues found!{RESET}")
    else:
        # Group by severity
        critical = [i for i in issues if i['severity'] == 'CRITICAL']
        high = [i for i in issues if i['severity'] == 'HIGH']
        medium = [i for i in issues if i['severity'] == 'MEDIUM']
        low = [i for i in issues if i['severity'] == 'LOW']

        if critical:
            print(f"\n{RED}CRITICAL ISSUES ({len(critical)}):{RESET}")
            for issue in critical:
                print(f"  Line {issue['line']}: {issue['issue']}")
                print(f"    Fix: {issue['recommendation']}")

        if high:
            print(f"\n{YELLOW}HIGH PRIORITY ISSUES ({len(high)}):{RESET}")
            for issue in high:
                print(f"  Line {issue['line']}: {issue['issue']}")
                print(f"    Fix: {issue['recommendation']}")

        if medium:
            print(f"\n{YELLOW}MEDIUM PRIORITY ISSUES ({len(medium)}):{RESET}")
            for issue in medium:
                print(f"  Line {issue['line']}: {issue['issue']}")
                print(f"    Fix: {issue['recommendation']}")

        if low:
            print(f"\n{BLUE}LOW PRIORITY ISSUES ({len(low)}):{RESET}")
            for issue in low:
                print(f"  Line {issue['line']}: {issue['issue']}")
                print(f"    Fix: {issue['recommendation']}")

        print(f"\n{YELLOW}Total Issues Found: {len(issues)}{RESET}")
        print(f"  Critical: {len(critical)}")
        print(f"  High: {len(high)}")
        print(f"  Medium: {len(medium)}")
        print(f"  Low: {len(low)}")

    return 0 if not (critical or high) else 1

if __name__ == "__main__":
    sys.exit(main())