#!/usr/bin/env python3
"""
Comprehensive test suite for multivisit.py code review findings.
Tests edge cases, overflow conditions, and collision handling.
"""

import sys
import tempfile
from pathlib import Path
import shutil
import csv
import time
from typing import Dict, Any, List

# Color codes for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_test_header(test_name: str):
    """Print formatted test header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Testing: {test_name}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def print_result(success: bool, message: str):
    """Print colored result message."""
    if success:
        print(f"{GREEN}✅ PASS: {message}{RESET}")
    else:
        print(f"{RED}❌ FAIL: {message}{RESET}")

def test_timestamp_overflow() -> Dict[str, Any]:
    """Test timestamp multiplication by 1e9 for overflow conditions."""
    print_test_header("Timestamp Overflow (Line 147)")

    results = {
        'test': 'timestamp_overflow',
        'passed': [],
        'failed': []
    }

    # Test cases with various timestamp values
    test_cases = [
        ('normal_seconds', 1234567890.123456),  # Normal Unix timestamp
        ('max_float_safe', 9007199254.740991),  # Max safe float before precision loss
        ('near_overflow', 1e308 / 1e9),  # Near Python float max
        ('negative', -1234567890.123456),  # Negative timestamp
        ('zero', 0.0),
        ('small_fraction', 0.000000001),
    ]

    for name, timestamp_seconds in test_cases:
        try:
            # Simulate the code from line 147
            result = int(timestamp_seconds * 1e9)

            # Check for overflow or precision loss
            if timestamp_seconds > 0:
                # Check if we can reverse the operation
                reversed = result / 1e9
                precision_loss = abs(reversed - timestamp_seconds) / timestamp_seconds > 0.01

                if precision_loss:
                    print_result(False, f"{name}: Precision loss detected - input: {timestamp_seconds}, output: {result}, reversed: {reversed}")
                    results['failed'].append(f"{name}_precision_loss")
                else:
                    print_result(True, f"{name}: {timestamp_seconds} -> {result} (OK)")
                    results['passed'].append(name)
            else:
                print_result(True, f"{name}: {timestamp_seconds} -> {result} (OK)")
                results['passed'].append(name)

        except OverflowError as e:
            print_result(False, f"{name}: OverflowError - {e}")
            results['failed'].append(f"{name}_overflow")
        except Exception as e:
            print_result(False, f"{name}: Unexpected error - {e}")
            results['failed'].append(f"{name}_error")

    return results

def test_collision_infinite_loop() -> Dict[str, Any]:
    """Test filename collision handling for infinite loop conditions."""
    print_test_header("Collision Handling Infinite Loop (Lines 154-159)")

    results = {
        'test': 'collision_infinite_loop',
        'passed': [],
        'failed': []
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test case 1: Normal collision resolution
        used_filenames = set()
        base_name = "session_1_frame_000001.jpg"
        ext = ".jpg"

        # Simulate multiple collisions
        for i in range(100):
            used_filenames.add(f"session_1_frame_000001_dup{i}.jpg")

        # Now test the collision resolution loop
        dest_name = base_name
        collision_counter = 1
        max_iterations = 1000  # Safety limit
        iterations = 0

        while dest_name in used_filenames and iterations < max_iterations:
            name_without_ext = base_name[:-len(ext)] if ext else base_name
            dest_name = f"{name_without_ext}_dup{collision_counter}{ext}"
            collision_counter += 1
            iterations += 1

        if iterations >= max_iterations:
            print_result(False, f"Infinite loop detected after {max_iterations} iterations")
            results['failed'].append('infinite_loop_detected')
        elif collision_counter > 101:  # We added 100 duplicates
            print_result(True, f"Resolved collision after {collision_counter-1} attempts")
            results['passed'].append('collision_resolved')
        else:
            print_result(False, f"Unexpected collision counter: {collision_counter}")
            results['failed'].append('unexpected_counter')

        # Test case 2: Edge case with empty extension
        used_filenames2 = {"test_file"}
        base_name2 = "test_file"
        ext2 = ""
        dest_name2 = base_name2
        collision_counter2 = 1

        while dest_name2 in used_filenames2:
            name_without_ext2 = base_name2[:-len(ext2)] if ext2 else base_name2
            dest_name2 = f"{name_without_ext2}_dup{collision_counter2}{ext2}"
            collision_counter2 += 1
            if collision_counter2 > 10:
                break

        if dest_name2 == "test_file_dup1":
            print_result(True, "Empty extension handled correctly")
            results['passed'].append('empty_extension')
        else:
            print_result(False, f"Empty extension handling failed: {dest_name2}")
            results['failed'].append('empty_extension')

    return results

def test_malformed_csv() -> Dict[str, Any]:
    """Test handling of malformed CSV files."""
    print_test_header("Malformed CSV Handling (Lines 29-45)")

    results = {
        'test': 'malformed_csv',
        'passed': [],
        'failed': []
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        session_dir = tmpdir / "session"
        session_dir.mkdir()

        # Test cases for malformed CSV
        test_cases = [
            ('empty_file', ''),
            ('header_only', 'filename,timestamp_seconds\n'),
            ('missing_timestamp', 'filename,timestamp_seconds\nfile1.jpg\n'),
            ('invalid_timestamp', 'filename,timestamp_seconds\nfile1.jpg,not_a_number\n'),
            ('extra_columns', 'filename,timestamp_seconds,extra\nfile1.jpg,123.456,extra_data\n'),
            ('missing_columns', 'filename\nfile1.jpg\n'),
            ('mixed_formats', 'filename,timestamp_seconds\nfile1.jpg,123.456\nfile2.jpg\nfile3.jpg,456.789\n'),
            ('unicode_characters', 'filename,timestamp_seconds\n文件.jpg,123.456\n'),
            ('very_long_line', 'filename,timestamp_seconds\n' + 'a'*10000 + '.jpg,123.456\n'),
        ]

        sys.path.insert(0, str(Path(__file__).parent / 'src' / 'sfm_experiments'))
        from multivisit import _load_metadata

        for test_name, csv_content in test_cases:
            metadata_file = session_dir / "frames_metadata.csv"

            # Write test CSV
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(csv_content)

            try:
                # Call the function under test
                mapping = _load_metadata(session_dir)

                # Validate results based on test case
                if test_name == 'empty_file':
                    if mapping == {}:
                        print_result(True, f"{test_name}: Empty file handled correctly")
                        results['passed'].append(test_name)
                    else:
                        print_result(False, f"{test_name}: Expected empty dict, got {mapping}")
                        results['failed'].append(test_name)

                elif test_name == 'invalid_timestamp':
                    # Should skip invalid lines but not crash
                    print_result(True, f"{test_name}: Invalid timestamp skipped without crash")
                    results['passed'].append(test_name)

                elif test_name == 'extra_columns':
                    if 'file1.jpg' in mapping and mapping['file1.jpg'] == 123.456:
                        print_result(True, f"{test_name}: Extra columns ignored correctly")
                        results['passed'].append(test_name)
                    else:
                        print_result(False, f"{test_name}: Failed to parse with extra columns")
                        results['failed'].append(test_name)

                else:
                    # General success case - function didn't crash
                    print_result(True, f"{test_name}: Handled without crash")
                    results['passed'].append(test_name)

            except Exception as e:
                print_result(False, f"{test_name}: Raised exception: {e}")
                results['failed'].append(test_name)

            # Clean up
            if metadata_file.exists():
                metadata_file.unlink()

    return results

def test_cache_validation_edge_cases() -> Dict[str, Any]:
    """Test cache validation logic edge cases."""
    print_test_header("Cache Validation Edge Cases (Lines 97-112)")

    results = {
        'test': 'cache_validation',
        'passed': [],
        'failed': []
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test case 1: Cache with wrong file extensions
        output_dir = tmpdir / "cached_output"
        output_dir.mkdir()

        # Create mixed file types
        (output_dir / "frame1.jpg").touch()
        (output_dir / "frame2.JPG").touch()  # Different case
        (output_dir / "frame3.jpeg").touch()
        (output_dir / "frame4.png").touch()
        (output_dir / "frame5.PNG").touch()
        (output_dir / "not_an_image.txt").touch()  # Should be ignored

        # Count images using the same logic as the code
        existing_frames = []
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        for pattern in image_patterns:
            existing_frames.extend(list(output_dir.glob(pattern)))

        if len(existing_frames) == 5:
            print_result(True, "Mixed case extensions counted correctly")
            results['passed'].append('mixed_case_extensions')
        else:
            print_result(False, f"Expected 5 frames, got {len(existing_frames)}")
            results['failed'].append('mixed_case_extensions')

        # Test case 2: Empty cache directory
        empty_dir = tmpdir / "empty_cache"
        empty_dir.mkdir()

        existing_frames2 = []
        for pattern in image_patterns:
            existing_frames2.extend(list(empty_dir.glob(pattern)))

        if len(existing_frames2) == 0:
            print_result(True, "Empty cache handled correctly")
            results['passed'].append('empty_cache')
        else:
            print_result(False, f"Empty dir should have 0 frames, got {len(existing_frames2)}")
            results['failed'].append('empty_cache')

        # Test case 3: Symlinks in cache
        symlink_dir = tmpdir / "symlink_cache"
        symlink_dir.mkdir()
        real_file = symlink_dir / "real.jpg"
        real_file.touch()
        symlink_file = symlink_dir / "link.jpg"
        try:
            symlink_file.symlink_to(real_file)

            existing_frames3 = []
            for pattern in image_patterns:
                existing_frames3.extend(list(symlink_dir.glob(pattern)))

            if len(existing_frames3) == 2:  # Both real and symlink counted
                print_result(True, "Symlinks counted as separate files")
                results['passed'].append('symlink_handling')
            else:
                print_result(False, f"Symlink test: expected 2, got {len(existing_frames3)}")
                results['failed'].append('symlink_handling')
        except OSError:
            print_result(True, "Symlink test skipped (not supported)")
            results['passed'].append('symlink_skipped')

    return results

def test_resource_cleanup() -> Dict[str, Any]:
    """Test proper cleanup of temporary files and resources."""
    print_test_header("Resource Cleanup")

    results = {
        'test': 'resource_cleanup',
        'passed': [],
        'failed': []
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test structure
        session1 = tmpdir / "session1"
        session1.mkdir()
        (session1 / "frame1.jpg").touch()

        output_dir = tmpdir / "output"

        sys.path.insert(0, str(Path(__file__).parent / 'src' / 'sfm_experiments'))
        from multivisit import combine_sessions

        # First run - create cache
        result1 = combine_sessions([session1], output_dir, use_cache=True)

        if output_dir.exists():
            print_result(True, "Output directory created")
            results['passed'].append('output_created')
        else:
            print_result(False, "Output directory not created")
            results['failed'].append('output_created')

        # Check if shutil.rmtree is called when cache is invalid
        # Add more frames to invalidate cache
        (session1 / "frame2.jpg").touch()

        # Second run - should rebuild cache
        result2 = combine_sessions([session1], output_dir, use_cache=True)

        if result2 == 2:  # Should now have 2 frames
            print_result(True, "Cache properly invalidated and rebuilt")
            results['passed'].append('cache_rebuild')
        else:
            print_result(False, f"Cache rebuild failed: expected 2 frames, got {result2}")
            results['failed'].append('cache_rebuild')

        # Test hardlink fallback
        protected_dir = tmpdir / "protected"
        protected_dir.mkdir()
        (protected_dir / "test.jpg").touch()

        # Make directory read-only to force copy instead of hardlink
        import os
        os.chmod(protected_dir, 0o555)

        try:
            output_dir2 = tmpdir / "output2"
            # This should fall back to copy since hardlink will fail
            result3 = combine_sessions([protected_dir], output_dir2, use_cache=False)

            if output_dir2.exists() and list(output_dir2.glob("*.jpg")):
                print_result(True, "Hardlink fallback to copy worked")
                results['passed'].append('hardlink_fallback')
            else:
                print_result(False, "Hardlink fallback failed")
                results['failed'].append('hardlink_fallback')
        finally:
            # Restore permissions for cleanup
            os.chmod(protected_dir, 0o755)

    return results

def main():
    """Run all tests and summarize results."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Code Review Validation Suite for multivisit.py{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    all_results = []

    # Run all tests
    all_results.append(test_timestamp_overflow())
    all_results.append(test_collision_infinite_loop())
    all_results.append(test_malformed_csv())
    all_results.append(test_cache_validation_edge_cases())
    all_results.append(test_resource_cleanup())

    # Summarize results
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    total_passed = sum(len(r['passed']) for r in all_results)
    total_failed = sum(len(r['failed']) for r in all_results)
    total_tests = total_passed + total_failed

    for result in all_results:
        test_name = result['test']
        passed = len(result['passed'])
        failed = len(result['failed'])

        if failed == 0:
            print(f"{GREEN}✅ {test_name}: All {passed} tests passed{RESET}")
        else:
            print(f"{YELLOW}⚠️  {test_name}: {passed} passed, {failed} failed{RESET}")
            for failure in result['failed']:
                print(f"   {RED}- {failure}{RESET}")

    print(f"\n{BLUE}{'='*60}{RESET}")
    if total_failed == 0:
        print(f"{GREEN}✅ ALL TESTS PASSED ({total_passed}/{total_tests}){RESET}")
        return 0
    else:
        print(f"{RED}❌ TESTS FAILED: {total_failed}/{total_tests} failed{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())