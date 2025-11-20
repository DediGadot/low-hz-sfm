"""
Utility functions and logging setup for SfM experiments.

This module provides:
- Logging configuration with loguru
- File I/O helpers
- Path utilities
- Common helper functions

Dependencies:
- loguru: https://loguru.readthedocs.io/

Sample Input: logger.info("Processing started")
Expected Output: Formatted log messages to console and file
"""

from pathlib import Path
from typing import Any, Dict, Optional
import sys

from loguru import logger


def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure logging with loguru.

    Args:
        log_file: Path to log file. If None, only console logging is used.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log files (e.g., "10 MB", "1 week")
        retention: How long to keep old logs

    Example:
        >>> setup_logging(Path("logs/experiment.log"), level="DEBUG")
    """
    # Remove default logger
    logger.remove()

    # Add console logger with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Add file logger if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logging initialized: level={level}, file={log_file}")


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to create

    Returns:
        The path that was created/verified

    Example:
        >>> output_dir = ensure_dir(Path("results/experiment_01"))
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_files(directory: Path, pattern: str = "*") -> int:
    """
    Count files matching pattern in directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.jpg", "*.txt")

    Returns:
        Number of matching files

    Example:
        >>> num_images = count_files(Path("data/images"), "*.jpg")
    """
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 15m 30s")

    Example:
        >>> print(format_duration(7530))
        2h 5m 30s
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def print_section(title: str, width: int = 80) -> None:
    """
    Print formatted section header.

    Args:
        title: Section title
        width: Total width of header line

    Example:
        >>> print_section("Phase 1: Data Loading")
    """
    logger.info("=" * width)
    logger.info(f" {title} ".center(width, "="))
    logger.info("=" * width)


def print_summary(data: Dict[str, Any], title: str = "Summary") -> None:
    """
    Print formatted summary of key-value pairs.

    Args:
        data: Dictionary of values to display
        title: Summary title

    Example:
        >>> print_summary({"Images": 150, "Duration": "2h 5m"}, "Results")
    """
    print_section(title)
    max_key_len = max(len(str(k)) for k in data.keys()) if data else 0

    for key, value in data.items():
        logger.info(f"  {str(key).ljust(max_key_len)} : {value}")

    logger.info("=" * 80)


# Validation
if __name__ == "__main__":
    """
    Validation of utils.py functionality with real usage scenarios.
    Tests logging setup, file operations, and formatting utilities.
    """
    import sys
    import tempfile
    import time

    all_validation_failures = []
    total_tests = 0

    # Test 1: Logging setup
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_file, level="DEBUG")

            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")

            # Check log file was created
            if not log_file.exists():
                all_validation_failures.append("Logging setup: Log file was not created")
            elif log_file.stat().st_size == 0:
                all_validation_failures.append("Logging setup: Log file is empty")
    except Exception as e:
        all_validation_failures.append(f"Logging setup: Exception raised: {e}")

    # Test 2: Directory creation
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "nested" / "test" / "dir"
            result = ensure_dir(test_dir)

            if not result.exists():
                all_validation_failures.append("Directory creation: Directory was not created")
            elif result != test_dir:
                all_validation_failures.append(f"Directory creation: Expected {test_dir}, got {result}")
    except Exception as e:
        all_validation_failures.append(f"Directory creation: Exception raised: {e}")

    # Test 3: File counting
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test files
            for i in range(5):
                (test_dir / f"test{i}.jpg").touch()
            for i in range(3):
                (test_dir / f"test{i}.txt").touch()

            jpg_count = count_files(test_dir, "*.jpg")
            txt_count = count_files(test_dir, "*.txt")

            if jpg_count != 5:
                all_validation_failures.append(f"File counting: Expected 5 JPG files, got {jpg_count}")
            if txt_count != 3:
                all_validation_failures.append(f"File counting: Expected 3 TXT files, got {txt_count}")
    except Exception as e:
        all_validation_failures.append(f"File counting: Exception raised: {e}")

    # Test 4: Duration formatting
    total_tests += 1
    try:
        test_cases = [
            (7530, "2h 5m 30s"),
            (65, "1m 5s"),
            (30, "30s"),
            (3665, "1h 1m 5s"),
        ]

        for seconds, expected in test_cases:
            result = format_duration(seconds)
            if result != expected:
                all_validation_failures.append(f"Duration formatting: Expected '{expected}' for {seconds}s, got '{result}'")
    except Exception as e:
        all_validation_failures.append(f"Duration formatting: Exception raised: {e}")

    # Test 5: Section printing (verify no exceptions)
    total_tests += 1
    try:
        print_section("Test Section")
        print_summary({"Test": "Value", "Count": 42}, "Test Summary")
    except Exception as e:
        all_validation_failures.append(f"Section printing: Exception raised: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Utility functions validated and ready for use")
        sys.exit(0)
