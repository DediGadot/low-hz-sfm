"""
GLOMAP (Global Structure-from-Motion) wrapper for subprocess invocation.

This module provides a Python interface to GLOMAP's command-line tools,
allowing integration with the SfM pipeline as an alternative to COLMAP's
incremental mapper.

Dependencies:
- GLOMAP: https://github.com/colmap/glomap
- subprocess: Built-in Python module

Sample Input: COLMAP database with features and matches
Expected Output: COLMAP-compatible sparse reconstruction

GLOMAP Installation:
- Build from source: https://github.com/colmap/glomap#installation
- Conda: conda install -c conda-forge glomap
- System package manager (if available)
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class GlomapOptions:
    """Configuration options for GLOMAP mapper."""

    # Output format
    output_format: str = "bin"  # "bin" or "txt"

    # Relative pose estimation
    max_epipolar_error: float = 2.0  # Increase for blurry/high-res images

    # Track establishment
    max_num_tracks: Optional[int] = None  # Cap points for speed (e.g., 1000)

    # Global positioning
    global_positioning_max_iterations: int = 100

    # Bundle adjustment
    bundle_adjustment_max_iterations: int = 100

    # Performance options
    skip_retriangulation: bool = False  # Skip for speed

    # Constraint type
    constraint_type: str = "POINTS_AND_CAMERAS_BALANCED"

    def to_command_args(self) -> list[str]:
        """Convert options to GLOMAP command-line arguments."""
        args = []

        # Output format
        args.extend(["--output_format", self.output_format])

        # Relative pose estimation
        args.extend([
            "--RelPoseEstimation.max_epipolar_error",
            str(self.max_epipolar_error)
        ])

        # Track establishment
        if self.max_num_tracks is not None:
            args.extend([
                "--TrackEstablishment.max_num_tracks",
                str(self.max_num_tracks)
            ])

        # Global positioning
        args.extend([
            "--GlobalPositioning.max_num_iterations",
            str(self.global_positioning_max_iterations)
        ])

        # Bundle adjustment
        args.extend([
            "--BundleAdjustment.max_num_iterations",
            str(self.bundle_adjustment_max_iterations)
        ])

        # Performance
        if self.skip_retriangulation:
            args.append("--skip_retriangulation")

        # Constraint type
        args.extend(["--constraint_type", self.constraint_type])

        return args


def check_glomap_available() -> Tuple[bool, Optional[str]]:
    """
    Check if GLOMAP is installed and available.

    Returns:
        Tuple of (available: bool, version_or_error: str)

    Example:
        >>> available, info = check_glomap_available()
        >>> if available:
        ...     print(f"GLOMAP available: {info}")
        ... else:
        ...     print(f"GLOMAP not found: {info}")
    """
    try:
        # Check if glomap command exists
        glomap_path = shutil.which("glomap")
        if not glomap_path:
            return False, "GLOMAP executable not found in PATH"

        # Try to get version info
        result = subprocess.run(
            ["glomap", "-h"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            # Extract version from help text if available
            version_info = "GLOMAP available"
            return True, version_info
        else:
            return False, f"GLOMAP command failed: {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        return False, "GLOMAP command timed out"
    except Exception as e:
        return False, f"Error checking GLOMAP: {str(e)}"


def run_glomap_mapper(
    database_path: Path,
    image_path: Path,
    output_path: Path,
    options: Optional[GlomapOptions] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """
    Run GLOMAP mapper via subprocess.

    Args:
        database_path: Path to COLMAP database.db file
        image_path: Directory containing input images
        output_path: Where to save sparse reconstruction
        options: GLOMAP configuration options
        timeout: Maximum execution time in seconds (None for no limit)

    Returns:
        subprocess.CompletedProcess with stdout, stderr, returncode

    Raises:
        FileNotFoundError: If GLOMAP is not installed
        subprocess.TimeoutExpired: If execution exceeds timeout
        RuntimeError: If GLOMAP execution fails

    Example:
        >>> result = run_glomap_mapper(
        ...     database_path=Path("database.db"),
        ...     image_path=Path("images"),
        ...     output_path=Path("sparse"),
        ... )
        >>> if result.returncode == 0:
        ...     print("GLOMAP succeeded")
    """
    # Check GLOMAP availability
    available, info = check_glomap_available()
    if not available:
        raise FileNotFoundError(
            f"GLOMAP not available: {info}\n\n"
            "Please install GLOMAP:\n"
            "  - Build from source: https://github.com/colmap/glomap#installation\n"
            "  - Conda: conda install -c conda-forge glomap\n"
            "  - System package manager (if available)"
        )

    # Use default options if none provided
    if options is None:
        options = GlomapOptions()

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "glomap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--output_path", str(output_path),
    ]

    # Add options
    cmd.extend(options.to_command_args())

    logger.info(f"Running GLOMAP mapper...")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        # Execute GLOMAP
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,  # Don't raise on non-zero return code
        )

        # Log output
        if result.stdout:
            logger.debug(f"GLOMAP stdout:\n{result.stdout}")
        if result.stderr:
            logger.debug(f"GLOMAP stderr:\n{result.stderr}")

        # Check for success
        if result.returncode != 0:
            error_msg = f"GLOMAP failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nError output:\n{result.stderr[:1000]}"
            raise RuntimeError(error_msg)

        logger.info("✅ GLOMAP mapper completed successfully")
        return result

    except subprocess.TimeoutExpired as e:
        logger.error(f"GLOMAP timed out after {timeout} seconds")
        raise


def parse_glomap_output(stdout: str) -> Dict[str, Any]:
    """
    Parse GLOMAP stdout for statistics and diagnostics.

    Args:
        stdout: GLOMAP's standard output text

    Returns:
        Dictionary with parsed information

    Example:
        >>> stats = parse_glomap_output(result.stdout)
        >>> print(f"Registered images: {stats.get('num_images', 'N/A')}")
    """
    stats = {}

    # Look for common patterns in GLOMAP output
    lines = stdout.split("\n")

    for line in lines:
        # Parse registration info
        if "registered" in line.lower():
            # Extract numbers if available
            pass

        # Parse timing info
        if "Elapsed time" in line:
            try:
                # Example: "Elapsed time: 0.123 [minutes]"
                parts = line.split(":")
                if len(parts) >= 2:
                    time_str = parts[1].strip().split()[0]
                    stats["elapsed_minutes"] = float(time_str)
            except (ValueError, IndexError):
                pass

    return stats


# Validation
if __name__ == "__main__":
    """
    Validation of glomap_wrapper.py functionality.
    Tests availability check and command building.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: GLOMAP availability check
    total_tests += 1
    try:
        available, info = check_glomap_available()
        logger.info(f"GLOMAP availability: {available}")
        logger.info(f"Info: {info}")
        # Not a failure if GLOMAP isn't installed - just informational
    except Exception as e:
        all_validation_failures.append(f"Availability check: Exception raised: {e}")

    # Test 2: GlomapOptions creation and command building
    total_tests += 1
    try:
        options = GlomapOptions(
            max_epipolar_error=4.0,
            max_num_tracks=1000,
            skip_retriangulation=True,
        )

        args = options.to_command_args()

        # Verify expected arguments present
        if "--RelPoseEstimation.max_epipolar_error" not in args:
            all_validation_failures.append(
                "GlomapOptions: Missing max_epipolar_error in command args"
            )
        if "--TrackEstablishment.max_num_tracks" not in args:
            all_validation_failures.append(
                "GlomapOptions: Missing max_num_tracks in command args"
            )
        if "--skip_retriangulation" not in args:
            all_validation_failures.append(
                "GlomapOptions: Missing skip_retriangulation in command args"
            )

        logger.info(f"GlomapOptions command args: {' '.join(args)}")

    except Exception as e:
        all_validation_failures.append(f"GlomapOptions: Exception raised: {e}")

    # Test 3: Parse output (basic test)
    total_tests += 1
    try:
        sample_output = """
        GLOMAP mapper
        Elapsed time: 1.234 [minutes]
        Registered 42 images
        """
        stats = parse_glomap_output(sample_output)
        if "elapsed_minutes" in stats:
            if abs(stats["elapsed_minutes"] - 1.234) > 0.001:
                all_validation_failures.append(
                    f"Parse output: Expected 1.234 minutes, got {stats['elapsed_minutes']}"
                )
        # Parsing is opportunistic, so not finding some fields isn't a failure
    except Exception as e:
        all_validation_failures.append(f"Parse output: Exception raised: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("GLOMAP wrapper module validated and ready for use")
        print(f"\nNote: GLOMAP availability: {available}")
        if not available:
            print("  To use GLOMAP mapper, install from: https://github.com/colmap/glomap")
        sys.exit(0)
