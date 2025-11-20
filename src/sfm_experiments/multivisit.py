"""
Multi-visit reconstruction orchestration module.

This module provides:
- Session frame combining
- Multi-visit experiment execution
- Result tracking and aggregation
- Progress monitoring

Dependencies:
- pathlib, shutil for file operations

Sample Input: Multiple session frame directories
Expected Output: Combined frames and reconstruction results for varying visit counts
"""

from pathlib import Path
from typing import List, Dict, Any
import shutil

from loguru import logger
from tqdm import tqdm

from .colmap_runner import run_sfm_reconstruction, ReconstructionResult

PAIR_PRIME = 2147483647  # COLMAP pair_id prime constant


def _load_metadata(session_dir: Path) -> dict:
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
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to read metadata from {metadata_file}: {exc}")
    return mapping


def _safe_link(src: Path, dst: Path) -> None:
    """Create a hardlink if possible, else copy."""
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def combine_sessions(
    session_dirs: List[Path],
    output_dir: Path,
    session_names: List[str] = None,
    use_cache: bool = True,
) -> int:
    """
    Combine multiple session frame directories into one.

    Copies frames from each session into a single directory with
    sequential numbering. This enables COLMAP to find loop closures
    between different visits.

    Args:
        session_dirs: List of directories containing frames
        output_dir: Where to save combined frames
        session_names: Optional list of session names for logging
        use_cache: If True, skip combining if output already exists with correct frame count

    Returns:
        Total number of frames copied

    Example:
        >>> num_frames = combine_sessions(
        ...     [Path("frames/session_01"), Path("frames/session_02")],
        ...     Path("combined/visit_1_2")
        ... )
        >>> print(f"Combined {num_frames} frames")
    """
    if session_names is None:
        session_names = [f"session_{i+1}" for i in range(len(session_dirs))]

    # Calculate expected frame count
    expected_frames = 0
    for session_dir in session_dirs:
        if session_dir.exists():
            image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
            for pattern in image_patterns:
                expected_frames += len(list(session_dir.glob(pattern)))

    # Check cache: if output exists with correct number of frames, skip
    if use_cache and output_dir.exists():
        existing_frames = []
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        for pattern in image_patterns:
            existing_frames.extend(list(output_dir.glob(pattern)))

        if expected_frames > 0 and len(existing_frames) == expected_frames:
            logger.info(
                f"✅ Using cached combined frames: {output_dir} ({expected_frames} frames)"
            )
            return expected_frames
        else:
            logger.info(
                f"Cache invalid for {output_dir} (expected {expected_frames} frames, "
                f"found {len(existing_frames)}) - rebuilding"
            )

    # Need to combine sessions from scratch when cache disabled/mismatched
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Need to combine sessions
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0

    logger.info(f"Combining {len(session_dirs)} sessions into {output_dir}")

    for session_dir, session_name in zip(session_dirs, session_names):
        if not session_dir.exists():
            logger.warning(f"Skipping missing session: {session_dir}")
            continue

        metadata = _load_metadata(session_dir)

        # Get all image files (jpg, jpeg, png)
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_files = []
        for pattern in image_patterns:
            image_files.extend(sorted(session_dir.glob(pattern)))

        logger.info(f"  {session_name}: {len(image_files)} frames")

        # Copy frames with sequential numbering
        for frame_path in tqdm(image_files, desc=f"Copying {session_name}", leave=False):
            # Preserve original extension and include timestamp if known to maintain GT alignment
            ext = frame_path.suffix
            timestamp_seconds = metadata.get(frame_path.name)
            if timestamp_seconds is not None:
                dest_name = f"{session_name}_{int(timestamp_seconds * 1e9)}{ext}"
            else:
                dest_name = f"{session_name}_frame_{frame_count:06d}{ext}"

            dest_path = output_dir / dest_name
            _safe_link(frame_path, dest_path)
            frame_count += 1

    logger.info(f"✅ Combined {frame_count} total frames")
    return frame_count


def run_multivisit_experiment(
    session_dirs: List[Path],
    output_base: Path,
    visit_counts: List[int],
    session_names: List[str] = None,
    use_cache: bool = True,
    **colmap_kwargs,
) -> Dict[int, ReconstructionResult]:
    """
    Run multi-visit reconstruction experiment.

    Tests how reconstruction quality improves with multiple visits
    to the same location. For each visit count, combines the first N
    sessions and runs COLMAP reconstruction.

    Args:
        session_dirs: List of session frame directories (in order)
        output_base: Base directory for outputs
        visit_counts: How many visits to test (e.g., [1, 2, 3, 5])
        session_names: Optional session names for logging
        use_cache: If True, reuse existing combined frames and reconstructions
        **colmap_kwargs: Additional arguments for run_colmap_reconstruction

    Returns:
        Dictionary mapping visit count to ReconstructionResult

    Example:
        >>> results = run_multivisit_experiment(
        ...     [Path("frames/seq01"), Path("frames/seq02"), Path("frames/seq03")],
        ...     Path("results"),
        ...     visit_counts=[1, 2, 3]
        ... )
        >>> for n, result in results.items():
        ...     print(f"{n} visits: {result.num_registered_images} images")
    """
    if session_names is None:
        session_names = [f"session_{i+1}" for i in range(len(session_dirs))]

    results = {}

    for n_visits in visit_counts:
        if n_visits > len(session_dirs):
            logger.warning(
                f"⚠️  Skipping {n_visits} visits (only {len(session_dirs)} sessions available)"
            )
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"Running reconstruction with {n_visits} visit(s)")
        logger.info(f"Sessions: {', '.join(session_names[:n_visits])}")
        logger.info(f"{'='*80}")

        # Combine first n sessions
        combined_dir = output_base / f"combined_{n_visits}_visits"
        num_frames = combine_sessions(
            session_dirs[:n_visits],
            combined_dir,
            session_names[:n_visits],
            use_cache=use_cache,
        )

        if num_frames == 0:
            logger.error(f"No frames to process for {n_visits} visits")
            continue

        # Run SfM reconstruction (COLMAP or GLOMAP based on config)
        output_dir = output_base / f"reconstruction_{n_visits}_visits"
        result = run_sfm_reconstruction(
            combined_dir, output_dir, use_cache=use_cache, **colmap_kwargs
        )

        if result.success:
            logger.info(
                f"✅ {n_visits} visit(s): {result.num_registered_images} images, "
                f"{result.num_3d_points} points, {result.execution_time:.1f}s, "
                f"reproj_error={result.mean_reprojection_error:.3f}px"
            )
        else:
            logger.warning(f"❌ {n_visits} visit(s) failed after {result.execution_time:.1f}s")

        results[n_visits] = result

    logger.info(f"\n{'='*80}")
    logger.info("Multi-visit experiment complete")
    logger.info(f"{'='*80}")

    return results


def summarize_multivisit_results(results: Dict[int, ReconstructionResult]) -> Dict[str, Any]:
    """
    Create summary of multi-visit experiment results.

    Args:
        results: Dictionary mapping visit count to ReconstructionResult

    Returns:
        Summary dictionary with statistics

    Example:
        >>> summary = summarize_multivisit_results(results)
        >>> print(summary["best_visit_count"])
    """
    if not results:
        return {"num_experiments": 0, "successful_experiments": 0}

    successful_results = {k: v for k, v in results.items() if v.success}

    summary = {
        "num_experiments": len(results),
        "successful_experiments": len(successful_results),
        "visit_counts": sorted(results.keys()),
        "results_by_visit": {},
    }

    for n_visits, result in results.items():
        summary["results_by_visit"][n_visits] = {
            "success": result.success,
            "num_images": result.num_registered_images,
            "num_points": result.num_3d_points,
            "reprojection_error": result.mean_reprojection_error,
            "execution_time": result.execution_time,
        }

    # Find best visit count (most registered images)
    if successful_results:
        best_visit_count = max(
            successful_results.keys(),
            key=lambda k: successful_results[k].num_registered_images,
        )
        summary["best_visit_count"] = best_visit_count
        summary["best_num_images"] = successful_results[best_visit_count].num_registered_images

    return summary


# Validation
if __name__ == "__main__":
    """
    Validation of multivisit.py functionality.
    Tests session combining and result summarization.
    """
    import sys
    import tempfile
    from colmap_runner import ReconstructionResult

    all_validation_failures = []
    total_tests = 0

    # Test 1: Session combining with temp files
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test sessions
            session1 = tmpdir / "session_01"
            session2 = tmpdir / "session_02"
            session1.mkdir()
            session2.mkdir()

            # Create test frames
            (session1 / "frame_001.jpg").touch()
            (session1 / "frame_002.jpg").touch()
            (session2 / "frame_001.jpg").touch()
            (session2 / "frame_002.jpg").touch()
            (session2 / "frame_003.jpg").touch()

            output_dir = tmpdir / "combined"

            # Combine sessions
            num_frames = combine_sessions([session1, session2], output_dir)

            if num_frames != 5:
                all_validation_failures.append(
                    f"Session combining: Expected 5 frames, got {num_frames}"
                )

            # Check output files exist
            combined_files = list(output_dir.glob("*.jpg"))
            if len(combined_files) != 5:
                all_validation_failures.append(
                    f"Session combining: Expected 5 output files, got {len(combined_files)}"
                )

    except Exception as e:
        all_validation_failures.append(f"Session combining: Exception raised: {e}")

    # Test 2: Result summarization
    total_tests += 1
    try:
        # Create mock results
        test_results = {
            1: ReconstructionResult(
                success=True,
                num_registered_images=50,
                num_3d_points=1000,
                mean_reprojection_error=0.5,
                execution_time=60.0,
                sparse_path=Path("/test/sparse"),
            ),
            2: ReconstructionResult(
                success=True,
                num_registered_images=95,
                num_3d_points=2500,
                mean_reprojection_error=0.4,
                execution_time=120.0,
                sparse_path=Path("/test/sparse"),
            ),
            3: ReconstructionResult(
                success=True,
                num_registered_images=140,
                num_3d_points=4000,
                mean_reprojection_error=0.35,
                execution_time=180.0,
                sparse_path=Path("/test/sparse"),
            ),
        }

        summary = summarize_multivisit_results(test_results)

        if summary["num_experiments"] != 3:
            all_validation_failures.append(
                f"Result summarization: Expected 3 experiments, got {summary['num_experiments']}"
            )

        if summary["successful_experiments"] != 3:
            all_validation_failures.append(
                f"Result summarization: Expected 3 successful, got {summary['successful_experiments']}"
            )

        if summary.get("best_visit_count") != 3:
            all_validation_failures.append(
                f"Result summarization: Expected best_visit_count=3, got {summary.get('best_visit_count')}"
            )

    except Exception as e:
        all_validation_failures.append(f"Result summarization: Exception raised: {e}")

    # Test 3: Empty results handling
    total_tests += 1
    try:
        empty_summary = summarize_multivisit_results({})

        if empty_summary["num_experiments"] != 0:
            all_validation_failures.append(
                f"Empty results: Expected 0 experiments, got {empty_summary['num_experiments']}"
            )

    except Exception as e:
        all_validation_failures.append(f"Empty results: Exception raised: {e}")

    # Final validation result
    if all_validation_failures:
        print(
            f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Multi-visit module validated and ready for use")
        sys.exit(0)
