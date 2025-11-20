#!/usr/bin/env python3
"""
LaMAR Dataset Handler

This module provides utilities for working with the LaMAR dataset, including
loading pre-built COLMAP reconstructions, reading metadata, and preparing
data for use with the SfM pipeline.

LaMAR Dataset Information:
- Source: https://cvg-data.inf.ethz.ch/lamar/
- GitHub: https://github.com/microsoft/lamar-benchmark
- License: CC BY-SA 4.0
- Documentation: https://github.com/microsoft/lamar-benchmark/blob/main/CAPTURE.md

Third-party Dependencies:
- pycolmap: https://github.com/colmap/pycolmap
- pathlib: https://docs.python.org/3/library/pathlib.html
- json: https://docs.python.org/3/library/json.html

Sample Input:
    scene = "CAB"
    colmap_path = Path("datasets/lamar/colmap/CAB")
    reconstruction = load_lamar_reconstruction(colmap_path)

Expected Output:
    Loaded COLMAP reconstruction with cameras, images, and 3D points
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pycolmap
from loguru import logger


@dataclass
class LamarScene:
    """LaMAR scene metadata."""
    name: str
    base_path: Path
    colmap_path: Optional[Path] = None
    benchmark_path: Optional[Path] = None
    description: str = ""
    num_images: int = 0
    num_points3d: int = 0
    num_cameras: int = 0


@dataclass
class LamarMetadata:
    """Metadata from LaMAR JSON files."""
    device_type: str  # 'hololens', 'phone', 'navvis'
    sessions: List[str]
    capture_info: Dict


def load_lamar_reconstruction(colmap_path: Path, model_name: str = "0") -> Optional[pycolmap.Reconstruction]:
    """
    Load pre-built COLMAP reconstruction from LaMAR dataset.

    The LaMAR dataset includes pre-built COLMAP reconstructions in the
    colmap/ directory. Each scene contains a sparse/ directory with the
    reconstruction data.

    Args:
        colmap_path: Path to LaMAR scene COLMAP directory (e.g., datasets/lamar/colmap/CAB)
        model_name: Reconstruction model subdirectory name (default: "0")

    Returns:
        pycolmap.Reconstruction object or None if loading fails

    Raises:
        FileNotFoundError: If COLMAP path doesn't exist
    """
    if not colmap_path.exists():
        raise FileNotFoundError(f"COLMAP path does not exist: {colmap_path}")

    # LaMAR COLMAP reconstructions are in sparse/ subdirectory
    sparse_path = colmap_path / "sparse" / model_name

    if not sparse_path.exists():
        # Try without model_name subdirectory
        sparse_path = colmap_path / "sparse"
        if not sparse_path.exists():
            logger.error(f"No sparse reconstruction found at {colmap_path}/sparse")
            return None

    logger.info(f"Loading COLMAP reconstruction from {sparse_path}")

    try:
        reconstruction = pycolmap.Reconstruction(str(sparse_path))

        logger.info(f"Loaded reconstruction with:")
        logger.info(f"  - {len(reconstruction.cameras)} cameras")
        logger.info(f"  - {len(reconstruction.images)} images")
        logger.info(f"  - {len(reconstruction.points3D)} 3D points")

        return reconstruction

    except Exception as e:
        logger.error(f"Failed to load reconstruction: {e}")
        return None


def load_lamar_metadata(scene_path: Path, device_type: str = "hololens") -> Optional[LamarMetadata]:
    """
    Load metadata JSON files from LaMAR raw data.

    The raw/ directory contains metadata files for each device type:
    - metadata_hololens.json
    - metadata_phone.json

    Args:
        scene_path: Path to LaMAR scene directory (e.g., datasets/lamar/raw/CAB)
        device_type: Type of device metadata to load ('hololens' or 'phone')

    Returns:
        LamarMetadata object or None if loading fails
    """
    metadata_file = scene_path / f"metadata_{device_type}.json"

    if not metadata_file.exists():
        logger.warning(f"Metadata file not found: {metadata_file}")
        return None

    try:
        with open(metadata_file, 'r') as f:
            data = json.load(f)

        # Extract session information
        sessions = list(data.get('sessions', {}).keys()) if 'sessions' in data else []

        metadata = LamarMetadata(
            device_type=device_type,
            sessions=sessions,
            capture_info=data
        )

        logger.info(f"Loaded {device_type} metadata: {len(sessions)} sessions")
        return metadata

    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_file}: {e}")
        return None


def get_lamar_scene_info(scene_name: str, base_dir: Path) -> LamarScene:
    """
    Get comprehensive information about a LaMAR scene.

    Args:
        scene_name: Name of scene ('CAB', 'HGE', or 'LIN')
        base_dir: Base directory for LaMAR dataset (e.g., datasets/lamar)

    Returns:
        LamarScene object with scene information
    """
    scene = LamarScene(
        name=scene_name,
        base_path=base_dir
    )

    # Check for COLMAP reconstruction
    colmap_path = base_dir / "colmap" / scene_name
    if colmap_path.exists():
        scene.colmap_path = colmap_path

        # Try to load reconstruction to get stats
        reconstruction = load_lamar_reconstruction(colmap_path)
        if reconstruction:
            scene.num_cameras = len(reconstruction.cameras)
            scene.num_images = len(reconstruction.images)
            scene.num_points3d = len(reconstruction.points3D)

    # Check for benchmark data
    benchmark_path = base_dir / "benchmark" / scene_name
    if benchmark_path.exists():
        scene.benchmark_path = benchmark_path

    return scene


def list_lamar_scenes(base_dir: Path) -> List[LamarScene]:
    """
    List all available LaMAR scenes in the dataset directory.

    Args:
        base_dir: Base directory for LaMAR dataset

    Returns:
        List of LamarScene objects
    """
    scene_names = ["CAB", "HGE", "LIN"]
    scenes = []

    for scene_name in scene_names:
        scene = get_lamar_scene_info(scene_name, base_dir)

        # Only include scenes that have at least COLMAP or benchmark data
        if scene.colmap_path or scene.benchmark_path:
            scenes.append(scene)
            logger.info(f"Found scene: {scene_name}")

    return scenes


def export_lamar_images_list(
    reconstruction: pycolmap.Reconstruction,
    output_file: Path
) -> int:
    """
    Export list of images from LaMAR COLMAP reconstruction.

    This is useful for creating image lists for further processing or
    for understanding which images were successfully registered.

    Args:
        reconstruction: COLMAP reconstruction object
        output_file: Path to output text file

    Returns:
        Number of images exported
    """
    images = reconstruction.images

    with open(output_file, 'w') as f:
        f.write("# LaMAR Image List\n")
        f.write("# Format: image_id, image_name, camera_id, num_points3d\n\n")

        for img_id, image in images.items():
            num_points = len(image.points2D)
            f.write(f"{img_id},{image.name},{image.camera_id},{num_points}\n")

    logger.info(f"Exported {len(images)} images to {output_file}")
    return len(images)


def get_lamar_camera_params(reconstruction: pycolmap.Reconstruction) -> Dict[int, Dict]:
    """
    Extract camera parameters from LaMAR COLMAP reconstruction.

    Args:
        reconstruction: COLMAP reconstruction object

    Returns:
        Dictionary mapping camera_id to camera parameters
    """
    camera_params = {}

    for cam_id, camera in reconstruction.cameras.items():
        camera_params[cam_id] = {
            'model': camera.model.name,
            'width': camera.width,
            'height': camera.height,
            'params': camera.params.tolist(),
        }

        logger.debug(f"Camera {cam_id}: {camera.model.name}, {camera.width}x{camera.height}")

    return camera_params


def get_lamar_images_path(base_dir: Path, scene_name: str) -> Optional[Path]:
    """
    Get the path to raw images directory for a LaMAR scene.

    LaMAR images are organized in a hierarchical structure:
    colmap/{scene}/images/{session_id}/{camera_name}/*.jpg

    Where:
    - session_id: HoloLens capture session (e.g., hl_2021-06-02-11-29-38-299)
    - camera_name: Camera identifier (hetlf, hetll, hetrf, hetrr)

    Args:
        base_dir: Base directory for LaMAR dataset
        scene_name: Name of scene ('CAB', 'HGE', or 'LIN')

    Returns:
        Path to images directory or None if not found
    """
    images_path = base_dir / "colmap" / scene_name / "images"

    if not images_path.exists():
        logger.error(f"Images directory not found: {images_path}")
        return None

    # Verify it contains images by checking for at least one session directory
    session_dirs = list(images_path.glob("hl_*"))
    if not session_dirs:
        logger.error(f"No session directories found in {images_path}")
        return None

    logger.info(f"Found images directory with {len(session_dirs)} sessions: {images_path}")
    return images_path


def count_lamar_images(images_path: Path) -> int:
    """
    Count total number of images in a LaMAR images directory.

    Args:
        images_path: Path to images directory (colmap/{scene}/images/)

    Returns:
        Total number of JPG images
    """
    if not images_path or not images_path.exists():
        return 0

    # Count all .jpg files recursively
    jpg_files = list(images_path.glob("**/*.jpg"))
    count = len(jpg_files)

    logger.info(f"Counted {count} images in {images_path}")
    return count


def list_lamar_sessions(images_path: Path) -> List[str]:
    """
    List all capture sessions in a LaMAR images directory.

    Args:
        images_path: Path to images directory (colmap/{scene}/images/)

    Returns:
        List of session IDs (e.g., ['hl_2021-06-02-11-29-38-299', ...])
    """
    if not images_path or not images_path.exists():
        return []

    # Find all session directories (start with "hl_")
    session_dirs = sorted(images_path.glob("hl_*"))
    sessions = [d.name for d in session_dirs if d.is_dir()]

    logger.info(f"Found {len(sessions)} sessions in {images_path}")
    return sessions


def get_image_timestamps(image_paths: List[Path]) -> List[float]:
    """
    Extract timestamps from LaMAR images.

    Attempts multiple strategies in order:
    1. Parse from image filename if it contains timestamp pattern (preferred so frames vary)
    2. Parse from session directory name (e.g., hl_2021-06-02-11-29-38-299)
    3. Fallback to sequential ordering (0, 1, 2, ...)

    Args:
        image_paths: List of image file paths

    Returns:
        List of timestamps in seconds (float). For sequential fallback,
        returns frame indices as timestamps (0.0, 1.0, 2.0, ...)

    Example:
        >>> images = list(Path("datasets/lamar/colmap/CAB/images").glob("**/*.jpg"))
        >>> timestamps = get_image_timestamps(images[:100])
        >>> print(f"First timestamp: {timestamps[0]:.2f}s")
    """
    import re
    from datetime import datetime

    timestamps = []

    for img_path in image_paths:
        timestamp = None

        # Strategy 1: Parse from filename if it contains timestamp-like digits
        # Example filenames: 326766299.jpg, 1000133345.jpg
        filename = img_path.stem
        match = re.search(r'(\d{8,})', filename)  # 8+ digits likely a timestamp
        if match:
            try:
                ts_value = int(match.group(1))
                # Treat numeric filenames as microsecond timestamps to preserve
                # realistic per-frame spacing when sampling by time.
                timestamp = ts_value / 1_000_000.0
            except (ValueError, OverflowError):
                timestamp = None

        # Strategy 2: Extract from session directory name when filenames lack usable time info
        if timestamp is None:
            # Format: hl_2021-06-02-11-29-38-299 -> datetime
            parts = img_path.parts
            for part in parts:
                if part.startswith("hl_") or part.startswith("phone_") or part.startswith("navvis_"):
                    # Try to parse datetime from session name
                    # Pattern: device_YYYY-MM-DD-HH-MM-SS-milliseconds
                    match = re.search(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-?(\d+)?', part)
                    if match:
                        try:
                            year, month, day, hour, minute, second = map(int, match.groups()[:6])
                            milliseconds = int(match.group(7)) if match.group(7) else 0

                            dt = datetime(year, month, day, hour, minute, second)
                            # Convert to timestamp (seconds since epoch)
                            timestamp = dt.timestamp() + (milliseconds / 1000.0)
                            break
                        except (ValueError, OverflowError):
                            pass

        # Strategy 3: Fallback to sequential index
        if timestamp is None:
            # Use index as timestamp (will maintain relative ordering)
            timestamp = float(len(timestamps))

        timestamps.append(timestamp)

    # Normalize timestamps to start from 0 if using session-based timestamps
    if timestamps and timestamps[0] > 1000:  # Likely real timestamps
        min_timestamp = min(timestamps)
        timestamps = [t - min_timestamp for t in timestamps]

    return timestamps


def sample_images_by_fps(
    image_paths: List[Path],
    timestamps: List[float],
    target_fps: float,
    session_name: str = "session"
) -> List[Path]:
    """
    Sample images based on target frame rate using time-based selection.

    Mirrors the sampling logic from dataset.py extract_frames_from_rosbag().
    Selects images such that the time interval between consecutive selected
    images is approximately 1.0 / target_fps.

    Args:
        image_paths: List of image file paths (must be sorted by time)
        timestamps: List of timestamps corresponding to each image (in seconds)
        target_fps: Target frame rate (e.g., 0.25 = 1 frame every 4 seconds)
        session_name: Name of session for logging

    Returns:
        Filtered list of image paths sampled at target_fps

    Example:
        >>> images = sorted(Path("session").glob("*.jpg"))
        >>> timestamps = get_image_timestamps(images)
        >>> sampled = sample_images_by_fps(images, timestamps, target_fps=0.25)
        >>> print(f"Sampled {len(sampled)} from {len(images)} images")
    """
    if not image_paths or not timestamps:
        logger.warning(f"{session_name}: No images to sample")
        return []

    if len(image_paths) != len(timestamps):
        raise ValueError(
            f"Mismatch: {len(image_paths)} images but {len(timestamps)} timestamps"
        )

    # Calculate sample interval
    sample_interval = 1.0 / target_fps

    # Sort by timestamp to ensure temporal ordering
    sorted_pairs = sorted(zip(timestamps, image_paths))
    timestamps_sorted = [t for t, _ in sorted_pairs]
    images_sorted = [img for _, img in sorted_pairs]

    # Sample images based on time intervals
    sampled_images = []
    last_saved_time = None

    for i, (current_time, img_path) in enumerate(zip(timestamps_sorted, images_sorted)):
        # Save first image and images where enough time has passed
        if last_saved_time is None or (current_time - last_saved_time) >= sample_interval:
            sampled_images.append(img_path)
            last_saved_time = current_time

    # Log sampling results
    original_count = len(image_paths)
    sampled_count = len(sampled_images)
    percentage = (sampled_count / original_count * 100) if original_count > 0 else 0

    # Format interval display based on magnitude
    if sample_interval >= 1.0:
        interval_str = f"{sample_interval:.1f}s"
    else:
        interval_str = f"{sample_interval:.3f}s"

    logger.info(
        f"{session_name}: Sampled {sampled_count} of {original_count} images "
        f"({percentage:.1f}%) at {target_fps} FPS (1 frame every {interval_str})"
    )

    return sampled_images


def sample_lamar_images_by_session(
    images_path: Path,
    target_fps: float,
) -> List[Path]:
    """
    Sample LaMAR images per session at target frame rate.

    LaMAR images are organized as: images/{session}/{camera}/*.jpg
    This function samples images independently within each session/camera
    combination to ensure balanced representation.

    Args:
        images_path: Path to LaMAR images directory (e.g., datasets/lamar/colmap/CAB/images)
        target_fps: Target frame rate for sampling (e.g., 0.25)

    Returns:
        List of sampled image paths from all sessions

    Example:
        >>> images_path = Path("datasets/lamar/colmap/CAB/images")
        >>> sampled = sample_lamar_images_by_session(images_path, target_fps=0.25)
        >>> print(f"Sampled {len(sampled)} total images")
    """
    if not images_path.exists():
        logger.error(f"Images path does not exist: {images_path}")
        return []

    all_sampled_images = []

    # Find all session directories (handle all device prefixes: hl_, phone_, navvis_, ios_, etc.)
    session_dirs = sorted([d for d in images_path.iterdir() if d.is_dir()])

    if not session_dirs:
        logger.warning(f"No session directories found in {images_path}")
        return []

    logger.info(f"Found {len(session_dirs)} sessions, sampling each at {target_fps} FPS")

    for session_dir in session_dirs:
        if not session_dir.is_dir():
            continue

        # Find all camera subdirectories in this session
        camera_dirs = [d for d in session_dir.iterdir() if d.is_dir()]

        for camera_dir in camera_dirs:
            # Get all images in this camera directory
            images = sorted(camera_dir.glob("*.jpg"))

            if not images:
                continue

            session_camera_name = f"{session_dir.name}/{camera_dir.name}"

            # Extract timestamps for these images
            timestamps = get_image_timestamps(images)

            # Sample images based on FPS
            sampled = sample_images_by_fps(
                images,
                timestamps,
                target_fps,
                session_name=session_camera_name
            )

            all_sampled_images.extend(sampled)

    total_original = sum(len(list(d.glob("**/*.jpg"))) for d in session_dirs if d.is_dir())
    total_sampled = len(all_sampled_images)
    percentage = (total_sampled / total_original * 100) if total_original > 0 else 0

    logger.info(
        f"Total: Sampled {total_sampled} of {total_original} images "
        f"({percentage:.1f}%) across all sessions at {target_fps} FPS"
    )

    return all_sampled_images


def validate_lamar_dataset(base_dir: Path, scene_name: str) -> Tuple[bool, List[str]]:
    """
    Validate that LaMAR dataset is properly downloaded and extracted.

    Args:
        base_dir: Base directory for LaMAR dataset
        scene_name: Name of scene to validate

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check base directory
    if not base_dir.exists():
        issues.append(f"Base directory does not exist: {base_dir}")
        return False, issues

    # Check COLMAP data
    colmap_path = base_dir / "colmap" / scene_name
    if not colmap_path.exists():
        issues.append(f"COLMAP directory not found: {colmap_path}")
    else:
        sparse_path = colmap_path / "sparse"
        if not sparse_path.exists():
            issues.append(f"Sparse reconstruction not found: {sparse_path}")
        else:
            # Check for required files
            required_files = ['cameras.bin', 'images.bin', 'points3D.bin']
            for filename in required_files:
                file_path = sparse_path / "0" / filename
                if not file_path.exists():
                    # Try without "0" subdirectory
                    file_path = sparse_path / filename
                    if not file_path.exists():
                        issues.append(f"Missing required file: {filename}")

    # Check images directory
    images_path = base_dir / "colmap" / scene_name / "images"
    if not images_path.exists():
        issues.append(f"Images directory not found: {images_path}")
    else:
        # Verify it contains at least one session
        session_dirs = list(images_path.glob("hl_*"))
        if not session_dirs:
            issues.append(f"No session directories found in {images_path}")

    # Check benchmark data (optional)
    benchmark_path = base_dir / "benchmark" / scene_name
    if not benchmark_path.exists():
        logger.warning(f"Benchmark directory not found (optional): {benchmark_path}")

    is_valid = len(issues) == 0
    return is_valid, issues


if __name__ == "__main__":
    """
    Validation function to test LaMAR dataset handling.

    Tests:
    1. Loading COLMAP reconstruction
    2. Reading camera parameters
    3. Exporting image lists
    4. Validating dataset structure
    """
    from loguru import logger

    # Setup logging
    logger.add("logs/lamar_handler_test.log", rotation="10 MB")

    # Track validation failures
    all_validation_failures = []
    total_tests = 0

    # Test configuration
    base_dir = Path("datasets/lamar")
    scene_name = "CAB"
    colmap_path = base_dir / "colmap" / scene_name

    print("="*80)
    print("LaMAR Dataset Handler Validation")
    print("="*80)

    # Test 1: Validate dataset structure
    total_tests += 1
    print(f"\nTest {total_tests}: Validating dataset structure...")
    is_valid, issues = validate_lamar_dataset(base_dir, scene_name)
    if not is_valid:
        all_validation_failures.append(f"Dataset validation: Found {len(issues)} issues: {issues}")
        print(f"❌ Dataset validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nNote: Download dataset first using:")
        print("  uv run python scripts/download_lamar_dataset.py")
    else:
        print(f"✅ Dataset structure is valid")

    # Only continue with other tests if dataset is valid
    if is_valid:
        # Test 2: Load COLMAP reconstruction
        total_tests += 1
        print(f"\nTest {total_tests}: Loading COLMAP reconstruction...")
        try:
            reconstruction = load_lamar_reconstruction(colmap_path)
            if reconstruction is None:
                all_validation_failures.append("COLMAP loading: Failed to load reconstruction")
                print("❌ Failed to load reconstruction")
            elif len(reconstruction.images) == 0:
                all_validation_failures.append("COLMAP loading: Reconstruction has no images")
                print("❌ Reconstruction has no images")
            else:
                print(f"✅ Loaded reconstruction with {len(reconstruction.images)} images")

                # Test 3: Extract camera parameters
                total_tests += 1
                print(f"\nTest {total_tests}: Extracting camera parameters...")
                camera_params = get_lamar_camera_params(reconstruction)
                if len(camera_params) == 0:
                    all_validation_failures.append("Camera params: No cameras found")
                    print("❌ No cameras found")
                else:
                    print(f"✅ Extracted {len(camera_params)} camera(s)")
                    for cam_id, params in camera_params.items():
                        print(f"  Camera {cam_id}: {params['model']}, {params['width']}x{params['height']}")

                # Test 4: Export images list
                total_tests += 1
                print(f"\nTest {total_tests}: Exporting images list...")
                output_file = Path("test_lamar_images.txt")
                try:
                    num_exported = export_lamar_images_list(reconstruction, output_file)
                    if num_exported == 0:
                        all_validation_failures.append("Image export: No images exported")
                        print("❌ No images exported")
                    else:
                        print(f"✅ Exported {num_exported} images to {output_file}")
                        # Clean up test file
                        output_file.unlink()
                except Exception as e:
                    all_validation_failures.append(f"Image export: Exception - {e}")
                    print(f"❌ Export failed: {e}")

                # Test 5: Get scene info
                total_tests += 1
                print(f"\nTest {total_tests}: Getting scene information...")
                try:
                    scene = get_lamar_scene_info(scene_name, base_dir)
                    if scene.num_images == 0:
                        all_validation_failures.append("Scene info: No images in scene")
                        print("❌ No images in scene")
                    else:
                        print(f"✅ Scene info retrieved:")
                        print(f"  - Scene: {scene.name}")
                        print(f"  - Images: {scene.num_images}")
                        print(f"  - Cameras: {scene.num_cameras}")
                        print(f"  - 3D Points: {scene.num_points3d}")
                except Exception as e:
                    all_validation_failures.append(f"Scene info: Exception - {e}")
                    print(f"❌ Failed to get scene info: {e}")

                # Test 6: Timestamp extraction
                total_tests += 1
                print(f"\nTest {total_tests}: Testing timestamp extraction...")
                try:
                    images_path = get_lamar_images_path(base_dir, scene_name)
                    if images_path:
                        # Get first 100 images from first session
                        test_images = list(images_path.glob("**/*.jpg"))[:100]
                        if test_images:
                            timestamps = get_image_timestamps(test_images)
                            if len(timestamps) != len(test_images):
                                all_validation_failures.append(f"Timestamp extraction: Expected {len(test_images)} timestamps, got {len(timestamps)}")
                                print(f"❌ Timestamp count mismatch: expected {len(test_images)}, got {len(timestamps)}")
                            elif not all(isinstance(t, float) for t in timestamps):
                                all_validation_failures.append("Timestamp extraction: Not all timestamps are floats")
                                print("❌ Some timestamps are not floats")
                            else:
                                print(f"✅ Extracted {len(timestamps)} timestamps")
                                print(f"  - First timestamp: {timestamps[0]:.2f}s")
                                print(f"  - Last timestamp: {timestamps[-1]:.2f}s")
                                print(f"  - Time span: {timestamps[-1] - timestamps[0]:.2f}s")
                        else:
                            all_validation_failures.append("Timestamp extraction: No images found for testing")
                            print("❌ No images found for testing")
                    else:
                        all_validation_failures.append("Timestamp extraction: Images path not found")
                        print("❌ Images path not found")
                except Exception as e:
                    all_validation_failures.append(f"Timestamp extraction: Exception - {e}")
                    print(f"❌ Exception: {e}")

                # Test 7: FPS sampling
                total_tests += 1
                print(f"\nTest {total_tests}: Testing FPS sampling...")
                try:
                    images_path = get_lamar_images_path(base_dir, scene_name)
                    if images_path:
                        # Test with 0.25 FPS (like Hilti default)
                        test_fps = 0.25
                        sampled_images = sample_lamar_images_by_session(images_path, test_fps)

                        if not sampled_images:
                            all_validation_failures.append("FPS sampling: No images after sampling")
                            print("❌ No images after sampling")
                        else:
                            # Calculate expected reduction
                            all_images = list(images_path.glob("**/*.jpg"))
                            reduction_ratio = len(sampled_images) / len(all_images) if all_images else 0

                            print(f"✅ FPS sampling at {test_fps} FPS:")
                            print(f"  - Original images: {len(all_images)}")
                            print(f"  - Sampled images: {len(sampled_images)}")
                            print(f"  - Reduction: {reduction_ratio*100:.1f}% of original")
                            print(f"  - Sample interval: {1.0/test_fps:.1f}s between frames")

                            # Verify sampling actually reduced image count
                            if len(sampled_images) >= len(all_images):
                                all_validation_failures.append("FPS sampling: Sampling did not reduce image count")
                                print("❌ Warning: Sampling did not reduce image count")
                    else:
                        all_validation_failures.append("FPS sampling: Images path not found")
                        print("❌ Images path not found")
                except Exception as e:
                    all_validation_failures.append(f"FPS sampling: Exception - {e}")
                    print(f"❌ Exception: {e}")

        except FileNotFoundError as e:
            all_validation_failures.append(f"COLMAP loading: FileNotFoundError - {e}")
            print(f"❌ {e}")
        except Exception as e:
            all_validation_failures.append(f"COLMAP loading: Exception - {e}")
            print(f"❌ Unexpected error: {e}")

    # Final validation result
    print("\n" + "="*80)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        print("="*80)
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("LaMAR handler is validated and ready to use")
        print("="*80)
        sys.exit(0)
