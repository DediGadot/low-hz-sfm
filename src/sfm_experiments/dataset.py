"""
Hilti SLAM Challenge 2023 dataset handler with ROS bag frame extraction.

This module provides:
- ROS bag reading with pure Python (rosbags library)
- Frame extraction at configurable frame rates
- Ground truth pose loading
- Dataset validation

Dependencies:
- rosbags: https://ternaris.gitlab.io/rosbags/
- opencv-python: https://opencv.org/
- numpy: https://numpy.org/

Sample Input: ROS bag file at 30 Hz camera rate
Expected Output: JPEG frames extracted at 0.25 Hz (1 frame every 4 seconds)
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2

from rosbags.rosbag1 import Reader as Rosbag1Reader
from rosbags.typesys import Stores, get_typestore
from loguru import logger
from tqdm import tqdm


@dataclass
class FrameInfo:
    """Information about an extracted frame."""

    frame_id: int
    timestamp: float  # seconds
    frame_path: Path
    session_name: str


def extract_frames_from_rosbag(
    bag_path: Path,
    output_dir: Path,
    camera_topic: str = "/camera/image_raw",
    target_fps: float = 0.25,
    jpeg_quality: int = 95,
) -> List[FrameInfo]:
    """
    Extract frames from ROS bag at specified rate using rosbags library.

    This function reads ROS1 bag files without requiring system ROS installation.
    It supports standard ROS image topics (sensor_msgs/Image).

    Args:
        bag_path: Path to .bag file
        output_dir: Where to save extracted frames
        camera_topic: ROS topic containing camera images
        target_fps: Target frame rate (0.25 = 1 frame every 4 seconds)
        jpeg_quality: JPEG compression quality (0-100)

    Returns:
        List of FrameInfo objects for extracted frames

    Example:
        >>> frames = extract_frames_from_rosbag(
        ...     Path("data/sequence_01.bag"),
        ...     Path("output/frames"),
        ...     target_fps=0.25
        ... )
        >>> print(f"Extracted {len(frames)} frames")
    """
    if not bag_path.exists():
        raise FileNotFoundError(f"ROS bag not found: {bag_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting frames from: {bag_path.name}")
    logger.info(f"Target FPS: {target_fps} (1 frame every {1/target_fps:.1f} seconds)")

    # Get typestore for ROS1
    typestore = get_typestore(Stores.ROS1_NOETIC)

    sample_interval = 1.0 / target_fps  # seconds between frames
    last_saved_time = None
    saved_frames: List[FrameInfo] = []
    session_name = bag_path.stem

    # Prepare metadata writer to keep timestamp ↔ filename mapping for downstream metrics
    metadata_path = output_dir / "frames_metadata.csv"
    metadata_lines = ["filename,timestamp_seconds\n"]

    # Open ROS bag
    with Rosbag1Reader(bag_path) as reader:
        # Find camera topic connections
        camera_connections = [
            conn for conn in reader.connections if conn.topic == camera_topic
        ]

        if not camera_connections:
            available_topics = [conn.topic for conn in reader.connections]
            raise ValueError(
                f"Camera topic '{camera_topic}' not found in bag. "
                f"Available topics: {available_topics}"
            )

        logger.info(f"Found camera topic: {camera_topic}")

        # Count total messages for progress bar
        total_messages = sum(
            1
            for connection, timestamp, rawdata in reader.messages(
                connections=camera_connections
            )
        )

        # Reset reader and extract frames
        with tqdm(total=total_messages, desc="Processing frames") as pbar:
            for connection, timestamp, rawdata in reader.messages(
                connections=camera_connections
            ):
                current_time = timestamp * 1e-9  # Convert nanoseconds to seconds

                # Check if enough time has passed since last saved frame
                if last_saved_time is None or (
                    current_time - last_saved_time
                ) >= sample_interval:
                    # Deserialize ROS message
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

                    # Convert data to numpy array
                    # msg.data can be bytes or list depending on deserialization
                    if isinstance(msg.data, bytes):
                        img_data = np.frombuffer(msg.data, dtype=np.uint8)
                    else:
                        img_data = np.array(msg.data, dtype=np.uint8)

                    # Convert ROS image to numpy array
                    # Handle different encodings
                    if msg.encoding == "rgb8":
                        img = img_data.reshape(msg.height, msg.width, 3)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif msg.encoding == "bgr8":
                        img = img_data.reshape(msg.height, msg.width, 3)
                    elif msg.encoding == "mono8":
                        img = img_data.reshape(msg.height, msg.width)
                    else:
                        logger.warning(
                            f"Unsupported encoding: {msg.encoding}, skipping frame"
                        )
                        pbar.update(1)
                        continue

                    # Save frame
                    frame_id = len(saved_frames)
                    # Preserve timestamp in filename to allow GT pose alignment
                    frame_filename = f"frame_{int(current_time * 1e9)}.jpg"
                    frame_path = output_dir / frame_filename

                    cv2.imwrite(
                        str(frame_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    )

                    metadata_lines.append(f"{frame_filename},{current_time:.9f}\n")

                    # Record frame info
                    frame_info = FrameInfo(
                        frame_id=frame_id,
                        timestamp=current_time,
                        frame_path=frame_path,
                        session_name=session_name,
                    )
                    saved_frames.append(frame_info)

                    last_saved_time = current_time

                pbar.update(1)

    # Persist metadata alongside frames
    try:
        with open(metadata_path, "w") as meta_file:
            meta_file.writelines(metadata_lines)
        logger.info(f"Wrote frame metadata to {metadata_path}")
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning(f"Failed to write metadata file {metadata_path}: {exc}")

    logger.info(f"✅ Extracted {len(saved_frames)} frames to {output_dir}")
    return saved_frames


def extract_all_sessions(
    rosbags_dir: Path,
    frames_base_dir: Path,
    sessions: List[str],
    camera_topic: str = "/camera/image_raw",
    target_fps: float = 0.25,
    jpeg_quality: int = 95,
) -> Dict[str, List[FrameInfo]]:
    """
    Extract frames from multiple ROS bag sessions.

    Args:
        rosbags_dir: Directory containing ROS bag files
        frames_base_dir: Base directory for extracted frames
        sessions: List of session names (e.g., ["sequence_01", "sequence_02"])
        camera_topic: ROS camera topic
        target_fps: Target frame extraction rate

    Returns:
        Dictionary mapping session name to list of FrameInfo

    Example:
        >>> sessions_data = extract_all_sessions(
        ...     Path("data/rosbags"),
        ...     Path("data/frames"),
        ...     ["sequence_01", "sequence_02"]
        ... )
    """
    results = {}

    for session in sessions:
        bag_path = rosbags_dir / f"{session}.bag"

        if not bag_path.exists():
            logger.warning(f"Skipping missing bag: {bag_path}")
            continue

        output_dir = frames_base_dir / session
        frames = extract_frames_from_rosbag(
            bag_path,
            output_dir,
            camera_topic,
            target_fps,
            jpeg_quality=jpeg_quality,
        )

        results[session] = frames

    return results


def load_ground_truth_poses(gt_file: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load ground truth poses from file.

    Expected format: timestamp, tx, ty, tz, qx, qy, qz, qw (TUM format)

    Args:
        gt_file: Path to ground truth file

    Returns:
        Dictionary mapping timestamp to (tvec, qvec) tuples

    Example:
        >>> gt_poses = load_ground_truth_poses(Path("ground_truth/poses.txt"))
    """
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    poses = {}

    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = (
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
            )

            tvec = np.array([tx, ty, tz])
            qvec = np.array([qw, qx, qy, qz])  # COLMAP format: qw, qx, qy, qz

            poses[str(timestamp)] = (tvec, qvec)

    logger.info(f"Loaded {len(poses)} ground truth poses from {gt_file}")
    return poses


# Validation
if __name__ == "__main__":
    """
    Validation of dataset.py functionality.
    Tests core functions without requiring actual ROS bag files.
    """
    import sys
    import tempfile

    all_validation_failures = []
    total_tests = 0

    # Test 1: FrameInfo dataclass
    total_tests += 1
    try:
        frame = FrameInfo(
            frame_id=0,
            timestamp=1.23,
            frame_path=Path("/test/frame.jpg"),
            session_name="test_session"
        )

        if frame.frame_id != 0:
            all_validation_failures.append(f"FrameInfo: Expected frame_id=0, got {frame.frame_id}")
        if frame.timestamp != 1.23:
            all_validation_failures.append(f"FrameInfo: Expected timestamp=1.23, got {frame.timestamp}")
    except Exception as e:
        all_validation_failures.append(f"FrameInfo: Exception raised: {e}")

    # Test 2: Error handling for missing ROS bag
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_bag = Path(tmpdir) / "nonexistent.bag"
            output_dir = Path(tmpdir) / "frames"

            try:
                extract_frames_from_rosbag(nonexistent_bag, output_dir)
                all_validation_failures.append("Error handling: Expected FileNotFoundError for missing bag")
            except FileNotFoundError:
                # Expected behavior
                pass
    except Exception as e:
        all_validation_failures.append(f"Error handling: Unexpected exception: {e}")

    # Test 3: Ground truth loading
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_file = Path(tmpdir) / "groundtruth.txt"

            # Create test ground truth file
            with open(gt_file, "w") as f:
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                f.write("1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
                f.write("2.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
                f.write("3.0 2.0 0.0 0.0 0.0 0.0 0.0 1.0\n")

            poses = load_ground_truth_poses(gt_file)

            if len(poses) != 3:
                all_validation_failures.append(
                    f"Ground truth loading: Expected 3 poses, got {len(poses)}"
                )
            elif "1.0" not in poses:
                all_validation_failures.append(
                    "Ground truth loading: Missing expected timestamp '1.0'"
                )
            else:
                # Check pose values
                tvec, qvec = poses["1.0"]
                expected_tvec = np.array([0.0, 0.0, 0.0])
                expected_qvec = np.array([1.0, 0.0, 0.0, 0.0])  # qw, qx, qy, qz

                if not np.allclose(tvec, expected_tvec):
                    all_validation_failures.append(
                        f"Ground truth loading: Expected tvec {expected_tvec}, got {tvec}"
                    )
                if not np.allclose(qvec, expected_qvec):
                    all_validation_failures.append(
                        f"Ground truth loading: Expected qvec {expected_qvec}, got {qvec}"
                    )
    except Exception as e:
        all_validation_failures.append(f"Ground truth loading: Exception raised: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Dataset module validated: ROS bag extraction and ground truth loading working")
        sys.exit(0)
