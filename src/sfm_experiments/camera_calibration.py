"""
Camera calibration utilities for COLMAP.

This module provides functions to initialize COLMAP database with known camera parameters,
preventing wild calibration variations across visits.

Dependencies:
- pycolmap: https://github.com/colmap/pycolmap

Sample Input:
    database_path: Path to COLMAP database.db
    camera_params: Dict with model, width, height, and intrinsic parameters

Expected Output:
    Camera initialized in database with fixed parameters
"""

from pathlib import Path
from typing import Dict, Optional
import pycolmap
from loguru import logger


def initialize_camera_in_database(
    database_path: Path,
    camera_model: str = "SIMPLE_RADIAL",
    width: int = 720,
    height: int = 540,
    focal_length: float = 400.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    k1: float = 0.0,
) -> int:
    """
    Initialize camera in COLMAP database with known parameters.

    This prevents COLMAP from solving wildly incorrect calibrations by
    providing a reasonable starting point based on sensor specifications.

    Args:
        database_path: Path to database.db file
        camera_model: COLMAP camera model (SIMPLE_RADIAL, PINHOLE, etc.)
        width: Image width in pixels
        height: Image height in pixels
        focal_length: Focal length in pixels (fx)
        cx: Principal point x (defaults to width/2)
        cy: Principal point y (defaults to height/2)
        k1: Radial distortion coefficient

    Returns:
        camera_id: ID of created camera in database

    Example:
        >>> initialize_camera_in_database(
        ...     Path("database.db"),
        ...     focal_length=400.0,
        ...     width=720,
        ...     height=540
        ... )
        1
    """
    # Set defaults for principal point
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    # Open database
    db = pycolmap.Database(str(database_path))

    # Create camera with parameters
    # SIMPLE_RADIAL: [fx, cx, cy, k1]
    if camera_model == "SIMPLE_RADIAL":
        params = [focal_length, cx, cy, k1]
        model_id = pycolmap.CameraModelId.SIMPLE_RADIAL
    elif camera_model == "PINHOLE":
        # PINHOLE: [fx, fy, cx, cy]
        params = [focal_length, focal_length, cx, cy]
        model_id = pycolmap.CameraModelId.PINHOLE
    elif camera_model == "RADIAL":
        # RADIAL: [fx, cx, cy, k1, k2]
        params = [focal_length, cx, cy, k1, 0.0]
        model_id = pycolmap.CameraModelId.RADIAL
    else:
        raise ValueError(f"Unsupported camera model: {camera_model}")

    # Add camera to database
    camera_id = db.add_camera(
        model=model_id,
        width=width,
        height=height,
        params=params,
        prior_focal_length=True,  # Use focal length as prior
    )

    db.commit()
    db.close()

    logger.info(
        f"✅ Initialized camera {camera_id} in database: "
        f"{camera_model} {width}x{height}, fx={focal_length:.1f}px"
    )

    return camera_id


def load_camera_calibration_from_config(config: Dict) -> Dict:
    """
    Load camera calibration from config dictionary.

    Args:
        config: Configuration dict with dataset.camera_calibration section

    Returns:
        Dict with camera parameters

    Example:
        >>> config = {"dataset": {"camera_calibration": {
        ...     "model": "SIMPLE_RADIAL",
        ...     "width": 720,
        ...     "height": 540,
        ...     "params": {"fx": 400.0, "cx": 360.0, "cy": 270.0, "k1": 0.0}
        ... }}}
        >>> load_camera_calibration_from_config(config)
        {'model': 'SIMPLE_RADIAL', 'width': 720, 'height': 540, ...}
    """
    if "dataset" not in config:
        return None

    if "camera_calibration" not in config["dataset"]:
        return None

    cal = config["dataset"]["camera_calibration"]

    return {
        "model": cal.get("model", "SIMPLE_RADIAL"),
        "width": cal.get("width", 720),
        "height": cal.get("height", 540),
        "focal_length": cal["params"].get("fx", 400.0),
        "cx": cal["params"].get("cx"),
        "cy": cal["params"].get("cy"),
        "k1": cal["params"].get("k1", 0.0),
    }


if __name__ == "__main__":
    """
    Validation: Test camera initialization in a test database.

    Expected results:
    - Camera should be created successfully
    - Parameters should match input
    - Database should be readable
    """
    import sys
    import tempfile
    import os

    all_validation_failures = []
    total_tests = 0

    # Test 1: Initialize camera in new database
    total_tests += 1
    print("\n🧪 Test 1: Initialize camera in new database")

    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)

        camera_id = initialize_camera_in_database(
            db_path,
            camera_model="SIMPLE_RADIAL",
            width=720,
            height=540,
            focal_length=400.0,
            cx=360.0,
            cy=270.0,
            k1=0.0,
        )

        # Verify camera was created
        db = pycolmap.Database(str(db_path))
        camera = db.read_camera(camera_id)

        if camera.width != 720:
            all_validation_failures.append(f"Test 1: Expected width 720, got {camera.width}")
        if camera.height != 540:
            all_validation_failures.append(f"Test 1: Expected height 540, got {camera.height}")

        params = camera.params
        if abs(params[0] - 400.0) > 0.1:
            all_validation_failures.append(f"Test 1: Expected fx=400.0, got {params[0]:.2f}")
        if abs(params[1] - 360.0) > 0.1:
            all_validation_failures.append(f"Test 1: Expected cx=360.0, got {params[1]:.2f}")
        if abs(params[2] - 270.0) > 0.1:
            all_validation_failures.append(f"Test 1: Expected cy=270.0, got {params[2]:.2f}")

        db.close()
        os.unlink(db_path)

        print(f"  ✅ Camera created: {camera.model}, {camera.width}x{camera.height}")
        print(f"  ✅ Parameters: fx={params[0]:.2f}, cx={params[1]:.2f}, cy={params[2]:.2f}, k1={params[3]:.6f}")

    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception: {type(e).__name__}: {e}")

    # Test 2: Load from config
    total_tests += 1
    print("\n🧪 Test 2: Load camera calibration from config")

    try:
        test_config = {
            "dataset": {
                "camera_calibration": {
                    "model": "SIMPLE_RADIAL",
                    "width": 720,
                    "height": 540,
                    "params": {
                        "fx": 400.0,
                        "cx": 360.0,
                        "cy": 270.0,
                        "k1": 0.0,
                    }
                }
            }
        }

        cal = load_camera_calibration_from_config(test_config)

        if cal["model"] != "SIMPLE_RADIAL":
            all_validation_failures.append(f"Test 2: Expected SIMPLE_RADIAL, got {cal['model']}")
        if cal["focal_length"] != 400.0:
            all_validation_failures.append(f"Test 2: Expected fx=400.0, got {cal['focal_length']}")

        print(f"  ✅ Loaded calibration: {cal['model']}, fx={cal['focal_length']:.1f}px")

    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception: {type(e).__name__}: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Camera calibration utilities validated and ready for use")
        sys.exit(0)
