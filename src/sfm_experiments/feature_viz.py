"""
Feature Extraction Visualization Module

Purpose:
    Visualize SIFT feature detection results to debug feature extraction quality.
    Provides three visualization types:
    1. Keypoint Overlay - Draw detected keypoints on images with size/color encoding
    2. Feature Distribution Heatmap - Spatial density of features across image
    3. Feature Statistics Dashboard - Histograms of scale, orientation, response strength

Dependencies:
    - pycolmap: Database access for keypoints/descriptors
      https://github.com/colmap/pycolmap
    - OpenCV: Image drawing and visualization
      https://docs.opencv.org/4.x/
    - matplotlib: Statistical plots
      https://matplotlib.org/stable/index.html

Sample Input:
    - database_path: Path to COLMAP database.db (SQLite)
    - images_dir: Directory containing original images
    - image_name: Name of specific image to visualize (e.g., "frame_000123.jpg")

Expected Output:
    - Keypoint overlay: Image with colored circles showing detected keypoints
    - Distribution heatmap: 2D heatmap showing feature density
    - Statistics dashboard: Multi-panel plot with scale/orientation/response histograms
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.gridspec import GridSpec


def _blob_to_array(blob: bytes, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert SQLite BLOB to numpy array.

    Args:
        blob: Binary data from database
        dtype: Target numpy dtype

    Returns:
        Numpy array decoded from blob
    """
    return np.frombuffer(blob, dtype=dtype)


def get_image_keypoints(
    database_path: Path,
    image_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract keypoints and descriptors for a specific image from COLMAP database.

    Args:
        database_path: Path to COLMAP database.db
        image_name: Name of image (e.g., "frame_000123.jpg")

    Returns:
        Tuple of (keypoints, descriptors)
        - keypoints: Nx6 array of [x, y, scale, orientation, response, octave]
        - descriptors: Nx128 array of SIFT descriptors

    Raises:
        ValueError: If image not found in database
    """
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Get image_id for the image name
    cursor.execute("SELECT image_id FROM images WHERE name = ?", (image_name,))
    row = cursor.fetchone()

    if row is None:
        conn.close()
        raise ValueError(f"Image '{image_name}' not found in database")

    image_id = row[0]
    logger.info(f"Found image_id={image_id} for '{image_name}'")

    # Get keypoints
    cursor.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id,))
    kp_row = cursor.fetchone()

    if kp_row is None:
        conn.close()
        raise ValueError(f"No keypoints found for image_id={image_id}")

    # COLMAP stores keypoints as: [x, y, scale, orientation] (4 values per keypoint)
    kp_data = _blob_to_array(kp_row[0], dtype=np.float32)
    num_keypoints = len(kp_data) // 6  # COLMAP uses 6 values: x, y, scale, orientation, response, octave
    keypoints = kp_data.reshape((num_keypoints, 6))

    # Get descriptors
    cursor.execute("SELECT data FROM descriptors WHERE image_id = ?", (image_id,))
    desc_row = cursor.fetchone()

    descriptors = None
    if desc_row is not None:
        desc_data = _blob_to_array(desc_row[0], dtype=np.uint8)
        descriptors = desc_data.reshape((num_keypoints, 128))

    conn.close()

    logger.info(f"Loaded {num_keypoints} keypoints for '{image_name}'")
    return keypoints, descriptors


def visualize_keypoints_overlay(
    image_path: Path,
    keypoints: np.ndarray,
    output_path: Optional[Path] = None,
    max_keypoints: int = 5000,
    color_by: str = "response",
) -> np.ndarray:
    """Draw SIFT keypoints on image with size and color encoding.

    Args:
        image_path: Path to original image
        keypoints: Nx6 array from get_image_keypoints() [x, y, scale, orientation, response, octave]
        output_path: Optional path to save visualization (PNG)
        max_keypoints: Maximum keypoints to draw (for performance)
        color_by: Coloring scheme - "response" or "scale"

    Returns:
        Image array with keypoints drawn (RGB, uint8)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]

    # Sample keypoints if too many
    if len(keypoints) > max_keypoints:
        logger.warning(f"Sampling {max_keypoints} of {len(keypoints)} keypoints for visualization")
        # Sort by response and take top keypoints
        sorted_indices = np.argsort(keypoints[:, 4])[::-1]  # response is column 4
        keypoints = keypoints[sorted_indices[:max_keypoints]]

    # Create overlay image
    overlay = img_rgb.copy()

    # Normalize color values
    if color_by == "response":
        color_values = keypoints[:, 4]  # response
    elif color_by == "scale":
        color_values = keypoints[:, 2]  # scale
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    # Normalize to [0, 1]
    if len(color_values) > 0:
        vmin, vmax = np.percentile(color_values, [5, 95])
        color_norm = np.clip((color_values - vmin) / (vmax - vmin + 1e-8), 0, 1)
    else:
        color_norm = np.array([])

    # Draw each keypoint
    for i, kp in enumerate(keypoints):
        x, y, scale, orientation, response, octave = kp

        # Convert to pixel coordinates
        x_px, y_px = int(round(x)), int(round(y))

        # Skip if outside image bounds
        if x_px < 0 or x_px >= width or y_px < 0 or y_px >= height:
            continue

        # Radius proportional to scale
        radius = max(1, int(scale * 2))

        # Color from blue (low) to red (high)
        if len(color_norm) > 0:
            t = color_norm[i]
            color = (
                int(255 * t),  # Red channel
                int(255 * (1 - abs(2 * t - 1))),  # Green channel (peaks at 0.5)
                int(255 * (1 - t)),  # Blue channel
            )
        else:
            color = (255, 255, 0)  # Yellow default

        # Draw circle
        cv2.circle(overlay, (x_px, y_px), radius, color, 1, cv2.LINE_AA)

        # Draw orientation line (optional, only for larger keypoints)
        if radius > 3:
            angle_rad = np.radians(orientation)
            end_x = int(x_px + radius * 1.5 * np.cos(angle_rad))
            end_y = int(y_px + radius * 1.5 * np.sin(angle_rad))
            cv2.line(overlay, (x_px, y_px), (end_x, end_y), color, 1, cv2.LINE_AA)

    # Add text overlay with info
    info_text = f"Keypoints: {len(keypoints)} | Color: {color_by}"
    cv2.putText(
        overlay, info_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        overlay, info_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA
    )

    # Save if requested
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 8))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"SIFT Keypoints - {image_path.name}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved keypoint overlay: {output_path}")

    return overlay


def visualize_feature_distribution(
    keypoints: np.ndarray,
    image_shape: Tuple[int, int],
    output_path: Optional[Path] = None,
    grid_size: int = 50,
) -> np.ndarray:
    """Create 2D heatmap showing spatial distribution of features.

    Args:
        keypoints: Nx6 array from get_image_keypoints()
        image_shape: (height, width) of original image
        output_path: Optional path to save visualization (PNG)
        grid_size: Grid cell size in pixels for density computation

    Returns:
        2D density heatmap array
    """
    height, width = image_shape

    # Create grid
    grid_h = (height + grid_size - 1) // grid_size
    grid_w = (width + grid_size - 1) // grid_size
    density = np.zeros((grid_h, grid_w), dtype=np.float32)

    # Accumulate keypoints into grid
    for kp in keypoints:
        x, y = kp[0], kp[1]
        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)

        if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
            density[grid_y, grid_x] += 1

    # Visualize
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(density, cmap="hot", interpolation="nearest", aspect="auto")
        ax.set_title(f"Feature Density Heatmap ({len(keypoints)} keypoints)")
        ax.set_xlabel(f"X (grid cell = {grid_size}px)")
        ax.set_ylabel(f"Y (grid cell = {grid_size}px)")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Keypoints per cell")

        # Add statistics
        stats_text = (
            f"Total: {len(keypoints)}\n"
            f"Mean: {density.mean():.1f}\n"
            f"Max: {density.max():.0f}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved distribution heatmap: {output_path}")

    return density


def visualize_feature_statistics(
    keypoints: np.ndarray,
    output_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Create multi-panel dashboard showing feature statistics.

    Args:
        keypoints: Nx6 array from get_image_keypoints() [x, y, scale, orientation, response, octave]
        output_path: Optional path to save visualization (PNG)

    Returns:
        Dictionary with histogram data for each statistic
    """
    scales = keypoints[:, 2]
    orientations = keypoints[:, 3]
    responses = keypoints[:, 4]
    octaves = keypoints[:, 5]

    stats = {
        "scales": scales,
        "orientations": orientations,
        "responses": responses,
        "octaves": octaves,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Scale distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(scales, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Scale")
        ax1.set_ylabel("Count")
        ax1.set_title("Scale Distribution")
        ax1.axvline(scales.mean(), color="red", linestyle="--", label=f"Mean: {scales.mean():.2f}")
        ax1.legend()

        # Orientation distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(orientations, bins=36, color="orange", edgecolor="black", alpha=0.7)
        ax2.set_xlabel("Orientation (degrees)")
        ax2.set_ylabel("Count")
        ax2.set_title("Orientation Distribution")
        ax2.set_xlim([0, 360])

        # Response distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(responses, bins=50, color="green", edgecolor="black", alpha=0.7)
        ax3.set_xlabel("Response Strength")
        ax3.set_ylabel("Count")
        ax3.set_title("Response Distribution")
        ax3.axvline(responses.mean(), color="red", linestyle="--", label=f"Mean: {responses.mean():.2f}")
        ax3.legend()

        # Octave distribution
        ax4 = fig.add_subplot(gs[1, 1])
        # Handle negative octaves by shifting to non-negative range for bincount
        octaves_int = octaves.astype(int)
        octave_min = octaves_int.min()
        octave_max = octaves_int.max()
        if octave_min < 0:
            octaves_shifted = octaves_int - octave_min
            octave_counts = np.bincount(octaves_shifted)
            octave_labels = list(range(octave_min, octave_max + 1))
        else:
            octave_counts = np.bincount(octaves_int)
            octave_labels = list(range(len(octave_counts)))

        ax4.bar(range(len(octave_counts)), octave_counts, color="purple", edgecolor="black", alpha=0.7)
        ax4.set_xlabel("Octave")
        ax4.set_ylabel("Count")
        ax4.set_title("Octave Distribution")
        ax4.set_xticks(range(len(octave_counts)))
        ax4.set_xticklabels(octave_labels)

        # 2D scatter: Scale vs Response
        ax5 = fig.add_subplot(gs[2, :])
        scatter = ax5.scatter(scales, responses, c=orientations, cmap="hsv", alpha=0.3, s=10)
        ax5.set_xlabel("Scale")
        ax5.set_ylabel("Response")
        ax5.set_title("Scale vs Response (colored by orientation)")
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label("Orientation (deg)")

        # Add summary statistics
        summary_text = (
            f"Total Keypoints: {len(keypoints)}\n"
            f"Scale: {scales.min():.2f} - {scales.max():.2f} (μ={scales.mean():.2f})\n"
            f"Response: {responses.min():.2f} - {responses.max():.2f} (μ={responses.mean():.2f})"
        )
        fig.text(
            0.5, 0.02, summary_text,
            ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved statistics dashboard: {output_path}")

    return stats


if __name__ == "__main__":
    """Validation with real COLMAP database."""
    import sys

    # Configuration - using actual project data
    test_database = Path("/home/fiod/sfm/results/lamar/CAB/reconstruction_glomap_fps_5.0/database.db")
    # Images are stored relative to database location
    test_images_base = test_database.parent
    output_dir = Path("/home/fiod/sfm/test_visualizations/features")

    all_validation_failures = []
    total_tests = 0

    # Check if test data exists
    if not test_database.exists():
        logger.error(f"Test database not found: {test_database}")
        logger.info("Please update the test_database path in __main__ to point to a real COLMAP database")
        sys.exit(1)

    # Get a test image name from database
    conn = sqlite3.connect(str(test_database))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM images LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if row is None:
        logger.error("No images found in database")
        sys.exit(1)

    test_image_name = row[0]
    # Images are stored with relative paths from database location
    test_image_path = test_images_base / test_image_name

    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        logger.error(f"Image name from database: {test_image_name}")
        sys.exit(1)

    logger.info(f"Using test image: {test_image_name}")
    logger.info(f"Full path: {test_image_path}")

    # Test 1: Load keypoints from database
    total_tests += 1
    try:
        keypoints, descriptors = get_image_keypoints(test_database, test_image_name)
        if keypoints is None or len(keypoints) == 0:
            all_validation_failures.append("Test 1: No keypoints loaded from database")
        elif keypoints.shape[1] != 6:
            all_validation_failures.append(f"Test 1: Expected keypoints shape (N, 6), got {keypoints.shape}")
        else:
            logger.info(f"✓ Test 1 passed: Loaded {len(keypoints)} keypoints with shape {keypoints.shape}")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Create keypoint overlay
    total_tests += 1
    try:
        output_path = output_dir / "test_keypoint_overlay.png"
        overlay = visualize_keypoints_overlay(
            test_image_path, keypoints, output_path, max_keypoints=1000
        )
        if overlay is None or overlay.size == 0:
            all_validation_failures.append("Test 2: Keypoint overlay returned empty image")
        elif not output_path.exists():
            all_validation_failures.append(f"Test 2: Output file not created: {output_path}")
        else:
            logger.info(f"✓ Test 2 passed: Created keypoint overlay at {output_path}")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Create distribution heatmap
    total_tests += 1
    try:
        img = cv2.imread(str(test_image_path))
        image_shape = (img.shape[0], img.shape[1])
        output_path = output_dir / "test_distribution_heatmap.png"
        density = visualize_feature_distribution(keypoints, image_shape, output_path)
        if density is None or density.size == 0:
            all_validation_failures.append("Test 3: Distribution heatmap returned empty array")
        elif not output_path.exists():
            all_validation_failures.append(f"Test 3: Output file not created: {output_path}")
        else:
            logger.info(f"✓ Test 3 passed: Created distribution heatmap at {output_path}")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Create statistics dashboard
    total_tests += 1
    try:
        output_path = output_dir / "test_statistics_dashboard.png"
        stats = visualize_feature_statistics(keypoints, output_path)
        if stats is None or len(stats) == 0:
            all_validation_failures.append("Test 4: Statistics dashboard returned empty dict")
        elif not output_path.exists():
            all_validation_failures.append(f"Test 4: Output file not created: {output_path}")
        elif "scales" not in stats or "responses" not in stats:
            all_validation_failures.append("Test 4: Missing expected statistics keys")
        else:
            logger.info(f"✓ Test 4 passed: Created statistics dashboard at {output_path}")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception raised: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print(f"Feature visualizations saved to: {output_dir}")
        print("Module is validated and ready for use")
        sys.exit(0)
