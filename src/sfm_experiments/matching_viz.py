"""
Feature Matching Visualization Module

Purpose:
    Visualize SIFT feature matching results to debug matching quality and patterns.
    Provides three visualization types:
    1. Match Pair Visualization - Draw lines between matching keypoints (inliers/outliers)
    2. Match Matrix Heatmap - NxN matrix showing match counts between all image pairs
    3. Sequential Match Gap Analysis - Plot match quality vs temporal frame gap

Dependencies:
    - pycolmap: Database access for matches and geometric verification
      https://github.com/colmap/pycolmap
    - OpenCV: Image drawing and match visualization
      https://docs.opencv.org/4.x/
    - matplotlib/seaborn: Heatmaps and statistical plots
      https://matplotlib.org/stable/index.html

Sample Input:
    - database_path: Path to COLMAP database.db (SQLite)
    - images_dir: Directory containing original images
    - image_pair: Tuple of two image names to visualize matches

Expected Output:
    - Match pair visualization: Side-by-side images with match lines (green=inlier, red=outlier)
    - Match matrix: Symmetric NxN heatmap of match counts
    - Gap analysis: Line plot showing match quality decay with temporal distance
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from sfm_experiments.feature_viz import _blob_to_array, get_image_keypoints


def _pair_id_to_image_ids(pair_id: int) -> Tuple[int, int]:
    """Convert COLMAP pair_id to (image_id1, image_id2).

    COLMAP stores matches using pair_id = image_id1 * 2147483647 + image_id2
    where image_id1 < image_id2.

    Args:
        pair_id: COLMAP pair identifier

    Returns:
        Tuple of (image_id1, image_id2)
    """
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return (image_id1, image_id2)


def _image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    """Convert (image_id1, image_id2) to COLMAP pair_id.

    Args:
        image_id1: First image ID
        image_id2: Second image ID

    Returns:
        COLMAP pair identifier
    """
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * 2147483647 + image_id2


def get_image_pair_matches(
    database_path: Path,
    image_name1: str,
    image_name2: str,
    verified_only: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract matches between two images from COLMAP database.

    Args:
        database_path: Path to COLMAP database.db
        image_name1: First image name
        image_name2: Second image name
        verified_only: If True, return only geometrically verified matches

    Returns:
        Tuple of (matches, inlier_mask)
        - matches: Nx2 array of [keypoint_idx1, keypoint_idx2]
        - inlier_mask: Boolean array indicating geometric inliers (None if verified_only=False)
    """
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Get image IDs
    cursor.execute("SELECT image_id FROM images WHERE name = ?", (image_name1,))
    row1 = cursor.fetchone()
    cursor.execute("SELECT image_id FROM images WHERE name = ?", (image_name2,))
    row2 = cursor.fetchone()

    if row1 is None or row2 is None:
        conn.close()
        raise ValueError(f"One or both images not found: {image_name1}, {image_name2}")

    image_id1, image_id2 = row1[0], row2[0]
    pair_id = _image_ids_to_pair_id(image_id1, image_id2)

    # Get matches
    cursor.execute("SELECT data FROM matches WHERE pair_id = ?", (pair_id,))
    match_row = cursor.fetchone()

    if match_row is None:
        conn.close()
        logger.warning(f"No matches found for pair {image_name1} <-> {image_name2}")
        return np.array([]), None

    # Decode matches (pairs of uint32)
    match_data = _blob_to_array(match_row[0], dtype=np.uint32)
    matches = match_data.reshape((-1, 2))

    # Get geometric verification if requested
    inlier_mask = None
    if verified_only:
        cursor.execute("SELECT data FROM two_view_geometries WHERE pair_id = ?", (pair_id,))
        geo_row = cursor.fetchone()

        if geo_row is not None:
            # two_view_geometries stores: qvec (4), tvec (3), then inlier indices
            geo_data = _blob_to_array(geo_row[0], dtype=np.uint32)
            # Skip first 7 elements (qvec + tvec stored as uint32)
            # Actually, COLMAP stores inlier matches directly
            inlier_indices = geo_data
            inlier_mask = np.zeros(len(matches), dtype=bool)
            # Map inlier indices to match indices
            for inlier_idx in inlier_indices:
                if inlier_idx < len(matches):
                    inlier_mask[inlier_idx] = True

    conn.close()

    logger.info(f"Found {len(matches)} matches between {image_name1} and {image_name2}")
    if inlier_mask is not None:
        logger.info(f"  └─ {inlier_mask.sum()} inliers ({100*inlier_mask.sum()/len(matches):.1f}%)")

    return matches, inlier_mask


def visualize_match_pair(
    image_path1: Path,
    image_path2: Path,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    max_matches: int = 200,
) -> np.ndarray:
    """Draw side-by-side images with match lines.

    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        keypoints1: Nx6 keypoints for image 1
        keypoints2: Nx6 keypoints for image 2
        matches: Mx2 array of [keypoint_idx1, keypoint_idx2]
        inlier_mask: Optional boolean mask for inliers (green) vs outliers (red)
        output_path: Optional path to save visualization (PNG)
        max_matches: Maximum matches to draw (for performance)

    Returns:
        Combined image with matches drawn
    """
    # Load images
    img1 = cv2.imread(str(image_path1))
    img2 = cv2.imread(str(image_path2))

    if img1 is None or img2 is None:
        raise ValueError(f"Failed to load images: {image_path1}, {image_path2}")

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Sample matches if too many
    if len(matches) > max_matches:
        logger.warning(f"Sampling {max_matches} of {len(matches)} matches for visualization")
        # Prioritize inliers if available
        if inlier_mask is not None:
            inlier_indices = np.where(inlier_mask)[0]
            outlier_indices = np.where(~inlier_mask)[0]

            # Take all inliers if possible, otherwise sample
            if len(inlier_indices) > max_matches // 2:
                sampled_inliers = np.random.choice(inlier_indices, max_matches // 2, replace=False)
            else:
                sampled_inliers = inlier_indices

            # Fill rest with outliers
            num_outliers = max_matches - len(sampled_inliers)
            if len(outlier_indices) > num_outliers:
                sampled_outliers = np.random.choice(outlier_indices, num_outliers, replace=False)
            else:
                sampled_outliers = outlier_indices

            sample_indices = np.concatenate([sampled_inliers, sampled_outliers])
        else:
            sample_indices = np.random.choice(len(matches), max_matches, replace=False)

        matches = matches[sample_indices]
        if inlier_mask is not None:
            inlier_mask = inlier_mask[sample_indices]

    # Create side-by-side canvas
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    h_max = max(h1, h2)
    canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_rgb
    canvas[:h2, w1:w1+w2] = img2_rgb

    # Draw matches
    for i, (idx1, idx2) in enumerate(matches):
        if idx1 >= len(keypoints1) or idx2 >= len(keypoints2):
            continue

        pt1 = (int(keypoints1[idx1, 0]), int(keypoints1[idx1, 1]))
        pt2 = (int(keypoints2[idx2, 0]) + w1, int(keypoints2[idx2, 1]))

        # Color: green for inliers, red for outliers
        if inlier_mask is not None:
            color = (0, 255, 0) if inlier_mask[i] else (255, 0, 0)
            thickness = 1 if inlier_mask[i] else 1
        else:
            color = (255, 255, 0)  # Yellow if no inlier info
            thickness = 1

        # Draw line
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Draw keypoint circles
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)

    # Add text overlay
    if inlier_mask is not None:
        num_inliers = inlier_mask.sum()
        num_outliers = (~inlier_mask).sum()
        info_text = f"Matches: {len(matches)} | Inliers: {num_inliers} | Outliers: {num_outliers}"
    else:
        info_text = f"Matches: {len(matches)}"

    cv2.putText(
        canvas, info_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas, info_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA
    )

    # Save if requested
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(16, 8))
        plt.imshow(canvas)
        plt.axis("off")
        plt.title(f"Feature Matches: {image_path1.name} ↔ {image_path2.name}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved match visualization: {output_path}")

    return canvas


def visualize_match_matrix(
    database_path: Path,
    output_path: Optional[Path] = None,
    max_images: int = 100,
    min_matches: int = 0,
) -> np.ndarray:
    """Create NxN heatmap showing match counts between all image pairs.

    Args:
        database_path: Path to COLMAP database.db
        output_path: Optional path to save visualization (PNG)
        max_images: Maximum images to include (for performance)
        min_matches: Minimum matches required to include pair

    Returns:
        NxN match count matrix
    """
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Get all images
    cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
    images = cursor.fetchall()

    if len(images) > max_images:
        logger.warning(f"Sampling {max_images} of {len(images)} images for matrix")
        images = images[:max_images]

    # Create ID to index mapping
    image_ids = [img[0] for img in images]
    image_names = [img[1] for img in images]
    id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}

    n = len(images)
    match_matrix = np.zeros((n, n), dtype=np.int32)

    # Query all matches
    cursor.execute("SELECT pair_id, data FROM matches")
    for row in cursor.fetchall():
        pair_id = row[0]
        match_data = _blob_to_array(row[1], dtype=np.uint32)
        num_matches = len(match_data) // 2

        if num_matches < min_matches:
            continue

        img_id1, img_id2 = _pair_id_to_image_ids(pair_id)

        if img_id1 in id_to_idx and img_id2 in id_to_idx:
            idx1 = id_to_idx[img_id1]
            idx2 = id_to_idx[img_id2]
            match_matrix[idx1, idx2] = num_matches
            match_matrix[idx2, idx1] = num_matches  # Symmetric

    conn.close()

    # Visualize
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Use log scale for better visualization
        match_matrix_log = np.log10(match_matrix + 1)

        sns.heatmap(
            match_matrix_log,
            cmap="YlOrRd",
            square=True,
            cbar_kws={"label": "log10(matches + 1)"},
            ax=ax,
            xticklabels=False,
            yticklabels=False,
        )

        ax.set_title(f"Match Matrix ({n} images)")
        ax.set_xlabel("Image Index")
        ax.set_ylabel("Image Index")

        # Add statistics
        non_zero = (match_matrix > 0).sum() // 2  # Divide by 2 for symmetric matrix
        total_pairs = n * (n - 1) // 2
        coverage = 100 * non_zero / total_pairs if total_pairs > 0 else 0

        stats_text = (
            f"Images: {n}\n"
            f"Pairs with matches: {non_zero}/{total_pairs} ({coverage:.1f}%)\n"
            f"Avg matches/pair: {match_matrix[match_matrix > 0].mean():.0f}"
        )
        ax.text(
            1.15, 0.5, stats_text,
            transform=ax.transAxes,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved match matrix: {output_path}")

    return match_matrix


def visualize_sequential_match_gap(
    database_path: Path,
    output_path: Optional[Path] = None,
    max_gap: int = 100,
) -> Dict[int, List[int]]:
    """Plot match counts vs temporal gap between sequential frames.

    Args:
        database_path: Path to COLMAP database.db
        output_path: Optional path to save visualization (PNG)
        max_gap: Maximum frame gap to analyze

    Returns:
        Dictionary mapping gap → [match_counts]
    """
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Get all images ordered by name (assuming sequential naming)
    cursor.execute("SELECT image_id, name FROM images ORDER BY name")
    images = cursor.fetchall()

    # Create mapping: image_id → index in sequence
    id_to_seq = {img[0]: idx for idx, img in enumerate(images)}

    # Collect matches by gap
    gap_matches: Dict[int, List[int]] = {i: [] for i in range(max_gap + 1)}

    cursor.execute("SELECT pair_id, data FROM matches")
    for row in cursor.fetchall():
        pair_id = row[0]
        match_data = _blob_to_array(row[1], dtype=np.uint32)
        num_matches = len(match_data) // 2

        img_id1, img_id2 = _pair_id_to_image_ids(pair_id)

        if img_id1 in id_to_seq and img_id2 in id_to_seq:
            seq1 = id_to_seq[img_id1]
            seq2 = id_to_seq[img_id2]
            gap = abs(seq2 - seq1)

            if gap <= max_gap:
                gap_matches[gap].append(num_matches)

    conn.close()

    # Compute statistics
    gaps = []
    means = []
    medians = []
    counts = []

    for gap in sorted(gap_matches.keys()):
        if gap_matches[gap]:
            gaps.append(gap)
            means.append(np.mean(gap_matches[gap]))
            medians.append(np.median(gap_matches[gap]))
            counts.append(len(gap_matches[gap]))

    # Visualize
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Top: Match counts vs gap
        ax1.plot(gaps, means, "o-", label="Mean", color="steelblue", linewidth=2)
        ax1.plot(gaps, medians, "s-", label="Median", color="orange", linewidth=2)
        ax1.set_xlabel("Frame Gap")
        ax1.set_ylabel("Number of Matches")
        ax1.set_title("Match Quality vs Temporal Frame Gap")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: Number of pairs per gap
        ax2.bar(gaps, counts, color="green", alpha=0.6)
        ax2.set_xlabel("Frame Gap")
        ax2.set_ylabel("Number of Image Pairs")
        ax2.set_title("Image Pair Count by Frame Gap")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved gap analysis: {output_path}")

    return gap_matches


if __name__ == "__main__":
    """Validation with real COLMAP database."""
    import sys

    # Configuration - using actual project data
    test_database = Path("/home/fiod/sfm/results/lamar/CAB/reconstruction_glomap_fps_5.0/database.db")
    test_images_base = test_database.parent
    output_dir = Path("/home/fiod/sfm/test_visualizations/matching")

    all_validation_failures = []
    total_tests = 0

    # Check if test data exists
    if not test_database.exists():
        logger.error(f"Test database not found: {test_database}")
        sys.exit(1)

    # Get two test images from database
    conn = sqlite3.connect(str(test_database))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM images ORDER BY name LIMIT 10")
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 2:
        logger.error("Need at least 2 images in database")
        sys.exit(1)

    test_image_name1 = rows[0][0]
    test_image_name2 = rows[1][0]
    test_image_path1 = test_images_base / test_image_name1
    test_image_path2 = test_images_base / test_image_name2

    logger.info(f"Using test images: {test_image_name1} and {test_image_name2}")

    # Test 1: Load matches between two images
    total_tests += 1
    try:
        matches, inlier_mask = get_image_pair_matches(test_database, test_image_name1, test_image_name2, verified_only=True)
        if matches is None or len(matches) == 0:
            all_validation_failures.append("Test 1: No matches loaded from database")
        elif matches.shape[1] != 2:
            all_validation_failures.append(f"Test 1: Expected matches shape (N, 2), got {matches.shape}")
        else:
            logger.info(f"✓ Test 1 passed: Loaded {len(matches)} matches with shape {matches.shape}")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception raised: {e}")

    # Test 2: Create match pair visualization
    total_tests += 1
    try:
        if len(matches) > 0:
            keypoints1, _ = get_image_keypoints(test_database, test_image_name1)
            keypoints2, _ = get_image_keypoints(test_database, test_image_name2)
            output_path = output_dir / "test_match_pair.png"
            result = visualize_match_pair(
                test_image_path1, test_image_path2,
                keypoints1, keypoints2,
                matches, inlier_mask, output_path
            )
            if result is None or result.size == 0:
                all_validation_failures.append("Test 2: Match pair visualization returned empty image")
            elif not output_path.exists():
                all_validation_failures.append(f"Test 2: Output file not created: {output_path}")
            else:
                logger.info(f"✓ Test 2 passed: Created match pair visualization at {output_path}")
        else:
            logger.warning("⊘ Test 2 skipped: No matches to visualize")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception raised: {e}")

    # Test 3: Create match matrix heatmap
    total_tests += 1
    try:
        output_path = output_dir / "test_match_matrix.png"
        matrix = visualize_match_matrix(test_database, output_path, max_images=50)
        if matrix is None or matrix.size == 0:
            all_validation_failures.append("Test 3: Match matrix returned empty array")
        elif not output_path.exists():
            all_validation_failures.append(f"Test 3: Output file not created: {output_path}")
        else:
            logger.info(f"✓ Test 3 passed: Created match matrix at {output_path}")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception raised: {e}")

    # Test 4: Create sequential gap analysis
    total_tests += 1
    try:
        output_path = output_dir / "test_gap_analysis.png"
        gap_data = visualize_sequential_match_gap(test_database, output_path, max_gap=30)
        if gap_data is None or len(gap_data) == 0:
            all_validation_failures.append("Test 4: Gap analysis returned empty dict")
        elif not output_path.exists():
            all_validation_failures.append(f"Test 4: Output file not created: {output_path}")
        else:
            logger.info(f"✓ Test 4 passed: Created gap analysis at {output_path}")
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
        print(f"Matching visualizations saved to: {output_dir}")
        print("Module is validated and ready for use")
        sys.exit(0)
