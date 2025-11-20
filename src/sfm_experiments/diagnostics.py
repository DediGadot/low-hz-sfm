"""
Diagnostic tools for analyzing COLMAP reconstruction results.

This module provides functions to:
- Analyze which images were successfully registered
- Examine feature extraction quality
- Visualize matching statistics
- Generate diagnostic reports

Third-party packages:
- pycolmap: https://github.com/colmap/pycolmap
- loguru: https://github.com/Delgan/loguru
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pycolmap
from loguru import logger


def analyze_reconstruction(
    reconstruction_path: Path,
    database_path: Path,
    images_dir: Optional[Path] = None,
) -> Dict[str, any]:
    """
    Analyze a COLMAP reconstruction and database to diagnose issues.

    Args:
        reconstruction_path: Path to sparse reconstruction (contains cameras.bin, images.bin, etc.)
        database_path: Path to COLMAP database.db file
        images_dir: Optional path to images directory for visualization

    Returns:
        Dictionary containing diagnostic information:
        - registered_images: List of registered image names
        - total_images: Total images in database
        - registration_rate: Percentage of registered images
        - avg_features_per_image: Average number of features
        - avg_matches_per_pair: Average number of matches
        - feature_stats: Detailed feature statistics
        - match_stats: Detailed matching statistics
    """
    logger.info(f"Analyzing reconstruction: {reconstruction_path}")

    # Load reconstruction
    reconstruction = pycolmap.Reconstruction(str(reconstruction_path))

    # Get registered images
    registered_images = [img.name for img in reconstruction.images.values()]
    num_registered = len(registered_images)
    num_points = len(reconstruction.points3D)

    logger.info(f"Registered images: {num_registered}")
    logger.info(f"3D points: {num_points}")

    # Analyze database
    db_stats = analyze_database(database_path)

    registration_rate = (num_registered / db_stats["total_images"]) * 100 if db_stats["total_images"] > 0 else 0

    # Calculate average observations per point
    total_observations = sum(len(point.track.elements) for point in reconstruction.points3D.values())
    avg_track_length = total_observations / num_points if num_points > 0 else 0

    # Calculate reprojection error statistics
    reprojection_errors = []
    for point in reconstruction.points3D.values():
        reprojection_errors.append(point.error)

    diagnostics = {
        "registered_images": registered_images,
        "num_registered": num_registered,
        "num_points": num_points,
        "total_images": db_stats["total_images"],
        "registration_rate": registration_rate,
        "avg_features_per_image": db_stats["avg_features"],
        "avg_matches_per_pair": db_stats["avg_matches"],
        "feature_distribution": db_stats["feature_distribution"],
        "match_distribution": db_stats["match_distribution"],
        "avg_track_length": avg_track_length,
        "reprojection_error_mean": np.mean(reprojection_errors) if reprojection_errors else 0,
        "reprojection_error_std": np.std(reprojection_errors) if reprojection_errors else 0,
    }

    return diagnostics


def analyze_database(database_path: Path) -> Dict[str, any]:
    """
    Analyze COLMAP database to extract feature and matching statistics.

    Args:
        database_path: Path to database.db file

    Returns:
        Dictionary with database statistics
    """
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Count total images
    cursor.execute("SELECT COUNT(*) FROM images")
    total_images = cursor.fetchone()[0]

    # Get feature counts per image
    cursor.execute("SELECT image_id, rows FROM keypoints")
    feature_counts = cursor.fetchall()

    feature_list = [count[1] for count in feature_counts]
    avg_features = np.mean(feature_list) if feature_list else 0

    # Get matching statistics
    cursor.execute("SELECT pair_id, rows FROM two_view_geometries")
    match_counts = cursor.fetchall()

    match_list = [count[1] for count in match_counts]
    avg_matches = np.mean(match_list) if match_list else 0
    total_pairs = len(match_list)

    # Image name lookup
    cursor.execute("SELECT image_id, name FROM images")
    image_names = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()

    return {
        "total_images": total_images,
        "avg_features": avg_features,
        "avg_matches": avg_matches,
        "total_pairs": total_pairs,
        "feature_distribution": {
            "min": int(np.min(feature_list)) if feature_list else 0,
            "max": int(np.max(feature_list)) if feature_list else 0,
            "median": int(np.median(feature_list)) if feature_list else 0,
            "std": float(np.std(feature_list)) if feature_list else 0,
        },
        "match_distribution": {
            "min": int(np.min(match_list)) if match_list else 0,
            "max": int(np.max(match_list)) if match_list else 0,
            "median": int(np.median(match_list)) if match_list else 0,
            "std": float(np.std(match_list)) if match_list else 0,
        },
        "image_names": image_names,
    }


def get_match_matrix(database_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Create a match matrix showing which images match which.

    Args:
        database_path: Path to database.db file

    Returns:
        Tuple of (match_matrix, image_names) where match_matrix[i,j] = number of matches
    """
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Get all images
    cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
    images = cursor.fetchall()
    image_ids = [img[0] for img in images]
    image_names = [img[1] for img in images]
    n_images = len(image_ids)

    # Create empty matrix
    match_matrix = np.zeros((n_images, n_images), dtype=int)

    # Fill matrix with match counts
    cursor.execute("SELECT pair_id, rows FROM two_view_geometries")
    for pair_id, match_count in cursor.fetchall():
        # Decode pair_id to get image IDs
        # COLMAP uses: pair_id = image_id1 * 2147483647 + image_id2
        # where image_id1 < image_id2
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) // 2147483647

        # Find indices
        try:
            idx1 = image_ids.index(image_id1)
            idx2 = image_ids.index(image_id2)
            match_matrix[idx1, idx2] = match_count
            match_matrix[idx2, idx1] = match_count
        except ValueError:
            continue

    conn.close()

    return match_matrix, image_names


def print_diagnostic_report(diagnostics: Dict[str, any], title: str = "Reconstruction Diagnostics"):
    """
    Print a formatted diagnostic report.

    Args:
        diagnostics: Dictionary from analyze_reconstruction()
        title: Report title
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")

    print(f"📊 Registration Statistics:")
    print(f"  Total images in database    : {diagnostics['total_images']}")
    print(f"  Registered images           : {diagnostics['num_registered']}")
    print(f"  Registration rate           : {diagnostics['registration_rate']:.1f}%")
    print(f"  3D points reconstructed     : {diagnostics['num_points']}")
    print()

    print(f"🔍 Feature Extraction Quality:")
    print(f"  Average features per image  : {diagnostics['avg_features_per_image']:.1f}")
    print(f"  Feature distribution:")
    print(f"    Min   : {diagnostics['feature_distribution']['min']}")
    print(f"    Median: {diagnostics['feature_distribution']['median']}")
    print(f"    Max   : {diagnostics['feature_distribution']['max']}")
    print(f"    StdDev: {diagnostics['feature_distribution']['std']:.1f}")
    print()

    print(f"🔗 Matching Statistics:")
    print(f"  Average matches per pair    : {diagnostics['avg_matches_per_pair']:.1f}")
    print(f"  Match distribution:")
    print(f"    Min   : {diagnostics['match_distribution']['min']}")
    print(f"    Median: {diagnostics['match_distribution']['median']}")
    print(f"    Max   : {diagnostics['match_distribution']['max']}")
    print(f"    StdDev: {diagnostics['match_distribution']['std']:.1f}")
    print()

    print(f"📐 Reconstruction Quality:")
    print(f"  Avg track length            : {diagnostics['avg_track_length']:.2f} observations/point")
    print(f"  Reprojection error (mean)   : {diagnostics['reprojection_error_mean']:.3f} px")
    print(f"  Reprojection error (std)    : {diagnostics['reprojection_error_std']:.3f} px")
    print()

    print(f"✅ Registered Images:")
    for i, img_name in enumerate(diagnostics['registered_images'][:10], 1):
        print(f"  {i:2d}. {img_name}")
    if len(diagnostics['registered_images']) > 10:
        print(f"  ... and {len(diagnostics['registered_images']) - 10} more")
    print()

    # Diagnostic interpretation
    print(f"💡 Diagnostic Interpretation:")

    if diagnostics['registration_rate'] < 10:
        print(f"  ❌ CRITICAL: Only {diagnostics['registration_rate']:.1f}% registration rate!")
        print(f"     → COLMAP parameters are too strict for this dataset")

    if diagnostics['avg_matches_per_pair'] < 15:
        print(f"  ⚠️  WARNING: Very low average matches ({diagnostics['avg_matches_per_pair']:.1f})")
        print(f"     → Possible causes:")
        print(f"       - Frames too far apart temporally")
        print(f"       - Low visual overlap between frames")
        print(f"       - Poor feature extraction (texture-less scenes)")

    if diagnostics['avg_features_per_image'] < 1000:
        print(f"  ⚠️  WARNING: Low feature count ({diagnostics['avg_features_per_image']:.0f}/image)")
        print(f"     → Images may lack distinctive visual features")

    if diagnostics['registration_rate'] >= 40:
        print(f"  ✅ GOOD: {diagnostics['registration_rate']:.1f}% registration rate")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    """
    Validation: Analyze the latest reconstruction results.

    Expected output:
    - Diagnostic report showing why only 2 images were registered
    - Feature and match statistics revealing parameter mismatch
    """
    import sys

    # Test with the latest reconstruction
    results_dir = Path("results")

    all_validation_failures = []
    total_tests = 0

    # Test 1: Analyze 1-visit reconstruction
    total_tests += 1
    print("\n" + "="*60)
    print("TEST 1: Analyzing 1-visit reconstruction")
    print("="*60)

    recon_1_path = results_dir / "reconstruction_1_visits" / "sparse" / "0"
    db_1_path = results_dir / "reconstruction_1_visits" / "database.db"

    if recon_1_path.exists() and db_1_path.exists():
        try:
            diag_1 = analyze_reconstruction(recon_1_path, db_1_path)
            print_diagnostic_report(diag_1, "1-Visit Reconstruction")

            # Verify we got the expected poor results
            if diag_1['registration_rate'] > 10:
                all_validation_failures.append(
                    f"1-visit: Expected low registration (<10%), got {diag_1['registration_rate']:.1f}%"
                )
            if diag_1['avg_matches_per_pair'] > 15:
                all_validation_failures.append(
                    f"1-visit: Expected low matches (<15), got {diag_1['avg_matches_per_pair']:.1f}"
                )
        except Exception as e:
            all_validation_failures.append(f"1-visit analysis failed: {e}")
    else:
        all_validation_failures.append(f"1-visit reconstruction not found at {recon_1_path}")

    # Test 2: Analyze 2-visit reconstruction
    total_tests += 1
    print("\n" + "="*60)
    print("TEST 2: Analyzing 2-visit reconstruction")
    print("="*60)

    recon_2_path = results_dir / "reconstruction_2_visits" / "sparse" / "0"
    db_2_path = results_dir / "reconstruction_2_visits" / "database.db"

    if recon_2_path.exists() and db_2_path.exists():
        try:
            diag_2 = analyze_reconstruction(recon_2_path, db_2_path)
            print_diagnostic_report(diag_2, "2-Visit Reconstruction")

            # Verify poor results
            if diag_2['registration_rate'] > 10:
                all_validation_failures.append(
                    f"2-visit: Expected low registration (<10%), got {diag_2['registration_rate']:.1f}%"
                )
        except Exception as e:
            all_validation_failures.append(f"2-visit analysis failed: {e}")
    else:
        all_validation_failures.append(f"2-visit reconstruction not found at {recon_2_path}")

    # Test 3: Analyze 3-visit reconstruction
    total_tests += 1
    print("\n" + "="*60)
    print("TEST 3: Analyzing 3-visit reconstruction")
    print("="*60)

    recon_3_path = results_dir / "reconstruction_3_visits" / "sparse" / "0"
    db_3_path = results_dir / "reconstruction_3_visits" / "database.db"

    if recon_3_path.exists() and db_3_path.exists():
        try:
            diag_3 = analyze_reconstruction(recon_3_path, db_3_path)
            print_diagnostic_report(diag_3, "3-Visit Reconstruction")

            # Verify poor results
            if diag_3['registration_rate'] > 10:
                all_validation_failures.append(
                    f"3-visit: Expected low registration (<10%), got {diag_3['registration_rate']:.1f}%"
                )
        except Exception as e:
            all_validation_failures.append(f"3-visit analysis failed: {e}")
    else:
        all_validation_failures.append(f"3-visit reconstruction not found at {recon_3_path}")

    # Test 4: Compare degradation across visits
    total_tests += 1
    print("\n" + "="*60)
    print("TEST 4: Comparing multi-visit degradation")
    print("="*60)

    try:
        if 'diag_1' in locals() and 'diag_2' in locals() and 'diag_3' in locals():
            print(f"\nRegistration Rate Comparison:")
            print(f"  1 visit: {diag_1['registration_rate']:5.1f}%")
            print(f"  2 visits: {diag_2['registration_rate']:5.1f}%")
            print(f"  3 visits: {diag_3['registration_rate']:5.1f}%")

            print(f"\nAverage Matches Per Pair:")
            print(f"  1 visit: {diag_1['avg_matches_per_pair']:5.1f}")
            print(f"  2 visits: {diag_2['avg_matches_per_pair']:5.1f}")
            print(f"  3 visits: {diag_3['avg_matches_per_pair']:5.1f}")

            # Expect degradation in matches as visits increase
            if not (diag_1['avg_matches_per_pair'] >= diag_2['avg_matches_per_pair'] >= diag_3['avg_matches_per_pair']):
                all_validation_failures.append(
                    "Expected degradation in avg matches per pair as visits increase"
                )
        else:
            all_validation_failures.append("Not all reconstructions available for comparison")
    except Exception as e:
        all_validation_failures.append(f"Comparison failed: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} diagnostic tests produced expected results")
        print("Diagnostic tool successfully identified the reconstruction issues:")
        print("  - Low registration rates (<10%)")
        print("  - Low average matches per pair (<15)")
        print("  - Degradation in match quality as visits increase")
        sys.exit(0)
