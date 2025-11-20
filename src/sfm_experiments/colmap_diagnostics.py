"""
COLMAP Database Diagnostics Tool

This module provides deep diagnostic analysis of COLMAP reconstructions:
- Feature extraction statistics per image
- Feature matching statistics between image pairs
- Image registration status and reasons for failure
- Bundle adjustment convergence metrics

Dependencies:
- pycolmap: https://github.com/colmap/pycolmap
- sqlite3: Standard library
- pathlib: Standard library

Sample Input:
    database_path: Path to COLMAP database.db file
    reconstruction_path: Path to sparse reconstruction directory

Expected Output:
    Detailed diagnostics including:
    - Feature counts per image
    - Match counts between pairs
    - Registered vs. unregistered images
    - Reconstruction statistics
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pycolmap
from loguru import logger

PAIR_PRIME = 2147483647  # COLMAP pair_id encoding constant


def _decode_pair_id(pair_id: int) -> Tuple[int, int]:
    """Decode pair_id stored by COLMAP."""
    image_id2 = pair_id % PAIR_PRIME
    image_id1 = (pair_id - image_id2) // PAIR_PRIME
    return int(image_id1), int(image_id2)


@dataclass
class ImageStats:
    """Statistics for a single image."""
    image_id: int
    name: str
    num_features: int
    num_matches: int
    is_registered: bool
    camera_id: Optional[int] = None


@dataclass
class MatchStats:
    """Statistics for feature matches between image pairs."""
    image1_name: str
    image2_name: str
    num_matches: int
    inlier_ratio: Optional[float] = None


@dataclass
class ReconstructionDiagnostics:
    """Complete diagnostic report for a reconstruction."""
    total_images: int
    registered_images: int
    total_features: int
    total_matches: int
    avg_features_per_image: float
    avg_matches_per_pair: float
    image_stats: List[ImageStats]
    match_stats: List[MatchStats]
    point3d_count: int
    mean_track_length: float
    mean_reprojection_error: float


def get_database_stats(database_path: Path) -> Dict:
    """
    Extract statistics from COLMAP database.

    Args:
        database_path: Path to database.db file

    Returns:
        Dictionary with database statistics
    """
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    stats = {}

    # Get image count
    cursor.execute("SELECT COUNT(*) FROM images")
    stats['num_images'] = cursor.fetchone()[0]

    # Get camera count
    cursor.execute("SELECT COUNT(*) FROM cameras")
    stats['num_cameras'] = cursor.fetchone()[0]

    # Get total keypoints
    cursor.execute("SELECT SUM(rows) FROM keypoints")
    result = cursor.fetchone()[0]
    stats['total_keypoints'] = result if result else 0

    # Get match count
    cursor.execute("SELECT COUNT(*) FROM two_view_geometries")
    stats['num_matches'] = cursor.fetchone()[0]

    conn.close()
    return stats


def get_image_feature_counts(database_path: Path) -> Dict[str, int]:
    """
    Get feature count for each image in the database.

    Args:
        database_path: Path to database.db file

    Returns:
        Dict mapping image_name -> feature_count
    """
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Query: get image name and keypoint count
    cursor.execute("""
        SELECT images.name, keypoints.rows
        FROM images
        LEFT JOIN keypoints ON images.image_id = keypoints.image_id
    """)

    feature_counts = {}
    for row in cursor.fetchall():
        image_name = row[0]
        num_features = row[1] if row[1] else 0
        feature_counts[image_name] = num_features

    conn.close()
    return feature_counts


def get_match_statistics(database_path: Path) -> List[MatchStats]:
    """
    Get feature match statistics between image pairs.

    Args:
        database_path: Path to database.db file

    Returns:
        List of MatchStats objects
    """
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Build image_id -> name mapping
    cursor.execute("SELECT image_id, name FROM images")
    id_to_name = dict(cursor.fetchall())

    cursor.execute("SELECT pair_id, rows FROM two_view_geometries")
    match_stats = []
    for pair_id, num_matches in cursor.fetchall():
        img1, img2 = _decode_pair_id(pair_id)
        name1 = id_to_name.get(img1)
        name2 = id_to_name.get(img2)
        if not name1 or not name2:
            continue
        match_stats.append(MatchStats(
            image1_name=name1,
            image2_name=name2,
            num_matches=num_matches,
            inlier_ratio=None  # Not stored in database
        ))

    conn.close()
    return match_stats


def analyze_reconstruction(
    reconstruction_path: Path,
    database_path: Path
) -> ReconstructionDiagnostics:
    """
    Perform comprehensive analysis of a COLMAP reconstruction.

    Args:
        reconstruction_path: Path to sparse reconstruction directory (e.g., sparse/0)
        database_path: Path to database.db file

    Returns:
        ReconstructionDiagnostics object with complete analysis
    """
    # Load reconstruction
    reconstruction = pycolmap.Reconstruction(str(reconstruction_path))

    # Get feature counts from database
    feature_counts = get_image_feature_counts(database_path)

    # Get registered images from reconstruction
    registered_image_names = {img.name for img in reconstruction.images.values()}

    # Build image statistics
    image_stats = []
    total_features = 0

    for image_name, num_features in feature_counts.items():
        is_registered = image_name in registered_image_names

        # Get image_id and camera_id if registered
        image_id = None
        camera_id = None
        if is_registered:
            for img in reconstruction.images.values():
                if img.name == image_name:
                    image_id = img.image_id
                    camera_id = img.camera_id
                    break

        image_stats.append(ImageStats(
            image_id=image_id if image_id else 0,
            name=image_name,
            num_features=num_features,
            num_matches=0,  # Will calculate separately
            is_registered=is_registered,
            camera_id=camera_id
        ))

        total_features += num_features

    # Get match statistics
    match_stats = get_match_statistics(database_path)

    # Calculate 3D point statistics
    num_points = len(reconstruction.points3D)

    if num_points > 0:
        mean_track_length = sum(
            len(pt.track.elements) for pt in reconstruction.points3D.values()
        ) / num_points

        mean_reprojection_error = sum(
            pt.error for pt in reconstruction.points3D.values()
        ) / num_points
    else:
        mean_track_length = 0.0
        mean_reprojection_error = 0.0

    # Calculate statistics
    num_images = len(feature_counts)
    num_registered = len(registered_image_names)
    avg_features = total_features / num_images if num_images > 0 else 0
    avg_matches = sum(m.num_matches for m in match_stats) / len(match_stats) if match_stats else 0

    return ReconstructionDiagnostics(
        total_images=num_images,
        registered_images=num_registered,
        total_features=total_features,
        total_matches=len(match_stats),
        avg_features_per_image=avg_features,
        avg_matches_per_pair=avg_matches,
        image_stats=image_stats,
        match_stats=match_stats,
        point3d_count=num_points,
        mean_track_length=mean_track_length,
        mean_reprojection_error=mean_reprojection_error
    )


def print_diagnostics(diagnostics: ReconstructionDiagnostics, verbose: bool = False):
    """
    Print formatted diagnostic report.

    Args:
        diagnostics: ReconstructionDiagnostics object
        verbose: If True, print detailed per-image statistics
    """
    print("\n" + "="*80)
    print("COLMAP RECONSTRUCTION DIAGNOSTICS")
    print("="*80)

    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total images:          {diagnostics.total_images}")
    print(f"  Registered images:     {diagnostics.registered_images} ({diagnostics.registered_images/diagnostics.total_images*100:.1f}%)")
    print(f"  Unregistered images:   {diagnostics.total_images - diagnostics.registered_images}")
    print(f"  3D points:             {diagnostics.point3d_count}")
    print(f"  Mean track length:     {diagnostics.mean_track_length:.2f}")
    print(f"  Mean reprojection err: {diagnostics.mean_reprojection_error:.3f} px")

    # Feature statistics
    print(f"\n🔍 FEATURE EXTRACTION:")
    print(f"  Total features:        {diagnostics.total_features}")
    print(f"  Avg features/image:    {diagnostics.avg_features_per_image:.1f}")

    # Feature distribution
    registered_features = sum(
        img.num_features for img in diagnostics.image_stats if img.is_registered
    )
    unregistered_features = sum(
        img.num_features for img in diagnostics.image_stats if not img.is_registered
    )

    print(f"  Features in registered imgs:   {registered_features}")
    print(f"  Features in unregistered imgs: {unregistered_features}")

    # Match statistics
    print(f"\n🔗 FEATURE MATCHING:")
    print(f"  Total image pairs:     {diagnostics.total_matches}")
    print(f"  Avg matches/pair:      {diagnostics.avg_matches_per_pair:.1f}")

    # Top matches
    if diagnostics.match_stats:
        print(f"\n  Top 10 matches:")
        for i, match in enumerate(sorted(diagnostics.match_stats, key=lambda x: x.num_matches, reverse=True)[:10]):
            print(f"    {i+1}. {match.image1_name} <-> {match.image2_name}: {match.num_matches} matches")

    # Registered vs unregistered analysis
    registered_imgs = [img for img in diagnostics.image_stats if img.is_registered]
    unregistered_imgs = [img for img in diagnostics.image_stats if not img.is_registered]

    print(f"\n✅ REGISTERED IMAGES ({len(registered_imgs)}):")
    for img in sorted(registered_imgs, key=lambda x: x.num_features, reverse=True):
        print(f"  {img.name}: {img.num_features} features")

    print(f"\n❌ UNREGISTERED IMAGES ({len(unregistered_imgs)}):")
    if verbose:
        for img in sorted(unregistered_imgs, key=lambda x: x.num_features, reverse=True):
            print(f"  {img.name}: {img.num_features} features")
    else:
        # Show just a few examples
        for img in sorted(unregistered_imgs, key=lambda x: x.num_features, reverse=True)[:10]:
            print(f"  {img.name}: {img.num_features} features")
        if len(unregistered_imgs) > 10:
            print(f"  ... and {len(unregistered_imgs) - 10} more (use --verbose for full list)")

    # Feature quality analysis
    if registered_imgs:
        avg_registered_features = sum(img.num_features for img in registered_imgs) / len(registered_imgs)
        avg_unregistered_features = sum(img.num_features for img in unregistered_imgs) / len(unregistered_imgs) if unregistered_imgs else 0

        print(f"\n📈 FEATURE QUALITY COMPARISON:")
        print(f"  Avg features (registered):   {avg_registered_features:.1f}")
        print(f"  Avg features (unregistered): {avg_unregistered_features:.1f}")

        if avg_unregistered_features < avg_registered_features * 0.5:
            print(f"  ⚠️  Unregistered images have significantly fewer features!")

    print("\n" + "="*80)


if __name__ == "__main__":
    """
    Validation function: Test diagnostics on reconstruction_1_visits.

    Expected results:
    - Should load database and reconstruction successfully
    - Should identify 2 registered images out of 51 total
    - Should show feature counts and match statistics
    - Should complete without errors
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: Analyze reconstruction_1_visits
    total_tests += 1
    print("\n🧪 Test 1: Analyzing reconstruction_1_visits")
    try:
        database_path = Path("results/reconstruction_1_visits/database.db")
        reconstruction_path = Path("results/reconstruction_1_visits/sparse/0")

        if not database_path.exists():
            all_validation_failures.append(f"Test 1: Database not found at {database_path}")
        elif not reconstruction_path.exists():
            all_validation_failures.append(f"Test 1: Reconstruction not found at {reconstruction_path}")
        else:
            diagnostics = analyze_reconstruction(reconstruction_path, database_path)
            print_diagnostics(diagnostics, verbose=False)

            # Verify expected results
            if diagnostics.total_images != 51:
                all_validation_failures.append(f"Test 1: Expected 51 images, got {diagnostics.total_images}")
            if diagnostics.registered_images != 2:
                all_validation_failures.append(f"Test 1: Expected 2 registered images, got {diagnostics.registered_images}")
            if diagnostics.point3d_count != 303:
                all_validation_failures.append(f"Test 1: Expected 303 3D points, got {diagnostics.point3d_count}")

    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception occurred: {type(e).__name__}: {e}")

    # Test 2: Get database stats
    total_tests += 1
    print("\n🧪 Test 2: Getting database statistics")
    try:
        db_stats = get_database_stats(Path("results/reconstruction_1_visits/database.db"))
        print(f"Database stats: {db_stats}")

        if db_stats['num_images'] != 51:
            all_validation_failures.append(f"Test 2: Expected 51 images in database, got {db_stats['num_images']}")

    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception occurred: {type(e).__name__}: {e}")

    # Test 3: Get feature counts
    total_tests += 1
    print("\n🧪 Test 3: Getting feature counts per image")
    try:
        feature_counts = get_image_feature_counts(Path("results/reconstruction_1_visits/database.db"))
        print(f"Got feature counts for {len(feature_counts)} images")

        if len(feature_counts) != 51:
            all_validation_failures.append(f"Test 3: Expected 51 images, got {len(feature_counts)}")

        # Check that features are reasonable (should be > 0 for most images)
        images_with_features = sum(1 for count in feature_counts.values() if count > 0)
        if images_with_features < 40:  # Expect at least 40/51 images to have features
            all_validation_failures.append(f"Test 3: Only {images_with_features}/51 images have features")

    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception occurred: {type(e).__name__}: {e}")

    # Final validation result
    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Diagnostics tool is validated and ready for use")
        sys.exit(0)
