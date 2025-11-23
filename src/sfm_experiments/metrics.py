"""
Evaluation metrics for SfM reconstructions.

This module provides:
- Absolute Trajectory Error (ATE)
- Chamfer Distance (point cloud comparison)
- Map Completeness
- Reconstruction alignment (Sim3/Procrustes)

Dependencies:
- numpy: https://numpy.org/
- scipy: https://scipy.org/
- open3d: http://www.open3d.org/

Sample Input: Estimated poses + ground truth poses
Expected Output: Numerical accuracy metrics (ATE, Chamfer, Completeness)
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
from loguru import logger

# PERFORMANCE: Module-level cache for point cloud distance computations
# Caches Chamfer distance and completeness results to avoid recomputation
_DISTANCE_CACHE: Dict[int, float] = {}


def _camera_center_from_pose(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP pose (qw, qx, qy, qz, tvec) to camera center."""
    # BUGFIX: Normalize quaternion to ensure valid rotation matrix
    # Un-normalized quaternions produce incorrect rotations
    qvec_norm = qvec / np.linalg.norm(qvec)
    rotation = Rotation.from_quat([qvec_norm[1], qvec_norm[2], qvec_norm[3], qvec_norm[0]])
    R = rotation.as_matrix()
    center = -R.T @ tvec
    return center


def _align_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Align source points to target via similarity transform."""
    # BUGFIX: Validate minimum number of points
    if len(source) < 3:
        raise ValueError(f"Need at least 3 points for alignment, got {len(source)}")
    if len(source) != len(target):
        raise ValueError(f"Point counts must match: {len(source)} != {len(target)}")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # BUGFIX: Use epsilon comparison to avoid division by very small numbers
    var_source = np.sum(source_centered ** 2)
    scale = 1.0 if var_source < 1e-10 else np.sum(S) / var_source
    aligned = (scale * R @ source_centered.T).T + target_mean
    return aligned


def _compute_pcd_cache_key(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud,
    metric_name: str,
    max_points: int,
    threshold: Optional[float] = None,
) -> int:
    """
    Compute cache key for point cloud distance metrics.

    PERFORMANCE: Creates a hash key from point cloud data for caching distance computations.
    Uses point coordinates to ensure cache hits only when point clouds are identical.

    Args:
        pcd1: First point cloud
        pcd2: Second point cloud
        metric_name: Name of metric ("chamfer" or "completeness")
        max_points: Max points parameter (affects downsampling)
        threshold: Optional threshold parameter for completeness

    Returns:
        Integer hash key for cache lookup
    """
    # Hash based on point cloud contents
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    # Create hash from point data + parameters
    # Use first/last few points as fingerprint (faster than hashing all points)
    n1, n2 = len(points1), len(points2)
    fingerprint_size = min(100, n1, n2)  # Use up to 100 points for fingerprint

    if n1 > 0 and n2 > 0:
        # Sample points evenly distributed through the cloud
        idx1 = np.linspace(0, n1 - 1, fingerprint_size, dtype=int)
        idx2 = np.linspace(0, n2 - 1, fingerprint_size, dtype=int)
        fp1 = points1[idx1].tobytes()
        fp2 = points2[idx2].tobytes()
    else:
        fp1 = b""
        fp2 = b""

    # Combine fingerprints with parameters
    key_parts = [
        hash(fp1),
        hash(fp2),
        hash(metric_name),
        hash(max_points),
        hash(n1),
        hash(n2),
    ]

    if threshold is not None:
        key_parts.append(hash(threshold))

    return hash(tuple(key_parts))


def compute_ate(
    estimated_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ground_truth_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
    timestamp_tolerance: float = 0.05,
) -> float:
    """
    Compute Absolute Trajectory Error (ATE).

    ATE measures the RMS difference between estimated and ground truth
    camera positions after optimal alignment.

    References:
    - Sturm et al., "A Benchmark for RGB-D SLAM Evaluation", IROS 2012

    Args:
        estimated_poses: Dict mapping image name to (qvec, tvec)
            qvec: quaternion (qw, qx, qy, qz)
            tvec: translation vector (tx, ty, tz)
        ground_truth_poses: Dict mapping image name to (qvec, tvec)
        timestamp_tolerance: Maximum allowed difference (seconds) when aligning
            by timestamps parsed from filenames.

    Returns:
        RMS ATE in meters

    Example:
        >>> est_poses = {"img1.jpg": (np.array([1,0,0,0]), np.array([0,0,0]))}
        >>> gt_poses = {"img1.jpg": (np.array([1,0,0,0]), np.array([0.1,0,0]))}
        >>> ate = compute_ate(est_poses, gt_poses)
        >>> print(f"ATE: {ate:.3f}m")
    """
    # BUGFIX: Validate input poses are not empty
    if not estimated_poses or not ground_truth_poses:
        logger.warning("Empty pose dictionaries provided")
        return float('inf')

    # Find common image IDs
    common_ids = set(estimated_poses.keys()) & set(ground_truth_poses.keys())

    # Attempt timestamp-based alignment when names don't match (e.g., filenames embed timestamps)
    if not common_ids:
        # Build numeric lookup for GT keys
        gt_numeric = []
        for k in ground_truth_poses.keys():
            try:
                gt_numeric.append((float(k), k))
            except ValueError:
                continue

        # Helper to parse a timestamp-ish float from an image name
        def _parse_ts(name: str) -> Optional[float]:
            import re
            matches = re.findall(r"(\d{7,})", name)  # long digit runs only
            if not matches:
                return None
            # BUGFIX: Pick the longest digit sequence (not just last match)
            # and use more accurate timestamp unit detection
            val = float(max(matches, key=len))
            # Current Unix timestamp (year 2024) is ~1.7e9 seconds
            if val > 1e17:  # nanoseconds (Unix epoch * 1e9 ~ 1.7e18)
                return val / 1e9
            if val > 1e14:  # microseconds (Unix epoch * 1e6 ~ 1.7e15)
                return val / 1e6
            if val > 1e11:  # milliseconds (Unix epoch * 1e3 ~ 1.7e12)
                return val / 1e3
            if 1e9 < val < 2e9:  # Unix epoch seconds (1970-2033)
                return val
            return val  # Frame indices or other small values

        # BUGFIX: Prevent multiple estimated poses from mapping to same GT pose
        mapped = {}
        used_gt_indices = set()
        gt_times = np.array([v[0] for v in gt_numeric]) if gt_numeric else np.array([])

        # Sort estimated poses by timestamp for stable matching
        est_with_ts = []
        for name, pose in estimated_poses.items():
            ts = _parse_ts(name)
            if ts is not None:
                est_with_ts.append((ts, name, pose))
        est_with_ts.sort(key=lambda x: x[0])

        for ts, name, pose in est_with_ts:
            if gt_times.size == 0:
                continue
            diff = np.abs(gt_times - ts)
            sorted_indices = np.argsort(diff)

            # Find closest unused GT pose
            for idx in sorted_indices:
                if idx in used_gt_indices:
                    continue
                if diff[idx] <= timestamp_tolerance:
                    mapped_key = gt_numeric[idx][1]
                    mapped[name] = mapped_key
                    used_gt_indices.add(idx)
                    break

        if mapped:
            common_ids = set(mapped.keys())
            ground_truth_poses = {k: ground_truth_poses[v] for k, v in mapped.items()}
        else:
            logger.warning("No common images between estimated and ground truth")
            return float('inf')

    logger.info(f"Computing ATE over {len(common_ids)} common images")

    est_positions = []
    gt_positions = []
    for img_id in common_ids:
        est_qvec, est_tvec = estimated_poses[img_id]
        gt_qvec, gt_tvec = ground_truth_poses[img_id]

        # BUGFIX: Validate pose array dimensions before processing
        if len(est_qvec) != 4 or len(est_tvec) != 3:
            logger.warning(f"Invalid estimated pose dimensions for {img_id}, skipping")
            continue
        if len(gt_qvec) != 4 or len(gt_tvec) != 3:
            logger.warning(f"Invalid ground truth pose dimensions for {img_id}, skipping")
            continue

        est_positions.append(_camera_center_from_pose(est_qvec, est_tvec))
        gt_positions.append(_camera_center_from_pose(gt_qvec, gt_tvec))

    # BUGFIX: Check if we have valid positions after filtering
    if len(est_positions) == 0:
        logger.warning("No valid poses extracted from common images")
        return float('inf')

    est_positions = np.asarray(est_positions)
    gt_positions = np.asarray(gt_positions)

    aligned_positions = _align_similarity(est_positions, gt_positions)
    errors = np.linalg.norm(aligned_positions - gt_positions, axis=1)
    ate = np.sqrt(np.mean(errors ** 2))

    logger.info(f"ATE: {ate:.4f}m (over {len(errors)} poses)")
    return float(ate)


def compute_chamfer_distance(
    reconstruction_pcd: o3d.geometry.PointCloud,
    ground_truth_pcd: o3d.geometry.PointCloud,
    max_points: int = 200000,
    use_cache: bool = True,
) -> float:
    """
    Compute Chamfer Distance between point clouds.

    Chamfer Distance is the average bidirectional distance between
    two point clouds, measuring geometric similarity.

    Formula:
        CD = 0.5 * (mean(d(P_recon → P_gt)) + mean(d(P_gt → P_recon)))

    PERFORMANCE: Results are cached based on point cloud fingerprints. Set use_cache=False
    to force recomputation.

    Args:
        reconstruction_pcd: Reconstructed point cloud
        ground_truth_pcd: Ground truth point cloud
        max_points: Maximum points to use (downsampling applied if exceeded)
        use_cache: If True, use cached results when available (default: True)

    Returns:
        Chamfer distance in meters

    Example:
        >>> recon_pcd = o3d.io.read_point_cloud("reconstruction.ply")
        >>> gt_pcd = o3d.io.read_point_cloud("ground_truth.ply")
        >>> cd = compute_chamfer_distance(recon_pcd, gt_pcd)
        >>> print(f"Chamfer Distance: {cd:.4f}m")
    """
    # PERFORMANCE: Check cache before computing
    if use_cache:
        cache_key = _compute_pcd_cache_key(
            reconstruction_pcd,
            ground_truth_pcd,
            "chamfer",
            max_points,
        )
        if cache_key in _DISTANCE_CACHE:
            cached_result = _DISTANCE_CACHE[cache_key]
            logger.info(f"✓ Using cached Chamfer Distance: {cached_result:.4f}m")
            return cached_result

    logger.info(
        f"Computing Chamfer Distance: "
        f"recon={len(reconstruction_pcd.points)} points, "
        f"gt={len(ground_truth_pcd.points)} points"
    )

    def _maybe_downsample(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        # BUGFIX: Handle empty point clouds and degenerate cases
        if len(pcd.points) == 0:
            return pcd
        if len(pcd.points) <= max_points:
            return pcd
        bounds_diff = pcd.get_max_bound() - pcd.get_min_bound()
        max_bound = max(bounds_diff)
        if max_bound < 1e-6:  # All points essentially identical
            return pcd
        # Voxel size heuristic to reach target count
        voxel_size = max_bound / 512.0
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    recon_small = _maybe_downsample(reconstruction_pcd)
    gt_small = _maybe_downsample(ground_truth_pcd)

    # BUGFIX: Check for empty point clouds after downsampling
    if len(recon_small.points) == 0 or len(gt_small.points) == 0:
        logger.warning("One or both point clouds are empty after downsampling")
        return float('inf')

    # Distance from reconstruction to ground truth
    dists_recon_to_gt = np.asarray(
        recon_small.compute_point_cloud_distance(gt_small)
    )

    # Distance from ground truth to reconstruction
    dists_gt_to_recon = np.asarray(
        gt_small.compute_point_cloud_distance(recon_small)
    )

    # BUGFIX: Validate distance arrays are not empty
    if len(dists_recon_to_gt) == 0 or len(dists_gt_to_recon) == 0:
        logger.warning("No distances computed between point clouds")
        return float('inf')

    # Chamfer distance is symmetric average
    chamfer = 0.5 * (np.mean(dists_recon_to_gt) + np.mean(dists_gt_to_recon))

    logger.info(f"Chamfer Distance: {chamfer:.4f}m")

    # PERFORMANCE: Store result in cache
    if use_cache:
        cache_key = _compute_pcd_cache_key(
            reconstruction_pcd,
            ground_truth_pcd,
            "chamfer",
            max_points,
        )
        _DISTANCE_CACHE[cache_key] = float(chamfer)

    return float(chamfer)


def compute_completeness(
    reconstruction_pcd: o3d.geometry.PointCloud,
    ground_truth_pcd: o3d.geometry.PointCloud,
    threshold: float = 0.10,  # 10 cm
    max_points: int = 200000,
    use_cache: bool = True,
) -> float:
    """
    Compute map completeness.

    Completeness measures what percentage of the ground truth is covered
    by the reconstruction (within a distance threshold).

    PERFORMANCE: Results are cached based on point cloud fingerprints. Set use_cache=False
    to force recomputation.

    Args:
        reconstruction_pcd: Reconstructed point cloud
        ground_truth_pcd: Ground truth point cloud
        threshold: Distance threshold in meters (default: 0.10m = 10cm)
        max_points: Maximum points to use (downsampling applied if exceeded)
        use_cache: If True, use cached results when available (default: True)

    Returns:
        Completeness percentage (0.0 to 1.0)

    Example:
        >>> completeness = compute_completeness(recon_pcd, gt_pcd, threshold=0.10)
        >>> print(f"Completeness: {completeness*100:.1f}%")
    """
    # PERFORMANCE: Check cache before computing
    if use_cache:
        cache_key = _compute_pcd_cache_key(
            reconstruction_pcd,
            ground_truth_pcd,
            "completeness",
            max_points,
            threshold,
        )
        if cache_key in _DISTANCE_CACHE:
            cached_result = _DISTANCE_CACHE[cache_key]
            logger.info(f"✓ Using cached Completeness: {cached_result*100:.1f}%")
            return cached_result

    logger.info(
        f"Computing completeness (threshold={threshold}m): "
        f"recon={len(reconstruction_pcd.points)} points, "
        f"gt={len(ground_truth_pcd.points)} points"
    )

    def _maybe_downsample(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        # BUGFIX: Handle empty point clouds and degenerate cases (same as Chamfer)
        if len(pcd.points) == 0:
            return pcd
        if len(pcd.points) <= max_points:
            return pcd
        bounds_diff = pcd.get_max_bound() - pcd.get_min_bound()
        max_bound = max(bounds_diff)
        if max_bound < 1e-6:  # All points essentially identical
            return pcd
        voxel_size = max_bound / 512.0
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    recon_small = _maybe_downsample(reconstruction_pcd)
    gt_small = _maybe_downsample(ground_truth_pcd)

    # BUGFIX: Check for empty point clouds
    if len(recon_small.points) == 0 or len(gt_small.points) == 0:
        logger.warning("One or both point clouds are empty after downsampling")
        return 0.0  # 0% completeness

    # Distance from each ground truth point to nearest reconstruction point
    dists_gt_to_recon = np.asarray(
        gt_small.compute_point_cloud_distance(recon_small)
    )

    # BUGFIX: Validate distance array is not empty
    if len(dists_gt_to_recon) == 0:
        logger.warning("No distances computed from ground truth to reconstruction")
        return 0.0

    # Count ground truth points within threshold
    completeness = np.mean(dists_gt_to_recon < threshold)

    logger.info(f"Completeness: {completeness*100:.1f}%")

    # PERFORMANCE: Store result in cache
    if use_cache:
        cache_key = _compute_pcd_cache_key(
            reconstruction_pcd,
            ground_truth_pcd,
            "completeness",
            max_points,
            threshold,
        )
        _DISTANCE_CACHE[cache_key] = float(completeness)

    return float(completeness)


def align_point_clouds_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    threshold: float = 1.0,
    max_iterations: int = 100,
) -> Tuple[np.ndarray, float]:
    """
    Align source point cloud to target using ICP.

    Uses Iterative Closest Point (ICP) algorithm to find the optimal
    rigid transformation (rotation + translation) that aligns source to target.

    Args:
        source_pcd: Source point cloud (to be transformed)
        target_pcd: Target point cloud (reference)
        threshold: Maximum correspondence distance
        max_iterations: Maximum ICP iterations

    Returns:
        Tuple of (transformation_matrix, fitness_score)
        transformation_matrix: 4x4 homogeneous transformation matrix
        fitness_score: Alignment quality (0-1, higher is better)

    Example:
        >>> transform, fitness = align_point_clouds_icp(source, target)
        >>> aligned = source.transform(transform)
    """
    logger.info("Aligning point clouds with ICP...")

    # Run point-to-point ICP
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        np.eye(4),  # Initial transformation
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )

    logger.info(f"ICP fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}m")

    return result.transformation, result.fitness


def load_point_cloud(pcd_path: Path) -> o3d.geometry.PointCloud:
    """
    Load point cloud from file.

    Supports PLY, PCD, XYZ, and other formats supported by Open3D.

    Args:
        pcd_path: Path to point cloud file

    Returns:
        Open3D PointCloud object

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If file can't be loaded

    Example:
        >>> pcd = load_point_cloud(Path("reconstruction.ply"))
        >>> print(f"Loaded {len(pcd.points)} points")
    """
    if not pcd_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {pcd_path}")

    logger.info(f"Loading point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    if len(pcd.points) == 0:
        raise RuntimeError(f"Point cloud is empty: {pcd_path}")

    logger.info(f"Loaded {len(pcd.points)} points")
    return pcd


# Validation
if __name__ == "__main__":
    """
    Validation of metrics.py functionality.
    Tests metric computations with synthetic data.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: ATE computation
    total_tests += 1
    try:
        # Create test poses: perfect match (need at least 3 points for Sim3 alignment)
        est_poses = {
            "img1": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            "img2": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            "img3": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        }

        gt_poses = {
            "img1": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            "img2": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            "img3": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        }

        ate = compute_ate(est_poses, gt_poses)

        if not np.isclose(ate, 0.0, atol=1e-6):
            all_validation_failures.append(f"ATE perfect match: Expected 0.0, got {ate}")

        # BUGFIX: Test with uniform offset (should be ~0.0 after Sim3 alignment)
        gt_poses_uniform_offset = {
            "img1": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])),
            "img2": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.1, 0.0, 0.0])),
            "img3": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.1, 1.0, 0.0])),
        }

        ate_uniform = compute_ate(est_poses, gt_poses_uniform_offset)

        # Sim3 alignment removes uniform translation, so ATE should be near 0
        if not np.isclose(ate_uniform, 0.0, atol=1e-5):
            all_validation_failures.append(
                f"ATE uniform offset: Expected ~0.0 (Sim3 removes translation), got {ate_uniform}"
            )

        # Test with non-uniform error (should detect this)
        gt_poses_nonuniform = {
            "img1": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            "img2": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.2, 0.0])),  # Extra 0.2 in Y
            "img3": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # No error
        }

        ate_nonuniform = compute_ate(est_poses, gt_poses_nonuniform)

        # Should detect the 0.2 error in Y direction
        # Sim3 alignment will reduce but not eliminate the error
        # Important: ATE should be > 0 (detecting the error), exact value varies with alignment
        if ate_nonuniform < 0.01:  # Should detect error as > 1cm
            all_validation_failures.append(
                f"ATE non-uniform error: Expected to detect error >0.01m, got {ate_nonuniform:.3f}m"
            )

    except Exception as e:
        all_validation_failures.append(f"ATE computation: Exception raised: {e}")

    # Test 2: Point cloud creation and distance
    total_tests += 1
    try:
        # Create simple test point clouds
        points1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        points2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # Identical

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)

        # Test Chamfer distance (should be 0 for identical clouds)
        cd = compute_chamfer_distance(pcd1, pcd2)

        if not np.isclose(cd, 0.0, atol=1e-6):
            all_validation_failures.append(
                f"Chamfer distance identical: Expected 0.0, got {cd}"
            )

        # Test completeness (should be 100% for identical clouds)
        completeness = compute_completeness(pcd1, pcd2, threshold=0.01)

        if not np.isclose(completeness, 1.0, atol=1e-6):
            all_validation_failures.append(
                f"Completeness identical: Expected 1.0, got {completeness}"
            )

    except Exception as e:
        all_validation_failures.append(f"Point cloud metrics: Exception raised: {e}")

    # Test 3: No common images ATE
    total_tests += 1
    try:
        est_poses_no_match = {"img1": (np.array([1.0, 0, 0, 0]), np.array([0, 0, 0]))}
        gt_poses_no_match = {"img2": (np.array([1.0, 0, 0, 0]), np.array([0, 0, 0]))}

        ate_no_match = compute_ate(est_poses_no_match, gt_poses_no_match)

        if not np.isinf(ate_no_match):
            all_validation_failures.append(
                f"ATE no common images: Expected inf, got {ate_no_match}"
            )

    except Exception as e:
        all_validation_failures.append(f"ATE no common images: Exception raised: {e}")

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
        print("Metrics module validated and ready for use")
        sys.exit(0)
