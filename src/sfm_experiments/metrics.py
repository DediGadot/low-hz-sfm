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


def _camera_center_from_pose(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP pose (qw, qx, qy, qz, tvec) to camera center."""
    rotation = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
    R = rotation.as_matrix()
    center = -R.T @ tvec
    return center


def _align_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Align source points to target via similarity transform."""
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

    var_source = np.sum(source_centered ** 2)
    scale = 1.0 if var_source == 0 else np.sum(S) / var_source
    aligned = (scale * R @ source_centered.T).T + target_mean
    return aligned


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
            # Pick the largest run; convert to seconds depending on magnitude
            val = float(matches[-1])
            if val > 1e15:  # nanoseconds
                return val / 1e9
            if val > 1e12:  # microseconds
                return val / 1e6
            if val > 1e9:  # milliseconds
                return val / 1e3
            return val

        mapped = {}
        gt_times = np.array([v[0] for v in gt_numeric]) if gt_numeric else np.array([])
        for name, pose in estimated_poses.items():
            ts = _parse_ts(name)
            if ts is None or gt_times.size == 0:
                continue
            diff = np.abs(gt_times - ts)
            idx = int(np.argmin(diff))
            if diff[idx] <= timestamp_tolerance:
                mapped_key = gt_numeric[idx][1]
                mapped[name] = mapped_key

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

        est_positions.append(_camera_center_from_pose(est_qvec, est_tvec))
        gt_positions.append(_camera_center_from_pose(gt_qvec, gt_tvec))

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
) -> float:
    """
    Compute Chamfer Distance between point clouds.

    Chamfer Distance is the average bidirectional distance between
    two point clouds, measuring geometric similarity.

    Formula:
        CD = 0.5 * (mean(d(P_recon → P_gt)) + mean(d(P_gt → P_recon)))

    Args:
        reconstruction_pcd: Reconstructed point cloud
        ground_truth_pcd: Ground truth point cloud

    Returns:
        Chamfer distance in meters

    Example:
        >>> recon_pcd = o3d.io.read_point_cloud("reconstruction.ply")
        >>> gt_pcd = o3d.io.read_point_cloud("ground_truth.ply")
        >>> cd = compute_chamfer_distance(recon_pcd, gt_pcd)
        >>> print(f"Chamfer Distance: {cd:.4f}m")
    """
    logger.info(
        f"Computing Chamfer Distance: "
        f"recon={len(reconstruction_pcd.points)} points, "
        f"gt={len(ground_truth_pcd.points)} points"
    )

    def _maybe_downsample(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if len(pcd.points) <= max_points:
            return pcd
        # Voxel size heuristic to reach target count
        voxel_size = max(pcd.get_max_bound() - pcd.get_min_bound()) / 512.0
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    recon_small = _maybe_downsample(reconstruction_pcd)
    gt_small = _maybe_downsample(ground_truth_pcd)

    # Distance from reconstruction to ground truth
    dists_recon_to_gt = np.asarray(
        recon_small.compute_point_cloud_distance(gt_small)
    )

    # Distance from ground truth to reconstruction
    dists_gt_to_recon = np.asarray(
        gt_small.compute_point_cloud_distance(recon_small)
    )

    # Chamfer distance is symmetric average
    chamfer = 0.5 * (np.mean(dists_recon_to_gt) + np.mean(dists_gt_to_recon))

    logger.info(f"Chamfer Distance: {chamfer:.4f}m")
    return float(chamfer)


def compute_completeness(
    reconstruction_pcd: o3d.geometry.PointCloud,
    ground_truth_pcd: o3d.geometry.PointCloud,
    threshold: float = 0.10,  # 10 cm
    max_points: int = 200000,
) -> float:
    """
    Compute map completeness.

    Completeness measures what percentage of the ground truth is covered
    by the reconstruction (within a distance threshold).

    Args:
        reconstruction_pcd: Reconstructed point cloud
        ground_truth_pcd: Ground truth point cloud
        threshold: Distance threshold in meters (default: 0.10m = 10cm)

    Returns:
        Completeness percentage (0.0 to 1.0)

    Example:
        >>> completeness = compute_completeness(recon_pcd, gt_pcd, threshold=0.10)
        >>> print(f"Completeness: {completeness*100:.1f}%")
    """
    logger.info(
        f"Computing completeness (threshold={threshold}m): "
        f"recon={len(reconstruction_pcd.points)} points, "
        f"gt={len(ground_truth_pcd.points)} points"
    )

    def _maybe_downsample(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if len(pcd.points) <= max_points:
            return pcd
        voxel_size = max(pcd.get_max_bound() - pcd.get_min_bound()) / 512.0
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    recon_small = _maybe_downsample(reconstruction_pcd)
    gt_small = _maybe_downsample(ground_truth_pcd)

    # Distance from each ground truth point to nearest reconstruction point
    dists_gt_to_recon = np.asarray(
        gt_small.compute_point_cloud_distance(recon_small)
    )

    # Count ground truth points within threshold
    completeness = np.mean(dists_gt_to_recon < threshold)

    logger.info(f"Completeness: {completeness*100:.1f}%")
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
        # Create test poses: perfect match
        est_poses = {
            "img1": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            "img2": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        }

        gt_poses = {
            "img1": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            "img2": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        }

        ate = compute_ate(est_poses, gt_poses)

        if not np.isclose(ate, 0.0, atol=1e-6):
            all_validation_failures.append(f"ATE perfect match: Expected 0.0, got {ate}")

        # Test with offset
        gt_poses_offset = {
            "img1": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])),
            "img2": (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.1, 0.0, 0.0])),
        }

        ate_offset = compute_ate(est_poses, gt_poses_offset)

        if not np.isclose(ate_offset, 0.1, atol=1e-6):
            all_validation_failures.append(f"ATE with offset: Expected 0.1, got {ate_offset}")

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
