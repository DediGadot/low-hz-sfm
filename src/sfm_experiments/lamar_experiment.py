#!/usr/bin/env python3
"""
LaMAR Dataset Experiment Orchestration

This module provides multi-scene analysis and experiments for the LaMAR dataset.
Unlike Hilti which uses multi-visit reconstruction, LaMAR experiments focus on:
- Comparing pre-built reconstructions across scenes
- Analyzing localization performance using query images
- Cross-scene point cloud analysis

Dependencies:
- pycolmap: https://github.com/colmap/pycolmap
- pathlib: https://docs.python.org/3/library/pathlib.html

Sample Input:
    scenes = ["CAB", "HGE", "LIN"]
    base_dir = Path("datasets/lamar")
    results = run_lamar_experiment(scenes, base_dir, output_dir)

Expected Output:
    Dictionary mapping scene names to reconstruction statistics and metrics
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time

from loguru import logger
import pycolmap
import open3d as o3d
import numpy as np

from .lamar_handler import (
    load_lamar_reconstruction,
    get_lamar_scene_info,
    validate_lamar_dataset,
    get_lamar_images_path,
    sample_lamar_images_by_session,
    LamarScene,
)
from .colmap_runner import run_sfm_reconstruction, ReconstructionResult
from .metrics import (
    compute_ate,
    compute_chamfer_distance,
    compute_completeness,
    align_point_clouds_icp,
)


@dataclass
class LamarSceneResult:
    """Results from analyzing a LaMAR scene."""
    scene_name: str
    success: bool
    num_cameras: int = 0
    num_images: int = 0
    num_points3d: int = 0
    reconstruction: Optional[pycolmap.Reconstruction] = None
    reconstruction_gt: Optional[pycolmap.Reconstruction] = None
    reconstruction_result: Optional[ReconstructionResult] = None
    colmap_path: Optional[Path] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    # Ground truth data
    num_images_gt: int = 0
    num_points3d_gt: int = 0
    num_cameras_gt: int = 0


def evaluate_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    ground_truth: pycolmap.Reconstruction,
    scene_name: str,
) -> Dict[str, float]:
    """
    Evaluate a reconstruction against ground truth.

    Computes multiple evaluation metrics:
    - ATE (Absolute Trajectory Error): camera pose accuracy
    - Chamfer Distance: point cloud geometric similarity
    - Completeness: percentage of ground truth covered
    - Registration ratio: % images successfully registered

    Args:
        reconstruction: Reconstructed COLMAP model
        ground_truth: Ground truth COLMAP model
        scene_name: Scene name for logging

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}

    def _is_registered(img: pycolmap.Image) -> bool:
        """Safe check for registered flag across pycolmap versions."""
        flag = getattr(img, "is_registered", None)
        if callable(flag):
            try:
                return bool(flag())
            except Exception:
                return False
        return bool(getattr(img, "registered", flag if flag is not None else False))

    def _registered_image_count(recon: pycolmap.Reconstruction) -> int:
        """Count only registered images to avoid overstating success."""
        return sum(
            1 for img in recon.images.values() if _is_registered(img)
        )

    # 1. Registration ratio (% images registered)
    num_images_recon = _registered_image_count(reconstruction)
    num_images_gt = _registered_image_count(ground_truth)
    registration_ratio = num_images_recon / num_images_gt if num_images_gt > 0 else 0.0
    metrics['registration_ratio'] = registration_ratio
    logger.info(f"Registration: {num_images_recon}/{num_images_gt} images ({registration_ratio*100:.1f}%)")

    # 2. Point cloud metrics (requires point clouds)
    num_points_recon = len(reconstruction.points3D)
    num_points_gt = len(ground_truth.points3D)
    metrics['num_points_recon'] = float(num_points_recon)
    metrics['num_points_gt'] = float(num_points_gt)
    metrics['points_ratio'] = num_points_recon / num_points_gt if num_points_gt > 0 else 0.0

    # 3. ATE (Absolute Trajectory Error) - requires matching images
    try:
        # Extract poses from reconstructions
        estimated_poses = {}
        ground_truth_poses = {}

        for img in reconstruction.images.values():
            if _is_registered(img):
                estimated_poses[img.name] = (img.qvec, img.tvec)

        for img in ground_truth.images.values():
            if _is_registered(img):
                ground_truth_poses[img.name] = (img.qvec, img.tvec)

        # Compute ATE for common images
        ate = compute_ate(estimated_poses, ground_truth_poses)
        metrics['ate'] = ate
        logger.info(f"ATE: {ate:.4f}m")
    except Exception as e:
        logger.warning(f"Failed to compute ATE: {e}")
        metrics['ate'] = -1.0

    # 4. Chamfer Distance and Completeness (requires point cloud conversion)
    try:
        # Convert COLMAP reconstructions to Open3D point clouds
        recon_points = np.array([p.xyz for p in reconstruction.points3D.values()])
        gt_points = np.array([p.xyz for p in ground_truth.points3D.values()])

        recon_pcd = o3d.geometry.PointCloud()
        recon_pcd.points = o3d.utility.Vector3dVector(recon_points)

        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_points)

        # Align point clouds using ICP
        logger.info("Aligning point clouds with ICP...")
        transformation, fitness = align_point_clouds_icp(recon_pcd, gt_pcd)
        recon_pcd.transform(transformation)
        metrics['icp_fitness'] = fitness

        # Compute Chamfer distance
        chamfer = compute_chamfer_distance(recon_pcd, gt_pcd)
        metrics['chamfer_distance'] = chamfer

        # Compute completeness (10cm threshold)
        completeness = compute_completeness(recon_pcd, gt_pcd, threshold=0.10)
        metrics['completeness'] = completeness
        logger.info(f"Completeness: {completeness*100:.1f}%")

    except Exception as e:
        logger.warning(f"Failed to compute point cloud metrics: {e}")
        metrics['chamfer_distance'] = -1.0
        metrics['completeness'] = -1.0
        metrics['icp_fitness'] = -1.0

    return metrics


def analyze_lamar_scene(
    scene_name: str,
    base_dir: Path,
    validate: bool = True,
    run_reconstruction: bool = False,
    output_dir: Optional[Path] = None,
    mapper_type: str = "colmap",
    evaluate_against_gt: bool = False,
    target_fps: Optional[float] = None,
) -> LamarSceneResult:
    """
    Analyze a single LaMAR scene by loading or reconstructing it.

    Args:
        scene_name: Name of scene ('CAB', 'HGE', or 'LIN')
        base_dir: Base directory for LaMAR dataset
        validate: Whether to validate dataset structure before loading
        run_reconstruction: Whether to run SfM reconstruction from raw images
        output_dir: Output directory for reconstruction (required if run_reconstruction=True)
        mapper_type: Reconstruction method ('colmap' or 'glomap')
        evaluate_against_gt: Whether to evaluate reconstruction against ground truth
        target_fps: Target frame rate for image sampling (e.g., 0.25). Only applies
                   when run_reconstruction=True. If None, uses all images.

    Returns:
        LamarSceneResult with scene statistics and optional evaluation metrics
    """
    start_time = time.time()

    logger.info(f"Analyzing LaMAR scene: {scene_name}")

    # Validate dataset structure if requested
    if validate:
        is_valid, issues = validate_lamar_dataset(base_dir, scene_name)
        if not is_valid:
            error_msg = f"Dataset validation failed: {', '.join(issues)}"
            logger.error(error_msg)
            return LamarSceneResult(
                scene_name=scene_name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=error_msg
            )

    # Get scene info
    scene_info = get_lamar_scene_info(scene_name, base_dir)

    if not scene_info.colmap_path or not scene_info.colmap_path.exists():
        error_msg = f"COLMAP reconstruction not found for {scene_name}"
        logger.error(error_msg)
        return LamarSceneResult(
            scene_name=scene_name,
            success=False,
            execution_time=time.time() - start_time,
            error_message=error_msg
        )

    try:
        # Load ground truth reconstruction (pre-built COLMAP)
        logger.info(f"Loading ground truth reconstruction for {scene_name}...")
        reconstruction_gt = load_lamar_reconstruction(scene_info.colmap_path)

        if reconstruction_gt is None:
            error_msg = f"Failed to load ground truth reconstruction for {scene_name}"
            logger.error(error_msg)
            return LamarSceneResult(
                scene_name=scene_name,
                success=False,
                colmap_path=scene_info.colmap_path,
                execution_time=time.time() - start_time,
                error_message=error_msg
            )

        # Store ground truth statistics
        num_cameras_gt = len(reconstruction_gt.cameras)
        num_images_gt = len(reconstruction_gt.images)
        num_points3d_gt = len(reconstruction_gt.points3D)

        logger.info(f"Ground truth: {num_images_gt} images, {num_points3d_gt} points")

        # If not running reconstruction, use ground truth as main reconstruction (old behavior)
        if not run_reconstruction:
            execution_time = time.time() - start_time
            logger.info(f"✅ {scene_name}: {num_images_gt} images, {num_points3d_gt} points ({execution_time:.2f}s)")

            return LamarSceneResult(
                scene_name=scene_name,
                success=True,
                num_cameras=num_cameras_gt,
                num_images=num_images_gt,
                num_points3d=num_points3d_gt,
                reconstruction=reconstruction_gt,
                reconstruction_gt=reconstruction_gt,
                colmap_path=scene_info.colmap_path,
                execution_time=execution_time,
                num_cameras_gt=num_cameras_gt,
                num_images_gt=num_images_gt,
                num_points3d_gt=num_points3d_gt,
            )

        # Run reconstruction from raw images
        if not output_dir:
            error_msg = "output_dir required when run_reconstruction=True"
            logger.error(error_msg)
            return LamarSceneResult(
                scene_name=scene_name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=error_msg
            )

        reconstruction_result = reconstruct_lamar_scene(
            scene_name=scene_name,
            base_dir=base_dir,
            output_dir=output_dir,
            mapper_type=mapper_type,
            target_fps=target_fps,
        )

        if not reconstruction_result or not reconstruction_result.success:
            error_msg = f"Reconstruction failed for {scene_name}"
            logger.error(error_msg)
            return LamarSceneResult(
                scene_name=scene_name,
                success=False,
                reconstruction_gt=reconstruction_gt,
                num_cameras_gt=num_cameras_gt,
                num_images_gt=num_images_gt,
                num_points3d_gt=num_points3d_gt,
                execution_time=time.time() - start_time,
                error_message=error_msg
            )

        reconstruction = reconstruction_result.reconstruction
        num_cameras = len(reconstruction.cameras)
        num_images = len(reconstruction.images)
        num_points3d = len(reconstruction.points3D)

        logger.info(f"Reconstructed: {num_images} images, {num_points3d} points")

        # Evaluate against ground truth if requested
        evaluation_metrics = {}
        if evaluate_against_gt:
            logger.info(f"Evaluating reconstruction against ground truth...")
            try:
                evaluation_metrics = evaluate_reconstruction(
                    reconstruction=reconstruction,
                    ground_truth=reconstruction_gt,
                    scene_name=scene_name
                )
                logger.info(f"Evaluation complete: {len(evaluation_metrics)} metrics computed")
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        execution_time = time.time() - start_time

        logger.info(f"✅ {scene_name}: {num_images} images, {num_points3d} points ({execution_time:.2f}s)")

        return LamarSceneResult(
            scene_name=scene_name,
            success=True,
            num_cameras=num_cameras,
            num_images=num_images,
            num_points3d=num_points3d,
            reconstruction=reconstruction,
            reconstruction_gt=reconstruction_gt,
            reconstruction_result=reconstruction_result,
            colmap_path=scene_info.colmap_path,
            execution_time=execution_time,
            evaluation_metrics=evaluation_metrics,
            num_cameras_gt=num_cameras_gt,
            num_images_gt=num_images_gt,
            num_points3d_gt=num_points3d_gt,
        )

    except Exception as e:
        error_msg = f"Exception processing {scene_name}: {str(e)}"
        logger.error(error_msg)
        return LamarSceneResult(
            scene_name=scene_name,
            success=False,
            colmap_path=scene_info.colmap_path,
            execution_time=time.time() - start_time,
            error_message=error_msg
        )


def reconstruct_lamar_scene(
    scene_name: str,
    base_dir: Path,
    output_dir: Path,
    mapper_type: str = "colmap",
    camera_model: str = "SIMPLE_RADIAL",
    max_num_features: int = 8192,
    use_cache: bool = True,
    target_fps: Optional[float] = None,
) -> Optional[ReconstructionResult]:
    """
    Run full SfM reconstruction on a LaMAR scene from raw images.

    Args:
        scene_name: Name of scene ('CAB', 'HGE', or 'LIN')
        base_dir: Base directory for LaMAR dataset
        output_dir: Output directory for reconstruction
        mapper_type: Reconstruction method ('colmap' or 'glomap')
        camera_model: Camera model to use (default: SIMPLE_RADIAL)
        max_num_features: Maximum SIFT features per image
        use_cache: Whether to use cached reconstruction if available
        target_fps: Target frame rate for image sampling (e.g., 0.25). If None, uses all images.

    Returns:
        ReconstructionResult or None if reconstruction fails
    """
    logger.info(f"Reconstructing LaMAR scene {scene_name} with {mapper_type}...")

    # Get images path
    images_path = get_lamar_images_path(base_dir, scene_name)
    if not images_path:
        logger.error(f"Images not found for scene {scene_name}")
        return None

    # Sample images if target_fps is specified
    actual_images_path = images_path
    if target_fps is not None:
        interval = 1.0 / target_fps
        if interval >= 1.0:
            interval_str = f"{interval:.1f}s"
        else:
            interval_str = f"{interval:.3f}s"
        logger.info(f"Sampling images at {target_fps} FPS (1 frame every {interval_str})...")

        # Sample images per session
        sampled_images = sample_lamar_images_by_session(images_path, target_fps)

        if not sampled_images:
            logger.error(f"No images after sampling at {target_fps} FPS")
            return None

        # Create temporary directory with symlinks to sampled images
        import shutil
        temp_images_dir = output_dir / scene_name / f"sampled_images_fps_{target_fps}"

        # Remove existing temp dir if it exists
        if temp_images_dir.exists():
            shutil.rmtree(temp_images_dir)

        temp_images_dir.mkdir(parents=True, exist_ok=True)

        # Create symlinks to sampled images (preserving directory structure)
        logger.info(f"Creating symlinks for {len(sampled_images)} sampled images...")
        for img_path in sampled_images:
            # Get relative path from images_path
            rel_path = img_path.relative_to(images_path)
            link_path = temp_images_dir / rel_path
            link_path.parent.mkdir(parents=True, exist_ok=True)

            # Create symlink (use absolute path to avoid broken symlinks)
            if not link_path.exists():
                link_path.symlink_to(img_path.resolve())

        actual_images_path = temp_images_dir
        logger.info(f"Using sampled images from {actual_images_path}")

    # Create scene-specific output directory
    scene_output_dir = output_dir / scene_name / f"reconstruction_{mapper_type}"
    if target_fps is not None:
        scene_output_dir = output_dir / scene_name / f"reconstruction_{mapper_type}_fps_{target_fps}"
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if cached reconstruction exists
    sparse_path = scene_output_dir / "sparse" / "0"
    if use_cache and sparse_path.exists():
        cameras_file = sparse_path / "cameras.bin"
        images_file = sparse_path / "images.bin"
        points_file = sparse_path / "points3D.bin"

        if all(f.exists() for f in [cameras_file, images_file, points_file]):
            logger.info(f"Loading cached {mapper_type} reconstruction from {sparse_path}")
            try:
                reconstruction = pycolmap.Reconstruction(str(sparse_path))
                num_registered = (
                    reconstruction.num_reg_images()
                    if hasattr(reconstruction, "num_reg_images")
                    else len(reconstruction.images)
                )
                num_points = (
                    reconstruction.num_points3D()
                    if hasattr(reconstruction, "num_points3D")
                    else len(reconstruction.points3D)
                )
                result = ReconstructionResult(
                    success=True,
                    num_registered_images=num_registered,
                    num_3d_points=num_points,
                    mean_reprojection_error=reconstruction.compute_mean_reprojection_error()
                    if hasattr(reconstruction, "compute_mean_reprojection_error")
                    else 0.0,
                    execution_time=0.0,
                    sparse_path=sparse_path,
                    mapper_type=mapper_type,
                    reconstruction=reconstruction,
                )
                logger.info(
                    f"✅ Loaded cached reconstruction: {result.num_registered_images} images, "
                    f"{result.num_3d_points} points"
                )
                return result
            except Exception as e:
                logger.warning(f"Failed to load cached reconstruction: {e}, running fresh reconstruction")

    # Run SfM reconstruction
    try:
        result = run_sfm_reconstruction(
            image_dir=actual_images_path,
            output_dir=scene_output_dir,
            mapper_type=mapper_type,
            camera_model=camera_model,
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
            max_num_features=max_num_features,
            use_cache=use_cache,
        )

        if result.success:
            logger.info(f"✅ Reconstruction complete: {result.num_registered_images} images, "
                       f"{result.num_3d_points} points ({result.execution_time:.2f}s)")
        else:
            logger.error(f"❌ Reconstruction failed for {scene_name}")

        return result

    except Exception as e:
        logger.error(f"Exception during reconstruction of {scene_name}: {e}")
        return None


def run_lamar_experiment(
    scenes: List[str],
    base_dir: Path,
    output_dir: Path,
    validate: bool = True,
    run_reconstruction: bool = False,
    mapper_type: str = "colmap",
    evaluate_against_gt: bool = False,
    target_fps: Optional[float] = None,
) -> Dict[str, LamarSceneResult]:
    """
    Run analysis or reconstruction experiment on multiple LaMAR scenes.

    This can either:
    1. Load pre-built COLMAP reconstructions for analysis (run_reconstruction=False)
    2. Run full SfM reconstruction from raw images (run_reconstruction=True)

    Args:
        scenes: List of scene names to analyze (e.g., ['CAB', 'HGE', 'LIN'])
        base_dir: Base directory for LaMAR dataset
        output_dir: Output directory for results
        validate: Whether to validate dataset structure
        run_reconstruction: Whether to run SfM reconstruction from raw images
        mapper_type: Reconstruction method ('colmap' or 'glomap')
        evaluate_against_gt: Whether to evaluate reconstruction against ground truth
        target_fps: Target frame rate for image sampling (e.g., 0.25). Only applies
                   when run_reconstruction=True. If None, uses all images.

    Returns:
        Dictionary mapping scene name to LamarSceneResult

    Example:
        >>> # Analysis mode (old behavior)
        >>> results = run_lamar_experiment(
        ...     ['CAB', 'HGE'],
        ...     Path('datasets/lamar'),
        ...     Path('results/lamar')
        ... )
        >>> # Reconstruction mode (new behavior)
        >>> results = run_lamar_experiment(
        ...     ['CAB'],
        ...     Path('datasets/lamar'),
        ...     Path('results/lamar'),
        ...     run_reconstruction=True,
        ...     mapper_type='colmap',
        ...     evaluate_against_gt=True,
        ...     target_fps=0.25
        ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    mode_str = "Reconstruction" if run_reconstruction else "Analysis"
    logger.info(f"\n{'='*80}")
    logger.info(f"Running LaMAR Multi-Scene {mode_str}")
    logger.info(f"Scenes: {', '.join(scenes)}")
    logger.info(f"Output: {output_dir}")
    if run_reconstruction:
        logger.info(f"Mapper: {mapper_type}")
        logger.info(f"Evaluate: {evaluate_against_gt}")
        if target_fps is not None:
            interval = 1.0 / target_fps
            if interval >= 1.0:
                logger.info(f"Target FPS: {target_fps} (1 frame every {interval:.1f}s)")
            else:
                logger.info(f"Target FPS: {target_fps} (1 frame every {interval:.3f}s)")
        else:
            logger.info(f"Target FPS: None (using all images)")
    logger.info(f"{'='*80}\n")

    for scene_name in scenes:
        if run_reconstruction:
            result = analyze_lamar_scene(
                scene_name,
                base_dir,
                validate,
                run_reconstruction=True,
                output_dir=output_dir,
                mapper_type=mapper_type,
                evaluate_against_gt=evaluate_against_gt,
                target_fps=target_fps,
            )
        else:
            result = analyze_lamar_scene(scene_name, base_dir, validate)
        results[scene_name] = result

    # Summary
    successful = sum(1 for r in results.values() if r.success)
    total = len(results)

    logger.info(f"\n{'='*80}")
    logger.info(f"Experiment Summary: {successful}/{total} scenes processed successfully")
    if evaluate_against_gt and run_reconstruction:
        for scene_name, result in results.items():
            if result.success and result.evaluation_metrics:
                logger.info(f"{scene_name} Evaluation:")
                for metric, value in result.evaluation_metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
    logger.info(f"{'='*80}")

    return results


def compare_lamar_scenes(
    results: Dict[str, LamarSceneResult]
) -> Dict[str, Dict[str, float]]:
    """
    Compare statistics across multiple LaMAR scenes.

    Args:
        results: Results from run_lamar_experiment

    Returns:
        Dictionary with comparative statistics

    Example:
        >>> comparison = compare_lamar_scenes(results)
        >>> print(f"Average images: {comparison['avg_images']}")
    """
    successful_results = {k: v for k, v in results.items() if v.success}

    if not successful_results:
        logger.warning("No successful scenes to compare")
        return {}

    comparison = {}

    # Calculate statistics
    total_images = sum(r.num_images for r in successful_results.values())
    total_points = sum(r.num_points3d for r in successful_results.values())
    num_scenes = len(successful_results)

    comparison['total'] = {
        'scenes': num_scenes,
        'images': total_images,
        'points3d': total_points,
    }

    comparison['average'] = {
        'images': total_images / num_scenes if num_scenes > 0 else 0,
        'points3d': total_points / num_scenes if num_scenes > 0 else 0,
    }

    # Per-scene statistics
    comparison['by_scene'] = {}
    for scene_name, result in successful_results.items():
        comparison['by_scene'][scene_name] = {
            'images': result.num_images,
            'points3d': result.num_points3d,
            'cameras': result.num_cameras,
        }

    # Find extremes
    max_images_scene = max(successful_results.items(), key=lambda x: x[1].num_images)
    max_points_scene = max(successful_results.items(), key=lambda x: x[1].num_points3d)

    comparison['max'] = {
        'images': {'scene': max_images_scene[0], 'count': max_images_scene[1].num_images},
        'points3d': {'scene': max_points_scene[0], 'count': max_points_scene[1].num_points3d},
    }

    return comparison


def summarize_lamar_results(results: Dict[str, LamarSceneResult]) -> Dict[str, any]:
    """
    Create summary statistics from LaMAR experiment results.

    Args:
        results: Results from run_lamar_experiment

    Returns:
        Summary dictionary with aggregate statistics
    """
    successful = [r for r in results.values() if r.success]
    failed = [r for r in results.values() if not r.success]

    summary = {
        'num_scenes': len(results),
        'successful_scenes': len(successful),
        'failed_scenes': len(failed),
        'scene_names': list(results.keys()),
        'total_images': sum(r.num_images for r in successful),
        'total_points3d': sum(r.num_points3d for r in successful),
        'total_execution_time': sum(r.execution_time for r in results.values()),
    }

    if successful:
        summary['avg_images'] = summary['total_images'] / len(successful)
        summary['avg_points3d'] = summary['total_points3d'] / len(successful)
        summary['max_images'] = max(r.num_images for r in successful)
        summary['max_points3d'] = max(r.num_points3d for r in successful)

    if failed:
        summary['failed_scene_names'] = [r.scene_name for r in failed]
        summary['failure_messages'] = {r.scene_name: r.error_message for r in failed}

    return summary


if __name__ == "__main__":
    """
    Validation function to test LaMAR experiment orchestration.

    Tests:
    1. Single scene analysis
    2. Multi-scene experiment
    3. Scene comparison
    4. Results summarization
    """
    from loguru import logger

    # Setup logging
    logger.add("logs/lamar_experiment_test.log", rotation="10 MB")

    # Track validation failures
    all_validation_failures = []
    total_tests = 0

    # Test configuration
    base_dir = Path("datasets/lamar")
    output_dir = Path("test_lamar_output")
    test_scene = "CAB"
    test_scenes = ["CAB", "HGE", "LIN"]

    print("="*80)
    print("LaMAR Experiment Orchestration Validation")
    print("="*80)

    # Test 1: Validate single scene analysis
    total_tests += 1
    print(f"\nTest {total_tests}: Analyzing single scene ({test_scene})...")

    if not base_dir.exists():
        all_validation_failures.append(f"Single scene analysis: Dataset directory not found: {base_dir}")
        print(f"❌ Dataset not found. Download first using:")
        print("  uv run python scripts/download_lamar_dataset.py")
    else:
        try:
            result = analyze_lamar_scene(test_scene, base_dir, validate=True)

            if not result.success:
                all_validation_failures.append(f"Single scene analysis: Failed - {result.error_message}")
                print(f"❌ Scene analysis failed: {result.error_message}")
            elif result.num_images == 0:
                all_validation_failures.append("Single scene analysis: No images in reconstruction")
                print("❌ No images in reconstruction")
            else:
                print(f"✅ Scene analyzed successfully:")
                print(f"  - Images: {result.num_images}")
                print(f"  - Points: {result.num_points3d}")
                print(f"  - Cameras: {result.num_cameras}")
                print(f"  - Time: {result.execution_time:.2f}s")

        except Exception as e:
            all_validation_failures.append(f"Single scene analysis: Exception - {e}")
            print(f"❌ Exception: {e}")

    # Only continue if dataset is available
    if base_dir.exists():
        # Test 2: Multi-scene experiment
        total_tests += 1
        print(f"\nTest {total_tests}: Running multi-scene experiment...")

        try:
            results = run_lamar_experiment(
                test_scenes,
                base_dir,
                output_dir,
                validate=False  # Already validated in test 1
            )

            successful = sum(1 for r in results.values() if r.success)
            if successful == 0:
                all_validation_failures.append("Multi-scene experiment: No scenes loaded successfully")
                print(f"❌ No scenes loaded successfully")
            else:
                print(f"✅ Loaded {successful}/{len(results)} scenes")

                # Test 3: Scene comparison
                total_tests += 1
                print(f"\nTest {total_tests}: Comparing scenes...")

                try:
                    comparison = compare_lamar_scenes(results)

                    if not comparison:
                        all_validation_failures.append("Scene comparison: Empty comparison results")
                        print("❌ Empty comparison results")
                    else:
                        print(f"✅ Comparison complete:")
                        print(f"  - Total images: {comparison['total']['images']}")
                        print(f"  - Total points: {comparison['total']['points3d']}")
                        print(f"  - Average images/scene: {comparison['average']['images']:.0f}")

                except Exception as e:
                    all_validation_failures.append(f"Scene comparison: Exception - {e}")
                    print(f"❌ Exception: {e}")

                # Test 4: Results summarization
                total_tests += 1
                print(f"\nTest {total_tests}: Summarizing results...")

                try:
                    summary = summarize_lamar_results(results)

                    if summary['num_scenes'] == 0:
                        all_validation_failures.append("Results summarization: No scenes in summary")
                        print("❌ No scenes in summary")
                    else:
                        print(f"✅ Summary created:")
                        print(f"  - Scenes: {summary['num_scenes']}")
                        print(f"  - Successful: {summary['successful_scenes']}")
                        print(f"  - Total images: {summary['total_images']}")
                        print(f"  - Execution time: {summary['total_execution_time']:.2f}s")

                except Exception as e:
                    all_validation_failures.append(f"Results summarization: Exception - {e}")
                    print(f"❌ Exception: {e}")

            # Clean up test output
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)

        except Exception as e:
            all_validation_failures.append(f"Multi-scene experiment: Exception - {e}")
            print(f"❌ Exception: {e}")

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
        print("LaMAR experiment orchestration is validated and ready to use")
        print("="*80)
        sys.exit(0)
