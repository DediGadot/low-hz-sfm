"""
Enhanced COLMAP Reconstruction with Tuned Parameters

This module provides improved COLMAP reconstruction with parameters optimized
for low-overlap multi-visit scenarios like the Hilti dataset.

Dependencies:
- pycolmap: https://github.com/colmap/pycolmap
- loguru: https://loguru.readthedocs.io/

Key improvements over default parameters:
1. Relaxed initialization thresholds for low-feature-match scenarios
2. More aggressive image registration attempts
3. Optimized matching parameters
4. Better handling of sequential frame data

Sample Input:
    image_dir: Path to directory with sequential frames (frame_000000.jpg, ...)
    output_dir: Where to save reconstruction

Expected Output:
    Higher registration rate (target: >20% vs current 4%)
    More 3D points with lower reprojection error
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import time

import pycolmap
from loguru import logger


@dataclass
class TunedReconstructionResult:
    """Results from tuned COLMAP reconstruction."""
    success: bool
    num_registered_images: int
    num_3d_points: int
    mean_reprojection_error: float
    execution_time: float
    sparse_path: Optional[Path]
    point_cloud_path: Optional[Path]
    reconstruction: Optional[pycolmap.Reconstruction] = None
    config_name: str = "default"  # Which parameter config was used


def run_colmap_tuned_reconstruction(
    image_dir: Path,
    output_dir: Path,
    camera_model: str = "SIMPLE_RADIAL",
    max_num_features: int = 8192,
    config: str = "aggressive",
    use_cache: bool = True,
) -> TunedReconstructionResult:
    """
    Run COLMAP with tuned parameters for better registration.

    Args:
        image_dir: Directory with input images
        output_dir: Where to save outputs
        camera_model: COLMAP camera model
        max_num_features: Max SIFT features per image
        config: Parameter configuration preset:
            - "aggressive": Very relaxed thresholds for difficult scenes
            - "moderate": Balanced between quality and registration
            - "conservative": Stricter quality, fewer registrations
        use_cache: Whether to use cached reconstruction

    Returns:
        TunedReconstructionResult with detailed statistics
    """
    start_time = time.time()

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    sparse_dir = output_dir / "sparse"
    point_cloud_path = output_dir / "point_cloud.ply"

    if use_cache and sparse_dir.exists() and point_cloud_path.exists():
        model_dirs = [d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()]

        if model_dirs:
            best_model_dir = max(model_dirs, key=lambda d: int(d.name))

            try:
                cached_recon = pycolmap.Reconstruction(str(best_model_dir))
                execution_time = time.time() - start_time

                logger.info(
                    f"✅ Using cached tuned reconstruction ({config}): {best_model_dir.name} "
                    f"({cached_recon.num_reg_images()} images, {cached_recon.num_points3D()} points)"
                )

                return TunedReconstructionResult(
                    success=True,
                    num_registered_images=cached_recon.num_reg_images(),
                    num_3d_points=cached_recon.num_points3D(),
                    mean_reprojection_error=cached_recon.compute_mean_reprojection_error(),
                    execution_time=execution_time,
                    sparse_path=best_model_dir,
                    point_cloud_path=point_cloud_path,
                    reconstruction=cached_recon,
                    config_name=config,
                )
            except Exception as e:
                logger.warning(f"Failed to load cached reconstruction: {e}. Re-running...")

    logger.info(f"Starting tuned COLMAP reconstruction ({config}): {image_dir.name}")

    # Setup paths
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Feature extraction with optimized parameters
        logger.info("Step 1/4: Extracting SIFT features (optimized)...")

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = max_num_features
        sift_options.first_octave = -1  # Higher resolution
        sift_options.num_octaves = 4
        sift_options.octave_resolution = 3
        sift_options.peak_threshold = 0.0066  # Default
        sift_options.edge_threshold = 10.0

        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.sift = sift_options

        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(image_dir),
            camera_mode=pycolmap.CameraMode.AUTO,
            camera_model=camera_model,
            extraction_options=extraction_options,
        )

        # Step 2: Feature matching with relaxed parameters
        logger.info("Step 2/4: Matching features (relaxed thresholds)...")

        # Create SIFT matching options
        sift_match_options = pycolmap.SiftMatchingOptions()

        if config == "aggressive":
            sift_match_options.max_ratio = 0.9  # More permissive (default: 0.8)
            sift_match_options.max_distance = 0.8  # More permissive (default: 0.7)
            sift_match_options.cross_check = True
        elif config == "moderate":
            sift_match_options.max_ratio = 0.85
            sift_match_options.max_distance = 0.75
            sift_match_options.cross_check = True
        else:  # conservative
            sift_match_options.max_ratio = 0.8
            sift_match_options.max_distance = 0.7
            sift_match_options.cross_check = True

        # Create FeatureMatchingOptions and set sift options
        match_options = pycolmap.FeatureMatchingOptions()
        match_options.sift = sift_match_options

        pycolmap.match_exhaustive(
            database_path=str(database_path),
            matching_options=match_options,
        )

        # Step 3: Incremental reconstruction with tuned mapper options
        logger.info("Step 3/4: Running incremental reconstruction (tuned params)...")

        # Create pipeline options and configure the mapper
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        mapper_options = pipeline_options.mapper

        if config == "aggressive":
            # Very relaxed thresholds for difficult scenes
            mapper_options.init_min_num_inliers = 30  # Default: 100 (way too high!)
            mapper_options.init_min_tri_angle = 8.0  # Default: 16.0
            mapper_options.init_max_reg_trials = 5  # Default: 2
            mapper_options.max_reg_trials = 5  # Default: 3
            mapper_options.filter_min_tri_angle = 1.0  # Default: 1.5
            mapper_options.filter_max_reproj_error = 6.0  # Default: 4.0
            mapper_options.abs_pose_min_num_inliers = 20  # Default: 30
            mapper_options.abs_pose_min_inlier_ratio = 0.15  # Default: 0.25

        elif config == "moderate":
            # Balanced approach
            mapper_options.init_min_num_inliers = 50
            mapper_options.init_min_num_inliers = 50
            mapper_options.init_min_tri_angle = 12.0
            mapper_options.init_max_reg_trials = 3
            mapper_options.max_reg_trials = 4
            mapper_options.filter_min_tri_angle = 1.2
            mapper_options.filter_max_reproj_error = 5.0
            mapper_options.abs_pose_min_num_inliers = 25
            mapper_options.abs_pose_min_inlier_ratio = 0.20

        else:  # conservative
            # Higher quality, fewer registrations
            mapper_options.init_min_num_inliers = 80
            mapper_options.init_min_tri_angle = 14.0
            mapper_options.init_max_reg_trials = 2
            mapper_options.max_reg_trials = 3
            mapper_options.filter_min_tri_angle = 1.5
            mapper_options.filter_max_reproj_error = 4.0

        # Common settings for all configs
        mapper_options.num_threads = -1  # Use all cores
        mapper_options.ba_local_num_images = 6
        mapper_options.ba_global_ignore_redundant_points3D = False

        # Pipeline-level settings
        pipeline_options.min_num_matches = 15  # Default
        pipeline_options.num_threads = -1

        reconstructions = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_dir),
            output_path=str(sparse_dir),
            options=pipeline_options,
        )

        if not reconstructions:
            logger.warning(f"Reconstruction failed with {config} config: no models generated")
            return TunedReconstructionResult(
                success=False,
                num_registered_images=0,
                num_3d_points=0,
                mean_reprojection_error=0.0,
                execution_time=time.time() - start_time,
                sparse_path=None,
                point_cloud_path=None,
                config_name=config,
            )

        # Get best reconstruction (most registered images)
        best_recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())

        logger.info(
            f"✅ Reconstruction complete: {best_recon.num_reg_images()} images, "
            f"{best_recon.num_points3D()} points"
        )

        # Step 4: Export point cloud
        logger.info("Step 4/4: Exporting point cloud...")
        best_recon.export_PLY(str(point_cloud_path))

        execution_time = time.time() - start_time

        # Get the path to the best model
        best_model_idx = max(
            (int(d.name) for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()),
            default=0
        )
        best_model_path = sparse_dir / str(best_model_idx)

        return TunedReconstructionResult(
            success=True,
            num_registered_images=best_recon.num_reg_images(),
            num_3d_points=best_recon.num_points3D(),
            mean_reprojection_error=best_recon.compute_mean_reprojection_error(),
            execution_time=execution_time,
            sparse_path=best_model_path,
            point_cloud_path=point_cloud_path,
            reconstruction=best_recon,
            config_name=config,
        )

    except Exception as e:
        logger.error(f"COLMAP reconstruction failed: {e}")
        import traceback
        traceback.print_exc()

        return TunedReconstructionResult(
            success=False,
            num_registered_images=0,
            num_3d_points=0,
            mean_reprojection_error=0.0,
            execution_time=time.time() - start_time,
            sparse_path=None,
            point_cloud_path=None,
            config_name=config,
        )


if __name__ == "__main__":
    """
    Validation: Test tuned reconstruction on visit 1 with different configs.

    Expected results:
    - Aggressive config should register more images than default
    - Should complete without errors
    - Should produce point cloud files
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    configs_to_test = ["aggressive", "moderate", "conservative"]
    results = {}

    for config in configs_to_test:
        total_tests += 1
        print(f"\n🧪 Test {total_tests}: Testing {config} configuration")

        try:
            image_dir = Path("results/combined_1_visits")
            output_dir = Path(f"results/tuned_reconstruction_1_{config}")

            if not image_dir.exists():
                all_validation_failures.append(
                    f"Test {total_tests}: Image directory not found: {image_dir}"
                )
                continue

            result = run_colmap_tuned_reconstruction(
                image_dir=image_dir,
                output_dir=output_dir,
                config=config,
                use_cache=False,  # Force re-run for testing
            )

            results[config] = result

            print(f"  Config: {config}")
            print(f"  Success: {result.success}")
            print(f"  Registered images: {result.num_registered_images}")
            print(f"  3D points: {result.num_3d_points}")
            print(f"  Mean reproj error: {result.mean_reprojection_error:.3f}px")
            print(f"  Execution time: {result.execution_time:.2f}s")

            if not result.success:
                all_validation_failures.append(
                    f"Test {total_tests}: {config} reconstruction failed"
                )

        except Exception as e:
            all_validation_failures.append(
                f"Test {total_tests}: Exception with {config}: {type(e).__name__}: {e}"
            )

    # Compare results
    if len(results) == 3:
        total_tests += 1
        print(f"\n🧪 Test {total_tests}: Comparing configurations")
        print("\n| Config | Registered Images | 3D Points | Reproj Error |")
        print("|--------|-------------------|-----------|--------------|")
        for config in configs_to_test:
            r = results[config]
            print(f"| {config:12} | {r.num_registered_images:17} | {r.num_3d_points:9} | {r.mean_reprojection_error:12.3f} |")

        # Aggressive should register more images
        if results["aggressive"].num_registered_images <= results["conservative"].num_registered_images:
            all_validation_failures.append(
                f"Test {total_tests}: Expected aggressive config to register more images than conservative"
            )

    # Final validation result
    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Tuned COLMAP runner is validated and ready for use")
        sys.exit(0)
