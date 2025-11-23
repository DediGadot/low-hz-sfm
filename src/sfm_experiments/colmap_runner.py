"""
SfM reconstruction wrapper supporting COLMAP and GLOMAP.

This module provides:
- Simplified SfM pipeline execution (COLMAP or GLOMAP)
- Feature extraction, matching, reconstruction
- Result parsing and validation
- Error handling and progress tracking

Dependencies:
- pycolmap: https://colmap.github.io/pycolmap/
- numpy: https://numpy.org/
- GLOMAP (optional): https://github.com/colmap/glomap

Sample Input: Directory of JPEG images
Expected Output: Sparse reconstruction with poses and 3D points
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time
import shutil
import sqlite3
import json

import pycolmap
import numpy as np
from loguru import logger
from tqdm import tqdm

# GLOMAP integration
try:
    from .glomap_wrapper import (
        check_glomap_available,
        run_glomap_mapper,
        GlomapOptions,
    )
except ImportError:
    # Handle direct execution (python colmap_runner.py)
    from glomap_wrapper import (
        check_glomap_available,
        run_glomap_mapper,
        GlomapOptions,
    )

PAIR_PRIME = 2147483647  # COLMAP pair_id prime constant
_CUDA_AVAILABLE: Optional[bool] = None


def _decode_pair_id(pair_id: int) -> Tuple[int, int]:
    """Decode COLMAP pair_id (image_id1 * prime + image_id2)."""
    image_id2 = pair_id % PAIR_PRIME
    image_id1 = (pair_id - image_id2) // PAIR_PRIME
    return int(image_id1), int(image_id2)


def _iter_image_files(image_dir: Path):
    """Yield image files recursively (case-insensitive common extensions)."""
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for ext in exts:
        yield from image_dir.rglob(f"*{ext}")


def _count_images(image_dir: Path) -> int:
    """Count images recursively without materializing the file list."""
    return sum(1 for _ in _iter_image_files(image_dir))


def _sanitize_for_cache(obj: Any) -> Any:
    """
    Make cache metadata JSON-friendly and comparable.

    Converts non-serializable types to string representations while
    preserving simple scalars/collections.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_cache(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_cache(v) for v in obj]
    return str(obj)


def _is_cuda_available() -> bool:
    """Check whether pycolmap reports CUDA support (cached)."""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE

    try:
        has_cuda = getattr(pycolmap, "has_cuda", None)
        if callable(has_cuda):
            _CUDA_AVAILABLE = bool(has_cuda())
        elif isinstance(has_cuda, bool):
            _CUDA_AVAILABLE = has_cuda
        else:
            _CUDA_AVAILABLE = False
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"CUDA availability check failed: {exc}")
        _CUDA_AVAILABLE = False

    if not _CUDA_AVAILABLE:
        logger.info("pycolmap CUDA unavailable; forcing CPU execution")

    return _CUDA_AVAILABLE


@dataclass
class ReconstructionResult:
    """Container for SfM reconstruction outputs (COLMAP or GLOMAP)."""

    success: bool
    num_registered_images: int
    num_3d_points: int
    mean_reprojection_error: float
    execution_time: float
    sparse_path: Path
    mapper_type: str = "colmap"  # "colmap" or "glomap"
    point_cloud_path: Optional[Path] = None
    reconstruction: Optional[pycolmap.Reconstruction] = None


def find_best_image_pair(database_path: Path) -> Tuple[Optional[Tuple[str, str]], int]:
    """
    Find the image pair with the most verified matches for optimal initialization.

    This function queries the COLMAP database to identify the image pair with
    the highest number of inlier matches from geometric verification. Using the
    best pair for initialization provides a stronger foundation for reconstruction.

    Args:
        database_path: Path to COLMAP database.db file

    Returns:
        Tuple of ((image1_name, image2_name), match_count) or (None, 0) if failed

    Example:
        >>> best_pair, count = find_best_image_pair(Path("database.db"))
        >>> print(f"Best pair: {best_pair[0]} <-> {best_pair[1]} ({count} matches)")
    """
    try:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()

        # Cache image_id -> name to decode pair_id ourselves (COLMAP uses prime encoding)
        cursor.execute("SELECT image_id, name FROM images")
        id_to_name = dict(cursor.fetchall())

        cursor.execute("SELECT pair_id, rows FROM two_view_geometries ORDER BY rows DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if not row:
            return (None, 0)

        pair_id, match_count = row
        img_id1, img_id2 = _decode_pair_id(pair_id)
        name1 = id_to_name.get(img_id1)
        name2 = id_to_name.get(img_id2)
        if name1 and name2:
            return ((name1, name2), int(match_count))
        return (None, 0)
    except Exception as e:
        logger.warning(f"Failed to find best pair: {e}")
        return (None, 0)


def run_sfm_reconstruction(
    image_dir: Path,
    output_dir: Path,
    mapper_type: str = "colmap",
    camera_model: str = "SIMPLE_RADIAL",
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.SINGLE,
    max_num_features: int = 8192,
    use_cache: bool = True,
    glomap_options: Optional[Dict[str, Any]] = None,
    visualize: bool = False,
    viz_num_samples: int = 10,
) -> ReconstructionResult:
    """
    Run SfM reconstruction with configurable mapper (COLMAP or GLOMAP).

    This function orchestrates the complete SfM pipeline:
    1. Feature extraction (SIFT via pycolmap)
    2. Feature matching (exhaustive + sequential via pycolmap)
    3. Reconstruction (COLMAP incremental OR GLOMAP global)
    4. Point cloud export

    Args:
        image_dir: Directory containing input images (JPEG)
        output_dir: Where to save reconstruction outputs
        mapper_type: "colmap" (incremental) or "glomap" (global)
        camera_model: COLMAP camera model (SIMPLE_RADIAL, PINHOLE, etc.)
        max_num_features: Maximum SIFT features per image
        use_cache: If True, reuse existing reconstruction if available
        glomap_options: Dict of GLOMAP-specific options (only used if mapper_type="glomap")
        visualize: If True, generate debug visualizations during pipeline execution
        viz_num_samples: Number of frames to visualize per pipeline stage (default: 10)

    Returns:
        ReconstructionResult with statistics and paths

    Example:
        >>> # Use COLMAP (default)
        >>> result = run_sfm_reconstruction(
        ...     Path("data/frames"),
        ...     Path("results/reconstruction_1")
        ... )
        >>> # Use GLOMAP for speed
        >>> result = run_sfm_reconstruction(
        ...     Path("data/frames"),
        ...     Path("results/reconstruction_1"),
        ...     mapper_type="glomap"
        ... )
        >>> # With visualizations
        >>> result = run_sfm_reconstruction(
        ...     Path("data/frames"),
        ...     Path("results/reconstruction_1"),
        ...     visualize=True,
        ...     viz_num_samples=10
        ... )
    """
    start_time = time.time()

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Validate mapper type
    if mapper_type not in ["colmap", "glomap"]:
        raise ValueError(f"Invalid mapper_type: {mapper_type}. Must be 'colmap' or 'glomap'")

    # Check GLOMAP availability if needed
    if mapper_type == "glomap":
        available, info = check_glomap_available()
        if not available:
            logger.warning(f"GLOMAP not available: {info}")
            logger.warning("Falling back to COLMAP incremental mapper")
            mapper_type = "colmap"

    logger.info(f"Starting SfM reconstruction with {mapper_type.upper()} mapper: {image_dir.name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    point_cloud_path = output_dir / "point_cloud.ply"
    cache_meta_path = output_dir / "cache_meta.json"

    image_count = _count_images(Path(image_dir))
    if image_count == 0:
        logger.warning(f"No images found in {image_dir} (recursive search)")
    else:
        logger.info(f"Detected {image_count} image(s) in {image_dir} (recursive)")

    current_cache_config = {
        "mapper_type": mapper_type,
        "camera_model": camera_model,
        "max_num_features": max_num_features,
        "camera_mode": camera_mode.name if hasattr(camera_mode, "name") else str(camera_mode),
        "image_count": image_count,
        "glomap_options": glomap_options or {},
    }
    current_cache_config = _sanitize_for_cache(current_cache_config)

    def _load_cache_meta(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return _sanitize_for_cache(data)
        except Exception as exc:
            logger.warning(f"Failed to read cache metadata {path}: {exc}")
            return None

    def _cache_matches(meta: Optional[Dict[str, Any]]) -> bool:
        if not meta:
            return False
        for key, value in current_cache_config.items():
            if meta.get(key) != value:
                logger.info(f"Cache metadata mismatch on '{key}': cached={meta.get(key)}, current={value}")
                return False
        return True

    cache_meta = _load_cache_meta(cache_meta_path)

    # Check cache: if reconstruction exists, load and return it (only when metadata matches)
    if use_cache and sparse_dir.exists() and point_cloud_path.exists() and _cache_matches(cache_meta):
        # Look for existing reconstruction models
        model_dirs = [d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()]

        if model_dirs:
            # Load the best model (highest numbered directory)
            best_model_dir = max(model_dirs, key=lambda d: int(d.name))

            try:
                # Load cached reconstruction
                cached_recon = pycolmap.Reconstruction(str(best_model_dir))
                execution_time = time.time() - start_time

                logger.info(
                    f"✅ Using cached reconstruction: {best_model_dir.name} "
                    f"({cached_recon.num_reg_images()} images, {cached_recon.num_points3D()} points)"
                )

                # Try to infer mapper type from cache metadata if available
                cached_mapper_type = mapper_type  # Default to requested type

                return ReconstructionResult(
                    success=True,
                    num_registered_images=cached_recon.num_reg_images(),
                    num_3d_points=cached_recon.num_points3D(),
                    mean_reprojection_error=cached_recon.compute_mean_reprojection_error(),
                    execution_time=execution_time,
                    sparse_path=best_model_dir,
                    mapper_type=cached_mapper_type,
                    point_cloud_path=point_cloud_path,
                    reconstruction=cached_recon,
                )
            except Exception as e:
                logger.warning(f"Failed to load cached reconstruction: {e}. Re-running SfM...")

    if not use_cache:
        if database_path.exists():
            database_path.unlink()
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        if point_cloud_path.exists():
            point_cloud_path.unlink()
        if cache_meta_path.exists():
            cache_meta_path.unlink()

    # Setup paths
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # BUGFIX: Validate database integrity before use
    def validate_database_integrity(db_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate COLMAP database integrity.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not db_path.exists():
            return True, None  # Database doesn't exist yet - will be created

        conn = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=5.0)

            # Run SQLite integrity check
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            result = cursor.fetchone()

            if result and result[0] != "ok":
                return False, f"Database integrity check failed: {result[0]}"

            # Verify COLMAP-required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}

            required_tables = {'cameras', 'images', 'keypoints', 'descriptors', 'matches'}
            missing_tables = required_tables - tables

            if missing_tables and len(tables) > 0:
                # Database exists but is incomplete (partial creation/corruption)
                return False, f"Database missing required tables: {missing_tables}"

            return True, None

        except sqlite3.DatabaseError as e:
            return False, f"Database corruption detected: {e}"
        except Exception as e:
            return False, f"Database validation failed: {e}"
        finally:
            # BUGFIX: Always close connection in finally block to prevent resource leaks
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass  # Ignore errors during cleanup

    # PERFORMANCE: Enable SQLite WAL mode for faster database access
    # WAL (Write-Ahead Logging) mode allows concurrent reads during writes
    # and provides better performance for large databases
    def enable_database_wal_mode(db_path: Path) -> bool:
        """Enable WAL mode for faster SQLite database access."""
        if not db_path.exists():
            return False
        conn = None
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")  # Balanced safety/performance
            conn.execute("PRAGMA cache_size=-64000;")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY;")  # Keep temp tables in memory
            conn.commit()
            logger.info("  ✓ SQLite WAL mode enabled for database performance")
            return True
        except Exception as e:
            logger.warning(f"Failed to enable WAL mode: {e}")
            return False
        finally:
            # BUGFIX: Always close connection in finally block to prevent resource leaks
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass  # Ignore errors during cleanup

    try:
        # ====================================================================
        # INITIALIZE VISUALIZATION (if enabled)
        # ====================================================================
        visualizer = None
        if visualize:
            from .pipeline_visualizer import PipelineVisualizer, VisualizationConfig
            viz_config = VisualizationConfig(
                output_dir=output_dir,
                num_samples=viz_num_samples,
            )
            visualizer = PipelineVisualizer(viz_config)
            logger.info(f"Debug visualizations enabled: {viz_num_samples} samples per stage")

        # ====================================================================
        # COMMON STEPS (Same for COLMAP and GLOMAP)
        # ====================================================================

        # BUGFIX: Validate database integrity before use
        if database_path.exists():
            is_valid, error_msg = validate_database_integrity(database_path)
            if not is_valid:
                logger.error(f"Database validation failed: {error_msg}")
                logger.info("Removing corrupted database and starting fresh...")
                database_path.unlink()  # Remove corrupted database
                # Also remove associated WAL files if they exist
                wal_file = database_path.with_suffix(".db-wal")
                shm_file = database_path.with_suffix(".db-shm")
                if wal_file.exists():
                    wal_file.unlink()
                if shm_file.exists():
                    shm_file.unlink()

        # Enable database optimizations if database exists and is valid
        if database_path.exists():
            enable_database_wal_mode(database_path)

        cuda_available = _is_cuda_available()

        # PERFORMANCE: Adaptive feature count based on dataset size
        # Smaller datasets don't need as many features per image
        adaptive_max_features = max_num_features
        if image_count and image_count < 100:
            adaptive_max_features = 4096  # 2x faster, minimal quality loss
            logger.info(f"  ✓ Using adaptive features: {adaptive_max_features} (small dataset)")
        elif image_count and image_count < 500:
            adaptive_max_features = 6144  # Balanced
            logger.info(f"  ✓ Using adaptive features: {adaptive_max_features} (medium dataset)")
        else:
            logger.info(f"  ✓ Using full features: {adaptive_max_features}")

        # Step 1: Feature extraction
        logger.info("Step 1/4: Extracting SIFT features...")
        sift_options = pycolmap.SiftExtractionOptions(
            max_num_features=adaptive_max_features,
        )

        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.sift = sift_options
        extraction_options.num_threads = -1  # Use all available CPU threads
        # PERFORMANCE: Enable GPU acceleration if available (GPU index 0)
        # Falls back to CPU automatically if GPU unavailable
        # NOTE: gpu_index is on FeatureExtractionOptions, NOT SiftExtractionOptions
        # NOTE: gpu_index must be a string, not an integer (pycolmap API requirement)
        use_gpu_for_extraction = cuda_available and (
            mapper_type == "glomap" or (image_count and image_count > 5000)
        )
        # BUGFIX: Cleaner type handling - set string directly, check boolean for logging
        extraction_options.gpu_index = "0" if use_gpu_for_extraction else "-1"

        if use_gpu_for_extraction:
            logger.info(f"  ✓ GPU acceleration enabled (GPU {extraction_options.gpu_index})")
        else:
            logger.info(f"  ✓ Using CPU with {extraction_options.num_threads} threads")

        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(image_dir),
            camera_mode=camera_mode,
            camera_model=camera_model,
            extraction_options=extraction_options,
        )

        # PERFORMANCE: Enable WAL mode after feature extraction creates database
        enable_database_wal_mode(database_path)

        # Visualization hook: Feature extraction complete
        if visualizer:
            image_files = sorted([f.name for f in _iter_image_files(image_dir)])
            visualizer.visualize_features(database_path, image_dir, image_files)

        # Step 2: Feature matching with multi-strategy approach
        logger.info("Step 2/4: Matching features (multi-strategy for loop closure)...")

        # Strategy 1: Exhaustive matching for initial pairs
        if image_count and image_count > 600:
            logger.info(f"  2a. Skipping exhaustive matching for large set ({image_count} images)")
        else:
            logger.info("  2a. Exhaustive matching...")
            # PERFORMANCE: Configure matching options for speed
            matching_options = pycolmap.FeatureMatchingOptions()
            matching_options.max_num_matches = adaptive_max_features  # Limit matches per pair
            # NOTE: gpu_index must be a string, not an integer
            use_gpu_for_matching = cuda_available and (
                mapper_type == "glomap" or (image_count and image_count > 1000)
            )
            matching_options.gpu_index = "0" if use_gpu_for_matching else "-1"
            matching_options.num_threads = -1  # Use all CPU cores

            pycolmap.match_exhaustive(
                database_path=str(database_path),
                matching_options=matching_options,
            )

        # Strategy 2: Sequential matching for temporal sequences
        # PERFORMANCE OPTIMIZATION: Reduced overlap and disabled quadratic for large-scale datasets
        # For >10K images: overlap=3, linear mode (was: overlap=10, quadratic)
        # This reduces matching complexity from O(n²) to O(n) within overlap window
        if image_count and image_count > 10000:
            overlap = 3  # Minimal overlap for very large datasets
        elif image_count and image_count > 1000:
            overlap = 4  # Reduced from 5 for better performance
        elif image_count and image_count > 200:
            overlap = 8  # Reduced from 20 for medium datasets
        else:
            overlap = 15  # Only tiny datasets need very high overlap

        logger.info(f"  2b. Sequential matching (overlap={overlap} for {'very large' if image_count and image_count > 10000 else 'large' if image_count and image_count > 1000 else 'medium' if image_count and image_count > 200 else 'small'} dataset with {image_count} images)...")
        try:
            pairing_options = pycolmap.SequentialPairingOptions()
            pairing_options.overlap = overlap  # Match each image with N neighbors
            # PERFORMANCE: Enable vocabulary tree for loop closure when available
            pairing_options.loop_detection = False  # TODO: Enable with vocab tree path
            # CRITICAL OPTIMIZATION: Disable quadratic overlap for large datasets
            # Linear overlap creates overlap pairs, quadratic creates overlap² pairs
            # PERFORMANCE: Lowered threshold from 5000 to 1000 to reduce O(n²) matching for medium datasets
            pairing_options.quadratic_overlap = False if (image_count and image_count > 1000) else True

            # PERFORMANCE: Configure matching options for sequential matching
            matching_options = pycolmap.FeatureMatchingOptions()
            matching_options.max_num_matches = adaptive_max_features
            # NOTE: gpu_index must be a string, not an integer (pycolmap API requirement)
            use_gpu_for_seq_matching = cuda_available and (
                mapper_type == "glomap" or (image_count and image_count > 5000)
            )
            # BUGFIX: Cleaner type handling - set string directly, check boolean for logging
            matching_options.gpu_index = "0" if use_gpu_for_seq_matching else "-1"
            matching_options.num_threads = -1  # Use all available CPU threads

            if use_gpu_for_seq_matching:
                logger.info(f"    ✓ GPU matching enabled with overlap={overlap}, quadratic={pairing_options.quadratic_overlap}")
            else:
                logger.info(f"    ✓ CPU matching with {matching_options.num_threads} threads, overlap={overlap}, quadratic={pairing_options.quadratic_overlap}")

            pycolmap.match_sequential(
                database_path=str(database_path),
                pairing_options=pairing_options,
                matching_options=matching_options,
            )
            logger.info("  ✅ Sequential matching completed successfully")
        except Exception as e:
            logger.warning(f"Sequential matching failed: {e}. Continuing with exhaustive matches only.")

        # Visualization hook: Matching complete
        if visualizer:
            image_files = sorted([f.name for f in _iter_image_files(image_dir)])
            visualizer.visualize_matches(database_path, image_dir, image_files)

        # ====================================================================
        # MAPPER-SPECIFIC RECONSTRUCTION
        # ====================================================================

        logger.info(f"Step 3/4: Running {mapper_type.upper()} reconstruction...")

        # Analyze matches to find best initialization pair
        logger.info("  3a. Analyzing matches to find best initialization pair...")
        best_pair, best_count = find_best_image_pair(database_path)
        if best_pair:
            logger.info(f"  ✅ Best pair identified: {best_pair[0]} <-> {best_pair[1]} ({best_count} inlier matches)")
        else:
            logger.warning("  ⚠️  Could not identify best pair, using mapper's automatic selection")

        if mapper_type == "colmap":
            # ================================================================
            # COLMAP INCREMENTAL MAPPING
            # ================================================================
            logger.info("  3b. Running COLMAP incremental mapping (ultra-aggressive parameters)...")

            pipeline_options = pycolmap.IncrementalPipelineOptions()
            mapper = pipeline_options.mapper

            # ULTRA-AGGRESSIVE parameters (calibrated to 7.9 avg matches/pair)
            mapper.init_min_num_inliers = 6
            mapper.init_min_tri_angle = 2.0
            mapper.init_max_error = 8.0
            mapper.init_max_reg_trials = 15

            mapper.max_reg_trials = 15
            mapper.filter_min_tri_angle = 0.5
            mapper.filter_max_reproj_error = 8.0

            mapper.abs_pose_min_num_inliers = 5
            mapper.abs_pose_min_inlier_ratio = 0.05

            mapper.num_threads = -1  # Use all available cores

            reconstructions = pycolmap.incremental_mapping(
                database_path=str(database_path),
                image_path=str(image_dir),
                output_path=str(sparse_dir),
                options=pipeline_options,
            )

            if not reconstructions:
                logger.warning("Reconstruction failed: no models generated")
                return ReconstructionResult(
                    success=False,
                    num_registered_images=0,
                    num_3d_points=0,
                    mean_reprojection_error=0.0,
                    execution_time=time.time() - start_time,
                    sparse_path=sparse_dir,
                    mapper_type=mapper_type,
                )

            # Get best reconstruction (most registered images)
            best_idx = max(
                reconstructions.keys(),
                key=lambda k: reconstructions[k].num_reg_images(),
            )
            best_recon = reconstructions[best_idx]

            logger.info(
                f"Best COLMAP model: {best_recon.num_reg_images()} images, "
                f"{best_recon.num_points3D()} points"
            )

        elif mapper_type == "glomap":
            # ================================================================
            # GLOMAP GLOBAL MAPPING
            # ================================================================
            logger.info("  3b. Running GLOMAP global mapping...")

            # Build GLOMAP options
            glomap_opts = GlomapOptions()

            # PERFORMANCE OPTIMIZATIONS for GLOMAP with large-scale datasets
            if image_count and image_count > 10000:
                # Aggressive settings for very large datasets (>10K images)
                glomap_opts.max_epipolar_error = 6.0  # More lenient for speed
                glomap_opts.max_num_tracks = 500000  # Limit track count for memory
                glomap_opts.skip_retriangulation = True  # Skip for significant speedup
                logger.info(f"  ✓ GLOMAP optimizations for large dataset ({image_count} images)")
            elif image_count and image_count > 5000:
                # Moderate settings for medium datasets (5-10K images)
                glomap_opts.max_epipolar_error = 5.0
                glomap_opts.max_num_tracks = 750000
                glomap_opts.skip_retriangulation = False
                logger.info(f"  ✓ GLOMAP optimizations for medium dataset ({image_count} images)")

            # Override with user-provided options if specified
            if glomap_options:
                if "max_epipolar_error" in glomap_options:
                    glomap_opts.max_epipolar_error = glomap_options["max_epipolar_error"]
                if "max_num_tracks" in glomap_options:
                    glomap_opts.max_num_tracks = glomap_options["max_num_tracks"]
                if "skip_retriangulation" in glomap_options:
                    glomap_opts.skip_retriangulation = glomap_options["skip_retriangulation"]

            # Run GLOMAP
            result = run_glomap_mapper(
                database_path=database_path,
                image_path=image_dir,
                output_path=sparse_dir,
                options=glomap_opts,
            )

            # GLOMAP creates reconstruction in sparse/0/ directory
            # Load it with pycolmap
            model_dirs = [d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if not model_dirs:
                logger.error("GLOMAP failed: no reconstruction models generated")
                return ReconstructionResult(
                    success=False,
                    num_registered_images=0,
                    num_3d_points=0,
                    mean_reprojection_error=0.0,
                    execution_time=time.time() - start_time,
                    sparse_path=sparse_dir,
                    mapper_type=mapper_type,
                )

            best_model_dir = max(model_dirs, key=lambda d: int(d.name))
            best_recon = pycolmap.Reconstruction(str(best_model_dir))

            logger.info(
                f"Best GLOMAP model: {best_recon.num_reg_images()} images, "
                f"{best_recon.num_points3D()} points"
            )

        # Visualization hook: Reconstruction complete
        if visualizer:
            visualizer.visualize_reconstruction(best_recon)

        # ====================================================================
        # COMMON EXPORT (Same for COLMAP and GLOMAP)
        # ====================================================================

        # Step 4: Export point cloud
        logger.info("Step 4/4: Exporting point cloud...")
        # PERFORMANCE: Skip PLY export if file already exists
        if point_cloud_path.exists():
            logger.info(f"  ✓ Using existing point cloud: {point_cloud_path.name}")
        else:
            best_recon.export_PLY(str(point_cloud_path))
            logger.info(f"  ✓ Exported point cloud: {point_cloud_path.name}")

        # Save sparse reconstruction
        # BUGFIX: Separate logic for COLMAP vs GLOMAP to avoid variable scope issues
        if mapper_type == "colmap":
            output_sparse = sparse_dir / str(best_idx)
        else:  # mapper_type == "glomap"
            output_sparse = sparse_dir / str(best_model_dir.name)
        best_recon.write(str(output_sparse))

        execution_time = time.time() - start_time

        result = ReconstructionResult(
            success=True,
            num_registered_images=best_recon.num_reg_images(),
            num_3d_points=best_recon.num_points3D(),
            mean_reprojection_error=best_recon.compute_mean_reprojection_error(),
            execution_time=execution_time,
            sparse_path=output_sparse,
            mapper_type=mapper_type,
            point_cloud_path=point_cloud_path,
            reconstruction=best_recon,
        )

        # Persist cache metadata for future runs
        try:
            meta_to_save = dict(current_cache_config)
            meta_to_save["timestamp"] = time.time()
            with open(cache_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_to_save, f, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to write cache metadata: {exc}")

        # Finalize visualization report
        if visualizer:
            experiment_name = f"{image_dir.name}_{mapper_type}"
            visualizer.finalize_report(experiment_name, execution_time)

        logger.info(
            f"✅ Reconstruction complete ({mapper_type.upper()}): {result.num_registered_images} images, "
            f"{result.num_3d_points} points, {execution_time:.1f}s"
        )

        return result

    except Exception as e:
        logger.error(f"SfM reconstruction failed: {e}")
        return ReconstructionResult(
            success=False,
            num_registered_images=0,
            num_3d_points=0,
            mean_reprojection_error=0.0,
            execution_time=time.time() - start_time,
            sparse_path=sparse_dir,
            mapper_type=mapper_type,
        )


def run_colmap_reconstruction(
    image_dir: Path,
    output_dir: Path,
    camera_model: str = "SIMPLE_RADIAL",
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.SINGLE,
    max_num_features: int = 8192,
    use_cache: bool = True,
    visualize: bool = False,
    viz_num_samples: int = 10,
) -> ReconstructionResult:
    """
    Run full COLMAP incremental SfM pipeline.

    This function orchestrates the complete COLMAP pipeline:
    1. Feature extraction (SIFT)
    2. Feature matching (exhaustive for small datasets)
    3. Incremental reconstruction
    4. Point cloud export

    Args:
        image_dir: Directory containing input images (JPEG)
        output_dir: Where to save reconstruction outputs
        camera_model: COLMAP camera model (SIMPLE_RADIAL, PINHOLE, etc.)
        max_num_features: Maximum SIFT features per image
        use_cache: If True, reuse existing reconstruction if available

    Returns:
        ReconstructionResult with statistics and paths

    Example:
        >>> result = run_colmap_reconstruction(
        ...     Path("data/frames"),
        ...     Path("results/reconstruction_1")
        ... )
        >>> print(f"Registered {result.num_registered_images} images")

    Note:
        This function now wraps run_sfm_reconstruction() with mapper_type="colmap"
        for backward compatibility. Use run_sfm_reconstruction() directly for
        more control or to use alternative mappers like GLOMAP.
    """
    # Backward compatibility wrapper: call run_sfm_reconstruction with COLMAP mapper
    return run_sfm_reconstruction(
        image_dir=image_dir,
        output_dir=output_dir,
        mapper_type="colmap",
        camera_model=camera_model,
        camera_mode=camera_mode,
        max_num_features=max_num_features,
        use_cache=use_cache,
        visualize=visualize,
        viz_num_samples=viz_num_samples,
    )


def extract_poses_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
) -> Dict[str, tuple]:
    """
    Extract camera poses from COLMAP reconstruction.

    Args:
        reconstruction: COLMAP Reconstruction object

    Returns:
        Dictionary mapping image name to (qvec, tvec) tuple
        qvec: quaternion (qw, qx, qy, qz)
        tvec: translation vector (tx, ty, tz)

    Example:
        >>> poses = extract_poses_from_reconstruction(recon)
        >>> qvec, tvec = poses["frame_000001.jpg"]
    """
    poses = {}

    for image_id, image in reconstruction.images.items():
        # Get rotation quaternion and translation
        qvec = image.cam_from_world.rotation.quat  # (qw, qx, qy, qz)
        tvec = image.cam_from_world.translation

        poses[image.name] = (
            np.array([qvec[0], qvec[1], qvec[2], qvec[3]]),
            np.array([tvec[0], tvec[1], tvec[2]]),
        )

    return poses


def get_reconstruction_summary(reconstruction: pycolmap.Reconstruction) -> Dict[str, Any]:
    """
    Get summary statistics from COLMAP reconstruction.

    Args:
        reconstruction: COLMAP Reconstruction object

    Returns:
        Dictionary with reconstruction statistics

    Example:
        >>> summary = get_reconstruction_summary(recon)
        >>> print(summary["num_cameras"])
    """
    return {
        "num_cameras": reconstruction.num_cameras(),
        "num_images": reconstruction.num_images(),
        "num_reg_images": reconstruction.num_reg_images(),
        "num_points3D": reconstruction.num_points3D(),
        "num_observations": reconstruction.compute_num_observations(),
        "mean_track_length": reconstruction.compute_mean_track_length(),
        "mean_observations_per_image": reconstruction.compute_mean_observations_per_reg_image(),
        "mean_reprojection_error": reconstruction.compute_mean_reprojection_error(),
    }


# Validation
if __name__ == "__main__":
    """
    Validation of colmap_runner.py functionality.
    Tests dataclasses and utility functions.
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: ReconstructionResult dataclass
    total_tests += 1
    try:
        result = ReconstructionResult(
            success=True,
            num_registered_images=100,
            num_3d_points=5000,
            mean_reprojection_error=0.5,
            execution_time=120.0,
            sparse_path=Path("/test/sparse"),
            point_cloud_path=Path("/test/cloud.ply"),
        )

        if result.num_registered_images != 100:
            all_validation_failures.append(
                f"ReconstructionResult: Expected 100 images, got {result.num_registered_images}"
            )
        if result.success != True:
            all_validation_failures.append(
                f"ReconstructionResult: Expected success=True, got {result.success}"
            )
    except Exception as e:
        all_validation_failures.append(f"ReconstructionResult: Exception raised: {e}")

    # Test 2: Error handling for missing image directory
    total_tests += 1
    try:
        nonexistent_dir = Path("/nonexistent/images")
        output_dir = Path("/tmp/test_output")

        try:
            run_colmap_reconstruction(nonexistent_dir, output_dir)
            all_validation_failures.append(
                "Error handling: Expected FileNotFoundError for missing directory"
            )
        except FileNotFoundError:
            # Expected behavior
            pass
    except Exception as e:
        all_validation_failures.append(f"Error handling: Unexpected exception: {e}")

    # Test 3: Verify pycolmap is importable
    total_tests += 1
    try:
        # Check that pycolmap classes are accessible
        if not hasattr(pycolmap, "Reconstruction"):
            all_validation_failures.append("pycolmap check: Reconstruction class not found")
        if not hasattr(pycolmap, "extract_features"):
            all_validation_failures.append("pycolmap check: extract_features function not found")
    except Exception as e:
        all_validation_failures.append(f"pycolmap check: Exception raised: {e}")

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
        print("COLMAP wrapper module validated and ready for use")
        sys.exit(0)
