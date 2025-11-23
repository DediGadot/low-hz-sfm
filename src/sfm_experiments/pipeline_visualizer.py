"""
Pipeline Visualizer - Orchestrates debug visualization generation for SfM pipeline.

This module provides the main PipelineVisualizer class that coordinates visualization
generation at each stage of the Structure from Motion pipeline. It creates visual
diagnostics for debugging feature extraction, matching, and reconstruction quality.

Key Features:
- Progressive visualization: generates outputs after each pipeline stage
- Intelligent sampling: selects representative frames distributed across dataset
- Debug-focused: highlights quality issues and errors
- HTML report generation: creates comprehensive summary report

Documentation:
- OpenCV: https://docs.opencv.org/4.x/
- Matplotlib: https://matplotlib.org/stable/contents.html
- Pycolmap: https://github.com/colmap/pycolmap

Sample Input:
- Database path: Path to COLMAP database.db
- Image directory: Directory containing input frames
- Reconstruction: pycolmap.Reconstruction object

Expected Output:
- Feature visualizations (PNG images)
- Match visualizations (PNG images)
- Reconstruction visualizations (HTML + PNG)
- HTML debug report with all visualizations
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import pycolmap
from loguru import logger


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation.

    Attributes:
        output_dir: Base directory for experiment outputs
        num_samples: Number of frames to visualize per stage (default: 10)
        max_matches_display: Maximum number of matches to display in visualizations (default: 200)
        image_scale: Scale factor for images in visualizations (default: 0.5 = 50%)
        enable_features: Whether to generate feature visualizations (default: True)
        enable_matching: Whether to generate matching visualizations (default: True)
        enable_reconstruction: Whether to generate reconstruction visualizations (default: True)
    """
    output_dir: Path
    num_samples: int = 10
    max_matches_display: int = 200
    image_scale: float = 0.5
    enable_features: bool = True
    enable_matching: bool = True
    enable_reconstruction: bool = True


class PipelineVisualizer:
    """Orchestrates visualization generation for SfM pipeline.

    This class manages the creation of debug visualizations at each stage of the
    Structure from Motion pipeline. It coordinates feature, matching, and reconstruction
    visualizations and generates a progressive HTML report.

    The visualizer operates incrementally - generating and saving visualizations
    immediately after each pipeline stage completes. This ensures debug information
    is available even if the pipeline fails mid-execution.

    Example usage:
        >>> config = VisualizationConfig(output_dir=Path("results/experiment"))
        >>> visualizer = PipelineVisualizer(config)
        >>> visualizer.visualize_features(db_path, img_dir, image_list)
        >>> visualizer.visualize_matches(db_path, img_dir, image_list)
        >>> visualizer.visualize_reconstruction(reconstruction)
        >>> report_path = visualizer.finalize_report("my_experiment", 120.5)
    """

    def __init__(self, config: VisualizationConfig):
        """Initialize the pipeline visualizer.

        Args:
            config: VisualizationConfig object with settings
        """
        self.config = config
        self.viz_dir = config.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each stage
        (self.viz_dir / "features").mkdir(exist_ok=True)
        (self.viz_dir / "matching").mkdir(exist_ok=True)
        (self.viz_dir / "reconstruction").mkdir(exist_ok=True)

        # Initialize report generator (will be imported when needed)
        self.report = None
        self._init_report()

        logger.info(f"Initialized PipelineVisualizer: output_dir={self.viz_dir}")
        logger.info(f"Settings: num_samples={config.num_samples}, "
                   f"max_matches={config.max_matches_display}, "
                   f"image_scale={config.image_scale}")

    def _init_report(self) -> None:
        """Initialize the HTML report generator."""
        try:
            from .debug_report_generator import DebugReportGenerator
            self.report = DebugReportGenerator(self.viz_dir)
        except ImportError:
            logger.warning("DebugReportGenerator not available, HTML report will not be generated")
            self.report = None

    def get_sample_indices(self, total_frames: int) -> List[int]:
        """Get deterministic sample indices distributed across dataset.

        Computes a fixed set of frame indices that are evenly distributed across
        the entire dataset. Always includes the first and last frame to capture
        temporal endpoints. For datasets smaller than num_samples, returns all indices.

        Args:
            total_frames: Total number of frames in the dataset

        Returns:
            List of frame indices to visualize (sorted, unique)

        Example:
            >>> visualizer.get_sample_indices(100)  # num_samples=10
            [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
        """
        if total_frames <= self.config.num_samples:
            return list(range(total_frames))

        # Evenly distribute samples, always include first and last
        step = (total_frames - 1) / (self.config.num_samples - 1)
        indices = [int(i * step) for i in range(self.config.num_samples)]

        # Ensure last frame is included (handle rounding)
        indices[-1] = total_frames - 1

        return sorted(set(indices))  # Remove any duplicates, sort

    def visualize_features(
        self,
        database_path: Path,
        image_dir: Path,
        image_list: List[str]
    ) -> None:
        """Generate feature extraction visualizations.

        Creates keypoint overlays and feature quality statistics for sampled frames.
        Generates two images per sampled frame:
        1. Keypoint overlay: detected features drawn on original image
        2. Statistics panel: histograms and metrics for feature quality

        Args:
            database_path: Path to COLMAP database.db file
            image_dir: Directory containing input images
            image_list: List of image filenames (in order)

        Side Effects:
            - Writes PNG files to visualizations/features/
            - Updates HTML report with feature section (if report enabled)
            - Logs progress to console
        """
        if not self.config.enable_features:
            logger.info("Feature visualization disabled, skipping")
            return

        logger.info("Generating feature visualizations...")

        sample_indices = self.get_sample_indices(len(image_list))
        logger.info(f"Visualizing {len(sample_indices)} sampled frames: {sample_indices}")

        from .feature_viz import get_image_keypoints, visualize_keypoints_overlay, visualize_feature_statistics

        for idx in sample_indices:
            image_name = image_list[idx]
            logger.debug(f"Processing frame {idx}: {image_name}")

            try:
                # Retrieve keypoints from database
                keypoints, _ = get_image_keypoints(database_path, image_name)

                # Generate keypoint overlay
                keypoint_output = self.viz_dir / "features" / f"frame_{idx:04d}_keypoints.png"
                visualize_keypoints_overlay(
                    image_path=image_dir / image_name,
                    keypoints=keypoints,
                    output_path=keypoint_output
                )

                # Generate statistics panel
                stats_output = self.viz_dir / "features" / f"frame_{idx:04d}_stats.png"
                visualize_feature_statistics(
                    keypoints=keypoints,
                    output_path=stats_output
                )

                logger.debug(f"  ✓ Saved visualizations for frame {idx}")

            except Exception as e:
                logger.error(f"Failed to visualize frame {idx} ({image_name}): {e}")
                continue

        # Update HTML report
        if self.report:
            self.report.update_features_section(sample_indices, image_list)

        logger.info("✓ Feature visualizations complete")

    def visualize_matches(
        self,
        database_path: Path,
        image_dir: Path,
        image_list: List[str]
    ) -> None:
        """Generate feature matching visualizations.

        Creates match pair visualizations showing correspondences between consecutive
        frames. Uses sampled frames and visualizes matches between consecutive pairs
        (e.g., frame 0→1, frame 50→51, etc.). Inliers are shown in green, outliers in red.

        Args:
            database_path: Path to COLMAP database.db file
            image_dir: Directory containing input images
            image_list: List of image filenames (in order)

        Side Effects:
            - Writes PNG files to visualizations/matching/
            - Updates HTML report with matching section (if report enabled)
            - Logs progress to console
        """
        if not self.config.enable_matching:
            logger.info("Matching visualization disabled, skipping")
            return

        logger.info("Generating match visualizations...")

        sample_indices = self.get_sample_indices(len(image_list))

        # Create consecutive pairs from sampled frames
        pairs = []
        for i in range(len(sample_indices)):
            idx1 = sample_indices[i]
            # Pair with next frame (if available)
            idx2 = idx1 + 1
            if idx2 < len(image_list):
                pairs.append((idx1, idx2))

        logger.info(f"Visualizing {len(pairs)} match pairs")

        from .feature_viz import get_image_keypoints
        from .matching_viz import get_image_pair_matches, visualize_match_pair

        for idx1, idx2 in pairs:
            image_name1 = image_list[idx1]
            image_name2 = image_list[idx2]

            logger.debug(f"Processing pair {idx1}↔{idx2}: {image_name1} ↔ {image_name2}")

            try:
                # Retrieve keypoints for both images
                keypoints1, _ = get_image_keypoints(database_path, image_name1)
                keypoints2, _ = get_image_keypoints(database_path, image_name2)

                # Retrieve matches
                matches, inlier_mask = get_image_pair_matches(
                    database_path, image_name1, image_name2, verified_only=True
                )

                output_path = self.viz_dir / "matching" / f"match_pair_{idx1:04d}_{idx2:04d}.png"
                visualize_match_pair(
                    image_path1=image_dir / image_name1,
                    image_path2=image_dir / image_name2,
                    keypoints1=keypoints1,
                    keypoints2=keypoints2,
                    matches=matches,
                    inlier_mask=inlier_mask,
                    output_path=output_path,
                    max_matches=self.config.max_matches_display
                )

                logger.debug(f"  ✓ Saved match visualization for pair {idx1}↔{idx2}")

            except Exception as e:
                logger.error(f"Failed to visualize pair {idx1}↔{idx2}: {e}")
                continue

        # Update HTML report
        if self.report:
            self.report.update_matching_section(pairs, image_list)

        logger.info("✓ Match visualizations complete")

    def visualize_reconstruction(
        self,
        reconstruction: pycolmap.Reconstruction,
        output_name: str = "reconstruction"
    ) -> None:
        """Generate reconstruction visualizations.

        Creates camera trajectory and reconstruction statistics visualizations.
        Generates:
        1. Interactive 3D camera trajectory (HTML with Plotly)
        2. Statistics dashboard with quality metrics (PNG)

        Args:
            reconstruction: pycolmap.Reconstruction object from COLMAP
            output_name: Base name for output files (default: "reconstruction")

        Side Effects:
            - Writes HTML and PNG files to visualizations/reconstruction/
            - Updates HTML report with reconstruction section (if report enabled)
            - Logs progress to console
        """
        if not self.config.enable_reconstruction:
            logger.info("Reconstruction visualization disabled, skipping")
            return

        logger.info("Generating reconstruction visualizations...")

        from .reconstruction_viz import (
            plot_camera_trajectory_with_errors,
            plot_reconstruction_statistics
        )

        try:
            # Interactive 3D camera trajectory
            html_output = self.viz_dir / "reconstruction" / f"{output_name}_cameras.html"
            plot_camera_trajectory_with_errors(
                reconstruction=reconstruction,
                output_path=html_output
            )
            logger.debug(f"  ✓ Saved camera trajectory: {html_output.name}")

            # Statistics dashboard
            stats_output = self.viz_dir / "reconstruction" / f"{output_name}_stats.png"
            plot_reconstruction_statistics(
                reconstruction=reconstruction,
                output_path=stats_output
            )
            logger.debug(f"  ✓ Saved reconstruction statistics: {stats_output.name}")

            # Update HTML report
            if self.report:
                self.report.update_reconstruction_section(reconstruction)

            logger.info("✓ Reconstruction visualizations complete")

        except Exception as e:
            logger.error(f"Failed to generate reconstruction visualizations: {e}")

    def finalize_report(self, experiment_name: str, total_runtime: float) -> Optional[Path]:
        """Finalize the HTML debug report.

        Generates the final HTML report with all visualization sections and
        summary statistics. Should be called after all pipeline stages complete.

        Args:
            experiment_name: Name of the experiment for the report title
            total_runtime: Total pipeline execution time in seconds

        Returns:
            Path to the generated HTML report, or None if report generation failed

        Side Effects:
            - Writes debug_report.html to visualizations/
            - Logs report location to console
        """
        if not self.report:
            logger.warning("Report generator not available, skipping HTML report")
            return None

        try:
            report_path = self.report.finalize(experiment_name, total_runtime)
            logger.info(f"✓ Debug report saved to: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to finalize HTML report: {e}")
            return None


if __name__ == "__main__":
    """
    Validation function to test PipelineVisualizer with real data.

    This validates:
    1. Sample index generation across various dataset sizes
    2. Directory structure creation
    3. Configuration parameter handling
    """
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: Sample index generation for various dataset sizes
    total_tests += 1
    print("Test 1: Sample index generation")
    config = VisualizationConfig(output_dir=Path("/tmp/sfm_viz_test"))
    visualizer = PipelineVisualizer(config)

    # Test case: 100 frames, expect 10 evenly distributed samples
    indices_100 = visualizer.get_sample_indices(100)
    expected_100 = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
    if indices_100 != expected_100:
        all_validation_failures.append(
            f"Sample indices for 100 frames: Expected {expected_100}, got {indices_100}"
        )

    # Test case: 5 frames (less than num_samples), expect all frames
    indices_5 = visualizer.get_sample_indices(5)
    expected_5 = [0, 1, 2, 3, 4]
    if indices_5 != expected_5:
        all_validation_failures.append(
            f"Sample indices for 5 frames: Expected {expected_5}, got {indices_5}"
        )

    # Test case: 500 frames, check first/last and count
    indices_500 = visualizer.get_sample_indices(500)
    if len(indices_500) != 10:
        all_validation_failures.append(
            f"Sample count for 500 frames: Expected 10, got {len(indices_500)}"
        )
    if indices_500[0] != 0 or indices_500[-1] != 499:
        all_validation_failures.append(
            f"First/last frames for 500 frames: Expected [0, ..., 499], got [{indices_500[0]}, ..., {indices_500[-1]}]"
        )

    # Test 2: Directory structure creation
    total_tests += 1
    print("Test 2: Directory structure creation")
    if not (visualizer.viz_dir / "features").exists():
        all_validation_failures.append("Features directory not created")
    if not (visualizer.viz_dir / "matching").exists():
        all_validation_failures.append("Matching directory not created")
    if not (visualizer.viz_dir / "reconstruction").exists():
        all_validation_failures.append("Reconstruction directory not created")

    # Test 3: Configuration handling
    total_tests += 1
    print("Test 3: Configuration parameter handling")
    custom_config = VisualizationConfig(
        output_dir=Path("/tmp/sfm_viz_custom"),
        num_samples=5,
        max_matches_display=100,
        image_scale=0.25,
        enable_features=False
    )
    custom_visualizer = PipelineVisualizer(custom_config)

    if custom_visualizer.config.num_samples != 5:
        all_validation_failures.append(f"Custom num_samples: Expected 5, got {custom_visualizer.config.num_samples}")
    if custom_visualizer.config.enable_features != False:
        all_validation_failures.append(f"Custom enable_features: Expected False, got {custom_visualizer.config.enable_features}")

    # Test case: 20 frames with num_samples=5
    indices_custom = custom_visualizer.get_sample_indices(20)
    if len(indices_custom) != 5:
        all_validation_failures.append(
            f"Custom sample count for 20 frames: Expected 5, got {len(indices_custom)}"
        )
    if indices_custom[0] != 0 or indices_custom[-1] != 19:
        all_validation_failures.append(
            f"Custom first/last frames: Expected [0, ..., 19], got [{indices_custom[0]}, ..., {indices_custom[-1]}]"
        )

    # Final validation result
    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("PipelineVisualizer is validated and ready for use")
        sys.exit(0)
