#!/usr/bin/env python3
"""
LaMAR Visualization Orchestrator

This module orchestrates the generation of comprehensive visualizations for
LaMAR experiments, including cross-scene comparisons, per-scene analysis,
point cloud visualizations, and camera trajectory plots.

Third-party Dependencies:
- plotly: https://plotly.com/python/
- pandas: https://pandas.pydata.org/docs/
- numpy: https://numpy.org/doc/stable/

Sample Input:
    results = {
        'CAB': LamarSceneResult(...),
        'HGE': LamarSceneResult(...),
        'LIN': LamarSceneResult(...)
    }
    generate_lamar_visualizations(results, output_dir)

Expected Output:
    Collection of HTML files and PNG images in organized directory structure
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from loguru import logger

from .lamar_experiment import LamarSceneResult
from .point_cloud_viz import (
    create_point_cloud_3d_plot,
    create_point_density_heatmap,
    create_point_quality_plots,
    extract_point_cloud_data
)
from .camera_viz import (
    create_camera_trajectory_3d,
    create_camera_topdown_map,
    create_camera_distribution_plot,
    extract_camera_poses
)


@dataclass
class VisualizationPaths:
    """Paths to generated visualization files."""
    output_dir: Path
    comparison_plots: List[Path]
    scene_dirs: Dict[str, Path]
    all_figures: List[Path]


def extract_temporal_statistics(reconstruction) -> Dict[str, float]:
    """
    Extract temporal coverage statistics from image timestamps.

    For LaMAR dataset, timestamps are embedded in image filenames (nanoseconds).
    Example: "95378303839.jpg" -> 95378303839 nanoseconds

    Args:
        reconstruction: pycolmap.Reconstruction object

    Returns:
        Dictionary of temporal statistics:
        - capture_duration_s: Total capture time span in seconds
        - mean_capture_interval_s: Average time between images in seconds
        - capture_frequency_hz: Average capture frequency in Hz
        - num_images_with_timestamp: Number of images with valid timestamps
    """
    images = reconstruction.images
    timestamps = []

    for img_id, image in images.items():
        # Extract timestamp from filename (e.g., "95378303839.jpg")
        try:
            ts_str = Path(image.name).stem
            ts = int(ts_str)
            timestamps.append(ts)
        except (ValueError, AttributeError):
            # Skip images without numeric timestamps
            continue

    if len(timestamps) < 2:
        return {
            'capture_duration_s': 0.0,
            'mean_capture_interval_s': 0.0,
            'capture_frequency_hz': 0.0,
            'num_images_with_timestamp': len(timestamps)
        }

    timestamps = sorted(timestamps)
    diffs = np.diff(timestamps) / 1e9  # Convert nanoseconds to seconds

    return {
        'capture_duration_s': float((timestamps[-1] - timestamps[0]) / 1e9),
        'mean_capture_interval_s': float(np.mean(diffs)),
        'capture_frequency_hz': float(1.0 / np.mean(diffs)) if np.mean(diffs) > 0 else 0.0,
        'num_images_with_timestamp': len(timestamps)
    }


def extract_scene_metrics(result: LamarSceneResult) -> Dict[str, float]:
    """
    Extract comprehensive metrics from a scene result.

    Args:
        result: LamarSceneResult with reconstruction

    Returns:
        Dictionary of metrics including:
        - Basic counts (images, points, cameras)
        - Quality metrics (track length, reprojection error with mean/median/p95)
        - Spatial coverage (scene volume, extents)
        - Temporal coverage (capture frequency, duration)
        - Efficiency metrics (points per image)
    """
    if not result.success or result.reconstruction is None:
        return {}

    recon = result.reconstruction
    point_data = extract_point_cloud_data(recon, max_points=100000)

    metrics = {
        'num_images': result.num_images,
        'num_points3d': result.num_points3d,
        'num_cameras': result.num_cameras,
        'execution_time': result.execution_time,
    }

    # Add point cloud quality metrics
    if len(point_data['track_lengths']) > 0:
        metrics['mean_track_length'] = float(np.mean(point_data['track_lengths']))
        metrics['median_track_length'] = float(np.median(point_data['track_lengths']))
        metrics['track_length_p95'] = float(np.percentile(point_data['track_lengths'], 95))
        metrics['mean_reprojection_error'] = float(np.mean(point_data['errors']))
        metrics['median_reprojection_error'] = float(np.median(point_data['errors']))
        metrics['reprojection_error_p95'] = float(np.percentile(point_data['errors'], 95))
        metrics['observations_per_image'] = metrics['mean_track_length'] * metrics['num_points3d'] / max(metrics['num_images'], 1)

        # High-quality points percentage (track length > 10)
        high_quality_count = np.sum(point_data['track_lengths'] > 10)
        metrics['high_quality_points_pct'] = float(high_quality_count / len(point_data['track_lengths']) * 100)

    # Add spatial coverage statistics from camera poses
    try:
        camera_poses = extract_camera_poses(recon)
        positions = camera_poses['positions']

        if len(positions) > 0:
            metrics['spatial_extent_x'] = float(positions[:, 0].max() - positions[:, 0].min())
            metrics['spatial_extent_y'] = float(positions[:, 1].max() - positions[:, 1].min())
            metrics['spatial_extent_z'] = float(positions[:, 2].max() - positions[:, 2].min())
            metrics['scene_volume_m3'] = metrics['spatial_extent_x'] * metrics['spatial_extent_y'] * metrics['spatial_extent_z']

            # Calculate trajectory length (total camera movement distance)
            if len(positions) > 1:
                diffs = np.diff(positions, axis=0)
                distances = np.linalg.norm(diffs, axis=1)
                metrics['trajectory_length_m'] = float(np.sum(distances))
            else:
                metrics['trajectory_length_m'] = 0.0
    except Exception as e:
        logger.warning(f"Could not extract spatial statistics: {e}")
        metrics['spatial_extent_x'] = 0.0
        metrics['spatial_extent_y'] = 0.0
        metrics['spatial_extent_z'] = 0.0
        metrics['scene_volume_m3'] = 0.0
        metrics['trajectory_length_m'] = 0.0

    # Add temporal statistics
    try:
        temporal_stats = extract_temporal_statistics(recon)
        metrics.update(temporal_stats)
    except Exception as e:
        logger.warning(f"Could not extract temporal statistics: {e}")
        metrics['capture_duration_s'] = 0.0
        metrics['mean_capture_interval_s'] = 0.0
        metrics['capture_frequency_hz'] = 0.0
        metrics['num_images_with_timestamp'] = 0

    # Add efficiency metrics
    metrics['points_per_image'] = metrics['num_points3d'] / max(metrics['num_images'], 1)

    return metrics


def create_scene_comparison_bar_chart(
    results: Dict[str, LamarSceneResult]
) -> go.Figure:
    """
    Create grouped bar chart comparing scenes across key metrics.

    Args:
        results: Dictionary mapping scene names to results

    Returns:
        Plotly Figure
    """
    successful = {k: v for k, v in results.items() if v.success}

    if not successful:
        return go.Figure()

    scene_names = list(successful.keys())
    images = [r.num_images for r in successful.values()]
    points = [r.num_points3d for r in successful.values()]
    cameras = [r.num_cameras for r in successful.values()]

    fig = go.Figure(data=[
        go.Bar(name='Images', x=scene_names, y=images, marker_color='steelblue'),
        go.Bar(name='3D Points', x=scene_names, y=points, marker_color='coral'),
        go.Bar(name='Cameras', x=scene_names, y=cameras, marker_color='mediumseagreen')
    ])

    fig.update_layout(
        title=dict(
            text="LaMAR Scene Comparison<br><sub>Images, 3D Points, and Cameras per Scene</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Scene",
        yaxis_title="Count",
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_images_vs_points_scatter(
    results: Dict[str, LamarSceneResult]
) -> go.Figure:
    """
    Create scatter plot showing correlation between images and 3D points.

    Args:
        results: Dictionary mapping scene names to results

    Returns:
        Plotly Figure
    """
    successful = {k: v for k, v in results.items() if v.success}

    if not successful:
        return go.Figure()

    scene_names = list(successful.keys())
    images = [r.num_images for r in successful.values()]
    points = [r.num_points3d for r in successful.values()]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=images,
        y=points,
        mode='markers+text',
        marker=dict(size=20, color='steelblue', opacity=0.7),
        text=scene_names,
        textposition='top center',
        textfont=dict(size=14),
        hovertemplate='<b>%{text}</b><br>Images: %{x:,}<br>Points: %{y:,}<extra></extra>'
    ))

    # Add trend line
    if len(images) > 1:
        z = np.polyfit(images, points, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(images), max(images), 100)
        y_line = p(x_line)

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=dict(
            text="Images vs 3D Points Correlation<br><sub>Relationship between image count and reconstruction density</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Number of Images",
        yaxis_title="Number of 3D Points",
        height=500,
        hovermode='closest'
    )

    return fig


def create_quality_metrics_bar_chart(
    results: Dict[str, LamarSceneResult]
) -> go.Figure:
    """
    Create bar chart comparing reconstruction quality metrics across scenes.

    Args:
        results: Dictionary mapping scene names to results

    Returns:
        Plotly Figure
    """
    successful = {k: v for k, v in results.items() if v.success and v.reconstruction}

    if not successful:
        return go.Figure()

    metrics_data = {name: extract_scene_metrics(result)
                    for name, result in successful.items()}

    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Mean Track Length', 'Mean Reprojection Error')
    )

    scene_names = list(metrics_data.keys())
    track_lengths = [m.get('mean_track_length', 0) for m in metrics_data.values()]
    reproj_errors = [m.get('mean_reprojection_error', 0) for m in metrics_data.values()]

    # Track length bars
    fig.add_trace(
        go.Bar(
            x=scene_names,
            y=track_lengths,
            marker_color='steelblue',
            showlegend=False,
            hovertemplate='%{x}<br>Track Length: %{y:.1f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Reprojection error bars
    fig.add_trace(
        go.Bar(
            x=scene_names,
            y=reproj_errors,
            marker_color='coral',
            showlegend=False,
            hovertemplate='%{x}<br>Error: %{y:.3f}px<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Scene", row=1, col=1)
    fig.update_xaxes(title_text="Scene", row=1, col=2)
    fig.update_yaxes(title_text="Observations per Point", row=1, col=1)
    fig.update_yaxes(title_text="Error (pixels)", row=1, col=2)

    fig.update_layout(
        title=dict(
            text="Reconstruction Quality Metrics<br><sub>Track length and reprojection error comparison</sub>",
            x=0.5,
            xanchor='center'
        ),
        height=400,
        showlegend=False
    )

    return fig


def generate_comparison_plots(
    results: Dict[str, LamarSceneResult],
    output_dir: Path
) -> List[Path]:
    """
    Generate cross-scene comparison plots.

    Args:
        results: Dictionary mapping scene names to results
        output_dir: Directory to save plots

    Returns:
        List of paths to generated plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    logger.info("Generating cross-scene comparison plots...")

    # 1. Bar chart comparison
    fig_bars = create_scene_comparison_bar_chart(results)
    if len(fig_bars.data) > 0:
        filepath = output_dir / "scene_comparison_bar_chart.html"
        fig_bars.write_html(str(filepath))
        generated_files.append(filepath)
        logger.info(f"  ✓ Scene comparison bar chart: {filepath.name}")

    # 2. Images vs Points scatter
    fig_scatter = create_images_vs_points_scatter(results)
    if len(fig_scatter.data) > 0:
        filepath = output_dir / "images_vs_points_scatter.html"
        fig_scatter.write_html(str(filepath))
        generated_files.append(filepath)
        logger.info(f"  ✓ Images vs Points scatter: {filepath.name}")

    # 3. Quality metrics
    fig_quality = create_quality_metrics_bar_chart(results)
    if len(fig_quality.data) > 0:
        filepath = output_dir / "quality_metrics_comparison.html"
        fig_quality.write_html(str(filepath))
        generated_files.append(filepath)
        logger.info(f"  ✓ Quality metrics comparison: {filepath.name}")

    return generated_files


def generate_scene_visualizations(
    scene_name: str,
    result: LamarSceneResult,
    output_dir: Path
) -> List[Path]:
    """
    Generate all visualizations for a single scene.

    Args:
        scene_name: Name of the scene
        result: LamarSceneResult with reconstruction
        output_dir: Directory to save visualizations

    Returns:
        List of paths to generated files
    """
    if not result.success or result.reconstruction is None:
        logger.warning(f"Skipping visualizations for {scene_name} (no valid reconstruction)")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    recon = result.reconstruction

    logger.info(f"Generating visualizations for {scene_name}...")

    # Point cloud visualizations
    try:
        # 3D point cloud
        fig_pc_3d = create_point_cloud_3d_plot(recon, scene_name, max_points=10000)
        if len(fig_pc_3d.data) > 0:
            filepath = output_dir / f"{scene_name}_point_cloud_3d.html"
            fig_pc_3d.write_html(str(filepath))
            generated_files.append(filepath)
            logger.info(f"  ✓ Point cloud 3D: {filepath.name}")

        # Density heatmap
        fig_density = create_point_density_heatmap(recon, scene_name, grid_size=100)
        if len(fig_density.data) > 0:
            filepath = output_dir / f"{scene_name}_point_density.html"
            fig_density.write_html(str(filepath))
            generated_files.append(filepath)
            logger.info(f"  ✓ Point density heatmap: {filepath.name}")

        # Quality plots
        fig_track, fig_error = create_point_quality_plots(recon, scene_name)
        if len(fig_track.data) > 0:
            filepath = output_dir / f"{scene_name}_track_length_hist.html"
            fig_track.write_html(str(filepath))
            generated_files.append(filepath)
            logger.info(f"  ✓ Track length histogram: {filepath.name}")

        if len(fig_error.data) > 0:
            filepath = output_dir / f"{scene_name}_reprojection_error_hist.html"
            fig_error.write_html(str(filepath))
            generated_files.append(filepath)
            logger.info(f"  ✓ Reprojection error histogram: {filepath.name}")

    except Exception as e:
        logger.error(f"Error generating point cloud visualizations for {scene_name}: {e}")

    # Camera visualizations
    try:
        # 3D trajectory
        fig_cam_3d = create_camera_trajectory_3d(recon, scene_name, sample_rate=10, show_orientation=True)
        if len(fig_cam_3d.data) > 0:
            filepath = output_dir / f"{scene_name}_camera_trajectory_3d.html"
            fig_cam_3d.write_html(str(filepath))
            generated_files.append(filepath)
            logger.info(f"  ✓ Camera trajectory 3D: {filepath.name}")

        # Top-down map
        fig_map = create_camera_topdown_map(recon, scene_name, sample_rate=5)
        if len(fig_map.data) > 0:
            filepath = output_dir / f"{scene_name}_camera_topdown_map.html"
            fig_map.write_html(str(filepath))
            generated_files.append(filepath)
            logger.info(f"  ✓ Camera top-down map: {filepath.name}")

        # Distribution plots
        fig_dist = create_camera_distribution_plot(recon, scene_name)
        if len(fig_dist.data) > 0:
            filepath = output_dir / f"{scene_name}_camera_distribution.html"
            fig_dist.write_html(str(filepath))
            generated_files.append(filepath)
            logger.info(f"  ✓ Camera distribution: {filepath.name}")

    except Exception as e:
        logger.error(f"Error generating camera visualizations for {scene_name}: {e}")

    return generated_files


def create_summary_dataframe(results: Dict[str, LamarSceneResult]) -> pd.DataFrame:
    """
    Create pandas DataFrame with comprehensive scene statistics.

    Args:
        results: Dictionary mapping scene names to results

    Returns:
        pandas DataFrame with all metrics from extract_scene_metrics
    """
    data = []
    for scene_name, result in results.items():
        if result.success:
            metrics = extract_scene_metrics(result)
            # Create a row with all available metrics
            row = {
                'Scene': scene_name,
                # Basic counts
                'Images': result.num_images,
                '3D Points': result.num_points3d,
                'Cameras': result.num_cameras,
                'Time (s)': result.execution_time,
                # Quality metrics
                'Mean Track Length': metrics.get('mean_track_length', 0),
                'Median Track Length': metrics.get('median_track_length', 0),
                'Track Length P95': metrics.get('track_length_p95', 0),
                'Mean Error (px)': metrics.get('mean_reprojection_error', 0),
                'Median Error (px)': metrics.get('median_reprojection_error', 0),
                'Reprojection Error P95': metrics.get('reprojection_error_p95', 0),
                'High Quality Points %': metrics.get('high_quality_points_pct', 0),
                # Spatial coverage
                'Spatial Extent X (m)': metrics.get('spatial_extent_x', 0),
                'Spatial Extent Y (m)': metrics.get('spatial_extent_y', 0),
                'Spatial Extent Z (m)': metrics.get('spatial_extent_z', 0),
                'Scene Volume (m³)': metrics.get('scene_volume_m3', 0),
                'Trajectory Length (m)': metrics.get('trajectory_length_m', 0),
                # Temporal coverage
                'Capture Duration (s)': metrics.get('capture_duration_s', 0),
                'Mean Capture Interval (s)': metrics.get('mean_capture_interval_s', 0),
                'Capture Frequency (Hz)': metrics.get('capture_frequency_hz', 0),
                'Images with Timestamp': metrics.get('num_images_with_timestamp', 0),
                # Efficiency metrics
                'Points per Image': metrics.get('points_per_image', 0),
                'Observations per Image': metrics.get('observations_per_image', 0),
            }
            data.append(row)

    return pd.DataFrame(data)


def generate_lamar_visualizations(
    results: Dict[str, LamarSceneResult],
    output_dir: Path,
    generate_dashboard: bool = True
) -> VisualizationPaths:
    """
    Generate comprehensive visualizations for LaMAR experiment results.

    This is the main orchestration function that generates:
    - Cross-scene comparison plots (3 plots)
    - Per-scene visualizations (7 plots × N scenes)
    - Summary statistics
    - Comprehensive HTML dashboard (optional)

    Args:
        results: Dictionary mapping scene names to LamarSceneResult objects
        output_dir: Base directory for all outputs
        generate_dashboard: If True, generate comprehensive HTML dashboard

    Returns:
        VisualizationPaths with paths to all generated files
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info("Generating LaMAR Experiment Visualizations")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Scenes: {', '.join(results.keys())}")
    logger.info(f"{'='*80}\n")

    # Generate comparison plots
    comparison_dir = output_dir / "comparisons"
    comparison_plots = generate_comparison_plots(results, comparison_dir)

    # Generate per-scene visualizations
    scene_dirs = {}
    all_figures = list(comparison_plots)

    for scene_name, result in results.items():
        if result.success:
            scene_dir = output_dir / scene_name
            scene_dirs[scene_name] = scene_dir
            scene_figures = generate_scene_visualizations(scene_name, result, scene_dir)
            all_figures.extend(scene_figures)

    # Create summary DataFrame
    df_summary = create_summary_dataframe(results)
    if not df_summary.empty:
        summary_file = output_dir / "scene_summary.csv"
        df_summary.to_csv(summary_file, index=False)
        logger.info(f"\n✓ Summary table: {summary_file}")
        all_figures.append(summary_file)

    # Generate comprehensive HTML dashboard if requested
    if generate_dashboard:
        from .html_report_generator import generate_comprehensive_report

        # Create temporary VisualizationPaths object for report generator
        temp_viz_paths = VisualizationPaths(
            output_dir=output_dir,
            comparison_plots=comparison_plots,
            scene_dirs=scene_dirs,
            all_figures=all_figures
        )

        try:
            dashboard_path = output_dir / "comprehensive_dashboard.html"
            generate_comprehensive_report(temp_viz_paths, df_summary, dashboard_path)
            all_figures.append(dashboard_path)
        except Exception as e:
            logger.error(f"Failed to generate comprehensive dashboard: {e}")
            logger.info("Individual visualizations are still available")

    execution_time = time.time() - start_time

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Visualization Generation Complete")
    logger.info(f"   Total files: {len(all_figures)}")
    logger.info(f"   Execution time: {execution_time:.2f}s")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"{'='*80}\n")

    return VisualizationPaths(
        output_dir=output_dir,
        comparison_plots=comparison_plots,
        scene_dirs=scene_dirs,
        all_figures=all_figures
    )


if __name__ == "__main__":
    """
    Validation function to test visualization orchestration with real LaMAR data.

    Tests:
    1. Run LaMAR experiment on all 3 scenes
    2. Generate comprehensive visualizations
    3. Verify all expected files are created
    """
    from loguru import logger
    from .lamar_experiment import run_lamar_experiment

    # Setup logging
    logger.add("logs/lamar_visualization_test.log", rotation="10 MB")

    # Track validation failures
    all_validation_failures = []
    total_tests = 0

    # Test configuration
    base_dir = Path("datasets/lamar")
    output_dir = Path("test_viz_output/full_visualization")
    scenes = ["CAB", "HGE", "LIN"]

    print("="*80)
    print("LaMAR Visualization Orchestrator Validation")
    print("="*80)

    if not base_dir.exists():
        print(f"❌ Dataset not found at {base_dir}")
        sys.exit(1)

    # Test 1: Run LaMAR experiment
    total_tests += 1
    print(f"\nTest {total_tests}: Running LaMAR experiment on {len(scenes)} scenes...")
    try:
        results = run_lamar_experiment(scenes, base_dir, output_dir, validate=False)

        successful = sum(1 for r in results.values() if r.success)
        if successful == 0:
            all_validation_failures.append("LaMAR experiment: No scenes loaded successfully")
            print("❌ No scenes loaded successfully")
        else:
            print(f"✅ Loaded {successful}/{len(scenes)} scenes")
            for scene, result in results.items():
                if result.success:
                    print(f"  - {scene}: {result.num_images} images, {result.num_points3d:,} points")
    except Exception as e:
        all_validation_failures.append(f"LaMAR experiment: Exception - {e}")
        print(f"❌ Exception: {e}")
        sys.exit(1)

    # Test 2: Generate visualizations
    total_tests += 1
    print(f"\nTest {total_tests}: Generating comprehensive visualizations...")
    try:
        viz_paths = generate_lamar_visualizations(results, output_dir)

        if len(viz_paths.all_figures) == 0:
            all_validation_failures.append("Visualization generation: No files generated")
            print("❌ No files generated")
        else:
            print(f"✅ Generated {len(viz_paths.all_figures)} files")
            print(f"  - Comparison plots: {len(viz_paths.comparison_plots)}")
            print(f"  - Scene directories: {len(viz_paths.scene_dirs)}")
    except Exception as e:
        all_validation_failures.append(f"Visualization generation: Exception - {e}")
        print(f"❌ Exception: {e}")

    # Test 3: Verify expected file structure
    total_tests += 1
    print(f"\nTest {total_tests}: Verifying file structure...")
    try:
        expected_comparison = 3  # Bar chart, scatter, quality
        expected_per_scene = 7  # 4 point cloud + 3 camera

        actual_comparison = len(viz_paths.comparison_plots)
        if actual_comparison < expected_comparison:
            all_validation_failures.append(f"File structure: Expected {expected_comparison} comparison plots, got {actual_comparison}")
            print(f"❌ Expected {expected_comparison} comparison plots, got {actual_comparison}")
        else:
            print(f"✅ Comparison plots: {actual_comparison}/{expected_comparison}")

        # Check per-scene files
        for scene_name, scene_dir in viz_paths.scene_dirs.items():
            scene_files = list(scene_dir.glob("*.html"))
            if len(scene_files) < expected_per_scene:
                all_validation_failures.append(f"File structure: {scene_name} has {len(scene_files)} files, expected {expected_per_scene}")
                print(f"❌ {scene_name}: {len(scene_files)}/{expected_per_scene} files")
            else:
                print(f"✅ {scene_name}: {len(scene_files)}/{expected_per_scene} files")

    except Exception as e:
        all_validation_failures.append(f"File structure verification: Exception - {e}")
        print(f"❌ Exception: {e}")

    # Final validation result
    print("\n" + "="*80)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        print("="*80)
        print(f"\nPartial outputs may be available in: {output_dir}/")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("LaMAR visualization orchestrator is validated and ready to use")
        print("="*80)
        print(f"\nGenerated {len(viz_paths.all_figures)} visualization files in: {output_dir}/")
        print("Open the HTML files in a browser to view interactive plots")
        sys.exit(0)
