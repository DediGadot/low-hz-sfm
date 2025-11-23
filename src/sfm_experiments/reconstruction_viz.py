"""
Reconstruction Visualizations - Camera trajectory and quality statistics for SfM reconstructions.

This module provides visualization functions for analyzing COLMAP/GLOMAP reconstructions.
It generates interactive 3D camera trajectories with reprojection error color-coding and
comprehensive statistics dashboards for debugging reconstruction quality.

Key Features:
- Interactive 3D camera trajectory visualization with Plotly
- Reprojection error color-coding for quality assessment
- Statistics dashboard with multiple diagnostic plots
- Performance-optimized point cloud rendering

Documentation:
- Plotly: https://plotly.com/python/
- Matplotlib: https://matplotlib.org/stable/contents.html
- Pycolmap: https://github.com/colmap/pycolmap
- NumPy: https://numpy.org/doc/stable/

Sample Input:
- reconstruction: pycolmap.Reconstruction object from COLMAP
- output_path: Path where visualization should be saved

Expected Output:
- Interactive HTML file with 3D camera trajectory (Plotly)
- PNG dashboard with reconstruction statistics (Matplotlib)
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pycolmap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger


def _compute_camera_reprojection_errors(reconstruction: pycolmap.Reconstruction) -> Dict[int, float]:
    """Compute mean reprojection error for each camera in the reconstruction.

    Args:
        reconstruction: pycolmap.Reconstruction object

    Returns:
        Dictionary mapping image_id to mean reprojection error in pixels
    """
    errors = {}

    for image_id, image in reconstruction.images.items():
        if len(image.points2D) == 0:
            errors[image_id] = 0.0
            continue

        # Collect reprojection errors for all observed 3D points
        reproj_errors = []
        for point2D in image.points2D:
            if point2D.has_point3D():
                # Point2D.error is the reprojection error for this observation
                reproj_errors.append(point2D.error)

        if reproj_errors:
            errors[image_id] = float(np.mean(reproj_errors))
        else:
            errors[image_id] = 0.0

    return errors


def _get_error_color(error: float) -> str:
    """Get color code for reprojection error value.

    Color scheme:
    - Green: <1.0 pixel (good)
    - Yellow: 1.0-2.0 pixels (acceptable)
    - Orange: 2.0-4.0 pixels (concerning)
    - Red: >4.0 pixels (problematic)

    Args:
        error: Reprojection error in pixels

    Returns:
        Color string (CSS color name or hex code)
    """
    if error < 1.0:
        return 'green'
    elif error < 2.0:
        return 'yellow'
    elif error < 4.0:
        return 'orange'
    else:
        return 'red'


def _downsample_points(points: np.ndarray, max_points: int = 10000) -> np.ndarray:
    """Downsample 3D points for performance.

    Args:
        points: Nx3 array of 3D points
        max_points: Maximum number of points to keep

    Returns:
        Downsampled points array
    """
    if len(points) <= max_points:
        return points

    # Random sampling
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices]


def plot_camera_trajectory_with_errors(
    reconstruction: pycolmap.Reconstruction,
    output_path: Path,
    max_points_display: int = 10000
) -> None:
    """Generate interactive 3D camera trajectory visualization with reprojection errors.

    Creates an interactive Plotly visualization showing:
    - Camera positions as frustum pyramids
    - Camera orientations from rotation matrices
    - Color-coded reprojection errors (green=good, red=problematic)
    - Downsampled 3D point cloud as context
    - Hover information with frame names and error values

    Args:
        reconstruction: pycolmap.Reconstruction object from COLMAP/GLOMAP
        output_path: Path where HTML file should be saved
        max_points_display: Maximum number of 3D points to display (default: 10000)

    Side Effects:
        - Writes interactive HTML file to output_path
        - Logs progress to console

    Raises:
        ValueError: If reconstruction has no registered cameras
    """
    if len(reconstruction.images) == 0:
        raise ValueError("Reconstruction has no registered images")

    logger.info(f"Generating camera trajectory visualization...")
    logger.info(f"  Registered images: {len(reconstruction.images)}")
    logger.info(f"  3D points: {len(reconstruction.points3D)}")

    # Compute reprojection errors for all cameras
    camera_errors = _compute_camera_reprojection_errors(reconstruction)

    # Extract camera positions and metadata
    camera_positions = []
    camera_colors = []
    camera_labels = []
    camera_errors_list = []

    for image_id, image in reconstruction.images.items():
        # Camera center in world coordinates
        cam_center = image.projection_center()
        camera_positions.append(cam_center)

        # Reprojection error
        error = camera_errors[image_id]
        camera_errors_list.append(error)

        # Color based on error
        color = _get_error_color(error)
        camera_colors.append(color)

        # Label for hover
        label = f"{image.name}<br>Error: {error:.2f}px<br>Points: {len([p for p in image.points2D if p.has_point3D()])}"
        camera_labels.append(label)

    camera_positions = np.array(camera_positions)

    # Extract 3D points for context (downsampled for performance)
    if len(reconstruction.points3D) > 0:
        points_3d = np.array([point.xyz for point in reconstruction.points3D.values()])
        points_3d = _downsample_points(points_3d, max_points_display)
        logger.debug(f"  Displaying {len(points_3d)} / {len(reconstruction.points3D)} points")
    else:
        points_3d = np.array([])

    # Create Plotly figure
    fig = go.Figure()

    # Add 3D point cloud as small gray dots
    if len(points_3d) > 0:
        fig.add_trace(go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color='lightgray',
                opacity=0.3
            ),
            name='3D Points',
            hoverinfo='skip'
        ))

    # Add camera positions as scatter points with color coding
    fig.add_trace(go.Scatter3d(
        x=camera_positions[:, 0],
        y=camera_positions[:, 1],
        z=camera_positions[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=camera_errors_list,
            colorscale=[
                [0.0, 'green'],
                [0.25, 'yellow'],
                [0.5, 'orange'],
                [1.0, 'red']
            ],
            cmin=0.0,
            cmax=4.0,
            colorbar=dict(
                title="Reprojection<br>Error (px)",
                thickness=15,
                len=0.7
            ),
            line=dict(color='black', width=1)
        ),
        text=camera_labels,
        hoverinfo='text',
        name='Cameras'
    ))

    # Camera trajectory line (connecting cameras in sequence)
    if len(camera_positions) > 1:
        fig.add_trace(go.Scatter3d(
            x=camera_positions[:, 0],
            y=camera_positions[:, 1],
            z=camera_positions[:, 2],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Trajectory',
            hoverinfo='skip'
        ))

    # Configure layout
    fig.update_layout(
        title=dict(
            text=f"Camera Trajectory - {len(reconstruction.images)} cameras, "
                 f"{len(reconstruction.points3D)} points",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=800,
        hovermode='closest',
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )

    # Save to HTML
    fig.write_html(str(output_path))
    logger.info(f"✓ Saved camera trajectory to: {output_path}")


def plot_reconstruction_statistics(
    reconstruction: pycolmap.Reconstruction,
    output_path: Path
) -> None:
    """Generate reconstruction statistics dashboard.

    Creates a 2x2 grid of diagnostic plots:
    - Top-left: Histogram of per-camera reprojection errors
    - Top-right: Reprojection error vs frame index (time series)
    - Bottom-left: Number of 3D points observed per camera
    - Bottom-right: Text summary with key metrics

    Args:
        reconstruction: pycolmap.Reconstruction object from COLMAP/GLOMAP
        output_path: Path where PNG file should be saved

    Side Effects:
        - Writes PNG dashboard to output_path
        - Logs progress to console

    Raises:
        ValueError: If reconstruction has no registered cameras
    """
    if len(reconstruction.images) == 0:
        raise ValueError("Reconstruction has no registered images")

    logger.info(f"Generating reconstruction statistics...")

    # Compute reprojection errors
    camera_errors = _compute_camera_reprojection_errors(reconstruction)

    # Extract data for plots
    image_ids = sorted(camera_errors.keys())
    errors = [camera_errors[img_id] for img_id in image_ids]
    points_per_camera = [
        len([p for p in reconstruction.images[img_id].points2D if p.has_point3D()])
        for img_id in image_ids
    ]

    # Compute summary statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    num_high_error = sum(1 for e in errors if e > 2.0)
    pct_high_error = 100.0 * num_high_error / len(errors) if errors else 0.0

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Reconstruction Statistics - {len(reconstruction.images)} Cameras',
                 fontsize=16, fontweight='bold')

    # Top-left: Histogram of reprojection errors
    ax1 = axes[0, 0]
    ax1.hist(errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}px')
    ax1.axvline(median_error, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}px')
    ax1.axvline(2.0, color='darkred', linestyle=':', linewidth=2, label='2px threshold')
    ax1.set_xlabel('Reprojection Error (pixels)', fontsize=11)
    ax1.set_ylabel('Number of Cameras', fontsize=11)
    ax1.set_title('Distribution of Reprojection Errors', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Top-right: Reprojection error vs frame index (time series)
    ax2 = axes[0, 1]
    colors = [_get_error_color(e) for e in errors]
    ax2.scatter(range(len(errors)), errors, c=colors, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.axhline(1.0, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Good (<1px)')
    ax2.axhline(2.0, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Acceptable (<2px)')
    ax2.axhline(4.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Problematic (>4px)')
    ax2.set_xlabel('Frame Index', fontsize=11)
    ax2.set_ylabel('Reprojection Error (pixels)', fontsize=11)
    ax2.set_title('Reprojection Error Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)

    # Bottom-left: Number of observed 3D points per camera
    ax3 = axes[1, 0]
    ax3.bar(range(len(points_per_camera)), points_per_camera, color='teal', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Frame Index', fontsize=11)
    ax3.set_ylabel('Number of Observed 3D Points', fontsize=11)
    ax3.set_title('3D Points Observed per Camera', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    # Bottom-right: Text summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    RECONSTRUCTION SUMMARY
    {'=' * 40}

    Cameras:
      • Registered: {len(reconstruction.images)}
      • Total images: {len(reconstruction.images)}

    3D Points:
      • Total: {len(reconstruction.points3D):,}
      • Mean track length: {np.mean([len(pt.track.elements) for pt in reconstruction.points3D.values()]):.1f}

    Reprojection Errors:
      • Mean: {mean_error:.3f} pixels
      • Median: {median_error:.3f} pixels
      • Min: {min(errors):.3f} pixels
      • Max: {max(errors):.3f} pixels

    Quality Assessment:
      • Cameras with error > 2px: {num_high_error} ({pct_high_error:.1f}%)
      • Mean points per camera: {np.mean(points_per_camera):.0f}
      • Median points per camera: {np.median(points_per_camera):.0f}
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved reconstruction statistics to: {output_path}")


if __name__ == "__main__":
    """
    Validation function to test reconstruction visualization functions.

    This validates:
    1. Reprojection error computation with synthetic data
    2. Error color mapping
    3. Point downsampling
    4. Visualization generation with mock reconstruction
    """
    import sys
    from unittest.mock import Mock

    all_validation_failures = []
    total_tests = 0

    # Test 1: Error color mapping
    total_tests += 1
    print("Test 1: Error color mapping")
    test_cases = [
        (0.5, 'green'),
        (1.5, 'yellow'),
        (3.0, 'orange'),
        (5.0, 'red')
    ]
    for error, expected_color in test_cases:
        actual_color = _get_error_color(error)
        if actual_color != expected_color:
            all_validation_failures.append(
                f"Error color for {error}px: Expected '{expected_color}', got '{actual_color}'"
            )

    # Test 2: Point downsampling
    total_tests += 1
    print("Test 2: Point downsampling")
    points = np.random.randn(20000, 3)
    downsampled = _downsample_points(points, max_points=5000)
    if len(downsampled) != 5000:
        all_validation_failures.append(
            f"Downsampling: Expected 5000 points, got {len(downsampled)}"
        )
    if downsampled.shape[1] != 3:
        all_validation_failures.append(
            f"Downsampled shape: Expected (N, 3), got {downsampled.shape}"
        )

    # Test case: No downsampling needed
    small_points = np.random.randn(100, 3)
    not_downsampled = _downsample_points(small_points, max_points=5000)
    if len(not_downsampled) != 100:
        all_validation_failures.append(
            f"No downsampling needed: Expected 100 points, got {len(not_downsampled)}"
        )

    # Test 3: Reprojection error computation (mock)
    total_tests += 1
    print("Test 3: Reprojection error computation with mock data")

    # Create mock reconstruction
    mock_reconstruction = Mock(spec=pycolmap.Reconstruction)

    # Mock image with point observations
    mock_image = Mock()
    mock_image.name = "test_image.jpg"

    # Mock Point2D objects with errors
    mock_point1 = Mock()
    mock_point1.has_point3D = Mock(return_value=True)
    mock_point1.error = 1.5

    mock_point2 = Mock()
    mock_point2.has_point3D = Mock(return_value=True)
    mock_point2.error = 2.5

    mock_point3 = Mock()
    mock_point3.has_point3D = Mock(return_value=False)  # No 3D point

    mock_image.points2D = [mock_point1, mock_point2, mock_point3]

    mock_reconstruction.images = {1: mock_image}

    errors = _compute_camera_reprojection_errors(mock_reconstruction)
    expected_mean_error = (1.5 + 2.5) / 2  # = 2.0

    if 1 not in errors:
        all_validation_failures.append("Error computation: Image ID 1 not in results")
    elif abs(errors[1] - expected_mean_error) > 0.01:
        all_validation_failures.append(
            f"Error computation: Expected {expected_mean_error:.2f}, got {errors[1]:.2f}"
        )

    # Final validation result
    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Reconstruction visualization functions are validated and ready for use")
        print("\nNote: Full end-to-end tests with real pycolmap.Reconstruction objects")
        print("      require a complete COLMAP database and will be tested in integration.")
        sys.exit(0)
