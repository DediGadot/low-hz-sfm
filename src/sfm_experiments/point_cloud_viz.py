#!/usr/bin/env python3
"""
Point Cloud Visualization for LaMAR Experiments

This module creates interactive 3D visualizations of COLMAP reconstructions
including point clouds, density heatmaps, and quality analysis plots.

Third-party Dependencies:
- plotly: https://plotly.com/python/
- pycolmap: https://github.com/colmap/pycolmap
- numpy: https://numpy.org/doc/stable/
- pandas: https://pandas.pydata.org/docs/

Sample Input:
    reconstruction = pycolmap.Reconstruction("path/to/sparse/0")
    fig = create_point_cloud_3d_plot(reconstruction, scene_name="CAB")

Expected Output:
    Interactive Plotly figure with 3D point cloud visualization
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import random

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import pycolmap
from loguru import logger


def sample_points(
    coords: np.ndarray,
    colors: np.ndarray,
    errors: np.ndarray,
    track_lengths: np.ndarray,
    max_points: int = 50000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Smart sampling of point cloud for performance optimization.

    If point cloud exceeds max_points, samples a representative subset
    while preserving spatial distribution.

    Args:
        coords: Nx3 array of xyz coordinates
        colors: Nx3 array of RGB colors (0-255)
        errors: N array of reprojection errors
        track_lengths: N array of track lengths
        max_points: Maximum points to display (default: 50000)

    Returns:
        Tuple of (sampled_coords, sampled_colors, sampled_errors, sampled_track_lengths)
    """
    n_points = len(coords)

    if n_points <= max_points:
        return coords, colors, errors, track_lengths

    # Random sampling
    indices = np.random.choice(n_points, max_points, replace=False)
    indices = np.sort(indices)  # Sort for better data locality

    logger.info(f"Sampled {max_points} of {n_points} points ({max_points/n_points*100:.1f}%)")

    return coords[indices], colors[indices], errors[indices], track_lengths[indices]


def extract_point_cloud_data(
    reconstruction: pycolmap.Reconstruction,
    max_points: int = 50000
) -> Dict[str, np.ndarray]:
    """
    Extract point cloud data from COLMAP reconstruction.

    Args:
        reconstruction: COLMAP Reconstruction object
        max_points: Maximum points to extract (for performance)

    Returns:
        Dictionary with keys:
            - 'xyz': Nx3 coordinates
            - 'rgb': Nx3 colors (0-255)
            - 'errors': N reprojection errors
            - 'track_lengths': N track lengths (observations per point)
    """
    points3D = reconstruction.points3D
    n_points = len(points3D)

    if n_points == 0:
        logger.warning("No 3D points in reconstruction")
        return {
            'xyz': np.array([]).reshape(0, 3),
            'rgb': np.array([]).reshape(0, 3),
            'errors': np.array([]),
            'track_lengths': np.array([])
        }

    # Pre-allocate arrays
    coords = np.zeros((n_points, 3))
    colors = np.zeros((n_points, 3))
    errors = np.zeros(n_points)
    track_lengths = np.zeros(n_points, dtype=int)

    # Extract data
    for idx, (point_id, point) in enumerate(points3D.items()):
        coords[idx] = point.xyz
        colors[idx] = point.color  # RGB 0-255
        errors[idx] = point.error  # Reprojection error
        track_lengths[idx] = len(point.track.elements)  # Number of observations

    # Sample if needed
    coords, colors, errors, track_lengths = sample_points(
        coords, colors, errors, track_lengths, max_points
    )

    logger.info(f"Extracted {len(coords)} points with {len(track_lengths)} track lengths")

    return {
        'xyz': coords,
        'rgb': colors,
        'errors': errors,
        'track_lengths': track_lengths
    }


def create_point_cloud_3d_plot(
    reconstruction: pycolmap.Reconstruction,
    scene_name: str,
    max_points: int = 50000
) -> go.Figure:
    """
    Create interactive 3D scatter plot of point cloud with RGB colors.

    Args:
        reconstruction: COLMAP Reconstruction object
        scene_name: Name of scene for title
        max_points: Maximum points to display

    Returns:
        Plotly Figure object
    """
    data = extract_point_cloud_data(reconstruction, max_points)

    if len(data['xyz']) == 0:
        logger.warning(f"No points to visualize for {scene_name}")
        return go.Figure()

    xyz = data['xyz']
    rgb = data['rgb']

    # Convert RGB to hex colors for plotly
    colors_hex = [
        f'rgb({int(r)},{int(g)},{int(b)})'
        for r, g, b in rgb
    ]

    # Create hover text with point info
    hover_text = [
        f"Point {i}<br>"
        f"XYZ: ({x:.2f}, {y:.2f}, {z:.2f})<br>"
        f"Track length: {tl}<br>"
        f"Error: {err:.4f}px"
        for i, (x, y, z, tl, err) in enumerate(zip(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            data['track_lengths'], data['errors']
        ))
    ]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors_hex,
                opacity=0.8,
                line=dict(width=0)
            ),
            text=hover_text,
            hoverinfo='text',
            name='3D Points'
        )
    ])

    fig.update_layout(
        title=dict(
            text=f"3D Point Cloud: {scene_name}<br><sub>{len(xyz):,} points</sub>",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=60),
        hovermode='closest'
    )

    return fig


def create_point_density_heatmap(
    reconstruction: pycolmap.Reconstruction,
    scene_name: str,
    grid_size: int = 100
) -> go.Figure:
    """
    Create 2D top-down density heatmap of point cloud.

    Args:
        reconstruction: COLMAP Reconstruction object
        scene_name: Name of scene for title
        grid_size: Resolution of grid (default: 100x100)

    Returns:
        Plotly Figure object
    """
    data = extract_point_cloud_data(reconstruction, max_points=100000)

    if len(data['xyz']) == 0:
        return go.Figure()

    xyz = data['xyz']

    # Create 2D histogram (top-down view, X-Y plane)
    x_range = (xyz[:, 0].min(), xyz[:, 0].max())
    y_range = (xyz[:, 1].min(), xyz[:, 1].max())

    hist, xedges, yedges = np.histogram2d(
        xyz[:, 0], xyz[:, 1],
        bins=grid_size,
        range=[x_range, y_range]
    )

    fig = go.Figure(data=go.Heatmap(
        z=hist.T,
        x=xedges,
        y=yedges,
        colorscale='Viridis',
        colorbar=dict(title='Point Count'),
        hovertemplate='X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Density: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=f"Point Density Heatmap: {scene_name}<br><sub>Top-down view (X-Y plane)</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='X (meters)',
        yaxis_title='Y (meters)',
        height=600,
        yaxis=dict(scaleanchor="x", scaleratio=1)  # Equal aspect ratio
    )

    return fig


def create_point_quality_plots(
    reconstruction: pycolmap.Reconstruction,
    scene_name: str
) -> Tuple[go.Figure, go.Figure]:
    """
    Create histograms of point quality metrics.

    Args:
        reconstruction: COLMAP Reconstruction object
        scene_name: Name of scene for title

    Returns:
        Tuple of (track_length_figure, reprojection_error_figure)
    """
    data = extract_point_cloud_data(reconstruction, max_points=100000)

    if len(data['track_lengths']) == 0:
        return go.Figure(), go.Figure()

    # Track length histogram
    fig_track = go.Figure(data=[
        go.Histogram(
            x=data['track_lengths'],
            nbinsx=50,
            marker=dict(color='steelblue'),
            hovertemplate='Track length: %{x}<br>Count: %{y}<extra></extra>'
        )
    ])

    mean_track = np.mean(data['track_lengths'])
    median_track = np.median(data['track_lengths'])

    fig_track.update_layout(
        title=dict(
            text=f"Track Length Distribution: {scene_name}<br>"
                 f"<sub>Mean: {mean_track:.1f} | Median: {median_track:.0f} observations/point</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Track Length (observations per point)',
        yaxis_title='Number of Points',
        height=400,
        showlegend=False
    )

    # Reprojection error histogram (filter outliers for better visualization)
    errors_filtered = data['errors'][data['errors'] < np.percentile(data['errors'], 95)]

    fig_error = go.Figure(data=[
        go.Histogram(
            x=errors_filtered,
            nbinsx=50,
            marker=dict(color='coral'),
            hovertemplate='Error: %{x:.3f}px<br>Count: %{y}<extra></extra>'
        )
    ])

    mean_error = np.mean(data['errors'])
    median_error = np.median(data['errors'])

    fig_error.update_layout(
        title=dict(
            text=f"Reprojection Error Distribution: {scene_name}<br>"
                 f"<sub>Mean: {mean_error:.3f}px | Median: {median_error:.3f}px</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Reprojection Error (pixels)',
        yaxis_title='Number of Points',
        height=400,
        showlegend=False
    )

    return fig_track, fig_error


if __name__ == "__main__":
    """
    Validation function to test point cloud visualization with real LaMAR data.

    Tests:
    1. Extract point cloud data from reconstruction
    2. Create 3D scatter plot
    3. Create density heatmap
    4. Create quality plots (track length, reprojection error)
    """
    from loguru import logger
    from .lamar_handler import load_lamar_reconstruction

    # Setup logging
    logger.add("logs/point_cloud_viz_test.log", rotation="10 MB")

    # Track validation failures
    all_validation_failures = []
    total_tests = 0

    # Test configuration
    base_dir = Path("datasets/lamar")
    scene_name = "CAB"
    colmap_path = base_dir / "colmap" / scene_name
    output_dir = Path("test_viz_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Point Cloud Visualization Validation")
    print("="*80)

    # Check if dataset exists
    if not colmap_path.exists():
        print(f"❌ Dataset not found at {colmap_path}")
        print("Download LaMAR dataset first using:")
        print("  uv run python scripts/download_lamar_dataset.py")
        sys.exit(1)

    # Load reconstruction
    print(f"\nLoading COLMAP reconstruction from {colmap_path}...")
    reconstruction = load_lamar_reconstruction(colmap_path)

    if reconstruction is None:
        print("❌ Failed to load reconstruction")
        sys.exit(1)

    print(f"✅ Loaded reconstruction with {len(reconstruction.points3D)} points")

    # Test 1: Extract point cloud data
    total_tests += 1
    print(f"\nTest {total_tests}: Extracting point cloud data...")
    try:
        data = extract_point_cloud_data(reconstruction, max_points=50000)

        if len(data['xyz']) == 0:
            all_validation_failures.append("Point cloud extraction: No points extracted")
            print("❌ No points extracted")
        elif data['xyz'].shape[1] != 3:
            all_validation_failures.append(f"Point cloud extraction: Invalid xyz shape {data['xyz'].shape}")
            print(f"❌ Invalid xyz shape: {data['xyz'].shape}")
        elif len(data['rgb']) != len(data['xyz']):
            all_validation_failures.append("Point cloud extraction: RGB length mismatch")
            print("❌ RGB length doesn't match xyz length")
        else:
            print(f"✅ Extracted {len(data['xyz'])} points")
            print(f"  - XYZ shape: {data['xyz'].shape}")
            print(f"  - RGB range: [{data['rgb'].min():.0f}, {data['rgb'].max():.0f}]")
            print(f"  - Track length range: [{data['track_lengths'].min()}, {data['track_lengths'].max()}]")
            print(f"  - Error range: [{data['errors'].min():.4f}, {data['errors'].max():.4f}]px")
    except Exception as e:
        all_validation_failures.append(f"Point cloud extraction: Exception - {e}")
        print(f"❌ Exception: {e}")

    # Test 2: Create 3D scatter plot
    total_tests += 1
    print(f"\nTest {total_tests}: Creating 3D scatter plot...")
    try:
        fig_3d = create_point_cloud_3d_plot(reconstruction, scene_name, max_points=10000)

        if len(fig_3d.data) == 0:
            all_validation_failures.append("3D scatter plot: No data in figure")
            print("❌ No data in figure")
        else:
            output_file = output_dir / f"{scene_name}_point_cloud_3d.html"
            fig_3d.write_html(str(output_file))
            print(f"✅ Created 3D scatter plot")
            print(f"  - Saved to: {output_file}")
            print(f"  - Data traces: {len(fig_3d.data)}")
    except Exception as e:
        all_validation_failures.append(f"3D scatter plot: Exception - {e}")
        print(f"❌ Exception: {e}")

    # Test 3: Create density heatmap
    total_tests += 1
    print(f"\nTest {total_tests}: Creating density heatmap...")
    try:
        fig_density = create_point_density_heatmap(reconstruction, scene_name, grid_size=100)

        if len(fig_density.data) == 0:
            all_validation_failures.append("Density heatmap: No data in figure")
            print("❌ No data in figure")
        else:
            output_file = output_dir / f"{scene_name}_density_heatmap.html"
            fig_density.write_html(str(output_file))
            print(f"✅ Created density heatmap")
            print(f"  - Saved to: {output_file}")
    except Exception as e:
        all_validation_failures.append(f"Density heatmap: Exception - {e}")
        print(f"❌ Exception: {e}")

    # Test 4: Create quality plots
    total_tests += 1
    print(f"\nTest {total_tests}: Creating quality plots...")
    try:
        fig_track, fig_error = create_point_quality_plots(reconstruction, scene_name)

        if len(fig_track.data) == 0 or len(fig_error.data) == 0:
            all_validation_failures.append("Quality plots: Missing data in one or both figures")
            print("❌ Missing data in figures")
        else:
            track_file = output_dir / f"{scene_name}_track_length_hist.html"
            error_file = output_dir / f"{scene_name}_reprojection_error_hist.html"
            fig_track.write_html(str(track_file))
            fig_error.write_html(str(error_file))
            print(f"✅ Created quality plots")
            print(f"  - Track length: {track_file}")
            print(f"  - Reprojection error: {error_file}")
    except Exception as e:
        all_validation_failures.append(f"Quality plots: Exception - {e}")
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
        print("Point cloud visualization module is validated and ready to use")
        print("="*80)
        print(f"\nGenerated visualizations saved to: {output_dir}/")
        print("Open the HTML files in a browser to view interactive plots")
        sys.exit(0)
