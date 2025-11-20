#!/usr/bin/env python3
"""
Camera Trajectory Visualization for LaMAR Experiments

This module creates interactive visualizations of camera trajectories from
COLMAP reconstructions including 3D paths and 2D top-down maps.

Third-party Dependencies:
- plotly: https://plotly.com/python/
- pycolmap: https://github.com/colmap/pycolmap
- numpy: https://numpy.org/doc/stable/

Sample Input:
    reconstruction = pycolmap.Reconstruction("path/to/sparse/0")
    fig = create_camera_trajectory_3d(reconstruction, scene_name="CAB")

Expected Output:
    Interactive Plotly figure with 3D camera trajectory visualization
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycolmap
from loguru import logger


def extract_camera_poses(reconstruction: pycolmap.Reconstruction) -> Dict[str, np.ndarray]:
    """
    Extract camera positions and orientations from COLMAP reconstruction.

    Args:
        reconstruction: COLMAP Reconstruction object

    Returns:
        Dictionary with keys:
            - 'positions': Nx3 array of camera positions (world coordinates)
            - 'orientations': Nx4 array of quaternions (x, y, z, w) pycolmap format
            - 'image_names': List of N image names
            - 'image_ids': List of N image IDs
    """
    images = reconstruction.images
    n_images = len(images)

    if n_images == 0:
        logger.warning("No images in reconstruction")
        return {
            'positions': np.array([]).reshape(0, 3),
            'orientations': np.array([]).reshape(0, 4),  # x, y, z, w format
            'image_names': [],
            'image_ids': []
        }

    # Pre-allocate arrays
    positions = np.zeros((n_images, 3))
    orientations = np.zeros((n_images, 4))  # quaternions: x, y, z, w (pycolmap format)
    image_names = []
    image_ids = []

    # Extract camera positions (world coordinates)
    for idx, (img_id, image) in enumerate(images.items()):
        # Get camera center in world coordinates
        # pycolmap uses cam_from_world transformation
        # IMPORTANT: cam_from_world is a METHOD, not a property
        cam_from_world = image.cam_from_world()
        quat = cam_from_world.rotation.quat  # Returns [x, y, z, w]

        # Camera position in world coordinates (using built-in method)
        camera_pos = image.projection_center()

        positions[idx] = camera_pos
        orientations[idx] = quat  # Store as [x, y, z, w]
        image_names.append(image.name)
        image_ids.append(img_id)

    logger.info(f"Extracted {n_images} camera poses")
    logger.info(f"  Position range: X=[{positions[:,0].min():.1f}, {positions[:,0].max():.1f}], "
                f"Y=[{positions[:,1].min():.1f}, {positions[:,1].max():.1f}], "
                f"Z=[{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")

    return {
        'positions': positions,
        'orientations': orientations,
        'image_names': image_names,
        'image_ids': image_ids
    }


def quaternion_to_direction(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to forward direction vector.

    Args:
        quat: Quaternion [x, y, z, w] (pycolmap format)

    Returns:
        3D unit vector representing camera forward direction
    """
    x, y, z, w = quat

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

    # Camera forward is typically -Z axis in camera coordinates
    # Transform to world coordinates
    forward = R @ np.array([0, 0, -1])

    return forward / np.linalg.norm(forward)


def create_camera_trajectory_3d(
    reconstruction: pycolmap.Reconstruction,
    scene_name: str,
    sample_rate: int = 1,
    show_orientation: bool = True
) -> go.Figure:
    """
    Create interactive 3D visualization of camera trajectory.

    Args:
        reconstruction: COLMAP Reconstruction object
        scene_name: Name of scene for title
        sample_rate: Show every Nth camera (default: 1 = all cameras)
        show_orientation: Whether to show camera orientation arrows

    Returns:
        Plotly Figure object
    """
    camera_data = extract_camera_poses(reconstruction)

    if len(camera_data['positions']) == 0:
        logger.warning(f"No camera poses to visualize for {scene_name}")
        return go.Figure()

    positions = camera_data['positions'][::sample_rate]
    orientations = camera_data['orientations'][::sample_rate]
    image_names = camera_data['image_names'][::sample_rate]

    # Create hover text
    hover_text = [
        f"Image: {name}<br>"
        f"Position: ({x:.2f}, {y:.2f}, {z:.2f})m"
        for name, (x, y, z) in zip(image_names, positions)
    ]

    # Create figure
    fig = go.Figure()

    # Add camera trajectory line
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines+markers',
        line=dict(color='steelblue', width=2),
        marker=dict(
            size=3,
            color='steelblue',
            symbol='circle'
        ),
        text=hover_text,
        hoverinfo='text',
        name='Camera Path',
        showlegend=True
    ))

    # Add orientation arrows (sample every 10th for clarity)
    if show_orientation and len(positions) > 0:
        arrow_sample = max(1, len(positions) // 20)  # Show ~20 arrows
        arrow_positions = positions[::arrow_sample]
        arrow_orientations = orientations[::arrow_sample]

        # Create arrows using cones
        for pos, quat in zip(arrow_positions, arrow_orientations):
            direction = quaternion_to_direction(quat)

            # Arrow endpoint
            arrow_length = 0.5  # meters
            end_pos = pos + direction * arrow_length

            # Add arrow line
            fig.add_trace(go.Scatter3d(
                x=[pos[0], end_pos[0]],
                y=[pos[1], end_pos[1]],
                z=[pos[2], end_pos[2]],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Camera Trajectory 3D: {scene_name}<br>"
                 f"<sub>{len(positions):,} cameras shown (sample rate: {sample_rate})</sub>",
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
        hovermode='closest',
        showlegend=True
    )

    return fig


def create_camera_topdown_map(
    reconstruction: pycolmap.Reconstruction,
    scene_name: str,
    sample_rate: int = 1
) -> go.Figure:
    """
    Create 2D top-down map of camera positions (X-Y plane).

    Args:
        reconstruction: COLMAP Reconstruction object
        scene_name: Name of scene for title
        sample_rate: Show every Nth camera (default: 1 = all cameras)

    Returns:
        Plotly Figure object
    """
    camera_data = extract_camera_poses(reconstruction)

    if len(camera_data['positions']) == 0:
        logger.warning(f"No camera poses to visualize for {scene_name}")
        return go.Figure()

    positions = camera_data['positions'][::sample_rate]
    image_names = camera_data['image_names'][::sample_rate]

    # Create hover text
    hover_text = [
        f"Image: {name}<br>"
        f"X: {x:.2f}m<br>"
        f"Y: {y:.2f}m<br>"
        f"Z: {z:.2f}m"
        for name, (x, y, z) in zip(image_names, positions)
    ]

    fig = go.Figure()

    # Add camera positions
    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers+lines',
        line=dict(color='steelblue', width=1),
        marker=dict(
            size=5,
            color=positions[:, 2],  # Color by height (Z)
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Height (m)'),
            symbol='circle'
        ),
        text=hover_text,
        hoverinfo='text',
        name='Camera Positions'
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Camera Top-Down Map: {scene_name}<br>"
                 f"<sub>{len(positions):,} cameras (colored by height)</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='X (meters)',
        yaxis_title='Y (meters)',
        height=600,
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
        hovermode='closest'
    )

    return fig


def create_camera_distribution_plot(
    reconstruction: pycolmap.Reconstruction,
    scene_name: str
) -> go.Figure:
    """
    Create histogram showing spatial distribution of cameras.

    Args:
        reconstruction: COLMAP Reconstruction object
        scene_name: Name of scene for title

    Returns:
        Plotly Figure with 3 subplots (X, Y, Z distributions)
    """
    camera_data = extract_camera_poses(reconstruction)

    if len(camera_data['positions']) == 0:
        return go.Figure()

    positions = camera_data['positions']

    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('X Distribution', 'Y Distribution', 'Z Distribution')
    )

    # X distribution
    fig.add_trace(
        go.Histogram(
            x=positions[:, 0],
            nbinsx=30,
            marker=dict(color='steelblue'),
            name='X',
            showlegend=False
        ),
        row=1, col=1
    )

    # Y distribution
    fig.add_trace(
        go.Histogram(
            x=positions[:, 1],
            nbinsx=30,
            marker=dict(color='coral'),
            name='Y',
            showlegend=False
        ),
        row=1, col=2
    )

    # Z distribution
    fig.add_trace(
        go.Histogram(
            x=positions[:, 2],
            nbinsx=30,
            marker=dict(color='mediumseagreen'),
            name='Z',
            showlegend=False
        ),
        row=1, col=3
    )

    fig.update_xaxes(title_text="X (meters)", row=1, col=1)
    fig.update_xaxes(title_text="Y (meters)", row=1, col=2)
    fig.update_xaxes(title_text="Z (meters)", row=1, col=3)

    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.update_layout(
        title=dict(
            text=f"Camera Spatial Distribution: {scene_name}",
            x=0.5,
            xanchor='center'
        ),
        height=400,
        showlegend=False
    )

    return fig


if __name__ == "__main__":
    """
    Validation function to test camera visualization with real LaMAR data.

    Tests:
    1. Extract camera poses from reconstruction
    2. Create 3D trajectory plot
    3. Create top-down map
    4. Create distribution plots
    """
    from loguru import logger
    from .lamar_handler import load_lamar_reconstruction

    # Setup logging
    logger.add("logs/camera_viz_test.log", rotation="10 MB")

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
    print("Camera Visualization Validation")
    print("="*80)

    # Check if dataset exists
    if not colmap_path.exists():
        print(f"❌ Dataset not found at {colmap_path}")
        print("Download LaMAR dataset first")
        sys.exit(1)

    # Load reconstruction
    print(f"\nLoading COLMAP reconstruction from {colmap_path}...")
    reconstruction = load_lamar_reconstruction(colmap_path)

    if reconstruction is None:
        print("❌ Failed to load reconstruction")
        sys.exit(1)

    print(f"✅ Loaded reconstruction with {len(reconstruction.images)} images")

    # Test 1: Extract camera poses
    total_tests += 1
    print(f"\nTest {total_tests}: Extracting camera poses...")
    try:
        camera_data = extract_camera_poses(reconstruction)

        if len(camera_data['positions']) == 0:
            all_validation_failures.append("Camera pose extraction: No poses extracted")
            print("❌ No poses extracted")
        elif camera_data['positions'].shape[1] != 3:
            all_validation_failures.append(f"Camera pose extraction: Invalid shape {camera_data['positions'].shape}")
            print(f"❌ Invalid positions shape: {camera_data['positions'].shape}")
        else:
            print(f"✅ Extracted {len(camera_data['positions'])} camera poses")
            print(f"  - Position shape: {camera_data['positions'].shape}")
            print(f"  - Orientation shape: {camera_data['orientations'].shape}")
            print(f"  - X range: [{camera_data['positions'][:,0].min():.1f}, {camera_data['positions'][:,0].max():.1f}]m")
            print(f"  - Y range: [{camera_data['positions'][:,1].min():.1f}, {camera_data['positions'][:,1].max():.1f}]m")
            print(f"  - Z range: [{camera_data['positions'][:,2].min():.1f}, {camera_data['positions'][:,2].max():.1f}]m")
    except Exception as e:
        all_validation_failures.append(f"Camera pose extraction: Exception - {e}")
        print(f"❌ Exception: {e}")

    # Test 2: Create 3D trajectory plot
    total_tests += 1
    print(f"\nTest {total_tests}: Creating 3D trajectory plot...")
    try:
        fig_3d = create_camera_trajectory_3d(
            reconstruction, scene_name,
            sample_rate=10,  # Sample for faster rendering
            show_orientation=True
        )

        if len(fig_3d.data) == 0:
            all_validation_failures.append("3D trajectory: No data in figure")
            print("❌ No data in figure")
        else:
            output_file = output_dir / f"{scene_name}_camera_trajectory_3d.html"
            fig_3d.write_html(str(output_file))
            print(f"✅ Created 3D trajectory plot")
            print(f"  - Saved to: {output_file}")
            print(f"  - Data traces: {len(fig_3d.data)}")
    except Exception as e:
        all_validation_failures.append(f"3D trajectory: Exception - {e}")
        print(f"❌ Exception: {e}")

    # Test 3: Create top-down map
    total_tests += 1
    print(f"\nTest {total_tests}: Creating top-down map...")
    try:
        fig_map = create_camera_topdown_map(reconstruction, scene_name, sample_rate=5)

        if len(fig_map.data) == 0:
            all_validation_failures.append("Top-down map: No data in figure")
            print("❌ No data in figure")
        else:
            output_file = output_dir / f"{scene_name}_camera_topdown_map.html"
            fig_map.write_html(str(output_file))
            print(f"✅ Created top-down map")
            print(f"  - Saved to: {output_file}")
    except Exception as e:
        all_validation_failures.append(f"Top-down map: Exception - {e}")
        print(f"❌ Exception: {e}")

    # Test 4: Create distribution plots
    total_tests += 1
    print(f"\nTest {total_tests}: Creating distribution plots...")
    try:
        fig_dist = create_camera_distribution_plot(reconstruction, scene_name)

        if len(fig_dist.data) == 0:
            all_validation_failures.append("Distribution plots: No data in figure")
            print("❌ No data in figure")
        else:
            output_file = output_dir / f"{scene_name}_camera_distribution.html"
            fig_dist.write_html(str(output_file))
            print(f"✅ Created distribution plots")
            print(f"  - Saved to: {output_file}")
            print(f"  - Subplots: 3 (X, Y, Z)")
    except Exception as e:
        all_validation_failures.append(f"Distribution plots: Exception - {e}")
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
        print("Camera visualization module is validated and ready to use")
        print("="*80)
        print(f"\nGenerated visualizations saved to: {output_dir}/")
        print("Open the HTML files in a browser to view interactive plots")
        sys.exit(0)
