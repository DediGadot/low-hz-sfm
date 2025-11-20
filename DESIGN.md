# SfM Multi-Visit Experimentation Pipeline - MVP Design

**Version:** 2.0 (MVP-Focused)
**Last Updated:** 2025-11-16
**Purpose:** Minimum viable pipeline to investigate how map accuracy improves with multiple visits to the same venue-scale location.

---

## Executive Summary

### MVP Goal

Build the **simplest possible** working pipeline that can:
1. Load multi-visit venue-scale data (Hilti SLAM 2023)
2. Sample at 0.25 Hz (1 frame every 4 seconds)
3. Run COLMAP reconstruction with 1, 2, 3, 5 visits
4. Measure accuracy improvement (ATE, Chamfer Distance, Completeness)
5. Generate plot: **Accuracy vs. Number of Visits**

**Start simple. Prove it works. Then expand.**

---

## Core Decisions (One Pick Per Category)

| Category | MVP Choice | Rationale | Future Expansion |
|----------|-----------|-----------|------------------|
| **Dataset** | Hilti SLAM 2023 Site 1 | Multi-visit, venue-scale, ground truth | Add TUM, NCLT, ScanNet |
| **SfM Pipeline** | COLMAP (pycolmap) | Most reliable, well-documented | Add GLOMAP, hloc |
| **Metrics** | ATE, Chamfer, Completeness | Cover pose and map quality | Add F-Score, RPE |
| **Visualization** | Open3D + matplotlib | Standard tools, easy to use | Add HTML reports |
| **Frame Rate** | 0.25 Hz (1 frame/4 sec) | Reduces compute, tests sparse data | - |

---

## System Architecture

### High-Level Flow

```
┌─────────────────────┐
│ Hilti SLAM 2023     │
│ ROS Bags → Frames   │ (0.25 Hz extraction)
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Session Splitter    │
│ Seq 1, 2, 3, 4, 5   │ (Separate multi-visit data)
└──────────┬──────────┘
           │
           v
┌─────────────────────────────────────────┐
│ Multi-Visit Reconstruction              │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │
│ │Visit1│ │ 1+2  │ │1+2+3 │ │1+2+..│   │ (COLMAP)
│ └──────┘ └──────┘ └──────┘ └──────┘   │
└──────────┬──────────────────────────────┘
           │
           v
┌─────────────────────┐
│ Evaluation          │
│ - Align to GT       │
│ - Compute ATE       │
│ - Chamfer Distance  │
│ - Completeness      │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Plot: Accuracy vs   │
│ Number of Visits    │
└─────────────────────┘
```

---

## Directory Structure (Simplified)

```
sfm_experiments/
├── pyproject.toml              # UV package config
├── README.md                   # Quick start
├── DESIGN.md                   # This file
├── datasets/
│   └── hilti/
│       ├── rosbags/           # Downloaded ROS bags
│       ├── frames/            # Extracted frames at 0.25 Hz
│       │   ├── session_01/
│       │   ├── session_02/
│       │   └── session_03/
│       └── ground_truth/      # GT poses and point clouds
├── src/
│   └── sfm_experiments/
│       ├── __init__.py
│       ├── cli.py             # Typer CLI
│       ├── dataset.py         # Hilti loader
│       ├── colmap_runner.py   # COLMAP wrapper
│       ├── multivist.py       # Multi-visit orchestrator
│       ├── metrics.py         # ATE, Chamfer, Completeness
│       ├── visualization.py   # Plotting
│       └── utils.py           # Logging, I/O
├── configs/
│   ├── hilti.yaml            # Dataset config
│   └── colmap.yaml           # Pipeline config
├── results/
│   ├── reconstructions/      # COLMAP outputs
│   │   ├── visit_1/
│   │   ├── visit_1_2/
│   │   ├── visit_1_2_3/
│   │   └── visit_1_2_3_4_5/
│   └── plots/                # Accuracy vs visits plots
└── tests/
    └── test_end_to_end.py    # MVP validation test
```

---

## Dataset: Hilti SLAM Challenge 2023

### Why Hilti?

✅ **Purpose-built for multi-session SLAM**
✅ **Venue-scale:** Multi-storey construction sites, parking structures
✅ **Multiple visits:** Different recording sessions of same areas
✅ **High-precision ground truth:** GCP markers with cm-level accuracy
✅ **Well-documented:** ROS bag format with calibration

### Download

```bash
# Official site
https://hilti-challenge.com/dataset-2023.html

# We'll use Site 1 for MVP (smallest, fastest to test)
```

### Data Structure

```
Hilti SLAM 2023/
├── site_1/
│   ├── sequence_01.bag    # Session 1
│   ├── sequence_02.bag    # Session 2
│   ├── sequence_03.bag    # Session 3
│   ├── calibration/       # Camera intrinsics
│   └── ground_truth/      # GCP-based poses
```

### Frame Extraction Strategy

**Input:** ROS bags at 30 Hz (camera)
**Output:** JPEGs at 0.25 Hz (1 frame every 4 seconds)

**Implementation:**
```python
import rosbag
import cv2
from cv_bridge import CvBridge

def extract_frames_from_rosbag(
    bag_path: Path,
    output_dir: Path,
    target_fps: float = 0.25
) -> list[Path]:
    """
    Extract frames from ROS bag at specified rate.

    Args:
        bag_path: Path to .bag file
        output_dir: Where to save frames
        target_fps: Target frame rate (0.25 = 1 frame/4 sec)

    Returns:
        List of saved frame paths
    """
    bridge = CvBridge()
    bag = rosbag.Bag(str(bag_path))

    sample_interval = 1.0 / target_fps  # 4 seconds
    last_saved_time = None
    saved_frames = []

    for topic, msg, t in bag.read_messages(topics=['/camera/image']):
        current_time = t.to_sec()

        if last_saved_time is None or (current_time - last_saved_time) >= sample_interval:
            # Convert ROS image to OpenCV
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Save frame
            frame_path = output_dir / f"frame_{len(saved_frames):04d}.jpg"
            cv2.imwrite(str(frame_path), cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

            saved_frames.append(frame_path)
            last_saved_time = current_time

    bag.close()
    return saved_frames
```

---

## COLMAP Pipeline

### Why COLMAP?

✅ **Industry standard** for SfM accuracy
✅ **Proven reliability** across diverse datasets
✅ **Python bindings (pycolmap)** for easy integration
✅ **Extensive documentation** and community support

### Implementation

```python
import pycolmap
from pathlib import Path
from dataclasses import dataclass
import time

@dataclass
class ReconstructionResult:
    """Container for COLMAP reconstruction outputs."""
    success: bool
    num_registered_images: int
    num_3d_points: int
    mean_reprojection_error: float
    execution_time: float
    sparse_path: Path
    point_cloud_path: Path

def run_colmap_reconstruction(
    image_dir: Path,
    output_dir: Path,
    camera_model: str = "PINHOLE"
) -> ReconstructionResult:
    """
    Run full COLMAP incremental SfM pipeline.

    Uses pycolmap Python API:
    - https://colmap.github.io/pycolmap/pycolmap

    Args:
        image_dir: Directory with input images
        output_dir: Where to save reconstruction
        camera_model: COLMAP camera model

    Returns:
        ReconstructionResult with statistics
    """
    start_time = time.time()

    # Setup paths
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Import images
    pycolmap.import_images(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.AUTO
    )

    # Step 2: Extract SIFT features
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),
        sift_options=pycolmap.SiftExtractionOptions(
            max_num_features=8192
        )
    )

    # Step 3: Match features (exhaustive for small datasets)
    pycolmap.match_exhaustive(
        database_path=str(database_path),
        sift_options=pycolmap.SiftMatchingOptions()
    )

    # Step 4: Incremental reconstruction
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(sparse_dir),
        options=pycolmap.IncrementalPipelineOptions()
    )

    if not reconstructions:
        return ReconstructionResult(
            success=False,
            num_registered_images=0,
            num_3d_points=0,
            mean_reprojection_error=0.0,
            execution_time=time.time() - start_time,
            sparse_path=sparse_dir,
            point_cloud_path=Path("")
        )

    # Get best reconstruction (most registered images)
    best_recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())

    # Export point cloud
    point_cloud_path = output_dir / "point_cloud.ply"
    best_recon.export_PLY(str(point_cloud_path))

    # Save sparse reconstruction
    best_idx = 0
    output_sparse = sparse_dir / str(best_idx)
    best_recon.write(str(output_sparse))

    return ReconstructionResult(
        success=True,
        num_registered_images=best_recon.num_reg_images(),
        num_3d_points=best_recon.num_points3D(),
        mean_reprojection_error=best_recon.compute_mean_reprojection_error(),
        execution_time=time.time() - start_time,
        sparse_path=output_sparse,
        point_cloud_path=point_cloud_path
    )
```

---

## Multi-Visit Processing

### Core Concept

**Cumulative Reconstruction:**
- Visit 1: Process session 1 only
- Visit 2: Process sessions 1+2 together (with loop closures)
- Visit 3: Process sessions 1+2+3 together (with loop closures)
- Visit N: Process sessions 1+2+...+N together

This mimics real-world incremental mapping where the robot/camera returns to known areas.

### Implementation

```python
from pathlib import Path
from typing import List
import shutil

def combine_sessions(
    session_dirs: List[Path],
    output_dir: Path
) -> Path:
    """
    Combine multiple session frame directories into one.

    Args:
        session_dirs: List of directories with frames
        output_dir: Where to save combined frames

    Returns:
        Path to combined directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0

    for session_dir in session_dirs:
        for frame_path in sorted(session_dir.glob("*.jpg")):
            dest_path = output_dir / f"frame_{frame_count:06d}.jpg"
            shutil.copy(frame_path, dest_path)
            frame_count += 1

    return output_dir

def run_multivisit_experiment(
    session_dirs: List[Path],
    output_base: Path,
    visit_counts: List[int] = [1, 2, 3, 5]
) -> dict:
    """
    Run multi-visit reconstruction experiment.

    Args:
        session_dirs: List of session frame directories (in order)
        output_base: Base directory for outputs
        visit_counts: How many visits to test (e.g., [1, 2, 3, 5])

    Returns:
        Dict mapping visit count to ReconstructionResult
    """
    results = {}

    for n_visits in visit_counts:
        if n_visits > len(session_dirs):
            print(f"⚠️  Skipping {n_visits} visits (only {len(session_dirs)} available)")
            continue

        print(f"\n🔄 Running reconstruction with {n_visits} visit(s)...")

        # Combine first n sessions
        combined_dir = output_base / f"combined_{n_visits}_visits"
        combine_sessions(session_dirs[:n_visits], combined_dir)

        # Run COLMAP
        output_dir = output_base / f"reconstruction_{n_visits}_visits"
        result = run_colmap_reconstruction(combined_dir, output_dir)

        if result.success:
            print(f"✅ Success: {result.num_registered_images} images, "
                  f"{result.num_3d_points} points, {result.execution_time:.1f}s")
        else:
            print(f"❌ Failed after {result.execution_time:.1f}s")

        results[n_visits] = result

    return results
```

---

## Evaluation Metrics

### 1. Absolute Trajectory Error (ATE)

**Definition:** RMS difference between estimated and ground truth camera poses.

**Formula:**
```
ATE = sqrt(mean((p_estimated - p_ground_truth)²))
```

**Implementation:**
```python
import numpy as np
from scipy.spatial.transform import Rotation

def compute_ate(
    estimated_poses: dict,  # {image_id: (qvec, tvec)}
    ground_truth_poses: dict  # {image_id: (qvec, tvec)}
) -> float:
    """
    Compute Absolute Trajectory Error.

    References:
    - Sturm et al., "A Benchmark for RGB-D SLAM Evaluation", IROS 2012

    Args:
        estimated_poses: Estimated camera poses
        ground_truth_poses: Ground truth poses

    Returns:
        RMS ATE in meters
    """
    # Find common image IDs
    common_ids = set(estimated_poses.keys()) & set(ground_truth_poses.keys())

    if not common_ids:
        return float('inf')

    errors = []
    for img_id in common_ids:
        est_qvec, est_tvec = estimated_poses[img_id]
        gt_qvec, gt_tvec = ground_truth_poses[img_id]

        # Euclidean distance between positions
        position_error = np.linalg.norm(np.array(est_tvec) - np.array(gt_tvec))
        errors.append(position_error)

    ate = np.sqrt(np.mean(np.array(errors) ** 2))
    return float(ate)
```

### 2. Chamfer Distance

**Definition:** Average bidirectional distance between point clouds.

**Formula:**
```
CD = 0.5 * (mean(d(P_recon → P_gt)) + mean(d(P_gt → P_recon)))
```

**Implementation:**
```python
import open3d as o3d
import numpy as np

def compute_chamfer_distance(
    reconstruction_pcd: o3d.geometry.PointCloud,
    ground_truth_pcd: o3d.geometry.PointCloud
) -> float:
    """
    Compute Chamfer Distance between point clouds.

    References:
    - Open3D: https://github.com/isl-org/open3d

    Args:
        reconstruction_pcd: Reconstructed point cloud
        ground_truth_pcd: Ground truth point cloud

    Returns:
        Chamfer distance in meters
    """
    # Distance from recon to GT
    dists_recon_to_gt = np.asarray(
        reconstruction_pcd.compute_point_cloud_distance(ground_truth_pcd)
    )

    # Distance from GT to recon
    dists_gt_to_recon = np.asarray(
        ground_truth_pcd.compute_point_cloud_distance(reconstruction_pcd)
    )

    # Chamfer distance is average of both directions
    chamfer = 0.5 * (np.mean(dists_recon_to_gt) + np.mean(dists_gt_to_recon))

    return float(chamfer)
```

### 3. Map Completeness

**Definition:** Percentage of ground truth covered by reconstruction.

**Implementation:**
```python
def compute_completeness(
    reconstruction_pcd: o3d.geometry.PointCloud,
    ground_truth_pcd: o3d.geometry.PointCloud,
    threshold: float = 0.10  # 10 cm
) -> float:
    """
    Compute map completeness.

    Args:
        reconstruction_pcd: Reconstructed point cloud
        ground_truth_pcd: Ground truth point cloud
        threshold: Distance threshold in meters

    Returns:
        Completeness percentage (0-1)
    """
    # Distance from each GT point to nearest recon point
    dists_gt_to_recon = np.asarray(
        ground_truth_pcd.compute_point_cloud_distance(reconstruction_pcd)
    )

    # Count GT points within threshold
    completeness = np.mean(dists_gt_to_recon < threshold)

    return float(completeness)
```

---

## Visualization

### Accuracy vs. Visits Plot

```python
import matplotlib.pyplot as plt
from typing import Dict

def plot_accuracy_vs_visits(
    results: Dict[int, dict],  # {n_visits: {'ate': X, 'chamfer': Y, 'completeness': Z}}
    output_path: Path
):
    """
    Create accuracy improvement plot.

    Args:
        results: Dict mapping visit count to metrics
        output_path: Where to save plot
    """
    visits = sorted(results.keys())
    ates = [results[v]['ate'] for v in visits]
    chamfers = [results[v]['chamfer'] for v in visits]
    completeness = [results[v]['completeness'] * 100 for v in visits]  # Convert to %

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ATE plot
    axes[0].plot(visits, ates, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Visits', fontsize=12)
    axes[0].set_ylabel('ATE (meters)', fontsize=12)
    axes[0].set_title('Absolute Trajectory Error', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Chamfer Distance plot
    axes[1].plot(visits, chamfers, 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Number of Visits', fontsize=12)
    axes[1].set_ylabel('Chamfer Distance (meters)', fontsize=12)
    axes[1].set_title('Point Cloud Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Completeness plot
    axes[2].plot(visits, completeness, '^-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Number of Visits', fontsize=12)
    axes[2].set_ylabel('Completeness (%)', fontsize=12)
    axes[2].set_title('Map Completeness', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to {output_path}")
```

---

## Atomic Task Breakdown (30 Tasks)

### Phase 1: Foundation (5 tasks)

| ID | Task | Input | Output | Est. Time |
|----|------|-------|--------|-----------|
| T001 | Initialize UV project | - | pyproject.toml | 30 min |
| T002 | Setup loguru logging | - | utils.py | 1 hour |
| T003 | Create directory structure | - | Complete tree | 15 min |
| T004 | Build typer CLI | - | cli.py | 1 hour |
| T005 | YAML config loader | YAML file | Config dict | 30 min |

### Phase 2: Hilti Dataset (5 tasks)

| ID | Task | Input | Output | Est. Time |
|----|------|-------|--------|-----------|
| T006 | Download Hilti Site 1 | URL | ROS bags | 2 hours |
| T007 | Extract ground truth | GT files | Pose dict | 1 hour |
| T008 | ROS bag frame extractor | .bag, fps | JPEGs | 3 hours |
| T009 | Session splitter | Bags | Session dirs | 1 hour |
| T010 | Dataset validator | Dataset path | Pass/Fail | 1 hour |

### Phase 3: COLMAP Pipeline (5 tasks)

| ID | Task | Input | Output | Est. Time |
|----|------|-------|--------|-----------|
| T011 | COLMAP wrapper function | Images, config | ReconstructionResult | 3 hours |
| T012 | Feature extraction test | Sample images | Features in DB | 1 hour |
| T013 | Matching validation | DB | Matches | 1 hour |
| T014 | Reconstruction parser | COLMAP output | Poses + cloud | 2 hours |
| T015 | Single-session test | 1 session | Working recon | 2 hours |

### Phase 4: Multi-Visit (5 tasks)

| ID | Task | Input | Output | Est. Time |
|----|------|-------|--------|-----------|
| T016 | Session combiner | Session dirs | Combined dir | 1 hour |
| T017 | Multi-visit runner | Sessions, counts | Results dict | 2 hours |
| T018 | GT alignment | Recon, GT | Transform | 2 hours |
| T019 | Result tracker | Results | JSON file | 1 hour |
| T020 | Visit orchestrator | Config | Full experiment | 2 hours |

### Phase 5: Evaluation (5 tasks)

| ID | Task | Input | Output | Est. Time |
|----|------|-------|--------|-----------|
| T021 | ICP alignment | Source, target PCDs | Transform | 2 hours |
| T022 | ATE computation | Est poses, GT poses | ATE value | 2 hours |
| T023 | Chamfer distance | Recon PCD, GT PCD | CD value | 1.5 hours |
| T024 | Completeness metric | Recon, GT, threshold | % value | 1 hour |
| T025 | Metrics aggregator | All metrics | Summary dict | 1 hour |

### Phase 6: Visualization (3 tasks)

| ID | Task | Input | Output | Est. Time |
|----|------|-------|--------|-----------|
| T026 | Point cloud viewer | PCD path | Open3D window | 1 hour |
| T027 | Accuracy vs visits plot | Results dict | PNG plot | 2 hours |
| T028 | Metric table generator | Results | Markdown table | 1 hour |

### Phase 7: Integration (2 tasks)

| ID | Task | Input | Output | Est. Time |
|----|------|-------|--------|-----------|
| T029 | End-to-end MVP test | Hilti Site 1 | Complete results | 4 hours |
| T030 | Generate final report | Results, plots | Summary doc | 2 hours |

**Total: 30 tasks, ~50-60 hours (~2 weeks)**

---

## Technology Stack (Minimal)

```toml
[project]
name = "sfm-experiments"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # Core SfM
    "pycolmap>=0.6.0",
    "open3d>=0.18.0",

    # Computer Vision
    "opencv-python>=4.9.0",
    "numpy>=1.26.0",

    # ROS bag handling
    "rosbag>=1.16.0",
    "rospkg>=1.5.0",
    "cv-bridge>=1.16.0",

    # Scientific computing
    "scipy>=1.11.0",

    # CLI & Config
    "typer>=0.9.0",
    "pyyaml>=6.0",
    "loguru>=0.7.0",

    # Visualization
    "matplotlib>=3.8.0",

    # Progress
    "tqdm>=4.66.0",
]
```

---

## Quick Start

### 1. Setup

```bash
# Clone/create project
cd sfm_experiments

# Initialize with UV
uv init
uv add pycolmap open3d opencv-python scipy typer pyyaml loguru matplotlib tqdm

# Install ROS dependencies (if needed)
sudo apt-get install ros-noetic-rosbag ros-noetic-cv-bridge
```

### 2. Download Data

```bash
# Download Hilti SLAM 2023 Site 1
# https://hilti-challenge.com/dataset-2023.html

# Extract to datasets/hilti/rosbags/
```

### 3. Extract Frames

```bash
uv run python -m sfm_experiments.cli extract-frames \
    --dataset hilti \
    --site 1 \
    --fps 0.25 \
    --output datasets/hilti/frames
```

### 4. Run Multi-Visit Experiment

```bash
uv run python -m sfm_experiments.cli run-experiment \
    --dataset hilti \
    --site 1 \
    --visits 1 2 3 5 \
    --output results/
```

### 5. Generate Report

```bash
uv run python -m sfm_experiments.cli report \
    --results results/ \
    --output results/plots/accuracy_vs_visits.png
```

---

## Expected Results

### Typical Accuracy Improvement Curve

```
ATE (meters)
   0.50 |●                        (1 visit)
        |
   0.30 |  ●                      (2 visits: ~40% reduction)
        |
   0.20 |    ●                    (3 visits: ~60% reduction)
        |
   0.15 |       ●                 (5 visits: ~70% reduction)
        |
   0.12 |          ●__●__●        (10+ visits: asymptotic)
        |
   0.00 +─────────────────────────────────────────
          1    2    3    5    10   15   20
                  Number of Visits
```

**Key Observations:**
- **1-3 visits:** Rapid improvement (40-60% error reduction)
- **3-5 visits:** Moderate gains (10-15% per visit)
- **5+ visits:** Diminishing returns (<5% per visit)
- **Asymptote:** Around 10-15 visits depending on overlap

---

## Validation Criteria

### MVP Success = All 3 Pass:

✅ **1. Pipeline Runs End-to-End**
- Downloads and processes Hilti Site 1
- Extracts frames at 0.25 Hz
- Completes COLMAP reconstruction for 1, 2, 3, 5 visits
- No crashes or exceptions

✅ **2. Metrics Computed Successfully**
- ATE values are reasonable (0.05-0.5m range)
- Chamfer distance shows improvement with visits
- Completeness increases monotonically

✅ **3. Plot Generated**
- Clear visualization showing accuracy improvement
- All three metrics (ATE, Chamfer, Completeness) plotted
- Saved as high-res PNG

---

## Expansion Roadmap (Post-MVP)

Once MVP is validated, expand in order:

### Week 3-4: Additional Datasets
- Add TUM RGB-D (small-scale validation)
- Add NCLT (long-term multi-session)
- Compare results across datasets

### Week 5-6: Alternative Pipelines
- Add GLOMAP (fast global SfM)
- Add hloc + SuperPoint + LightGlue
- Compare pipeline performance

### Week 7-8: Advanced Metrics & Visualization
- Add F-Score at multiple thresholds
- Add Relative Pose Error (RPE)
- Create HTML interactive reports
- Add Jupyter notebooks for analysis

### Week 9+: Production Features
- Comprehensive testing suite
- CI/CD pipeline
- Documentation generation
- Example gallery

---

## References

### Documentation
- **pycolmap:** https://colmap.github.io/pycolmap/pycolmap
- **Open3D:** https://github.com/isl-org/open3d
- **Hilti SLAM 2023:** https://hilti-challenge.com/dataset-2023.html

### Key Papers
- Campos et al., "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM", IEEE T-RO 2021
- Helmberger et al., "The Hilti SLAM Challenge Dataset", IEEE RA-L 2022
- Zhang et al., "Hilti SLAM Challenge 2023", arXiv 2024

---

**END OF MVP DESIGN DOCUMENT**

*Start simple. Prove it works. Then expand.*
