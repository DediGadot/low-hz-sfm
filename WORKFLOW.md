# SfM Multi-Visit Pipeline Workflow

## Overview
This pipeline measures how 3D reconstruction accuracy improves with multiple visits to the same location. Two datasets are supported: **Hilti** (traditional, full pipeline) and **LaMAR** (fast, pre-built models).

---

## HILTI PIPELINE: Traditional Multi-Visit SfM

### 1. Data Acquisition & Frame Extraction
**Module**: `dataset.py::extract_frames_from_bag()`

- **Input**: ROS bag file (20-300 GB, 30 Hz camera data)
- **Parameters**:
  - `camera_topic`: "/alphasense/cam0/image_raw" (ROS topic to extract)
  - `target_fps`: 0.25 Hz (1 frame every 4 seconds)
  - `jpeg_quality`: 95 (compression quality 0-100)
- **Process**:
  - Open ROS bag using `rosbags` library (pure Python, no ROS installation needed)
  - Iterate through camera messages at 30 Hz
  - Sample frames at target FPS (keep every 120th frame at 0.25 Hz)
  - Decode compressed image data to RGB
  - Save as JPEG with sequential numbering (frame_0000.jpg, frame_0001.jpg, ...)
- **Output**: ~245 JPEG frames per 16-minute session
- **Time**: 10-15 minutes per session

### 2. Session Organization
**Module**: `multivisit.py::organize_sessions()`

- **Input**: Extracted frames from multiple visits
- **Structure**:
  ```
  frames/
  ├── sequence_01/  (245 frames - visit 1)
  ├── sequence_02/  (267 frames - visit 2)
  └── sequence_03/  (289 frames - visit 3)
  ```
- **Cache Key**: MD5 hash of session list + frame directory

### 3. Multi-Visit Frame Combining
**Module**: `multivisit.py::combine_sessions()`

- **Input**: Organized session directories
- **Parameters**:
  - `num_visits`: Number of sessions to combine (1, 2, 3, ..., N)
  - `use_cache`: Enable/disable caching (default: true)
- **Process**:
  - For each visit count (1 to N):
    - Create combined directory: `combined_{num_visits}_visits/`
    - Copy frames from first `num_visits` sessions
    - Rename with global sequential numbering to avoid collisions
      - Session 1, frame_0000.jpg → 0000.jpg
      - Session 2, frame_0000.jpg → 0245.jpg (offset by previous session count)
      - Session 3, frame_0000.jpg → 0512.jpg
- **Caching Logic**:
  - Check if `combined_{num_visits}_visits/` exists
  - Verify frame count matches expected total
  - If valid, skip copying (50x speedup)
- **Output**: N combined directories with cumulative frames
  - 1 visit: 245 frames
  - 2 visits: 512 frames (245 + 267)
  - 3 visits: 801 frames (245 + 267 + 289)

### 4. SfM Reconstruction (Per Visit Count)
**Module**: `colmap_runner.py::run_sfm_reconstruction()` OR `glomap_wrapper.py::run_glomap_reconstruction()`

#### 4a. COLMAP Pipeline (Default - Robust, Slower)
- **Input**: Combined frame directory
- **Parameters** (from `configs/hilti.yaml`):
  - **Camera Model**: "SIMPLE_RADIAL" (fx, cx, cy, k1 distortion)
  - **Max Features**: 8192 SIFT features per image
  - **GPU Index**: 0 (use -1 for CPU fallback)
  - **Initialization Thresholds** (ultra-aggressive for challenging data):
    - `init_min_num_inliers`: 6 (standard: 15-30)
    - `abs_pose_min_num_inliers`: 5
    - `abs_pose_min_inlier_ratio`: 0.05 (5% inliers required)
- **Process Steps**:
  1. **Feature Extraction**:
     - Run SIFT on all images (GPU accelerated)
     - Extract up to 8192 keypoints per image
     - Compute 128-dimensional descriptors
  2. **Feature Matching**:
     - Use exhaustive matcher (all image pairs for small datasets)
     - SNN ratio: 0.8 (Lowe's ratio test)
     - Geometric verification with RANSAC
  3. **Incremental Reconstruction**:
     - Find best image pair for initialization (most matches, good baseline)
     - Triangulate initial points
     - Incrementally register remaining images
     - Bundle adjustment after every N images
     - Retriangulate and filter outliers
  4. **Output Files** in `sparse/0/`:
     - `cameras.bin`: Camera intrinsics
     - `images.bin`: Registered image poses + 2D-3D correspondences
     - `points3D.bin`: 3D point cloud with color, error, track length
- **Caching Logic**:
  - Check if `{workspace}/sparse/0/` exists
  - Verify reconstruction validity (registered images > 0)
  - If valid, skip reconstruction (200-600x speedup)
- **Time**: 20-60 seconds per visit count

#### 4b. GLOMAP Pipeline (Alternative - Fast, Less Robust)
- **Input**: COLMAP features + matches (reuse from 4a)
- **Parameters**:
  - `max_epipolar_error`: 2.0 pixels (increase to 4.0-10.0 for blurry images)
  - `max_num_tracks`: null (unlimited, set to 1000 for speed)
  - `skip_retriangulation`: false (set true for faster, less accurate results)
- **Process**:
  1. Load COLMAP database with features/matches
  2. **Global positioning** (solve all camera poses simultaneously)
  3. Triangulate points
  4. Global bundle adjustment
- **Speedup**: 10-100x faster than COLMAP (0.5-2 seconds vs 20-60 seconds)
- **Trade-off**: Less robust with sparse matches or extreme viewpoints

### 5. Metrics Computation
**Module**: `metrics.py::compute_metrics()`

- **Input**:
  - Reconstructed model (`sparse/0/`)
  - Ground truth poses (from `/tf` topic in ROS bag)
- **Parameters**:
  - `alignment_method`: "similarity" (Sim3 transform: translation, rotation, scale)
  - `completeness_threshold`: 0.10 meters (10 cm distance for coverage)
- **Process**:
  1. **Load Ground Truth**:
     - Extract camera poses from ROS bag `/tf` topic
     - Match to reconstructed image timestamps
  2. **Alignment** (Procrustes/Sim3):
     - Pair reconstructed poses with ground truth
     - Compute optimal 7-DoF transform (3 translation, 3 rotation, 1 scale)
     - Apply transform to reconstructed poses and 3D points
  3. **ATE (Absolute Trajectory Error)**:
     - Compute Euclidean distance between aligned poses and GT
     - RMS error across all registered images
     - Formula: `sqrt(mean((x_est - x_gt)^2 + (y_est - y_gt)^2 + (z_est - z_gt)^2))`
  4. **Chamfer Distance** (Point Cloud Similarity):
     - For each reconstructed point, find nearest GT point (forward distance)
     - For each GT point, find nearest reconstructed point (backward distance)
     - Chamfer = mean(forward) + mean(backward)
     - Uses KD-Tree for efficient nearest neighbor search
  5. **Map Completeness**:
     - For each GT point, check if reconstructed point within threshold (10 cm)
     - Completeness = (covered GT points) / (total GT points) × 100%
- **Output**: Dictionary with ATE, Chamfer, Completeness per visit count

### 6. Visualization & Reporting
**Module**: `visualization.py::plot_accuracy_vs_visits()`

- **Input**: Metrics dictionary from all visit counts
- **Process**:
  1. **Accuracy vs. Visits Plot** (3 subplots):
     - Subplot 1: ATE vs. number of visits
     - Subplot 2: Chamfer distance vs. number of visits
     - Subplot 3: Completeness % vs. number of visits
     - X-axis: Number of visits (1, 2, 3, ..., N)
     - Y-axis: Metric value
     - Show improvement curve with markers
  2. **Results Table** (Markdown):
     - Columns: Visits | ATE (m) | Chamfer (m) | Completeness (%)
     - Sort by visit count
  3. **Point Cloud Export** (optional):
     - Export aligned reconstruction as PLY file
     - Include RGB color, confidence scores
- **Output**:
  - `results/plots/accuracy_vs_visits.png`
  - `results/summary.md`
  - `results/models/reconstruction_{N}_visits.ply`

---

## LAMAR PIPELINE: Fast Pre-Built Model Analysis

### 1. Download Pre-Built Reconstructions
**Script**: `scripts/download_lamar_dataset.py`

- **Input**: Scene selection (CAB, HGE, LIN, etc.)
- **Parameters**:
  - `base_url`: "https://cvg-data.inf.ethz.ch/lamar_dataset/benchmark/"
  - `scenes`: List of scene names to download
  - `download_sessions`: false (skip raw images, only get COLMAP models)
- **Process**:
  - Download `{scene}_colmap_models.zip` (5-20 GB per scene)
  - Extract to `datasets/lamar/colmap/{scene}/`
  - Verify structure: `sparse/0/` with cameras.bin, images.bin, points3D.bin
- **Time**: ~30 minutes for 3 scenes (18 GB total at 100 Mbps)

### 2. Load Reconstructions
**Module**: `lamar_handler.py::load_lamar_reconstruction()`

- **Input**: Path to `sparse/0/` directory
- **Process**:
  - Load COLMAP model using `pycolmap.Reconstruction(sparse_path)`
  - Extract reconstruction statistics:
    - Number of registered images
    - Number of 3D points
    - Camera models and intrinsics
    - Scene bounding box (min/max XYZ)
- **Time**: <1 second per scene

### 3. Statistics Extraction
**Module**: `lamar_handler.py::extract_lamar_statistics()`

- **Input**: Loaded pycolmap.Reconstruction object
- **Parameters**:
  - `max_sample_points`: 100,000 (limit for statistical analysis)
- **Process**:
  1. **Point Cloud Analysis**:
     - Sample up to 100K points for stats (avoid memory issues)
     - Extract per-point data:
       - 3D position (X, Y, Z)
       - RGB color
       - Track length (number of images observing this point)
       - Reprojection error (pixel error across observations)
     - Compute statistics:
       - Mean/median/P95 track length
       - Mean/median/P95 reprojection error
       - Color distribution histogram
  2. **Camera Analysis**:
     - Extract all camera poses (rotation matrices, translation vectors)
     - Convert to positions (camera centers)
     - Extract timestamps from image filenames
     - Compute spatial metrics:
       - Bounding box (min/max X/Y/Z)
       - Scene volume (XYZ range product)
       - Trajectory length (sum of inter-frame distances)
  3. **Temporal Analysis**:
     - Parse timestamps from filenames (YYYY-MM-DD format)
     - Compute capture duration (first to last image)
     - Compute average capture frequency (images per day)
- **Output**: Dictionary with comprehensive scene statistics

### 4. Visualization Generation
**Module**: `lamar_visualization.py::generate_lamar_visualizations()`

- **Input**: Reconstruction + statistics
- **Parameters**:
  - `max_points`: 10,000 (points for 3D plot, browser rendering limit)
  - `grid_size`: 100 (density heatmap resolution 100×100)
  - `sample_rate`: 10 (show every Nth camera in 3D trajectory)
- **Process**:

  #### 4a. Point Cloud Visualizations (via `point_cloud_viz.py`)
  1. **3D Scatter Plot**:
     - Sample 10K points uniformly from full cloud
     - Create Plotly 3D scatter with RGB colors
     - Add axes, grid, camera controls
  2. **Density Heatmap**:
     - Project points to 2D (top-down XY view)
     - Create 100×100 grid
     - Count points per cell
     - Render as heatmap with colorscale
  3. **Quality Histograms**:
     - Track length distribution (bins: 0-2, 2-5, 5-10, 10-20, 20+)
     - Reprojection error distribution (bins: 0-0.5, 0.5-1.0, 1.0-2.0, 2.0+)

  #### 4b. Camera Visualizations (via `camera_viz.py`)
  1. **3D Trajectory Plot**:
     - Plot camera positions as scatter points
     - Connect sequential positions with lines
     - Sample every Nth camera (reduce clutter)
     - Add coordinate axes and grid
  2. **Top-Down Map**:
     - Project camera positions to XY plane
     - Show movement pattern from bird's eye view
     - Color by time or trajectory segment
  3. **Multi-View Distribution**:
     - Subplot 1: X vs Y positions
     - Subplot 2: X vs Z positions
     - Subplot 3: Y vs Z positions

  #### 4c. Cross-Scene Comparisons
  1. **Bar Chart Comparisons**:
     - Number of images per scene
     - Number of points per scene
     - Mean track length per scene
     - Mean reprojection error per scene
  2. **Scatter Plots**:
     - Images vs. Points (scene size correlation)
     - Track length vs. Reprojection error (quality trade-off)
- **Output**: Individual HTML files per visualization (10-15 files per scene)
- **Time**: 30-60 seconds per scene

### 5. HTML Dashboard Generation
**Module**: `html_report_generator.py::generate_comprehensive_report()`

- **Input**:
  - Statistics dictionary for all scenes
  - Paths to individual visualization HTML files
- **Parameters**:
  - `template_path`: Path to Jinja2 template
  - `output_path`: Where to save final dashboard
- **Process**:
  1. **Extract Plotly Content**:
     - Read each visualization HTML file
     - Use regex to extract Plotly div and script sections
     - Strip out redundant Plotly.js includes (use single CDN link)
  2. **Prepare Summary Table**:
     - Convert statistics dict to pandas DataFrame
     - Compute derived metrics (density = points/images)
     - Format numbers (commas, decimal places)
  3. **Render Template**:
     - Load `comprehensive_report.html.jinja2`
     - Inject summary table, embedded visualizations
     - Add expandable sections for each scene
     - Include navigation menu, statistics cards
  4. **Optimize File Size**:
     - Inline all visualizations (single-file report)
     - Compress Plotly JSON data
     - Minimize whitespace
- **Output**: `comprehensive_dashboard.html` (5-10 MB)
- **Time**: 1-5 seconds

---

## KEY PARAMETERS SUMMARY

### Performance Trade-offs
| Parameter | Low Value | High Value | Impact |
|-----------|-----------|------------|---------|
| `target_fps` | 0.1 Hz | 1.0 Hz | Faster extraction, sparser data vs. Denser data, slower processing |
| `max_num_features` | 2048 | 16384 | Faster matching, fewer correspondences vs. More robust, slower |
| `init_min_num_inliers` | 15 (standard) | 6 (aggressive) | Fails on hard data vs. More initializations, lower quality |
| `max_points` (viz) | 1000 | 50000 | Fast rendering, less detail vs. Detailed, slow browser |
| `completeness_threshold` | 0.05 m | 0.20 m | Strict coverage metric vs. Lenient coverage metric |

### Caching Control
- **Enable caching**: 100-600x speedup, reuse existing results
- **Disable caching** (`--no-cache`): Fresh computation, verify parameter changes

### Mapper Selection
- **COLMAP**: Incremental, robust, handles difficult data, slower (20-60s)
- **GLOMAP**: Global, fast (0.5-2s), requires good initialization, less robust

### Quality vs. Speed
- **High Quality**: COLMAP mapper, 8192 features, no track limit, 1.0 FPS extraction
- **Fast Prototyping**: GLOMAP mapper, 4096 features, track limit 1000, 0.1 FPS extraction
- **LaMAR**: Skip extraction/reconstruction entirely, analyze pre-built models (<2 seconds)

---

## EXPECTED RESULTS

### Accuracy Improvement Curve
```
Visits  | ATE (m) | Chamfer (m) | Completeness (%)
--------|---------|-------------|------------------
1       | 0.45    | 0.18        | 62%
2       | 0.28    | 0.12        | 75%              (↓38% error)
3       | 0.19    | 0.08        | 84%              (↓58% error)
5       | 0.14    | 0.06        | 89%              (↓69% error)
10      | 0.11    | 0.05        | 93%              (↓76% error)
```

### Diminishing Returns
- **1-3 visits**: Rapid improvement (20-40% error reduction per visit)
- **3-5 visits**: Moderate gains (10-15% per visit)
- **5+ visits**: Asymptotic (<5% per visit, approaching physical sensor limits)

---

## ERROR HANDLING & FALLBACKS

### Reconstruction Failures
1. **Check initialization quality**: Increase `init_min_num_inliers` if too many failed attempts
2. **Try GLOMAP**: May succeed where COLMAP incremental fails
3. **Verify feature matches**: Run `scripts/analyze_matches.py` to diagnose
4. **Adjust FPS**: Increase to 0.5 Hz for denser temporal sampling

### GPU Unavailable
- Automatic CPU fallback (slower but functional)
- Set `gpu_index: -1` in config to force CPU

### Cache Corruption
- Delete cached directories and rerun
- Use `--no-cache` flag to bypass

### Memory Issues (Large Scenes)
- Reduce `max_sample_points` in statistics extraction
- Reduce `max_points` in visualizations
- Process scenes individually instead of batch
