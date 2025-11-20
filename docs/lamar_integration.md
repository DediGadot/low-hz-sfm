# LaMAR Dataset Integration Guide

## Overview

This document describes how to download, configure, and use the LaMAR (Localization and Mapping for Augmented Reality) dataset with the SfM multi-visit experimentation pipeline.

**Dataset Information:**
- **Source:** https://cvg-data.inf.ethz.ch/lamar/
- **GitHub:** https://github.com/microsoft/lamar-benchmark
- **License:** CC BY-SA 4.0
- **Provider:** ETH Zurich Computer Vision and Geometry Lab
- **Version:** 2.2 (Released: September 28, 2023)

## Dataset Description

### Scenes

LaMAR includes three indoor scenes:
- **CAB** - Indoor office/lab environment
- **HGE** - Large indoor space
- **LIN** - Multi-floor indoor environment

### Data Types

1. **Benchmark Data** (~19.8 GB total)
   - Query images for localization testing
   - Camera poses for evaluation
   - Used for testing localization algorithms

2. **COLMAP Reconstructions** (~34 GB total)
   - Pre-built Structure-from-Motion models
   - Sparse 3D point clouds
   - Camera poses and intrinsic parameters
   - Ready to use without reconstruction

3. **Raw Sensor Data** (150-300+ GB, not included in standard download)
   - HoloLens 2 recordings (RGB + depth + IMU)
   - iPhone recordings (RGB)
   - NavVis M6 laser scans (ground truth)
   - Available at: https://cvg-data.inf.ethz.ch/lamar/raw/

### Capture Devices

- **HoloLens 2** - Mixed reality headset with RGB and depth cameras
- **iPhone** - Mobile phone cameras
- **NavVis M6** - Professional laser scanning system (for ground truth)

## Installation & Download

### Step 1: Download the Dataset

Use the provided download script to get the LaMAR dataset:

```bash
# Interactive download with menu
uv run python scripts/download_lamar_dataset.py
```

**Download Options:**

1. **Benchmark data only** (19.8 GB) - For localization testing
2. **COLMAP reconstructions only** (34 GB) - For using pre-built models
3. **Benchmark + COLMAP** (53.8 GB) - Recommended for full functionality
4. **Single scene** (CAB, HGE, or LIN) - Start with one scene (~18 GB each)
5. **Custom selection** - Choose specific files

**Quick Start Recommendation:**
```bash
# Option 4: Download single scene (CAB) with benchmark + COLMAP
# Select option 4 when prompted
uv run python scripts/download_lamar_dataset.py
```

### Step 2: Verify Download

After downloading, your directory structure should look like:

```
datasets/lamar/
├── benchmark/
│   ├── CAB/
│   ├── HGE/
│   └── LIN/
└── colmap/
    ├── CAB/
    │   └── sparse/
    │       └── 0/
    │           ├── cameras.bin
    │           ├── images.bin
    │           └── points3D.bin
    ├── HGE/
    └── LIN/
```

### Step 3: Validate the Dataset

Test that the dataset is properly loaded:

```bash
uv run python -m sfm_experiments.lamar_handler
```

Expected output:
```
LaMAR Dataset Handler Validation
================================================================================

Test 1: Validating dataset structure...
✅ Dataset structure is valid

Test 2: Loading COLMAP reconstruction...
✅ Loaded reconstruction with XXX images

Test 3: Extracting camera parameters...
✅ Extracted X camera(s)

Test 4: Exporting images list...
✅ Exported XXX images

Test 5: Getting scene information...
✅ Scene info retrieved

================================================================================
✅ VALIDATION PASSED - All 5 tests produced expected results
```

## Usage with Pipeline

### Option 1: Use Pre-built COLMAP Reconstructions

The LaMAR dataset comes with pre-built COLMAP reconstructions, which you can use directly without running reconstruction:

```python
from pathlib import Path
from sfm_experiments.lamar_handler import load_lamar_reconstruction, get_lamar_camera_params

# Load pre-built reconstruction
scene = "CAB"
colmap_path = Path("datasets/lamar/colmap") / scene
reconstruction = load_lamar_reconstruction(colmap_path)

# Get camera parameters
camera_params = get_lamar_camera_params(reconstruction)

# Access reconstruction data
print(f"Images: {len(reconstruction.images)}")
print(f"Cameras: {len(reconstruction.cameras)}")
print(f"3D Points: {len(reconstruction.points3D)}")

# Iterate through images
for img_id, image in reconstruction.images.items():
    print(f"Image: {image.name}")
    print(f"  Camera: {image.camera_id}")
    print(f"  Position: {image.tvec}")
    print(f"  Rotation: {image.qvec}")
```

### Option 2: Analyze Multiple Scenes

Compare reconstructions across different scenes:

```python
from pathlib import Path
from sfm_experiments.lamar_handler import list_lamar_scenes

base_dir = Path("datasets/lamar")
scenes = list_lamar_scenes(base_dir)

for scene in scenes:
    print(f"\nScene: {scene.name}")
    print(f"  Images: {scene.num_images}")
    print(f"  Cameras: {scene.num_cameras}")
    print(f"  3D Points: {scene.num_points3d}")
```

### Option 3: Export Data for Analysis

Export image lists and camera parameters:

```python
from pathlib import Path
from sfm_experiments.lamar_handler import (
    load_lamar_reconstruction,
    export_lamar_images_list,
    get_lamar_camera_params
)

scene = "CAB"
colmap_path = Path("datasets/lamar/colmap") / scene
reconstruction = load_lamar_reconstruction(colmap_path)

# Export images list
export_lamar_images_list(reconstruction, Path("lamar_cab_images.txt"))

# Get camera parameters
camera_params = get_lamar_camera_params(reconstruction)
```

### Option 4: Integration with Multi-Visit Pipeline

While LaMAR doesn't use ROS bags like Hilti, you can treat different scenes as "visits" for multi-visit experiments:

```python
from pathlib import Path
from sfm_experiments.config import Config
from sfm_experiments.lamar_handler import load_lamar_reconstruction
from sfm_experiments.metrics import compute_chamfer_distance

# Load configuration
config = Config.from_yaml("configs/lamar.yaml")

# Load reconstructions from different scenes
scenes = ["CAB", "HGE", "LIN"]
reconstructions = {}

for scene in scenes:
    colmap_path = Path(config.dataset.colmap_dir) / scene
    reconstructions[scene] = load_lamar_reconstruction(colmap_path)

# Compare point clouds between scenes
# (This is just an example - you'd need to align them first)
```

## Configuration

The LaMAR dataset configuration is in `configs/lamar.yaml`. Key settings:

```yaml
dataset:
  name: lamar
  type: colmap
  base_dir: datasets/lamar
  scenes: [CAB, HGE, LIN]
  default_scene: CAB

mapper:
  type: colmap
  colmap:
    use_prebuilt: true  # Use pre-built reconstructions
```

## Data Format Details

### COLMAP Reconstruction Format

Each scene's COLMAP reconstruction includes:

**cameras.bin** - Camera intrinsic parameters
- Format: Binary COLMAP cameras file
- Contains: Camera model, focal length, principal point, distortion parameters

**images.bin** - Registered images with poses
- Format: Binary COLMAP images file
- Contains: Image name, camera ID, quaternion rotation, translation vector, 2D keypoints

**points3D.bin** - Sparse 3D point cloud
- Format: Binary COLMAP points3D file
- Contains: 3D coordinates, RGB color, error, track information

### Metadata Format (Raw Data)

If you download raw data, each scene includes:

**metadata_hololens.json** - HoloLens session information
```json
{
  "sessions": {
    "session_id": {
      "timestamp": "2021-06-15T10:30:00",
      "duration": 120,
      "num_frames": 3600
    }
  }
}
```

**metadata_phone.json** - iPhone session information
- Similar format to HoloLens metadata

## Common Tasks

### Task 1: Inspect a Scene

```bash
# Validate and inspect CAB scene
uv run python -m sfm_experiments.lamar_handler
```

### Task 2: Export Camera Poses

```python
from pathlib import Path
from sfm_experiments.lamar_handler import load_lamar_reconstruction

reconstruction = load_lamar_reconstruction(Path("datasets/lamar/colmap/CAB"))

# Export to TUM format (timestamp tx ty tz qx qy qz qw)
with open("cab_poses.txt", 'w') as f:
    for img_id, image in reconstruction.images.items():
        # Use image_id as timestamp
        timestamp = img_id
        tx, ty, tz = image.tvec
        qw, qx, qy, qz = image.qvec  # COLMAP uses qw, qx, qy, qz order
        f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
```

### Task 3: Visualize Point Cloud

```python
import pycolmap
import open3d as o3d
from pathlib import Path
from sfm_experiments.lamar_handler import load_lamar_reconstruction

# Load reconstruction
reconstruction = load_lamar_reconstruction(Path("datasets/lamar/colmap/CAB"))

# Export to PLY
output_path = Path("cab_pointcloud.ply")
pycolmap.write_model(reconstruction, str(output_path.parent), ext=".ply")

# Visualize with Open3D
pcd = o3d.io.read_point_cloud(str(output_path))
o3d.visualization.draw_geometries([pcd])
```

### Task 4: Extract Image Paths

```python
from pathlib import Path
from sfm_experiments.lamar_handler import load_lamar_reconstruction

reconstruction = load_lamar_reconstruction(Path("datasets/lamar/colmap/CAB"))

# List all image names
image_names = [img.name for img in reconstruction.images.values()]
print(f"Found {len(image_names)} images:")
for name in image_names[:10]:  # Print first 10
    print(f"  - {name}")
```

## Differences from Hilti Dataset

| Feature | Hilti Dataset | LaMAR Dataset |
|---------|--------------|---------------|
| **Data Format** | ROS bags | Pre-built COLMAP models |
| **Frame Extraction** | Required (from ROS bags) | Not needed (images in COLMAP) |
| **Reconstruction** | Must be built | Pre-built available |
| **Ground Truth** | GCP markers | NavVis laser scans |
| **Environment** | Construction sites | Indoor office/lab spaces |
| **Capture Device** | Alphasense 5-camera rig | HoloLens 2, iPhone, NavVis |
| **Multi-Visit** | Explicit multi-visit sequences | Different scenes/sessions |

## Troubleshooting

### Issue: "COLMAP path does not exist"

**Solution:** Make sure you've downloaded and extracted the dataset:
```bash
uv run python scripts/download_lamar_dataset.py
# Select "Extract zip files after download? yes"
```

### Issue: "No sparse reconstruction found"

**Solution:** Check that the COLMAP data was properly extracted:
```bash
ls -la datasets/lamar/colmap/CAB/sparse/
# Should show: 0/ directory containing cameras.bin, images.bin, points3D.bin
```

### Issue: "Failed to load reconstruction"

**Solution:** Verify the COLMAP binary files are not corrupted:
```bash
# Re-download the scene
rm -rf datasets/lamar/colmap/CAB
uv run python scripts/download_lamar_dataset.py
# Select the specific scene again
```

### Issue: Raw sensor data not available

**Note:** Raw sensor data is very large (150-300+ GB) and not included in the standard download script. To download raw data:

1. Visit https://cvg-data.inf.ethz.ch/lamar/raw/
2. Download sessions manually using wget or curl
3. Extract to `datasets/lamar/raw/`

## Performance Notes

- **Pre-built COLMAP models** load very quickly (<1 second)
- **No reconstruction needed** - saves hours compared to Hilti
- **Smaller dataset** - 53.8 GB total vs Hilti's 328 GB
- **Good for testing** - Fast iteration and experimentation

## Citation

If you use the LaMAR dataset in your research, please cite:

```bibtex
@inproceedings{lamar2022,
  title={LaMAR: Benchmarking Localization and Mapping for Augmented Reality},
  author={...},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

Check the [GitHub repository](https://github.com/microsoft/lamar-benchmark) for the complete citation.

## Next Steps

1. **Download a scene** - Start with CAB (smallest, good for testing)
2. **Validate the data** - Run the validation script
3. **Explore the COLMAP model** - Use the handler functions
4. **Integrate with your pipeline** - Use pre-built reconstructions
5. **Experiment** - Compare scenes, analyze camera poses, visualize point clouds

## Additional Resources

- **Dataset Homepage:** https://cvg-data.inf.ethz.ch/lamar/
- **GitHub Repository:** https://github.com/microsoft/lamar-benchmark
- **COLMAP Documentation:** https://colmap.github.io/
- **PyColmap Documentation:** https://github.com/colmap/pycolmap
- **Paper:** Check GitHub repository for publication details

## License

The LaMAR dataset is licensed under CC BY-SA 4.0. You must:
- ✅ Give appropriate credit
- ✅ Provide a link to the license
- ✅ Indicate if changes were made
- ✅ License derivative works under the same license

See the LICENSE file in the dataset for full details.
