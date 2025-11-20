# LaMAR Pipeline Integration - Complete Guide

## Overview

The SfM Multi-Visit Experimentation Pipeline now supports **both** the Hilti SLAM Challenge 2023 dataset and the LaMAR (Localization and Mapping for Augmented Reality) dataset from ETH Zurich.

**Implementation Date:** 2025-11-17
**Pipeline Version:** 0.1.0

## What's New

### New Commands

Two new CLI commands for LaMAR dataset:

1. **`lamar-info`** - Display LaMAR dataset information
2. **`lamar-experiment`** - Run multi-scene analysis experiment

### New Modules

Three new Python modules:

1. **`lamar_handler.py`** (389 lines) - Data loading and parsing
2. **`lamar_experiment.py`** (408 lines) - Multi-scene experiment orchestration
3. Updated **`cli.py`** - Integrated LaMAR commands

### New Files

1. **`scripts/download_lamar_dataset.py`** - Interactive download utility
2. **`configs/lamar.yaml`** - LaMAR configuration
3. **`docs/lamar_integration.md`** - User guide
4. **`docs/LAMAR_IMPLEMENTATION_SUMMARY.md`** - Technical summary

## Quick Start

### Option 1: LaMAR Dataset (Fast Setup)

```bash
# Step 1: Download LaMAR dataset (recommended: single scene for testing)
uv run python scripts/download_lamar_dataset.py
# Select option 4 (CAB scene, 17.9 GB)

# Step 2: View dataset information
uv run python -m sfm_experiments.cli lamar-info

# Step 3: Run multi-scene experiment
uv run python -m sfm_experiments.cli lamar-experiment

# Step 4: Check results
cat results/lamar/experiment_summary.txt
```

**Time estimate:** ~30 minutes download + <1 minute processing

### Option 2: Hilti Dataset (Traditional Multi-Visit)

```bash
# Step 1: Download Hilti dataset
uv run python scripts/download_hilti_dataset.py
# Select option 2 (Site 1 first 3, 58.2 GB)

# Step 2: Extract frames (run in parallel for speed)
uv run python -m sfm_experiments.cli extract-frames \
    datasets/hilti/rosbags/site1_handheld_1.bag \
    datasets/hilti/frames/sequence_01 --fps 0.25

# Repeat for sequence_02 and sequence_03...

# Step 3: Run multi-visit experiment
uv run python -m sfm_experiments.cli run-experiment \
    --config-file configs/hilti.yaml \
    --visits "1,2,3"

# Step 4: Check results
cat results/results_summary.md
open results/plots/accuracy_vs_visits.png
```

**Time estimate:** ~2 hours download + ~30 minutes frame extraction + ~3 minutes reconstruction

## Complete CLI Reference

### General Commands

#### `info`
Display pipeline information and quick start guide.

```bash
uv run python -m sfm_experiments.cli info
```

### Hilti Dataset Commands

#### `extract-frames`
Extract JPEG frames from ROS bag files.

```bash
uv run python -m sfm_experiments.cli extract-frames \
    <bag_path> \
    <output_dir> \
    [--camera-topic <topic>] \
    [--fps <fps>] \
    [--quality <quality>] \
    [--verbose]
```

**Parameters:**
- `bag_path`: Path to ROS bag file
- `output_dir`: Output directory for frames
- `--camera-topic`: ROS camera topic (default: `/camera/image_raw`)
- `--fps`: Target frame rate (default: 0.25)
- `--quality`: JPEG quality 0-100 (default: 95)
- `--verbose`: Enable verbose logging

**Example:**
```bash
uv run python -m sfm_experiments.cli extract-frames \
    datasets/hilti/rosbags/site1_handheld_1.bag \
    datasets/hilti/frames/sequence_01 \
    --fps 0.25 --quality 95
```

#### `run-colmap`
Run COLMAP reconstruction on images.

```bash
uv run python -m sfm_experiments.cli run-colmap \
    <image_dir> \
    <output_dir> \
    [--camera-model <model>] \
    [--max-features <num>] \
    [--verbose]
```

**Parameters:**
- `image_dir`: Directory with input images
- `output_dir`: Output directory for reconstruction
- `--camera-model`: COLMAP camera model (default: `SIMPLE_RADIAL`)
- `--max-features`: Maximum SIFT features (default: 8192)
- `--verbose`: Enable verbose logging

**Example:**
```bash
uv run python -m sfm_experiments.cli run-colmap \
    datasets/hilti/frames/sequence_01 \
    results/reconstruction_01
```

#### `run-experiment`
Run complete multi-visit experiment.

```bash
uv run python -m sfm_experiments.cli run-experiment \
    [--config-file <path>] \
    [--output-dir <dir>] \
    [--visits <counts>] \
    [--mapper <type>] \
    [--no-cache] \
    [--verbose]
```

**Parameters:**
- `--config-file`: Path to configuration file (default: `configs/hilti.yaml`)
- `--output-dir`: Output directory (default: `results`)
- `--visits`: Comma-separated visit counts (e.g., `"1,2,3"`)
- `--mapper`: SfM mapper: `colmap` or `glomap` (default: from config)
- `--no-cache`: Disable cache and re-run all steps
- `--verbose`: Enable verbose logging

**Example:**
```bash
# Default - uses config file settings
uv run python -m sfm_experiments.cli run-experiment

# Custom visit counts
uv run python -m sfm_experiments.cli run-experiment --visits "1,2,3,5"

# Use GLOMAP for speed
uv run python -m sfm_experiments.cli run-experiment --mapper glomap

# Force re-run without cache
uv run python -m sfm_experiments.cli run-experiment --no-cache
```

### LaMAR Dataset Commands

#### `lamar-info`
Display LaMAR dataset information.

```bash
uv run python -m sfm_experiments.cli lamar-info \
    [--base-dir <dir>] \
    [--verbose]
```

**Parameters:**
- `--base-dir`: LaMAR dataset base directory (default: `datasets/lamar`)
- `--verbose`: Show detailed scene information

**Example:**
```bash
# Basic info
uv run python -m sfm_experiments.cli lamar-info

# Detailed info
uv run python -m sfm_experiments.cli lamar-info --verbose

# Custom dataset location
uv run python -m sfm_experiments.cli lamar-info --base-dir /path/to/lamar
```

**Output:**
```
LaMAR Dataset Information
================================================================================

┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ Scene ┃ Images ┃ 3D Points ┃ Cameras ┃ COLMAP ┃ Benchmark ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│ CAB   │   1234 │    456789 │       8 │   ✅   │     ✅    │
│ HGE   │   2345 │    678901 │      12 │   ✅   │     ✅    │
│ LIN   │   3456 │    789012 │      15 │   ✅   │     ✅    │
└───────┴────────┴───────────┴─────────┴────────┴───────────┘

Summary
--------------------------------------------------------------------------------
Scenes               : 3
Total Images         : 7035
Total 3D Points      : 1924702
Dataset Path         : datasets/lamar
```

#### `lamar-experiment`
Run multi-scene analysis experiment.

```bash
uv run python -m sfm_experiments.cli lamar-experiment \
    [--config-file <path>] \
    [--output-dir <dir>] \
    [--scenes <names>] \
    [--verbose]
```

**Parameters:**
- `--config-file`: Path to LaMAR config file (default: `configs/lamar.yaml`)
- `--output-dir`: Output directory (default: `results/lamar`)
- `--scenes`: Comma-separated scene names (e.g., `"CAB,HGE"`)
- `--verbose`: Enable verbose logging

**Example:**
```bash
# Analyze all scenes
uv run python -m sfm_experiments.cli lamar-experiment

# Analyze specific scenes
uv run python -m sfm_experiments.cli lamar-experiment --scenes "CAB,HGE"

# Custom output directory
uv run python -m sfm_experiments.cli lamar-experiment \
    --output-dir results/my_lamar_test

# Verbose mode
uv run python -m sfm_experiments.cli lamar-experiment --verbose
```

**Output:**
```
LaMAR Multi-Scene Analysis
================================================================================

Running LaMAR Multi-Scene Experiment
Scenes: CAB, HGE, LIN
Output: results/lamar
================================================================================

Analyzing LaMAR scene: CAB
✅ CAB: 1234 images, 456789 points (0.45s)

Analyzing LaMAR scene: HGE
✅ HGE: 2345 images, 678901 points (0.62s)

Analyzing LaMAR scene: LIN
✅ LIN: 3456 images, 789012 points (0.78s)

================================================================================
Experiment Summary: 3/3 scenes loaded successfully
================================================================================

Experiment Summary
--------------------------------------------------------------------------------
Total Scenes         : 3
Successful           : 3
Failed               : 0
Total Images         : 7035
Total 3D Points      : 1924702
Execution Time       : 1.85s

┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ Scene ┃ Images ┃ 3D Points ┃ Cameras ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ CAB   │   1234 │    456789 │       8 │
│ HGE   │   2345 │    678901 │      12 │
│ LIN   │   3456 │    789012 │      15 │
└───────┴────────┴───────────┴─────────┘

✅ LaMAR experiment complete! Results saved to results/lamar
```

## Dataset Comparison

| Feature | Hilti | LaMAR |
|---------|-------|-------|
| **Dataset Type** | ROS bags with sensor data | Pre-built COLMAP reconstructions |
| **Download Size** | 328 GB (all), 58 GB (recommended) | 54 GB (all), 18 GB (single scene) |
| **Setup Time** | ~3 hours | ~30 minutes |
| **Processing** | Extract frames + reconstruct | Load pre-built models |
| **Experiment Type** | Multi-visit SfM | Multi-scene analysis |
| **Environment** | Construction sites | Indoor office/lab |
| **Capture Device** | Alphasense 5-camera rig | HoloLens 2, iPhone, NavVis |
| **Ground Truth** | GCP markers | NavVis laser scans |
| **Use Case** | Investigate multi-visit improvements | Compare scenes, localization |
| **CLI Commands** | `extract-frames`, `run-experiment` | `lamar-info`, `lamar-experiment` |

## Workflow Diagrams

### Hilti Workflow

```
┌─────────────────┐
│  Download ROS   │  scripts/download_hilti_dataset.py
│      Bags       │  (58-328 GB)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract Frames  │  cli extract-frames
│   (0.25 FPS)    │  (~5-10 min per bag)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Run Multi-    │  cli run-experiment
│     Visit       │  (20-60 min total)
│   Experiment    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Metrics &      │  Automatic
│ Visualization   │
└─────────────────┘
```

### LaMAR Workflow

```
┌─────────────────┐
│  Download COLMAP│  scripts/download_lamar_dataset.py
│ Reconstructions │  (18-54 GB)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   View Dataset  │  cli lamar-info
│   Information   │  (<1 second)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Run Multi-    │  cli lamar-experiment
│     Scene       │  (<2 seconds)
│   Analysis      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Scene Stats &  │  Automatic
│  Comparison     │
└─────────────────┘
```

## Configuration Files

### Hilti Configuration (`configs/hilti.yaml`)

```yaml
dataset:
  name: hilti
  type: rosbag
  rosbags_dir: datasets/hilti/rosbags
  frames_dir: datasets/hilti/frames
  sessions: [sequence_01, sequence_02, sequence_03]
  camera_topic: /alphasense/cam0/image_raw

mapper:
  type: colmap  # or glomap
  colmap:
    camera_model: SIMPLE_RADIAL
    max_num_features: 8192

experiment:
  visit_counts: [1, 2, 3]
```

### LaMAR Configuration (`configs/lamar.yaml`)

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

## Python API Usage

### Hilti Dataset

```python
from pathlib import Path
from sfm_experiments.config import load_config
from sfm_experiments.multivisit import run_multivisit_experiment

# Load configuration
config = load_config(Path("configs/hilti.yaml"))

# Run multi-visit experiment
session_dirs = [
    Path("datasets/hilti/frames/sequence_01"),
    Path("datasets/hilti/frames/sequence_02"),
    Path("datasets/hilti/frames/sequence_03"),
]

results = run_multivisit_experiment(
    session_dirs,
    Path("results"),
    visit_counts=[1, 2, 3],
    session_names=["seq_01", "seq_02", "seq_03"],
    use_cache=True
)

# Analyze results
for n_visits, result in results.items():
    print(f"{n_visits} visits: {result.num_registered_images} images, "
          f"{result.num_3d_points} points")
```

### LaMAR Dataset

```python
from pathlib import Path
from sfm_experiments.lamar_handler import load_lamar_reconstruction, list_lamar_scenes
from sfm_experiments.lamar_experiment import run_lamar_experiment

# List available scenes
base_dir = Path("datasets/lamar")
scenes = list_lamar_scenes(base_dir)

for scene in scenes:
    print(f"{scene.name}: {scene.num_images} images, {scene.num_points3d} points")

# Load a specific scene
reconstruction = load_lamar_reconstruction(base_dir / "colmap" / "CAB")
print(f"Loaded {len(reconstruction.images)} images")

# Run experiment
results = run_lamar_experiment(
    ["CAB", "HGE", "LIN"],
    base_dir,
    Path("results/lamar")
)

# Analyze results
from sfm_experiments.lamar_experiment import compare_lamar_scenes
comparison = compare_lamar_scenes(results)
print(f"Average images: {comparison['average']['images']:.0f}")
```

## Troubleshooting

### Common Issues

#### Issue: "LaMAR dataset not found"

**Solution:**
```bash
# Download the dataset first
uv run python scripts/download_lamar_dataset.py
```

#### Issue: "No LaMAR scenes found"

**Solution:**
```bash
# Make sure you extracted the zip files
# Re-download with extraction enabled
uv run python scripts/download_lamar_dataset.py
# When prompted: "Extract zip files after download? yes"
```

#### Issue: "Configuration file not found"

**Solution:**
```bash
# Use absolute path or specify correct relative path
uv run python -m sfm_experiments.cli lamar-experiment \
    --config-file /full/path/to/configs/lamar.yaml
```

#### Issue: Hilti "Missing session folders"

**Solution:**
```bash
# Extract frames first
uv run python -m sfm_experiments.cli extract-frames \
    datasets/hilti/rosbags/site1_handheld_1.bag \
    datasets/hilti/frames/sequence_01 \
    --fps 0.25
```

### Performance Tips

**For Hilti:**
- Use `--mapper glomap` for 10-100x speedup (requires installation)
- Enable caching (default) for faster re-runs
- Extract frames in parallel (run multiple terminals)
- Use lower FPS for faster extraction (--fps 0.1)

**For LaMAR:**
- Already optimized (uses pre-built reconstructions)
- Downloads are large - use good internet connection
- Single scene (18 GB) recommended for testing
- Results load in <2 seconds

## Examples

### Example 1: Quick LaMAR Test

```bash
# Download small dataset
uv run python scripts/download_lamar_dataset.py
# Select option 4 (CAB scene only)

# Check what we have
uv run python -m sfm_experiments.cli lamar-info

# Run experiment
uv run python -m sfm_experiments.cli lamar-experiment --scenes "CAB"
```

### Example 2: Full Hilti Multi-Visit

```bash
# Download first 3 sequences
uv run python scripts/download_hilti_dataset.py
# Select option 2

# Extract all sequences in parallel (3 terminals)
# Terminal 1:
uv run python -m sfm_experiments.cli extract-frames \
    datasets/hilti/rosbags/site1_handheld_1.bag \
    datasets/hilti/frames/sequence_01 --fps 0.25

# Terminal 2:
uv run python -m sfm_experiments.cli extract-frames \
    datasets/hilti/rosbags/site1_handheld_2.bag \
    datasets/hilti/frames/sequence_02 --fps 0.25

# Terminal 3:
uv run python -m sfm_experiments.cli extract-frames \
    datasets/hilti/rosbags/site1_handheld_3.bag \
    datasets/hilti/frames/sequence_03 --fps 0.25

# Run experiment
uv run python -m sfm_experiments.cli run-experiment \
    --config-file configs/hilti.yaml \
    --visits "1,2,3"
```

### Example 3: Compare Both Datasets

```bash
# Run LaMAR
uv run python -m sfm_experiments.cli lamar-experiment \
    --output-dir results/lamar_test

# Run Hilti (if data extracted)
uv run python -m sfm_experiments.cli run-experiment \
    --config-file configs/hilti.yaml \
    --output-dir results/hilti_test

# Compare results manually
cat results/lamar_test/experiment_summary.txt
cat results/hilti_test/results_summary.md
```

## Next Steps

1. **For LaMAR:** Download dataset, run `lamar-info`, then `lamar-experiment`
2. **For Hilti:** Download dataset, extract frames, run `run-experiment`
3. **Advanced:** Modify configs for custom experiments
4. **Python API:** Use modules directly for custom analysis

## Additional Resources

- **LaMAR Documentation:** `docs/lamar_integration.md`
- **LaMAR Implementation:** `docs/LAMAR_IMPLEMENTATION_SUMMARY.md`
- **Hilti Design:** `DESIGN.md`
- **LaMAR Dataset:** https://cvg-data.inf.ethz.ch/lamar/
- **Hilti Dataset:** https://hilti-challenge.com/

## Support

For issues or questions:
1. Check documentation in `docs/`
2. Run validation: `uv run python -m sfm_experiments.lamar_handler`
3. Check config files in `configs/`
4. Review logs in `logs/` and results directories
