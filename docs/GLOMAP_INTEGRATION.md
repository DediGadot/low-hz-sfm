# GLOMAP Integration Guide

## Overview

This SfM pipeline now supports dual-mode mapping with both **COLMAP** (incremental) and **GLOMAP** (global) as reconstruction backends. GLOMAP offers 10-100x speedup on large datasets while maintaining accuracy.

### Key Benefits

- **Performance**: 10-100x faster than COLMAP incremental mapping
- **Accuracy**: Comparable quality to COLMAP on most datasets
- **Seamless Integration**: Same configuration format, automatic fallback
- **Backward Compatible**: Existing code continues to work unchanged

## Installation

### GLOMAP Installation

GLOMAP is not available via pip/uv. Choose one of these installation methods:

#### Option 1: Build from Source (Recommended)

```bash
# Install dependencies
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

# Clone and build GLOMAP
git clone https://github.com/colmap/glomap.git
cd glomap
mkdir build && cd build
cmake .. -GNinja
ninja
sudo ninja install

# Verify installation
glomap -h
```

#### Option 2: Conda Installation

```bash
conda install -c conda-forge glomap
```

**Note**: The pipeline will automatically fall back to COLMAP if GLOMAP is not installed.

## Configuration

### Basic Configuration

Edit your config file (e.g., `configs/hilti.yaml`):

```yaml
# SfM Mapper configuration
mapper:
  # Mapper type: "colmap" (incremental) or "glomap" (global)
  type: "colmap"  # Default: COLMAP for robustness

  # COLMAP-specific options (used when mapper.type = "colmap")
  colmap:
    camera_model: "SIMPLE_RADIAL"
    max_num_features: 8192
    init_min_num_inliers: 6
    abs_pose_min_num_inliers: 5
    abs_pose_min_inlier_ratio: 0.05

  # GLOMAP-specific options (used when mapper.type = "glomap")
  glomap:
    max_epipolar_error: 2.0      # Increase for blurry/high-res images (e.g., 4.0, 10.0)
    max_num_tracks: null          # Cap points for speed (e.g., 1000), null = no limit
    skip_retriangulation: false   # true = faster but less accurate
```

### GLOMAP Performance Tuning

#### For High-Resolution Images (>2MP)

```yaml
mapper:
  type: "glomap"
  glomap:
    max_epipolar_error: 4.0  # Increase tolerance for high-res
    max_num_tracks: null
    skip_retriangulation: false
```

#### For Speed-Critical Applications

```yaml
mapper:
  type: "glomap"
  glomap:
    max_epipolar_error: 2.0
    max_num_tracks: 1000         # Cap points to 1000
    skip_retriangulation: true   # Skip for speed
```

#### For Blurry/Motion-Blurred Images

```yaml
mapper:
  type: "glomap"
  glomap:
    max_epipolar_error: 10.0  # Very high tolerance
    max_num_tracks: null
    skip_retriangulation: false
```

## Usage

### Command-Line Interface

#### Use Mapper from Config File

```bash
# Uses mapper type specified in config (default: colmap)
uv run python -m sfm_experiments.cli run-experiment \
    --config-file configs/hilti.yaml \
    --output-dir results \
    --visits "1,2,3"
```

#### Override with GLOMAP

```bash
# Force GLOMAP mapper (ignores config file mapper type)
uv run python -m sfm_experiments.cli run-experiment \
    --config-file configs/hilti.yaml \
    --output-dir results \
    --visits "1,2,3" \
    --mapper glomap
```

#### Override with COLMAP

```bash
# Force COLMAP mapper
uv run python -m sfm_experiments.cli run-experiment \
    --config-file configs/hilti.yaml \
    --output-dir results \
    --visits "1,2,3" \
    --mapper colmap
```

### Python API

#### Basic Usage

```python
from pathlib import Path
from sfm_experiments.colmap_runner import run_sfm_reconstruction

# Run with GLOMAP
result = run_sfm_reconstruction(
    image_dir=Path("images"),
    output_dir=Path("output"),
    mapper_type="glomap",
    camera_model="SIMPLE_RADIAL",
    max_num_features=8192,
)

if result.success:
    print(f"✅ Registered {result.num_registered_images} images")
    print(f"   {result.num_3d_points} 3D points")
    print(f"   Execution time: {result.execution_time:.1f}s")
    print(f"   Mapper used: {result.mapper_type}")
```

#### With GLOMAP-Specific Options

```python
from pathlib import Path
from sfm_experiments.colmap_runner import run_sfm_reconstruction

# Advanced GLOMAP configuration
glomap_options = {
    "max_epipolar_error": 4.0,
    "max_num_tracks": 1000,
    "skip_retriangulation": True,
}

result = run_sfm_reconstruction(
    image_dir=Path("images"),
    output_dir=Path("output"),
    mapper_type="glomap",
    camera_model="SIMPLE_RADIAL",
    max_num_features=8192,
    glomap_options=glomap_options,
)
```

#### Automatic Fallback

```python
from pathlib import Path
from sfm_experiments.colmap_runner import run_sfm_reconstruction

# Request GLOMAP - will auto-fallback to COLMAP if not installed
result = run_sfm_reconstruction(
    image_dir=Path("images"),
    output_dir=Path("output"),
    mapper_type="glomap",  # Auto-fallback to COLMAP if GLOMAP unavailable
    camera_model="SIMPLE_RADIAL",
)

# Check which mapper was actually used
print(f"Mapper used: {result.mapper_type}")  # "glomap" or "colmap"
```

#### Backward Compatibility

```python
from pathlib import Path
from sfm_experiments.colmap_runner import run_colmap_reconstruction

# Old code continues to work unchanged
result = run_colmap_reconstruction(
    image_dir=Path("images"),
    output_dir=Path("output"),
    camera_model="SIMPLE_RADIAL",
)
```

## Performance Comparison

### COLMAP (Incremental Mapping)

**Strengths:**
- Very robust to difficult scenes
- Better handling of large baseline changes
- More mature, well-tested

**Weaknesses:**
- Slower (can take hours on large datasets)
- Sequential nature limits parallelization

### GLOMAP (Global Mapping)

**Strengths:**
- 10-100x faster than COLMAP
- Global optimization approach
- Better scalability to large datasets

**Weaknesses:**
- May struggle with very challenging scenes
- Requires good initial matches
- Less mature than COLMAP

### When to Use Each

| Scenario | Recommended Mapper | Reason |
|----------|-------------------|--------|
| Large dataset (>1000 images) | GLOMAP | 10-100x speedup |
| Good quality images | GLOMAP | Fast and accurate |
| Challenging scenes | COLMAP | More robust |
| Production environment | COLMAP | More mature |
| Research/experimentation | GLOMAP | Faster iteration |
| Blurry/low-quality images | COLMAP | Better handling |

## Troubleshooting

### GLOMAP Not Found

**Symptom:**
```
WARNING: GLOMAP not available: GLOMAP executable not found in PATH
WARNING: Falling back to COLMAP incremental mapper
```

**Solution:**
1. Install GLOMAP using one of the methods above
2. Verify installation: `glomap -h`
3. Ensure `glomap` is in your PATH

### GLOMAP Fails with "Too few inliers"

**Symptom:**
```
ERROR: GLOMAP failed with return code 1
Error output: Too few inlier matches
```

**Solutions:**
1. Increase `max_epipolar_error` in config:
   ```yaml
   glomap:
     max_epipolar_error: 4.0  # or higher
   ```

2. Use more features:
   ```yaml
   colmap:
     max_num_features: 16384  # Increase from 8192
   ```

3. Fall back to COLMAP for this dataset

### Performance Worse Than Expected

**Check these factors:**

1. **Image resolution**: GLOMAP performs best on high-res images (>2MP)
2. **Feature matches**: Ensure good overlap between images
3. **Configuration**: Try adjusting `max_epipolar_error`
4. **Dataset size**: GLOMAP speedup is more apparent on larger datasets (>500 images)

### Memory Issues

**Symptom:**
```
GLOMAP killed (out of memory)
```

**Solutions:**
1. Limit track count:
   ```yaml
   glomap:
     max_num_tracks: 1000  # Reduce memory usage
   ```

2. Reduce feature count:
   ```yaml
   colmap:
     max_num_features: 4096  # Reduce from 8192
   ```

## Architecture Details

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│ run_sfm_reconstruction(mapper_type="glomap")            │
└─────────────────────────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────────┐
    │ 1. Feature Extraction (SIFT via pycolmap)│
    │    - Same for both COLMAP and GLOMAP     │
    └─────────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────────┐
    │ 2. Feature Matching (via pycolmap)       │
    │    - Exhaustive + Sequential              │
    │    - Same for both mappers                │
    └─────────────────────────────────────────┘
                      ↓
        ┌─────────────────────────┐
        │ mapper_type == "glomap"? │
        └─────────────────────────┘
          Yes ↓              ↓ No
    ┌──────────────┐   ┌──────────────────┐
    │ 3a. GLOMAP   │   │ 3b. COLMAP       │
    │  (subprocess)│   │  (pycolmap API)  │
    └──────────────┘   └──────────────────┘
                ↓             ↓
    ┌─────────────────────────────────────────┐
    │ 4. Load Reconstruction (pycolmap)        │
    │    - Both output same format              │
    └─────────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────────┐
    │ 5. Export Point Cloud (PLY)              │
    └─────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `src/sfm_experiments/glomap_wrapper.py` | GLOMAP subprocess wrapper |
| `src/sfm_experiments/colmap_runner.py` | Dual-mode mapper implementation |
| `src/sfm_experiments/cli.py` | CLI with `--mapper` flag |
| `src/sfm_experiments/multivisit.py` | Multi-visit experiments (uses dual-mode) |
| `configs/hilti.yaml` | Example configuration |
| `tests/test_glomap_integration.py` | Integration validation |

## Validation

Validate the GLOMAP integration:

```bash
# Run integration tests
uv run python tests/test_glomap_integration.py

# Expected output:
# ✅ VALIDATION PASSED - All 5 tests produced expected results
# GLOMAP integration validated:
#   ✓ Dual-mode mapper function (COLMAP/GLOMAP)
#   ✓ Configuration parsing
#   ✓ Backward compatibility maintained
#   ✓ Command argument building
```

## References

- **GLOMAP Repository**: https://github.com/colmap/glomap
- **GLOMAP Paper**: [Global Structure-from-Motion Revisited](https://arxiv.org/abs/2407.20219)
- **COLMAP Documentation**: https://colmap.github.io/
- **pycolmap API**: https://github.com/colmap/pycolmap

## Migration Guide

### Migrating Existing Code

If you have existing code using `run_colmap_reconstruction()`, no changes are required. The function continues to work as before.

To opt-in to GLOMAP:

**Before:**
```python
from sfm_experiments.colmap_runner import run_colmap_reconstruction

result = run_colmap_reconstruction(image_dir, output_dir)
```

**After:**
```python
from sfm_experiments.colmap_runner import run_sfm_reconstruction

result = run_sfm_reconstruction(
    image_dir,
    output_dir,
    mapper_type="glomap"  # Add this parameter
)
```

### Migrating Config Files

Add the `mapper` section to your config file:

```yaml
# Add this section
mapper:
  type: "colmap"  # or "glomap"

  colmap:
    camera_model: "SIMPLE_RADIAL"
    max_num_features: 8192
    # ... existing COLMAP options ...

  glomap:
    max_epipolar_error: 2.0
    max_num_tracks: null
    skip_retriangulation: false
```

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Validate installation: `glomap -h`
3. Run integration test: `uv run python tests/test_glomap_integration.py`
4. Review GLOMAP logs in output directory
5. File issue on GLOMAP repository: https://github.com/colmap/glomap/issues
