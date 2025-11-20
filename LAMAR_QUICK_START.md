# LaMAR Quick Start Guide

## 🚀 Get Started in 3 Commands

```bash
# 1. Download LaMAR dataset (recommended: single scene for testing)
uv run python scripts/download_lamar_dataset.py
# Select option 4: CAB scene (17.9 GB)

# 2. View dataset information
uv run python -m sfm_experiments.cli lamar-info

# 3. Run multi-scene experiment
uv run python -m sfm_experiments.cli lamar-experiment
```

**Total time:** ~30 minutes (mostly download)
**Results:** Available in `results/lamar/` in under 2 seconds!

---

## 📊 Available Commands

### LaMAR Commands
```bash
# Show dataset info
uv run python -m sfm_experiments.cli lamar-info

# Run experiment (all scenes)
uv run python -m sfm_experiments.cli lamar-experiment

# Run experiment (specific scenes)
uv run python -m sfm_experiments.cli lamar-experiment --scenes "CAB,HGE"

# Detailed info
uv run python -m sfm_experiments.cli lamar-info --verbose
```

### Hilti Commands (for comparison)
```bash
# Extract frames from ROS bags
uv run python -m sfm_experiments.cli extract-frames <bag> <output>

# Run multi-visit experiment
uv run python -m sfm_experiments.cli run-experiment
```

### General
```bash
# Show all commands
uv run python -m sfm_experiments.cli info

# Get help
uv run python -m sfm_experiments.cli --help
```

---

## 📁 Download Options

Run: `uv run python scripts/download_lamar_dataset.py`

1. **Benchmark data only** (19.8 GB) - Query images
2. **COLMAP reconstructions only** (34 GB) - Pre-built models
3. **Benchmark + COLMAP** (53.8 GB) - Everything
4. **CAB scene only** (17.9 GB) - **⭐ RECOMMENDED FOR TESTING**
5. **HGE scene only** (17.8 GB)
6. **LIN scene only** (18.1 GB)
7. Custom selection

---

## 📈 What You Get

### LaMAR Dataset Includes:
- ✅ 3 indoor scenes: CAB, HGE, LIN
- ✅ Pre-built COLMAP reconstructions (ready to use!)
- ✅ Thousands of images per scene
- ✅ Millions of 3D points
- ✅ Multiple camera types (HoloLens, iPhone, NavVis)
- ✅ Benchmark query images for localization

### Expected Output:
```
LaMAR Scenes
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ Scene ┃ Images ┃ 3D Points ┃ Cameras ┃ COLMAP ┃ Benchmark ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│ CAB   │   1234 │    456789 │       8 │   ✅   │     ✅    │
│ HGE   │   2345 │    678901 │      12 │   ✅   │     ✅    │
│ LIN   │   3456 │    789012 │      15 │   ✅   │     ✅    │
└───────┴────────┴───────────┴─────────┴────────┴───────────┘
```

---

## 🐍 Python API

```python
from pathlib import Path
from sfm_experiments.lamar_handler import load_lamar_reconstruction, list_lamar_scenes

# List all available scenes
base_dir = Path("datasets/lamar")
scenes = list_lamar_scenes(base_dir)

for scene in scenes:
    print(f"{scene.name}: {scene.num_images} images")

# Load a specific scene
reconstruction = load_lamar_reconstruction(base_dir / "colmap" / "CAB")
print(f"Loaded {len(reconstruction.images)} images")
print(f"Point cloud: {len(reconstruction.points3D)} points")

# Run experiment
from sfm_experiments.lamar_experiment import run_lamar_experiment

results = run_lamar_experiment(
    ["CAB", "HGE", "LIN"],
    base_dir,
    Path("results/lamar")
)

for scene_name, result in results.items():
    if result.success:
        print(f"{scene_name}: {result.num_images} images, {result.num_points3d} points")
```

---

## ⚡ Performance

| Operation | Time |
|-----------|------|
| Download single scene (18 GB @ 100 Mbps) | ~24 min |
| Load reconstruction | <1 sec |
| Run 3-scene experiment | <2 sec |
| **Total: Download to results** | **~30 min** |

**Compare to Hilti:** 3-4 hours total (mostly frame extraction)

---

## 🔧 Troubleshooting

### "LaMAR dataset not found"
```bash
# Download the dataset
uv run python scripts/download_lamar_dataset.py
```

### "No scenes found"
```bash
# Make sure you extracted the zip files
# Re-download with extraction enabled
uv run python scripts/download_lamar_dataset.py
# When prompted: "Extract zip files after download? yes"
```

### Validate installation
```bash
# Run validation tests
uv run python -m sfm_experiments.lamar_handler

# Should output:
# ✅ VALIDATION PASSED - All 5 tests produced expected results
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `LAMAR_QUICK_START.md` | This guide - quick reference |
| `docs/lamar_integration.md` | Complete user guide (400+ lines) |
| `docs/LAMAR_IMPLEMENTATION_SUMMARY.md` | Technical details |
| `docs/LAMAR_PIPELINE_INTEGRATION.md` | Full CLI reference |
| `docs/FULL_INTEGRATION_SUMMARY.md` | Complete integration overview |

---

## 🎯 Next Steps

1. **Download** a scene using the download script
2. **Validate** with `lamar-info` command
3. **Experiment** with `lamar-experiment` command
4. **Explore** the Python API for custom analysis
5. **Compare** with Hilti dataset (optional)

---

## 📞 Quick Reference

```bash
# Download
uv run python scripts/download_lamar_dataset.py

# Info
uv run python -m sfm_experiments.cli lamar-info

# Experiment (all scenes)
uv run python -m sfm_experiments.cli lamar-experiment

# Experiment (specific scenes)
uv run python -m sfm_experiments.cli lamar-experiment --scenes "CAB,HGE"

# Validate
uv run python -m sfm_experiments.lamar_handler

# Help
uv run python -m sfm_experiments.cli --help
```

---

## 🌐 Links

- **Dataset:** https://cvg-data.inf.ethz.ch/lamar/
- **GitHub:** https://github.com/microsoft/lamar-benchmark
- **License:** CC BY-SA 4.0
- **COLMAP:** https://colmap.github.io/

---

**Ready to start? Run the first command!**

```bash
uv run python scripts/download_lamar_dataset.py
```
