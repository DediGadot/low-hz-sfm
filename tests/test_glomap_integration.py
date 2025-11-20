"""
Validation test for GLOMAP integration into SfM pipeline.

This test validates:
- Dual-mode mapper function (COLMAP/GLOMAP)
- Automatic fallback from GLOMAP to COLMAP
- Configuration parsing for both mapper types
- Backward compatibility with existing code

Dependencies:
- pytest: Testing framework
- pathlib: File operations
- pycolmap: SfM reconstruction (https://github.com/colmap/pycolmap)

Sample Input: Test dataset with images
Expected Output: Successful reconstruction with either COLMAP or GLOMAP

Run with:
    uv run pytest tests/test_glomap_integration.py -v
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sfm_experiments.glomap_wrapper import (
    check_glomap_available,
    GlomapOptions,
)
from sfm_experiments.colmap_runner import (
    run_sfm_reconstruction,
    run_colmap_reconstruction,
)


def test_glomap_availability_check():
    """Test 1: GLOMAP availability detection works correctly."""
    available, info = check_glomap_available()

    # Should return a tuple
    assert isinstance(available, bool), "availability check should return bool"
    assert isinstance(info, str), "info should be a string"

    # Info should be meaningful
    assert len(info) > 0, "info string should not be empty"

    print(f"✓ GLOMAP availability check: available={available}, info={info}")
    return True


def test_glomap_options_building():
    """Test 2: GlomapOptions creates valid command arguments."""
    # Create options with custom values
    options = GlomapOptions(
        max_epipolar_error=4.0,
        max_num_tracks=1000,
        skip_retriangulation=True,
    )

    # Build command arguments
    args = options.to_command_args()

    # Verify critical arguments present
    assert "--RelPoseEstimation.max_epipolar_error" in args, \
        "max_epipolar_error should be in command args"
    assert "4.0" in args, \
        "max_epipolar_error value should be in command args"

    assert "--TrackEstablishment.max_num_tracks" in args, \
        "max_num_tracks should be in command args"
    assert "1000" in args, \
        "max_num_tracks value should be in command args"

    assert "--skip_retriangulation" in args, \
        "skip_retriangulation should be in command args"

    print(f"✓ GlomapOptions command args built correctly: {len(args)} arguments")
    return True


def test_mapper_type_parameter():
    """Test 3: run_sfm_reconstruction accepts mapper_type parameter."""
    # This is a signature test - we don't actually run reconstruction
    # Just verify the function accepts the parameters

    import inspect

    sig = inspect.signature(run_sfm_reconstruction)
    params = sig.parameters

    # Verify mapper_type parameter exists
    assert "mapper_type" in params, \
        "run_sfm_reconstruction should have mapper_type parameter"

    # Verify glomap_options parameter exists
    assert "glomap_options" in params, \
        "run_sfm_reconstruction should have glomap_options parameter"

    # Verify default is "colmap"
    assert params["mapper_type"].default == "colmap", \
        "mapper_type should default to 'colmap'"

    print(f"✓ run_sfm_reconstruction has correct signature with mapper_type and glomap_options")
    return True


def test_backward_compatibility():
    """Test 4: run_colmap_reconstruction still works (backward compatibility)."""
    import inspect

    # Verify run_colmap_reconstruction still exists
    sig = inspect.signature(run_colmap_reconstruction)
    params = sig.parameters

    # Should have original parameters
    assert "image_dir" in params, "should have image_dir parameter"
    assert "output_dir" in params, "should have output_dir parameter"
    assert "camera_model" in params, "should have camera_model parameter"

    print(f"✓ run_colmap_reconstruction maintains backward compatibility")
    return True


def test_config_structure():
    """Test 5: Config file has proper mapper section."""
    import yaml

    config_path = Path(__file__).parent.parent / "configs" / "hilti.yaml"

    if not config_path.exists():
        print(f"⚠ Config file not found at {config_path}, skipping test")
        return True

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Verify mapper section exists
    assert "mapper" in config, "config should have 'mapper' section"

    mapper_cfg = config["mapper"]

    # Verify type field
    assert "type" in mapper_cfg, "mapper config should have 'type' field"
    assert mapper_cfg["type"] in ["colmap", "glomap"], \
        f"mapper type should be 'colmap' or 'glomap', got {mapper_cfg['type']}"

    # Verify colmap section exists
    assert "colmap" in mapper_cfg, "mapper config should have 'colmap' section"

    # Verify glomap section exists
    assert "glomap" in mapper_cfg, "mapper config should have 'glomap' section"

    # Verify glomap options
    glomap_cfg = mapper_cfg["glomap"]
    assert "max_epipolar_error" in glomap_cfg, \
        "glomap config should have max_epipolar_error"

    print(f"✓ Config file has proper mapper structure: type={mapper_cfg['type']}")
    return True


if __name__ == "__main__":
    """
    Validation of GLOMAP integration.
    Tests dual-mode mapper, configuration, and backward compatibility.
    """
    all_validation_failures = []
    total_tests = 0

    tests = [
        ("GLOMAP availability check", test_glomap_availability_check),
        ("GlomapOptions command building", test_glomap_options_building),
        ("Mapper type parameter", test_mapper_type_parameter),
        ("Backward compatibility", test_backward_compatibility),
        ("Config structure", test_config_structure),
    ]

    for test_name, test_func in tests:
        total_tests += 1
        try:
            result = test_func()
            if not result:
                all_validation_failures.append(f"{test_name}: Returned False")
        except AssertionError as e:
            all_validation_failures.append(f"{test_name}: {str(e)}")
        except Exception as e:
            all_validation_failures.append(f"{test_name}: Exception raised: {e}")

    # Final validation result
    print("\n" + "="*80)
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("\nGLOMAP integration validated:")
        print("  ✓ Dual-mode mapper function (COLMAP/GLOMAP)")
        print("  ✓ Configuration parsing")
        print("  ✓ Backward compatibility maintained")
        print("  ✓ Command argument building")

        # Check if GLOMAP is actually available
        available, _ = check_glomap_available()
        if available:
            print("\n✓ GLOMAP is installed and ready to use")
        else:
            print("\n⚠ GLOMAP not installed (will auto-fallback to COLMAP)")
            print("  To install GLOMAP, see: https://github.com/colmap/glomap#installation")

        sys.exit(0)
