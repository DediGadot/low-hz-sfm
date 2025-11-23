#!/usr/bin/env python
"""
Security vulnerability test cases for config.py
Tests for identified security issues in the configuration module.
"""

from pathlib import Path
import tempfile
import os
import yaml
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sfm_experiments.config import load_config, Config, _resolve_placeholders


def test_env_var_injection():
    """Test 1: Environment variable injection vulnerability (Line 48)"""
    print("\n[TEST 1] Environment Variable Injection")
    print("=" * 50)

    # Set a sensitive environment variable
    os.environ["SECRET_API_KEY"] = "sk-1234567890abcdef"
    os.environ["DATABASE_PASSWORD"] = "super_secret_password"

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "malicious.yaml"

        # Create a config that references environment variables
        malicious_config = """
dataset:
  name: "test"
  # Direct environment variable reference
  api_key: "${SECRET_API_KEY}"
  db_password: "${DATABASE_PASSWORD}"
  # Nested references
  connection_string: "postgres://user:${DATABASE_PASSWORD}@host:5432/db"
  # Try to access system paths
  home_dir: "${HOME}"
  path_var: "${PATH}"
"""

        with open(config_file, "w") as f:
            f.write(malicious_config)

        config = load_config(config_file)

        print(f"✓ API Key extracted: {config.dataset.api_key[:10]}...")
        print(f"✓ DB Password extracted: {config.dataset.db_password[:5]}...")
        print(f"✓ Connection string: {config.dataset.connection_string[:30]}...")
        print(f"✓ Home directory: {config.dataset.home_dir}")

        # Clean up
        del os.environ["SECRET_API_KEY"]
        del os.environ["DATABASE_PASSWORD"]

    return True


def test_yaml_billion_laughs():
    """Test 2: YAML Billion Laughs Attack (Exponential Entity Expansion)"""
    print("\n[TEST 2] YAML Billion Laughs Attack")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "billion_laughs.yaml"

        # Create a YAML bomb using anchors and aliases
        yaml_bomb = """
lol1: &lol1 "lol"
lol2: &lol2 [*lol1, *lol1, *lol1, *lol1, *lol1]
lol3: &lol3 [*lol2, *lol2, *lol2, *lol2, *lol2]
lol4: &lol4 [*lol3, *lol3, *lol3, *lol3, *lol3]
lol5: &lol5 [*lol4, *lol4, *lol4, *lol4, *lol4]
lol6: &lol6 [*lol5, *lol5, *lol5, *lol5, *lol5]
lol7: &lol7 [*lol6, *lol6, *lol6, *lol6, *lol6]
lol8: &lol8 [*lol7, *lol7, *lol7, *lol7, *lol7]
lol9: &lol9 [*lol8, *lol8, *lol8, *lol8, *lol8]
dataset:
  name: *lol9
"""

        with open(config_file, "w") as f:
            f.write(yaml_bomb)

        try:
            # This will consume significant memory due to exponential expansion
            # safe_load prevents some attacks but not all memory exhaustion
            config = load_config(config_file)
            print(f"✗ YAML bomb loaded - memory exhaustion possible!")
            # Check the actual size
            import json
            json_str = json.dumps(config.to_dict())
            print(f"  Expanded size: {len(json_str):,} bytes")
        except Exception as e:
            print(f"✓ YAML bomb prevented: {e}")

    return True


def test_circular_reference_bypass():
    """Test 3: Circular reference detection bypass attempts"""
    print("\n[TEST 3] Circular Reference Bypass")
    print("=" * 50)

    test_cases = [
        # Case 1: Direct circular reference
        {
            "a": "${b}",
            "b": "${a}"
        },
        # Case 2: Indirect circular reference through multiple levels
        {
            "x": "${y}",
            "y": "${z}",
            "z": "${x}"
        },
        # Case 3: Self-reference
        {
            "self": "prefix_${self}_suffix"
        },
        # Case 4: Complex nested circular reference
        {
            "level1": {"value": "${level2.value}"},
            "level2": {"value": "${level3.value}"},
            "level3": {"value": "${level1.value}"}
        }
    ]

    for i, test_data in enumerate(test_cases, 1):
        print(f"\n  Test case {i}: {list(test_data.keys())}")
        try:
            result = _resolve_placeholders(test_data, test_data)
            print(f"  ✗ No cycle detected! Result: {result}")
        except ValueError as e:
            if "Circular reference" in str(e):
                print(f"  ✓ Cycle detected correctly")
            else:
                print(f"  ✗ Different error: {e}")

    return True


def test_path_traversal():
    """Test 4: Path traversal vulnerability"""
    print("\n[TEST 4] Path Traversal")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "path_traversal.yaml"

        # Create a config with path traversal attempts
        path_config = """
dataset:
  name: "test"
  # Path traversal attempts
  base_path: "../../../etc"
  password_file: "../../../etc/passwd"
  ssh_keys: "~/.ssh/id_rsa"
  rosbags_dir: "/etc/../etc/../etc/shadow"

  # Using environment variable for path traversal
  custom_path: "${HOME}/../../root"

  # Absolute paths
  abs_path: "/etc/passwd"

  # Windows path traversal
  win_path: "..\\..\\..\\windows\\system32"
"""

        with open(config_file, "w") as f:
            f.write(path_config)

        config = load_config(config_file)

        print(f"✓ Path traversal values loaded:")
        print(f"  base_path: {config.dataset.base_path}")
        print(f"  password_file: {config.dataset.password_file}")
        print(f"  ssh_keys: {config.dataset.ssh_keys}")
        print(f"  abs_path: {config.dataset.abs_path}")

        # Check if any validation occurs
        print(f"\n  Note: No path validation performed - paths accepted as-is")

        # Try to access files using these paths
        dangerous_path = Path(config.dataset.password_file)
        if dangerous_path.exists():
            print(f"  ✗ CRITICAL: Can access {dangerous_path}")

    return True


def test_recursive_placeholder_bomb():
    """Test 5: Recursive placeholder expansion bomb"""
    print("\n[TEST 5] Recursive Placeholder Bomb")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "recursive_bomb.yaml"

        # Create deeply nested placeholders that expand exponentially
        recursive_config = """
# Each level doubles the previous one
level1: "A"
level2: "${level1}${level1}"
level3: "${level2}${level2}"
level4: "${level3}${level3}"
level5: "${level4}${level4}"
level6: "${level5}${level5}"
level7: "${level6}${level6}"
level8: "${level7}${level7}"
level9: "${level8}${level8}"
level10: "${level9}${level9}"

dataset:
  # This will expand to 2^10 = 1024 A's
  name: "${level10}"
  # Multiple expansions
  combined: "${level8}_${level8}_${level8}"
"""

        with open(config_file, "w") as f:
            f.write(recursive_config)

        try:
            config = load_config(config_file)
            name_len = len(config.dataset.name)
            combined_len = len(config.dataset.combined)

            print(f"✓ Recursive expansion completed:")
            print(f"  name length: {name_len} characters")
            print(f"  combined length: {combined_len} characters")

            if name_len == 1024:  # 2^10
                print(f"  ✓ Exponential expansion confirmed (2^10 = {name_len})")

            # Test with more levels would cause memory issues
            print(f"  ⚠️  More levels could cause memory exhaustion")

        except Exception as e:
            print(f"  Error during expansion: {e}")

    return True


def test_missing_input_validation():
    """Test 6: Missing input validation"""
    print("\n[TEST 6] Missing Input Validation")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test various invalid inputs
        test_cases = [
            ("empty.yaml", ""),
            ("null.yaml", "null"),
            ("invalid_types.yaml", """
dataset:
  name: 123  # Should be string?
  path: null
  list: "should_be_list"
  dict: ["should", "be", "dict"]
  negative_value: -9999
  huge_value: 999999999999999999999999999999
"""),
            ("special_chars.yaml", """
dataset:
  name: "'; DROP TABLE users; --"
  path: "../../etc/passwd\x00.jpg"
  command: "$(rm -rf /)"
  unicode: "\U0001F4A9\U0001F4A9\U0001F4A9"
"""),
        ]

        for filename, content in test_cases:
            config_file = Path(tmpdir) / filename

            if content:
                with open(config_file, "w") as f:
                    f.write(content)

                try:
                    config = load_config(config_file)
                    if filename == "empty.yaml":
                        print(f"  ✗ {filename}: Loaded empty config")
                    else:
                        print(f"  ✓ {filename}: Loaded without validation")
                        if hasattr(config, 'dataset'):
                            if hasattr(config.dataset, 'command'):
                                print(f"    Command injection string: {config.dataset.command}")
                            if hasattr(config.dataset, 'path'):
                                print(f"    Path with null byte: {repr(config.dataset.path)}")
                except ValueError as e:
                    if "Empty configuration" in str(e):
                        print(f"  ✓ {filename}: Rejected empty config")
                except Exception as e:
                    print(f"  ! {filename}: Error: {e}")

    return True


def test_max_iterations_bypass():
    """Test 7: Max iterations limit bypass attempt"""
    print("\n[TEST 7] Max Iterations Bypass")
    print("=" * 50)

    # Create a chain that requires exactly 10 iterations (the max)
    test_data = {
        "v0": "final_value",
        "v1": "${v0}",
        "v2": "${v1}",
        "v3": "${v2}",
        "v4": "${v3}",
        "v5": "${v4}",
        "v6": "${v5}",
        "v7": "${v6}",
        "v8": "${v7}",
        "v9": "${v8}",
        "v10": "${v9}",  # This would require 11 iterations
        "dataset": {
            "max_chain": "${v9}",  # Should work (10 iterations)
            "over_max": "${v10}"   # Should fail or be incomplete
        }
    }

    try:
        result = _resolve_placeholders(test_data, test_data)

        print(f"  Max chain result: {result['dataset']['max_chain']}")
        print(f"  Over max result: {result['dataset']['over_max']}")

        if result['dataset']['max_chain'] == 'final_value':
            print(f"  ✓ 10-iteration chain resolved correctly")

        if result['dataset']['over_max'] == 'final_value':
            print(f"  ✗ 11-iteration chain also resolved (limit not enforced)")
        elif "${" in result['dataset']['over_max']:
            print(f"  ✓ 11-iteration chain left unresolved (limit enforced)")

    except Exception as e:
        print(f"  Error: {e}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("SECURITY VULNERABILITY ANALYSIS: config.py")
    print("=" * 60)

    tests = [
        test_env_var_injection,
        test_yaml_billion_laughs,
        test_circular_reference_bypass,
        test_path_traversal,
        test_recursive_placeholder_bomb,
        test_missing_input_validation,
        test_max_iterations_bypass
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\nTest {test.__name__} failed with exception: {e}")
            results.append((test.__name__, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")