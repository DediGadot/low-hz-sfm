#!/usr/bin/env python3
"""
Download Hilti SLAM Challenge 2023 Dataset

This script provides an interactive interface to download sequences from the
Hilti SLAM Challenge 2023 dataset.

Dependencies:
- requests: https://docs.python-requests.org/
- tqdm: https://tqdm.github.io/

Usage:
    uv run python scripts/download_hilti_dataset.py

Sample Input:
    User selects sequences to download interactively

Expected Output:
    Downloaded ROS bag files in datasets/hilti/rosbags/
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class DatasetFile:
    """Dataset file metadata."""
    name: str
    url: str
    size_gb: float
    category: str  # 'site1', 'site2', 'site3', 'calibration', 'additional'
    platform: str  # 'handheld', 'robot'
    description: str


# Complete dataset catalog
DATASET_FILES = [
    # Site 1 - Handheld (Multi-visit sequences - RECOMMENDED)
    DatasetFile(
        "site1_handheld_1.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site1_handheld_1.bag",
        22.0,
        "site1",
        "handheld",
        "Floor 0"
    ),
    DatasetFile(
        "site1_handheld_2.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site1_handheld_2.bag",
        17.9,
        "site1",
        "handheld",
        "Floor 1"
    ),
    DatasetFile(
        "site1_handheld_3.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site1_handheld_3.bag",
        18.3,
        "site1",
        "handheld",
        "Floor 2"
    ),
    DatasetFile(
        "site1_handheld_4.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site1_handheld_4.bag",
        31.9,
        "site1",
        "handheld",
        "Underground"
    ),
    DatasetFile(
        "site1_handheld_5.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site1_handheld_5.bag",
        17.1,
        "site1",
        "handheld",
        "Stairs"
    ),

    # Site 2 - Robot
    DatasetFile(
        "site2_robot_1.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site2_robot_1.bag",
        63.3,
        "site2",
        "robot",
        "Parking (3 floors)"
    ),
    DatasetFile(
        "site2_robot_2.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site2_robot_2.bag",
        27.7,
        "site2",
        "robot",
        "Floor 1 Large room"
    ),
    DatasetFile(
        "site2_robot_3.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site2_robot_3.bag",
        32.5,
        "site2",
        "robot",
        "Floor 2 Large room"
    ),

    # Site 2 - Handheld
    DatasetFile(
        "site2_handheld_4.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site2_handheld_4.bag",
        9.3,
        "site2",
        "handheld",
        "Central staircase"
    ),
    DatasetFile(
        "site2_handheld_5.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site2_handheld_5.bag",
        20.6,
        "site2",
        "handheld",
        "Vault Staircase"
    ),
    DatasetFile(
        "site2_handheld_6.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site2_handheld_6.bag",
        13.8,
        "site2",
        "handheld",
        "Large room connector"
    ),

    # Site 3 - Handheld
    DatasetFile(
        "site3_handheld_1.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site3_handheld_1.bag",
        9.7,
        "site3",
        "handheld",
        "Underground 1"
    ),
    DatasetFile(
        "site3_handheld_2.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site3_handheld_2.bag",
        14.6,
        "site3",
        "handheld",
        "Underground 2"
    ),
    DatasetFile(
        "site3_handheld_3.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site3_handheld_3.bag",
        18.7,
        "site3",
        "handheld",
        "Underground 3"
    ),
    DatasetFile(
        "site3_handheld_4.bag",
        "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2023/site3_handheld_4.bag",
        10.6,
        "site3",
        "handheld",
        "Underground 4"
    ),
]


def download_file(url: str, output_path: Path, expected_size_gb: Optional[float] = None) -> bool:
    """
    Download a file with progress bar and resume capability.

    Args:
        url: URL to download from
        output_path: Local path to save file
        expected_size_gb: Expected file size in GB for validation

    Returns:
        True if download successful, False otherwise
    """
    # Check if file already exists and is complete
    if output_path.exists():
        existing_size = output_path.stat().st_size
        if expected_size_gb:
            expected_bytes = int(expected_size_gb * 1024 * 1024 * 1024)
            size_diff_mb = abs(existing_size - expected_bytes) / (1024 * 1024)

            # More lenient tolerance for large files (100 MB)
            if size_diff_mb < 100:
                print(f"✅ {output_path.name} already downloaded ({existing_size / 1024**3:.1f} GB, diff: {size_diff_mb:.1f} MB)")
                return True
            elif existing_size > expected_bytes:
                # File is larger than expected - might be corrupted, delete and restart
                print(f"⚠️  {output_path.name} is larger than expected, deleting and restarting...")
                output_path.unlink()
                existing_size = 0
                resume_header = {}
                mode = 'wb'
            else:
                # File exists but incomplete - try to resume
                print(f"📥 Resuming {output_path.name} from {existing_size / 1024**3:.1f} GB...")
                resume_header = {'Range': f'bytes={existing_size}-'}
                mode = 'ab'
        else:
            # No expected size, try to resume
            resume_header = {'Range': f'bytes={existing_size}-'}
            mode = 'ab'
    else:
        resume_header = {}
        mode = 'wb'
        existing_size = 0

    try:
        # Make request with stream=True for large files
        response = requests.get(url, headers=resume_header, stream=True, timeout=30)
        response.raise_for_status()

        # Get total size
        total_size = int(response.headers.get('content-length', 0)) + existing_size

        # Download with progress bar
        with open(output_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=existing_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Validate size if expected
        if expected_size_gb:
            final_size = output_path.stat().st_size
            expected_bytes = int(expected_size_gb * 1024 * 1024 * 1024)
            size_diff_mb = abs(final_size - expected_bytes) / (1024 * 1024)

            if size_diff_mb > 100:  # Allow 100 MB tolerance for large files
                print(f"⚠️  Size mismatch: expected {expected_size_gb:.1f} GB, got {final_size / 1024**3:.1f} GB (diff: {size_diff_mb:.1f} MB)")
                return False

        return True

    except requests.exceptions.HTTPError as e:
        # Handle 416 Range Not Satisfiable - file is already complete
        if e.response.status_code == 416:
            print(f"✅ {output_path.name} already downloaded (verified by server)")
            return True
        print(f"❌ Download failed: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏸️  Download interrupted. Run again to resume.")
        return False


def download_files_parallel(files: List[DatasetFile], output_dir: Path, max_workers: int = 2) -> Dict[str, bool]:
    """
    Download multiple files in parallel.

    Args:
        files: List of files to download
        output_dir: Output directory
        max_workers: Maximum parallel downloads

    Returns:
        Dict mapping filename to success status
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                download_file,
                f.url,
                output_dir / f.name,
                f.size_gb
            ): f.name
            for f in files
        }

        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                success = future.result()
                results[filename] = success
            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")
                results[filename] = False

    return results


def print_menu():
    """Print interactive menu."""
    print("\n" + "="*80)
    print("  Hilti SLAM Challenge 2023 Dataset Downloader")
    print("="*80)
    print("\nPreset Options:")
    print("  1. Site 1 All (Multi-visit, recommended) - 5 files, 107.2 GB")
    print("  2. Site 1 First 3 (Quick start) - 3 files, 58.2 GB")
    print("  3. Site 2 Robot - 3 files, 123.5 GB")
    print("  4. Site 2 Handheld - 3 files, 43.7 GB")
    print("  5. Site 3 All - 4 files, 53.6 GB")
    print("  6. Custom selection")
    print("  7. Download ALL sequences - 15 files, 328.0 GB ⚠️")
    print("  0. Exit")
    print("="*80)


def get_user_selection() -> List[DatasetFile]:
    """Get user file selection."""
    print_menu()

    choice = input("\nEnter your choice (0-7): ").strip()

    if choice == "0":
        sys.exit(0)

    elif choice == "1":
        # Site 1 All - Recommended for multi-visit experiments
        return [f for f in DATASET_FILES if f.category == "site1"]

    elif choice == "2":
        # Site 1 First 3 - Quick start
        return [f for f in DATASET_FILES if f.category == "site1"][:3]

    elif choice == "3":
        # Site 2 Robot
        return [f for f in DATASET_FILES if f.category == "site2" and f.platform == "robot"]

    elif choice == "4":
        # Site 2 Handheld
        return [f for f in DATASET_FILES if f.category == "site2" and f.platform == "handheld"]

    elif choice == "5":
        # Site 3 All
        return [f for f in DATASET_FILES if f.category == "site3"]

    elif choice == "6":
        # Custom selection
        return custom_selection()

    elif choice == "7":
        # Download all
        confirm = input("⚠️  This will download 328 GB. Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            return DATASET_FILES
        else:
            return get_user_selection()

    else:
        print("❌ Invalid choice. Please try again.")
        return get_user_selection()


def custom_selection() -> List[DatasetFile]:
    """Interactive custom file selection."""
    print("\n" + "="*80)
    print("Available Files:")
    print("="*80)

    for idx, f in enumerate(DATASET_FILES, 1):
        print(f"{idx:2d}. {f.name:30s} | {f.size_gb:5.1f} GB | {f.description}")

    print("="*80)
    print("\nEnter file numbers separated by commas (e.g., 1,2,3)")
    print("Or enter 'all' to select all files")

    selection = input("Your selection: ").strip()

    if selection.lower() == "all":
        return DATASET_FILES

    try:
        indices = [int(x.strip()) - 1 for x in selection.split(",")]
        selected = [DATASET_FILES[i] for i in indices if 0 <= i < len(DATASET_FILES)]

        if not selected:
            print("❌ No valid files selected.")
            return custom_selection()

        return selected

    except (ValueError, IndexError):
        print("❌ Invalid selection. Please try again.")
        return custom_selection()


def main():
    """Main execution function."""
    # Determine output directory (relative to script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "datasets" / "hilti" / "rosbags"

    print(f"\n📁 Output directory: {output_dir}")

    # Get user selection
    selected_files = get_user_selection()

    if not selected_files:
        print("❌ No files selected.")
        sys.exit(1)

    # Display selection summary
    total_size = sum(f.size_gb for f in selected_files)
    print("\n" + "="*80)
    print(f"Selected {len(selected_files)} file(s), total size: {total_size:.1f} GB")
    print("="*80)

    for f in selected_files:
        print(f"  • {f.name} ({f.size_gb:.1f} GB) - {f.description}")

    # Confirm download
    confirm = input("\nProceed with download? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("❌ Download cancelled.")
        sys.exit(0)

    # Ask about parallel downloads
    try:
        max_workers = int(input("\nParallel downloads (1-4, recommended 2): ").strip() or "2")
        max_workers = max(1, min(4, max_workers))
    except ValueError:
        max_workers = 2

    print(f"\n🚀 Starting download with {max_workers} parallel worker(s)...")

    # Download files
    results = download_files_parallel(selected_files, output_dir, max_workers)

    # Print summary
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful

    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed downloads:")
        for filename, success in results.items():
            if not success:
                print(f"  • {filename}")
        print("\n💡 Tip: Run the script again to resume failed downloads.")

    print(f"\n📁 Files saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Extract frames:")
    print("     uv run python -m sfm_experiments.cli extract-frames \\")
    print(f"         {output_dir}/site1_handheld_1.bag \\")
    print("         datasets/hilti/frames/sequence_01 \\")
    print("         --fps 0.25")
    print("\n  2. Run experiment:")
    print("     uv run python -m sfm_experiments.cli run-experiment \\")
    print("         --config-file configs/hilti.yaml \\")
    print("         --output-dir results \\")
    print("         --visits '1,2,3'")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
