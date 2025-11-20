#!/usr/bin/env python3
"""
Download LaMAR (Localization and Mapping for Augmented Reality) Dataset

This script provides an interactive interface to download files from the
LaMAR benchmark dataset from ETH Zurich CVG lab.

Dataset Information:
- Source: https://cvg-data.inf.ethz.ch/lamar/
- GitHub: https://github.com/microsoft/lamar-benchmark
- License: CC BY-SA 4.0
- Scenes: CAB, HGE, LIN (3 indoor environments)
- Data types: Benchmark queries, COLMAP reconstructions, Raw sensor data

Dependencies:
- requests: https://docs.python-requests.org/
- tqdm: https://tqdm.github.io/

Usage:
    uv run python scripts/download_lamar_dataset.py

Sample Input:
    User selects dataset type and scenes to download interactively

Expected Output:
    Downloaded and extracted dataset files in datasets/lamar/
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile


@dataclass
class DatasetFile:
    """Dataset file metadata."""
    name: str
    url: str
    size_gb: float
    category: str  # 'benchmark', 'colmap', 'raw'
    scene: str     # 'CAB', 'HGE', 'LIN'
    description: str


# LaMAR dataset catalog
DATASET_FILES = [
    # Benchmark data (query images for localization)
    DatasetFile(
        "CAB.zip",
        "https://cvg-data.inf.ethz.ch/lamar/benchmark/CAB.zip",
        5.9,
        "benchmark",
        "CAB",
        "CAB scene - Benchmark queries"
    ),
    DatasetFile(
        "HGE.zip",
        "https://cvg-data.inf.ethz.ch/lamar/benchmark/HGE.zip",
        6.8,
        "benchmark",
        "HGE",
        "HGE scene - Benchmark queries"
    ),
    DatasetFile(
        "LIN.zip",
        "https://cvg-data.inf.ethz.ch/lamar/benchmark/LIN.zip",
        7.1,
        "benchmark",
        "LIN",
        "LIN scene - Benchmark queries"
    ),

    # COLMAP reconstructions (pre-built SfM models)
    DatasetFile(
        "CAB.zip",
        "https://cvg-data.inf.ethz.ch/lamar/colmap/CAB.zip",
        12.0,
        "colmap",
        "CAB",
        "CAB scene - COLMAP reconstruction"
    ),
    DatasetFile(
        "HGE.zip",
        "https://cvg-data.inf.ethz.ch/lamar/colmap/HGE.zip",
        11.0,
        "colmap",
        "HGE",
        "HGE scene - COLMAP reconstruction"
    ),
    DatasetFile(
        "LIN.zip",
        "https://cvg-data.inf.ethz.ch/lamar/colmap/LIN.zip",
        11.0,
        "colmap",
        "LIN",
        "LIN scene - COLMAP reconstruction"
    ),
]


def download_file(
    url: str,
    output_path: Path,
    expected_size_gb: Optional[float] = None,
    extract: bool = False
) -> bool:
    """
    Download a file with progress bar and resume capability.

    Args:
        url: URL to download from
        output_path: Local path to save file
        expected_size_gb: Expected file size in GB for validation
        extract: Whether to extract zip file after download

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
                print(f"✅ {output_path.name} already downloaded ({existing_size / 1024**3:.1f} GB)")

                # Extract if requested and not already extracted
                if extract and output_path.suffix == '.zip':
                    extract_dir = output_path.parent / output_path.stem
                    if not extract_dir.exists():
                        print(f"📦 Extracting {output_path.name}...")
                        try:
                            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                                zip_ref.extractall(output_path.parent)
                            print(f"✅ Extracted to {extract_dir}")
                        except Exception as e:
                            print(f"❌ Extraction failed: {e}")
                            return False
                    else:
                        print(f"✅ Already extracted to {extract_dir}")

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
            # No expected size, assume complete
            print(f"✅ {output_path.name} already exists")
            return True
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
                print(f"⚠️  Size mismatch: expected {expected_size_gb:.1f} GB, got {final_size / 1024**3:.1f} GB")
                return False

        # Extract if requested
        if extract and output_path.suffix == '.zip':
            print(f"📦 Extracting {output_path.name}...")
            try:
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(output_path.parent)
                extract_dir = output_path.parent / output_path.stem
                print(f"✅ Extracted to {extract_dir}")
            except Exception as e:
                print(f"❌ Extraction failed: {e}")
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


def download_files_sequential(
    files: List[DatasetFile],
    output_dir: Path,
    extract: bool = False
) -> Dict[str, bool]:
    """
    Download files sequentially (recommended for large files).

    Args:
        files: List of files to download
        output_dir: Output directory
        extract: Whether to extract zip files after download

    Returns:
        Dict mapping filename to success status
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for f in files:
        # Create category-specific subdirectory
        category_dir = output_dir / f.category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Download file
        output_path = category_dir / f.name
        success = download_file(f.url, output_path, f.size_gb, extract)
        results[f"{f.category}/{f.name}"] = success

    return results


def print_menu():
    """Print interactive menu."""
    print("\n" + "="*80)
    print("  LaMAR Dataset Downloader (ETH Zurich CVG Lab)")
    print("  License: CC BY-SA 4.0")
    print("="*80)
    print("\nPreset Options:")
    print("  1. Benchmark data - All scenes (CAB, HGE, LIN) - 3 files, 19.8 GB")
    print("  2. COLMAP reconstructions - All scenes - 3 files, 34.0 GB")
    print("  3. Benchmark + COLMAP - All scenes - 6 files, 53.8 GB")
    print("  4. Single scene - CAB (Benchmark + COLMAP) - 2 files, 17.9 GB")
    print("  5. Single scene - HGE (Benchmark + COLMAP) - 2 files, 17.8 GB")
    print("  6. Single scene - LIN (Benchmark + COLMAP) - 2 files, 18.1 GB")
    print("  7. Custom selection")
    print("  0. Exit")
    print("="*80)
    print("\nNote: Raw sensor data (HoloLens/phone sessions) not included.")
    print("      Visit https://cvg-data.inf.ethz.ch/lamar/raw/ for raw data.")


def get_user_selection() -> List[DatasetFile]:
    """Get user file selection."""
    print_menu()

    choice = input("\nEnter your choice (0-7): ").strip()

    if choice == "0":
        sys.exit(0)

    elif choice == "1":
        # All benchmark data
        return [f for f in DATASET_FILES if f.category == "benchmark"]

    elif choice == "2":
        # All COLMAP reconstructions
        return [f for f in DATASET_FILES if f.category == "colmap"]

    elif choice == "3":
        # All benchmark + COLMAP
        return DATASET_FILES

    elif choice == "4":
        # CAB scene (Benchmark + COLMAP)
        return [f for f in DATASET_FILES if f.scene == "CAB"]

    elif choice == "5":
        # HGE scene (Benchmark + COLMAP)
        return [f for f in DATASET_FILES if f.scene == "HGE"]

    elif choice == "6":
        # LIN scene (Benchmark + COLMAP)
        return [f for f in DATASET_FILES if f.scene == "LIN"]

    elif choice == "7":
        # Custom selection
        return custom_selection()

    else:
        print("❌ Invalid choice. Please try again.")
        return get_user_selection()


def custom_selection() -> List[DatasetFile]:
    """Interactive custom file selection."""
    print("\n" + "="*80)
    print("Available Files:")
    print("="*80)

    for idx, f in enumerate(DATASET_FILES, 1):
        print(f"{idx:2d}. {f.category:10s} | {f.scene:3s} | {f.size_gb:5.1f} GB | {f.description}")

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
    output_dir = project_root / "datasets" / "lamar"

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
        print(f"  • {f.category}/{f.name} ({f.size_gb:.1f} GB) - {f.description}")

    # Confirm download
    confirm = input("\nProceed with download? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("❌ Download cancelled.")
        sys.exit(0)

    # Ask about extraction
    extract = input("\nExtract zip files after download? (yes/no): ").strip().lower() == "yes"

    print(f"\n🚀 Starting download...")

    # Download files sequentially (large files work better sequentially)
    results = download_files_sequential(selected_files, output_dir, extract)

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
    print("\n" + "="*80)
    print("Dataset Information:")
    print("="*80)
    print("  Source: https://cvg-data.inf.ethz.ch/lamar/")
    print("  GitHub: https://github.com/microsoft/lamar-benchmark")
    print("  License: CC BY-SA 4.0")
    print("  Citation: Please cite the LaMAR paper if using this dataset")
    print("\nNext steps:")
    print("  1. Explore the dataset structure")
    print("  2. Use COLMAP data with the existing pipeline")
    print("  3. See docs/lamar_usage.md for integration details")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
