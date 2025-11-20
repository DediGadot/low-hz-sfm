#!/usr/bin/env python3
"""Download Hilti SLAM Challenge 2023 ground-truth assets.

This script mirrors ``download_hilti_dataset.py`` but targets the smaller
survey/ground-truth artifacts (poses, fused point clouds, calibration packs).

Usage
-----
    uv run python scripts/download_hilti_ground_truth.py [options]

Key features
------------
* Reads a manifest (JSON/YAML) describing files, target site, and download URL
* Interactive picker (site-based presets) plus ``--selection`` CLI override
* Resumable `requests` downloads with progress bars (`tqdm`)
* Optional parallel fetches (`--max-workers`)
* Automatic extraction of ``.zip``/``.tar.*`` archives into
  ``datasets/hilti/ground_truth`` (disable via ``--skip-extract``)

Ground-truth access
-------------------
Hilti distributes the GT assets via signed S3 links behind the participant
portal. Populate ``scripts/hilti_ground_truth_manifest.json`` with the links you
receive (or pass ``--manifest`` pointing to your own file). Any entry whose URL
contains ``REPLACE`` will be skipped with a helpful warning.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import shutil

import requests
from tqdm import tqdm

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - yaml optional
    yaml = None


@dataclass
class GroundTruthFile:
    """Metadata for a single ground-truth artifact."""

    name: str
    url: str
    size_mb: Optional[float]
    site: str
    artifact_type: str
    description: str

    @property
    def size_bytes(self) -> Optional[int]:
        if self.size_mb is None:
            return None
        return int(self.size_mb * 1024 * 1024)


def load_manifest(path: Path) -> List[GroundTruthFile]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise RuntimeError("pyyaml not installed; cannot read YAML manifest")
            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)

    files: List[GroundTruthFile] = []
    for entry in data:
        files.append(
            GroundTruthFile(
                name=entry["name"],
                url=entry["url"],
                size_mb=entry.get("size_mb"),
                site=entry.get("site", "unknown"),
                artifact_type=entry.get("type", "misc"),
                description=entry.get("description", ""),
            )
        )
    return files


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_size(bytes_total: Optional[int]) -> str:
    if not bytes_total:
        return "unknown"
    gb = bytes_total / (1024 ** 3)
    if gb >= 0.1:
        return f"{gb:.2f} GB"
    mb = bytes_total / (1024 ** 2)
    return f"{mb:.1f} MB"


def download_file(file: GroundTruthFile, output_path: Path) -> Path:
    ensure_output_dir(output_path.parent)

    if "REPLACE" in file.url:
        raise RuntimeError(
            f"URL for {file.name} contains placeholder text. Update the manifest with the "
            "signed link from the Hilti portal."
        )

    headers = {}
    mode = "wb"
    resume_bytes = 0

    if output_path.exists():
        resume_bytes = output_path.stat().st_size
        headers = {"Range": f"bytes={resume_bytes}-"}
        mode = "ab"

    with requests.get(file.url, headers=headers, stream=True, timeout=30) as resp:
        if resp.status_code == 416:
            return output_path
        resp.raise_for_status()
        total = resp.headers.get("content-length")
        total_bytes = int(total) + resume_bytes if total else None

        with open(output_path, mode) as fh, tqdm(
            total=total_bytes,
            initial=resume_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=file.name,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))

    return output_path


def extract_if_needed(path: Path, target_dir: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in {".zip", ".tar", ".gz", ".bz2", ".xz"} and not path.name.endswith(
        (".tar.gz", ".tar.bz2", ".tar.xz")
    ):
        return

    dest = target_dir / path.stem
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    shutil.unpack_archive(str(path), str(dest))


def select_files_interactively(files: Sequence[GroundTruthFile]) -> List[GroundTruthFile]:
    sites = sorted({f.site for f in files})
    print("\nAvailable sites:")
    for idx, site in enumerate(sites, start=1):
        site_files = [f for f in files if f.site == site]
        total = sum((f.size_bytes or 0) for f in site_files)
        print(f"  {idx}. {site} ({len(site_files)} files, {format_size(total)})")
    print("  0. Custom selection")

    choice = input("Select site preset (0 for custom): ").strip()
    if choice.isdigit() and int(choice) in range(1, len(sites) + 1):
        selected_site = sites[int(choice) - 1]
        return [f for f in files if f.site == selected_site]

    print("\nEnter comma-separated file numbers to download:")
    for idx, f in enumerate(files, start=1):
        print(f"  {idx:2d}. {f.name:35s} | {f.site:<6s} | {f.artifact_type:<12s} | {f.description}")

    raw = input("Files: ").strip()
    indices = []
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if 1 <= idx <= len(files):
            indices.append(idx - 1)
    return [files[i] for i in indices]


def filter_by_cli(files: Sequence[GroundTruthFile], selection: Sequence[str]) -> List[GroundTruthFile]:
    if not selection:
        return list(files)

    selected: List[GroundTruthFile] = []
    lookup = {f.name: f for f in files}
    for token in selection:
        token_lower = token.lower()
        if token in lookup:
            selected.append(lookup[token])
            continue
        # allow site/type filters
        matches = [
            f for f in files if f.site.lower() == token_lower or f.artifact_type.lower() == token_lower
        ]
        if matches:
            selected.extend(matches)
            continue
        raise ValueError(f"Unknown selection token: {token}")
    # deduplicate while preserving order
    seen = set()
    result = []
    for f in selected:
        if f.name in seen:
            continue
        seen.add(f.name)
        result.append(f)
    return result


def download_files(
    files: Sequence[GroundTruthFile],
    output_dir: Path,
    max_workers: int,
    extract_archives: bool,
    extract_root: Path,
) -> Dict[str, bool]:
    ensure_output_dir(output_dir)
    results: Dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(download_file, f, output_dir / f.name): f for f in files
        }

        for future in as_completed(future_map):
            file = future_map[future]
            try:
                downloaded_path = future.result()
                if extract_archives:
                    try:
                        extract_if_needed(downloaded_path, extract_root)
                    except shutil.ReadError:
                        pass
                results[file.name] = True
            except Exception as exc:  # noqa: BLE001
                print(f"❌ Failed {file.name}: {exc}")
                results[file.name] = False
    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "hilti_ground_truth_manifest.json",
        help="Path to manifest JSON/YAML describing ground-truth files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "hilti" / "ground_truth",
        help="Directory where downloaded archives/raw files are stored",
    )
    parser.add_argument(
        "--selection",
        nargs="*",
        default=None,
        help="Optional list of site names, artifact types, or explicit filenames",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Parallel download workers (default: 2)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download only; do not unpack archives",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    files = load_manifest(args.manifest)
    if not files:
        print("❌ Manifest is empty")
        return 1

    if args.selection:
        try:
            selected = filter_by_cli(files, args.selection)
        except ValueError as exc:
            print(f"❌ {exc}")
            return 1
    else:
        selected = select_files_interactively(files)

    if not selected:
        print("⚠️  No files selected. Exiting.")
        return 0

    print("\nSelected files:")
    for f in selected:
        print(
            f"  • {f.name} | {f.site} | {f.artifact_type} | {format_size(f.size_bytes)} | {f.description}"
        )

    confirm = input("\nProceed with download? (yes/no): ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("❌ Cancelled by user")
        return 0

    results = download_files(
        selected,
        args.output_dir,
        max(1, min(args.max_workers, 4)),
        extract_archives=not args.skip_extract,
        extract_root=args.output_dir,
    )

    success = sum(1 for ok in results.values() if ok)
    print(f"\n✅ Completed {success}/{len(results)} downloads")
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print("❌ Failed downloads:")
        for name in failed:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
