"""
Command-line interface for SfM multi-visit experiments.

This module provides:
- Frame extraction from ROS bags
- Multi-visit reconstruction execution
- Metric computation and visualization
- End-to-end experiment orchestration

Dependencies:
- typer: https://typer.tiangolo.com/

Usage:
    uv run python -m sfm_experiments.cli --help
"""

from pathlib import Path
from typing import List, Optional, Dict
import math
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from loguru import logger

from .utils import setup_logging, print_section, print_summary
from .config import load_config
from .dataset import extract_frames_from_rosbag, extract_all_sessions, load_ground_truth_poses
from .colmap_runner import (
    run_colmap_reconstruction,
    run_sfm_reconstruction,
    extract_poses_from_reconstruction,
)
from .multivisit import run_multivisit_experiment, summarize_multivisit_results
from .metrics import compute_ate, compute_chamfer_distance, compute_completeness, load_point_cloud
from .visualization import plot_accuracy_vs_visits, create_results_table
from .lamar_handler import list_lamar_scenes, get_lamar_scene_info
from .lamar_experiment import run_lamar_experiment, summarize_lamar_results, compare_lamar_scenes

app = typer.Typer(
    name="sfm-experiments",
    help="SfM Multi-Visit Experimentation Pipeline",
    add_completion=False,
)
console = Console()


@app.command()
def extract_frames(
    bag_path: Path = typer.Argument(..., help="Path to ROS bag file"),
    output_dir: Path = typer.Argument(..., help="Output directory for frames"),
    camera_topic: str = typer.Option(
        "/camera/image_raw", help="ROS camera topic"
    ),
    fps: float = typer.Option(0.25, help="Target frame rate (frames per second)"),
    quality: int = typer.Option(95, help="JPEG quality (0-100)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Extract frames from ROS bag file.

    Example:
        uv run python -m sfm_experiments.cli extract-frames \\
            data/sequence_01.bag \\
            data/frames/session_01 \\
            --fps 0.25
    """
    setup_logging(level="DEBUG" if verbose else "INFO")

    print_section("Frame Extraction from ROS Bag")

    from .dataset import extract_frames_from_rosbag

    frames = extract_frames_from_rosbag(
        bag_path, output_dir, camera_topic, fps, quality
    )

    print_summary({
        "ROS Bag": bag_path.name,
        "Output Directory": str(output_dir),
        "Frames Extracted": len(frames),
        "Target FPS": fps,
    }, "Extraction Summary")

    rprint(f"[green]✅ Extracted {len(frames)} frames to {output_dir}[/green]")


@app.command()
def run_colmap(
    image_dir: Path = typer.Argument(..., help="Directory with input images"),
    output_dir: Path = typer.Argument(..., help="Output directory for reconstruction"),
    camera_model: str = typer.Option("SIMPLE_RADIAL", help="COLMAP camera model"),
    max_features: int = typer.Option(8192, help="Maximum SIFT features per image"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run COLMAP reconstruction on a directory of images.

    Example:
        uv run python -m sfm_experiments.cli run-colmap \\
            data/frames/session_01 \\
            results/reconstruction_01
    """
    setup_logging(level="DEBUG" if verbose else "INFO")

    print_section("COLMAP Reconstruction")

    result = run_colmap_reconstruction(image_dir, output_dir, camera_model, max_features)

    if result.success:
        print_summary({
            "Registered Images": result.num_registered_images,
            "3D Points": result.num_3d_points,
            "Reprojection Error": f"{result.mean_reprojection_error:.3f}px",
            "Execution Time": f"{result.execution_time:.1f}s",
            "Point Cloud": str(result.point_cloud_path),
        }, "Reconstruction Summary")

        rprint(f"[green]✅ Reconstruction successful![/green]")
    else:
        rprint(f"[red]❌ Reconstruction failed after {result.execution_time:.1f}s[/red]")
        raise typer.Exit(code=1)


@app.command()
def run_experiment(
    config_file: Path = typer.Option(
        "configs/hilti.yaml", help="Path to configuration file"
    ),
    output_dir: Path = typer.Option(
        "results", help="Base output directory"
    ),
    visits: Optional[str] = typer.Option(
        None, help="Comma-separated visit counts (e.g., '1,2,3')"
    ),
    mapper: Optional[str] = typer.Option(
        None, help="SfM mapper: 'colmap' (incremental) or 'glomap' (global). Overrides config file."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Disable cache and re-run all steps"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run complete multi-visit experiment.

    This orchestrates the full pipeline:
    1. Load configuration
    2. Extract frames from ROS bags (if needed)
    3. Run reconstructions for different visit counts (COLMAP or GLOMAP)
    4. Compute metrics
    5. Generate visualizations

    By default, the experiment uses caching to skip:
    - Already combined session frames
    - Already completed reconstructions

    Use --no-cache to force re-running all steps.

    Example:
        # Use default mapper from config (COLMAP)
        uv run python -m sfm_experiments.cli run-experiment \\
            --config-file configs/hilti.yaml \\
            --output-dir results \\
            --visits "1,2,3"

        # Use GLOMAP for 10-100x speedup
        uv run python -m sfm_experiments.cli run-experiment \\
            --config-file configs/hilti.yaml \\
            --output-dir results \\
            --mapper glomap

        # Force re-run without cache
        uv run python -m sfm_experiments.cli run-experiment \\
            --config-file configs/hilti.yaml \\
            --output-dir results \\
            --visits "1,2,3" \\
            --no-cache
    """
    setup_logging(
        log_file=Path(output_dir) / "experiment.log",
        level="DEBUG" if verbose else "INFO",
    )

    print_section("Multi-Visit SfM Experiment")

    # Load configuration
    config = load_config(config_file)

    # Parse visit counts
    if visits:
        visit_counts = [int(v.strip()) for v in visits.split(',')]
    else:
        visit_counts = config.experiment.visit_counts

    # Get session directories
    dataset_cfg = config.dataset
    frames_base = Path(dataset_cfg.frames_dir)
    session_names = list(dataset_cfg.sessions)
    session_dirs = [frames_base / session for session in session_names]

    # Auto-extract missing sessions when bags are available
    missing_sessions = [name for name, path in zip(session_names, session_dirs) if not path.exists()]

    if missing_sessions:
        rosbags_dir = getattr(dataset_cfg, "rosbags_dir", None)
        if rosbags_dir is None:
            rprint(
                f"[yellow]⚠️  Missing session folders and no rosbags_dir configured - cannot extract: {missing_sessions}[/yellow]"
            )
        else:
            rosbags_path = Path(rosbags_dir)
            if not rosbags_path.exists():
                rprint(
                    f"[yellow]⚠️  Missing session folders but rosbags dir {rosbags_path} not found[/yellow]"
                )
            else:
                rprint(
                    f"[blue]Extracting frames for sessions: {', '.join(missing_sessions)}[/blue]"
                )
                extract_all_sessions(
                    rosbags_path,
                    frames_base,
                    missing_sessions,
                    camera_topic=getattr(dataset_cfg, "camera_topic", "/camera/image_raw"),
                    target_fps=getattr(dataset_cfg, "target_fps", 0.25),
                    jpeg_quality=getattr(dataset_cfg, "jpeg_quality", 95),
                )

    # Re-evaluate session directories after optional extraction
    session_dirs = [frames_base / session for session in session_names]
    available_sessions = [
        (name, path) for name, path in zip(session_names, session_dirs) if path.exists()
    ]
    existing_session_names = [name for name, _ in available_sessions]
    existing_sessions = [path for _, path in available_sessions]

    if not existing_sessions:
        rprint(f"[red]❌ No session directories found in {frames_base}[/red]")
        rprint("[yellow]Run frame extraction first![/yellow]")
        raise typer.Exit(code=1)

    rprint(f"[blue]Found {len(existing_sessions)} session(s)[/blue]")

    # Show cache status
    if no_cache:
        rprint("[yellow]⚠️  Cache disabled - all steps will be re-run[/yellow]")
    else:
        rprint("[green]✅ Cache enabled - reusing existing results when possible[/green]")

    # Configure mapper type (COLMAP or GLOMAP)
    mapper_cfg = getattr(config, "mapper", None)
    mapper_type = mapper or (getattr(mapper_cfg, "type", "colmap") if mapper_cfg else "colmap")

    # Validate mapper type
    if mapper_type not in ["colmap", "glomap"]:
        rprint(f"[red]❌ Invalid mapper type: {mapper_type}. Must be 'colmap' or 'glomap'[/red]")
        raise typer.Exit(code=1)

    rprint(f"[blue]Using {mapper_type.upper()} mapper[/blue]")

    # Configure SfM parameters from config when available
    colmap_kwargs = {"mapper_type": mapper_type}

    # Get COLMAP config (backward compatibility)
    colmap_cfg = getattr(config, "colmap", None)
    if colmap_cfg is not None:
        camera_model = getattr(colmap_cfg, "camera_model", None)
        if camera_model:
            colmap_kwargs["camera_model"] = camera_model

        features_cfg = getattr(colmap_cfg, "features", None)
        max_features = getattr(features_cfg, "max_num_features", None) if features_cfg else None
        if max_features:
            colmap_kwargs["max_num_features"] = max_features

    # Get mapper-specific config from new structure
    if mapper_cfg:
        if mapper_type == "colmap":
            colmap_specific = getattr(mapper_cfg, "colmap", None)
            if colmap_specific:
                camera_model = getattr(colmap_specific, "camera_model", None)
                if camera_model:
                    colmap_kwargs["camera_model"] = camera_model
                max_features = getattr(colmap_specific, "max_num_features", None)
                if max_features:
                    colmap_kwargs["max_num_features"] = max_features
        elif mapper_type == "glomap":
            glomap_specific = getattr(mapper_cfg, "glomap", None)
            if glomap_specific:
                glomap_options = {}
                max_epipolar_error = getattr(glomap_specific, "max_epipolar_error", None)
                if max_epipolar_error:
                    glomap_options["max_epipolar_error"] = max_epipolar_error
                max_num_tracks = getattr(glomap_specific, "max_num_tracks", None)
                if max_num_tracks:
                    glomap_options["max_num_tracks"] = max_num_tracks
                skip_retriangulation = getattr(glomap_specific, "skip_retriangulation", None)
                if skip_retriangulation is not None:
                    glomap_options["skip_retriangulation"] = skip_retriangulation
                if glomap_options:
                    colmap_kwargs["glomap_options"] = glomap_options

    def _resolve_path(value: Optional[str]) -> Optional[Path]:
        if value is None:
            return None
        return Path(value)

    def _first_existing(candidates: List[Optional[Path]]) -> Optional[Path]:
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        return None

    gt_dir_attr = getattr(dataset_cfg, "ground_truth_dir", None)
    gt_dir = Path(gt_dir_attr) if gt_dir_attr else None
    gt_pose_candidates: List[Optional[Path]] = []
    manual_pose = getattr(dataset_cfg, "ground_truth_poses_file", None)
    manual_pose_path = _resolve_path(manual_pose) if manual_pose else None
    if manual_pose_path is not None:
        gt_pose_candidates.append(manual_pose_path)
        if not manual_pose_path.is_absolute() and gt_dir is not None:
            gt_pose_candidates.append(gt_dir / manual_pose_path)
    if gt_dir is not None:
        gt_pose_candidates.append(gt_dir / "poses.txt")

    gt_pose_path = _first_existing(gt_pose_candidates)
    ground_truth_poses = None
    if gt_pose_path:
        logger.info(f"Loading ground truth poses from {gt_pose_path}")
        ground_truth_poses = load_ground_truth_poses(gt_pose_path)
    else:
        logger.warning("Ground truth poses not found - ATE will be skipped")

    gt_map_candidates: List[Optional[Path]] = []
    manual_map = getattr(dataset_cfg, "ground_truth_point_cloud", None)
    manual_map_path = _resolve_path(manual_map) if manual_map else None
    if manual_map_path is not None:
        gt_map_candidates.append(manual_map_path)
        if not manual_map_path.is_absolute() and gt_dir is not None:
            gt_map_candidates.append(gt_dir / manual_map_path)
    if gt_dir is not None:
        gt_map_candidates.append(gt_dir / "map.ply")

    ground_truth_pcd = None
    gt_map_path = _first_existing(gt_map_candidates)
    if gt_map_path:
        try:
            ground_truth_pcd = load_point_cloud(gt_map_path)
        except Exception as exc:
            logger.warning(f"Failed to load ground truth point cloud: {exc}")
    else:
        logger.warning("Ground truth point cloud not found - point cloud metrics will be skipped")

    # Run multi-visit experiment
    results = run_multivisit_experiment(
        existing_sessions,
        Path(output_dir),
        visit_counts,
        existing_session_names,
        use_cache=not no_cache,
        **colmap_kwargs,
    )

    # Create summary
    summary = summarize_multivisit_results(results)

    # Generate metrics and visualization
    plot_path = Path(output_dir) / "plots" / "accuracy_vs_visits.png"
    metrics_for_plot: Dict[int, Dict[str, float]] = {}
    results_for_table = {}
    metrics_available = False

    for n_visits, result in results.items():
        metrics_entry = {
            "success": result.success,
            "num_images": result.num_registered_images,
            "num_points": result.num_3d_points,
            "ate": math.nan,
            "chamfer": math.nan,
            "completeness": math.nan,
            "execution_time": result.execution_time,
        }

        if result.success and result.reconstruction and ground_truth_poses:
            try:
                est_poses = extract_poses_from_reconstruction(result.reconstruction)
                metrics_entry["ate"] = compute_ate(est_poses, ground_truth_poses)
                metrics_available = True
            except Exception as exc:
                logger.warning(f"Failed to compute ATE for {n_visits} visits: {exc}")

        if result.success and ground_truth_pcd and result.point_cloud_path and result.point_cloud_path.exists():
            try:
                recon_pcd = load_point_cloud(result.point_cloud_path)
                metrics_entry["chamfer"] = compute_chamfer_distance(recon_pcd, ground_truth_pcd)
                metrics_entry["completeness"] = compute_completeness(recon_pcd, ground_truth_pcd)
                metrics_available = True
            except Exception as exc:
                logger.warning(f"Failed to compute point cloud metrics for {n_visits} visits: {exc}")

        results_for_table[n_visits] = metrics_entry

        metrics_for_plot[n_visits] = {
            "ate": metrics_entry["ate"],
            "chamfer": metrics_entry["chamfer"],
            "completeness": metrics_entry["completeness"],
        }

    if metrics_available:
        plot_accuracy_vs_visits(metrics_for_plot, plot_path)
    else:
        logger.warning("Skipping accuracy plot because no metrics were computed")

    # Create results table
    table_path = Path(output_dir) / "results_summary.md"
    create_results_table(results_for_table, table_path)

    # Display summary
    print_summary({
        "Total Experiments": summary["num_experiments"],
        "Successful": summary["successful_experiments"],
        "Visit Counts Tested": ", ".join(map(str, summary["visit_counts"])),
        "Best Visit Count": summary.get("best_visit_count", "N/A"),
        "Results Table": str(table_path),
        "Plot": str(plot_path),
    }, "Experiment Summary")

    rprint(f"[green]✅ Experiment complete! Results saved to {output_dir}[/green]")


@app.command()
def lamar_info(
    base_dir: Path = typer.Option(
        "datasets/lamar", help="LaMAR dataset base directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed scene information"),
):
    """
    Display information about LaMAR dataset.

    Example:
        uv run python -m sfm_experiments.cli lamar-info
        uv run python -m sfm_experiments.cli lamar-info --base-dir datasets/lamar --verbose
    """
    setup_logging(level="DEBUG" if verbose else "INFO")

    print_section("LaMAR Dataset Information")

    if not base_dir.exists():
        rprint(f"[red]❌ LaMAR dataset not found at {base_dir}[/red]")
        rprint("\n[yellow]Download the dataset first:[/yellow]")
        rprint("  uv run python scripts/download_lamar_dataset.py")
        raise typer.Exit(code=1)

    # List available scenes
    scenes = list_lamar_scenes(base_dir)

    if not scenes:
        rprint(f"[yellow]⚠️  No LaMAR scenes found in {base_dir}[/yellow]")
        rprint("\n[yellow]Download the dataset first:[/yellow]")
        rprint("  uv run python scripts/download_lamar_dataset.py")
        raise typer.Exit(code=1)

    # Create summary table
    table = Table(title="LaMAR Scenes")
    table.add_column("Scene", style="cyan", no_wrap=True)
    table.add_column("Images", justify="right", style="green")
    table.add_column("3D Points", justify="right", style="blue")
    table.add_column("Cameras", justify="right", style="magenta")
    table.add_column("COLMAP", justify="center")
    table.add_column("Benchmark", justify="center")

    for scene in scenes:
        table.add_row(
            scene.name,
            str(scene.num_images) if scene.num_images > 0 else "-",
            str(scene.num_points3d) if scene.num_points3d > 0 else "-",
            str(scene.num_cameras) if scene.num_cameras > 0 else "-",
            "✅" if scene.colmap_path else "❌",
            "✅" if scene.benchmark_path else "❌",
        )

    console.print(table)

    # Show detailed info if verbose
    if verbose:
        for scene in scenes:
            print(f"\n{scene.name} Details:")
            if scene.colmap_path:
                print(f"  COLMAP: {scene.colmap_path}")
            if scene.benchmark_path:
                print(f"  Benchmark: {scene.benchmark_path}")

    # Summary
    total_images = sum(s.num_images for s in scenes)
    total_points = sum(s.num_points3d for s in scenes)

    print_summary({
        "Scenes": len(scenes),
        "Total Images": total_images,
        "Total 3D Points": total_points,
        "Dataset Path": str(base_dir),
    }, "Summary")


@app.command()
def lamar_experiment(
    config_file: Path = typer.Option(
        "configs/lamar.yaml", help="Path to LaMAR configuration file"
    ),
    output_dir: Path = typer.Option(
        "results/lamar", help="Output directory for results"
    ),
    scenes: Optional[str] = typer.Option(
        None, help="Comma-separated scene names (e.g., 'CAB,HGE,LIN')"
    ),
    dashboard: bool = typer.Option(
        True, "--dashboard/--no-dashboard", help="Generate comprehensive HTML dashboard"
    ),
    reconstruct: bool = typer.Option(
        False, "--reconstruct/--no-reconstruct", help="Run SfM reconstruction from raw images instead of loading pre-built models"
    ),
    mapper: str = typer.Option(
        "colmap", "--mapper", "-m", help="SfM reconstruction method: 'colmap' (incremental) or 'glomap' (global)"
    ),
    evaluate: bool = typer.Option(
        False, "--evaluate/--no-evaluate", help="Evaluate reconstruction against ground truth (requires --reconstruct)"
    ),
    fps: Optional[float] = typer.Option(
        None, "--fps", "--target-fps", help="Target frame rate for image sampling (e.g., 0.25 = 1 frame every 4 seconds). Only applies with --reconstruct. If not specified, uses all images."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run LaMAR multi-scene analysis or reconstruction experiment.

    Two modes:
    1. Analysis mode (default): Load pre-built COLMAP reconstructions
    2. Reconstruction mode (--reconstruct): Run full SfM from raw images

    Example:
        # Analyze all scenes with dashboard (default)
        uv run python -m sfm_experiments.cli lamar-experiment

        # Analyze specific scenes
        uv run python -m sfm_experiments.cli lamar-experiment --scenes "CAB,HGE"

        # Run full reconstruction with COLMAP
        uv run python -m sfm_experiments.cli lamar-experiment --scenes "CAB" \\
            --reconstruct --mapper colmap --evaluate

        # Run reconstruction with GLOMAP (faster but less robust)
        uv run python -m sfm_experiments.cli lamar-experiment --scenes "CAB" \\
            --reconstruct --mapper glomap --evaluate

        # Generate individual visualizations without dashboard
        uv run python -m sfm_experiments.cli lamar-experiment --no-dashboard

        # Custom config and output
        uv run python -m sfm_experiments.cli lamar-experiment \\
            --config-file configs/lamar.yaml \\
            --output-dir results/my_lamar_experiment
    """
    setup_logging(
        log_file=Path(output_dir) / "lamar_experiment.log",
        level="DEBUG" if verbose else "INFO",
    )

    mode_str = "Reconstruction" if reconstruct else "Analysis"
    print_section(f"LaMAR Multi-Scene {mode_str}")

    # Validate mapper choice
    if mapper not in ["colmap", "glomap"]:
        rprint(f"[red]❌ Invalid mapper: {mapper}. Must be 'colmap' or 'glomap'[/red]")
        raise typer.Exit(code=1)

    # Validate evaluation flag
    if evaluate and not reconstruct:
        rprint(f"[yellow]⚠️  --evaluate requires --reconstruct. Ignoring evaluate flag.[/yellow]")
        evaluate = False

    # Validate FPS parameter
    if fps is not None:
        if fps <= 0:
            rprint(f"[red]❌ Invalid fps: {fps}. Must be greater than 0[/red]")
            raise typer.Exit(code=1)
        if not reconstruct:
            rprint(f"[yellow]⚠️  --fps requires --reconstruct. Ignoring fps parameter.[/yellow]")
            fps = None

    # Load configuration
    if not config_file.exists():
        rprint(f"[red]❌ Configuration file not found: {config_file}[/red]")
        raise typer.Exit(code=1)

    config = load_config(config_file)

    # Get base directory from config
    dataset_cfg = config.dataset
    base_dir = Path(dataset_cfg.base_dir)

    if not base_dir.exists():
        rprint(f"[red]❌ LaMAR dataset not found at {base_dir}[/red]")
        rprint("\n[yellow]Download the dataset first:[/yellow]")
        rprint("  uv run python scripts/download_lamar_dataset.py")
        raise typer.Exit(code=1)

    # Parse scene list
    if scenes:
        scene_list = [s.strip() for s in scenes.split(',')]
    else:
        scene_list = list(dataset_cfg.scenes)

    # Build mode details string
    mode_details_parts = []
    if reconstruct:
        mode_details_parts.append(f"{mapper} mapper")
        mode_details_parts.append(f"evaluate={evaluate}")
        if fps is not None:
            mode_details_parts.append(f"fps={fps}")
    else:
        mode_details_parts.append("pre-built models")

    mode_details = f"({', '.join(mode_details_parts)})"
    rprint(f"[blue]{mode_str} scenes: {', '.join(scene_list)} {mode_details}[/blue]")

    # Run experiment
    results = run_lamar_experiment(
        scene_list,
        base_dir,
        Path(output_dir),
        validate=True,
        run_reconstruction=reconstruct,
        mapper_type=mapper,
        evaluate_against_gt=evaluate,
        target_fps=fps,
    )

    # Generate visualizations
    from .lamar_visualization import generate_lamar_visualizations
    viz_paths = generate_lamar_visualizations(
        results,
        Path(output_dir),
        generate_dashboard=dashboard
    )

    # Create summary
    summary = summarize_lamar_results(results)

    # Generate comparison
    comparison = compare_lamar_scenes(results)

    # Display results
    print_summary({
        "Total Scenes": summary["num_scenes"],
        "Successful": summary["successful_scenes"],
        "Failed": summary.get("failed_scenes", 0),
        "Total Images": summary.get("total_images", 0),
        "Total 3D Points": summary.get("total_points3d", 0),
        "Execution Time": f"{summary['total_execution_time']:.2f}s",
    }, "Experiment Summary")

    # Show per-scene results
    if comparison and 'by_scene' in comparison:
        table = Table(title="Scene Comparison")
        table.add_column("Scene", style="cyan")
        table.add_column("Images", justify="right", style="green")
        table.add_column("3D Points", justify="right", style="blue")
        table.add_column("Cameras", justify="right", style="magenta")

        for scene_name, stats in comparison['by_scene'].items():
            table.add_row(
                scene_name,
                str(stats['images']),
                str(stats['points3d']),
                str(stats['cameras']),
            )

        console.print(table)

    # Show failures if any
    if summary.get("failed_scenes", 0) > 0:
        rprint(f"\n[yellow]Failed scenes:[/yellow]")
        for scene_name, error_msg in summary.get("failure_messages", {}).items():
            rprint(f"  • {scene_name}: {error_msg}")

    rprint(f"\n[green]✅ LaMAR experiment complete! Results saved to {output_dir}[/green]")


@app.command()
def info():
    """
    Display information about the SfM experiments pipeline.
    """
    rprint("[bold blue]SfM Multi-Visit Experimentation Pipeline[/bold blue]")
    rprint("\n[bold]Version:[/bold] 0.1.0")
    rprint("[bold]Purpose:[/bold] Investigate how map accuracy improves with multiple visits")

    rprint("\n[bold]Supported Datasets:[/bold]")
    rprint("  • Hilti SLAM Challenge 2023 - Multi-visit ROS bag data")
    rprint("  • LaMAR - Pre-built COLMAP reconstructions (ETH Zurich)")

    rprint("\n[bold]Available Commands:[/bold]")
    rprint("\n[cyan]Hilti Dataset:[/cyan]")
    rprint("  • extract-frames : Extract frames from ROS bags")
    rprint("  • run-colmap     : Run COLMAP reconstruction")
    rprint("  • run-experiment : Run complete multi-visit experiment")

    rprint("\n[cyan]LaMAR Dataset:[/cyan]")
    rprint("  • lamar-info       : Show LaMAR dataset information")
    rprint("  • lamar-experiment : Run multi-scene analysis")

    rprint("\n[cyan]General:[/cyan]")
    rprint("  • info           : Show this information")

    rprint("\n[bold]Quick Start - Hilti:[/bold]")
    rprint("  1. Download: uv run python scripts/download_hilti_dataset.py")
    rprint("  2. Extract: uv run python -m sfm_experiments.cli extract-frames <bag> <output>")
    rprint("  3. Run: uv run python -m sfm_experiments.cli run-experiment")

    rprint("\n[bold]Quick Start - LaMAR:[/bold]")
    rprint("  1. Download: uv run python scripts/download_lamar_dataset.py")
    rprint("  2. Info: uv run python -m sfm_experiments.cli lamar-info")
    rprint("  3. Run: uv run python -m sfm_experiments.cli lamar-experiment")

    rprint("\n[bold]Documentation:[/bold]")
    rprint("  • Hilti: See DESIGN.md")
    rprint("  • LaMAR: See docs/lamar_integration.md")


if __name__ == "__main__":
    app()
