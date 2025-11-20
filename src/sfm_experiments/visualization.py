"""
Visualization module for multi-visit experiment results.

This module provides:
- Accuracy vs. visits plots
- Metric comparison visualizations
- Result tables and summaries

Dependencies:
- matplotlib: https://matplotlib.org/
- numpy: https://numpy.org/

Sample Input: Dict of results mapping visit counts to metrics
Expected Output: High-resolution plots showing accuracy improvements
"""

from pathlib import Path
from typing import Dict, Any, Optional
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from loguru import logger


# Configure matplotlib for high-quality output
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10


def plot_accuracy_vs_visits(
    results: Dict[int, Dict[str, float]],
    output_path: Path,
    title: str = "SfM Accuracy Improvement with Multiple Visits",
) -> None:
    """
    Create accuracy improvement plot showing ATE, Chamfer, and Completeness.

    Generates a 3-subplot figure showing how each metric improves
    with increasing number of visits.

    Args:
        results: Dict mapping visit count to metrics dict
            Expected keys in metrics: 'ate', 'chamfer', 'completeness'
        output_path: Where to save the plot (PNG)
        title: Overall plot title

    Example:
        >>> results = {
        ...     1: {"ate": 0.50, "chamfer": 0.12, "completeness": 0.65},
        ...     2: {"ate": 0.30, "chamfer": 0.08, "completeness": 0.82},
        ...     3: {"ate": 0.20, "chamfer": 0.05, "completeness": 0.91},
        ... }
        >>> plot_accuracy_vs_visits(results, Path("results/accuracy_plot.png"))
    """
    if not results:
        logger.warning("No results to plot")
        return

    logger.info(f"Creating accuracy vs visits plot: {output_path}")

    visits = sorted(results.keys())

    def _metric_array(key: str, scale: float = 1.0) -> np.ndarray:
        values = []
        for v in visits:
            value = results[v].get(key)
            if value is None:
                values.append(np.nan)
            else:
                values.append(value * scale)
        return np.array(values, dtype=float)

    ates = _metric_array('ate')
    chamfers = _metric_array('chamfer')
    completeness = _metric_array('completeness', scale=100.0)

    if (
        np.all(np.isnan(ates))
        and np.all(np.isnan(chamfers))
        and np.all(np.isnan(completeness))
    ):
        logger.warning("No valid metrics available for plotting")
        return

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: ATE (lower is better)
    axes[0].plot(visits, ates, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_xlabel('Number of Visits', fontsize=12)
    axes[0].set_ylabel('ATE (meters)', fontsize=12)
    axes[0].set_title('Absolute Trajectory Error', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xticks(visits)

    # Add value labels
    for x, y in zip(visits, ates):
        if not np.isnan(y):
            axes[0].text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Chamfer Distance (lower is better)
    axes[1].plot(visits, chamfers, 's-', linewidth=2, markersize=8, color='#F77F00')
    axes[1].set_xlabel('Number of Visits', fontsize=12)
    axes[1].set_ylabel('Chamfer Distance (meters)', fontsize=12)
    axes[1].set_title('Point Cloud Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xticks(visits)

    # Add value labels
    for x, y in zip(visits, chamfers):
        if not np.isnan(y):
            axes[1].text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Completeness (higher is better)
    axes[2].plot(visits, completeness, '^-', linewidth=2, markersize=8, color='#06A77D')
    axes[2].set_xlabel('Number of Visits', fontsize=12)
    axes[2].set_ylabel('Completeness (%)', fontsize=12)
    axes[2].set_title('Map Completeness', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xticks(visits)
    axes[2].set_ylim([0, 105])  # 0-100% plus margin

    # Add value labels
    for x, y in zip(visits, completeness):
        if not np.isnan(y):
            axes[2].text(x, y, f'{y:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Plot saved to {output_path}")


def create_results_table(
    results: Dict[int, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> str:
    """
    Create markdown table of results.

    Args:
        results: Dict mapping visit count to result metrics
        output_path: Optional path to save markdown file

    Returns:
        Markdown formatted table string

    Example:
        >>> table = create_results_table(results)
        >>> print(table)
    """
    if not results:
        return "No results available"

    # Build markdown table
    lines = [
        "# Multi-Visit Reconstruction Results\n",
        "| Visits | Success | Images | Points | ATE (m) | Chamfer (m) | Completeness (%) | Time (s) |",
        "|--------|---------|--------|--------|---------|-------------|------------------|----------|",
    ]

    def _format_metric(value: Optional[float], precision: int = 4, percent: bool = False) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "N/A"
        if percent:
            return f"{value * 100:.1f}"
        return f"{value:.{precision}f}"

    for n_visits in sorted(results.keys()):
        r = results[n_visits]

        success_icon = "✅" if r.get('success', False) else "❌"
        num_images = r.get('num_images', 0)
        num_points = r.get('num_points', 0)
        ate = _format_metric(r.get('ate'))
        chamfer = _format_metric(r.get('chamfer'))
        completeness = _format_metric(r.get('completeness'), precision=1, percent=True)
        exec_time = r.get('execution_time', 0.0)

        line = (
            f"| {n_visits} | {success_icon} | {num_images} | {num_points} | "
            f"{ate} | {chamfer} | {completeness} | {exec_time:.1f} |"
        )
        lines.append(line)

    table = "\n".join(lines)

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table)
        logger.info(f"Results table saved to {output_path}")

    return table


def plot_metric_comparison(
    results: Dict[int, Dict[str, float]],
    metric_name: str,
    output_path: Path,
    ylabel: str = "Metric Value",
    title: Optional[str] = None,
) -> None:
    """
    Create single metric comparison plot.

    Args:
        results: Dict mapping visit count to metrics
        metric_name: Name of metric to plot (key in metrics dict)
        output_path: Where to save plot
        ylabel: Y-axis label
        title: Plot title (auto-generated if None)

    Example:
        >>> plot_metric_comparison(
        ...     results,
        ...     "ate",
        ...     Path("results/ate_comparison.png"),
        ...     ylabel="ATE (meters)",
        ...     title="Trajectory Error vs Visits"
        ... )
    """
    if not results:
        logger.warning("No results to plot")
        return

    visits = sorted(results.keys())
    values = [results[v].get(metric_name, 0) for v in visits]

    if title is None:
        title = f"{metric_name.upper()} vs Number of Visits"

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(visits, values, 'o-', linewidth=2, markersize=10)
    ax.set_xlabel('Number of Visits', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(visits)

    # Add value labels
    for x, y in zip(visits, values):
        ax.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to {output_path}")


# Validation
if __name__ == "__main__":
    """
    Validation of visualization.py functionality.
    Creates test plots to verify matplotlib functionality.
    """
    import sys
    import tempfile

    all_validation_failures = []
    total_tests = 0

    # Test 1: Create test plot
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_results = {
                1: {"ate": 0.50, "chamfer": 0.12, "completeness": 0.65},
                2: {"ate": 0.30, "chamfer": 0.08, "completeness": 0.82},
                3: {"ate": 0.20, "chamfer": 0.05, "completeness": 0.91},
            }

            output_path = Path(tmpdir) / "test_plot.png"
            plot_accuracy_vs_visits(test_results, output_path)

            if not output_path.exists():
                all_validation_failures.append("Plot creation: Output file not created")
            elif output_path.stat().st_size == 0:
                all_validation_failures.append("Plot creation: Output file is empty")

    except Exception as e:
        all_validation_failures.append(f"Plot creation: Exception raised: {e}")

    # Test 2: Results table creation
    total_tests += 1
    try:
        test_results = {
            1: {
                "success": True,
                "num_images": 50,
                "num_points": 1000,
                "ate": 0.5,
                "chamfer": 0.1,
                "completeness": 0.7,
                "execution_time": 60.0,
            },
        }

        table = create_results_table(test_results)

        if "Visits" not in table:
            all_validation_failures.append("Results table: Missing 'Visits' column header")
        if "50" not in table:
            all_validation_failures.append("Results table: Missing expected data (50 images)")

    except Exception as e:
        all_validation_failures.append(f"Results table: Exception raised: {e}")

    # Test 3: Empty results handling
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty_plot.png"
            plot_accuracy_vs_visits({}, output_path)

            # Should not create file for empty results
            # (This is acceptable behavior - just testing it doesn't crash)

    except Exception as e:
        all_validation_failures.append(f"Empty results: Exception raised: {e}")

    # Final validation result
    if all_validation_failures:
        print(
            f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Visualization module validated and ready for use")
        sys.exit(0)
