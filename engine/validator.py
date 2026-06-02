"""Walk-forward validation module for trading strategies.

Implements walk-forward optimization with rolling windows, Monte Carlo
simulation on OOS trades, and regime analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.table import Table

from engine.backtester import run_backtest
from engine.metrics import (
    calculate_expectancy,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_recovery_factor,
    calculate_sharpe,
    calculate_walk_forward_efficiency,
    calculate_win_rate,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_sortino,
    stitch_equity_curves,
)
from engine.optimizer import optimize_params, select_params_by_island_volume

# Console for rich output - disable legacy Windows mode for UTF-8 support
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
console = Console(force_terminal=True, legacy_windows=False)


def _select_window_params(
    opt_results: pl.DataFrame,
    strategy_class: type,
    train_data: pl.DataFrame,
    config: dict,
    method: str = "best",
) -> dict:
    """Select parameters for a WFO window using specified method.

    Args:
        opt_results: Optimization results from optimize_params() (sorted by objective desc)
        strategy_class: Strategy class for stability checking
        train_data: Training data for stability checking
        config: Full config dict (contains param_space and method-specific settings)
        method: Selection method - 'best', 'ivs', 'stability', 'multi_objective'

    Returns:
        Dict with selected parameter values

    Raises:
        ValueError: If opt_results is empty or method is invalid
    """
    if opt_results.is_empty():
        raise ValueError("Cannot select params from empty optimization results")

    param_space = config.get("param_space", {})
    validation_cfg = config.get("validation", {})
    wf_cfg = validation_cfg.get("walk_forward", {})

    # Get method-specific config
    ivs_cfg = wf_cfg.get("ivs", {})
    stability_cfg = wf_cfg.get("stability", {})
    mo_cfg = wf_cfg.get("multi_objective", {})

    # Extract param keys
    all_params = opt_results.row(0, named=True)
    param_keys = [
        k
        for k in all_params.keys()
        if k not in ["sharpe", "max_dd", "trade_count", "total_pnl"]
    ]

    if method == "best":
        # Current behavior: select highest IS Sharpe (row 0)
        best = all_params
        return {k: best[k] for k in param_keys}

    elif method == "ivs":
        # Option A: Island Volume Selection at window level
        from engine.optimizer import select_params_by_island_volume

        min_threshold = ivs_cfg.get("min_threshold", 0.0)

        # Dynamic threshold: top 80% of performers if not specified
        if min_threshold == 0.0:
            max_sharpe = opt_results["sharpe"].max()
            min_threshold = float(max_sharpe * 0.8)

        ivs_result = select_params_by_island_volume(
            results=opt_results,
            param_space=param_space,
            objective="sharpe",
            min_threshold=float(min_threshold),
        )

        selected = ivs_result.get("selected_params", {})

        # Fallback to best if IVS failed
        if not selected:
            best = all_params
            return {k: best[k] for k in param_keys}

        return selected

    elif method == "stability":
        # Option B: Stability-based selection
        # Check top N params, select one with best robustness score
        top_n = stability_cfg.get("top_n", 5)
        top_n = min(top_n, len(opt_results))

        top_results = opt_results.head(top_n)

        best_robust_score = -float("inf")
        best_stable_params = None

        for i in range(top_results.height):
            row = top_results.row(i, named=True)
            params = {k: row[k] for k in param_keys}

            # Reuse existing _check_flat_region for stability testing
            robustness = _check_flat_region(
                selected_params=params,
                strategy_class=strategy_class,
                train_data=train_data,
                config=config,
                perturbation_pct=stability_cfg.get("perturbation_pct", 0.1),
            )

            robust_score = robustness["flat_region_score"] * row["sharpe"]

            if robust_score > best_robust_score:
                best_robust_score = robust_score
                best_stable_params = params

        return (
            best_stable_params
            if best_stable_params
            else {k: all_params[k] for k in param_keys}
        )

    elif method == "multi_objective":
        # Option C: Sharpe × Stability optimization
        # Calculate stability for top N, pick max combined score

        stability_weight = mo_cfg.get("stability_weight", 0.3)
        top_n = stability_cfg.get("top_n", 10)  # Reuse stability config for efficiency
        top_n = min(top_n, len(opt_results))

        top_results = opt_results.head(top_n)

        # Collect Sharpe values for normalization
        sharpes = top_results["sharpe"].to_numpy()
        sharpe_mean = sharpes.mean()
        sharpe_std = sharpes.std() if len(sharpes) > 1 else 1.0
        sharpe_std = sharpe_std if sharpe_std > 0 else 1.0

        best_combined_score = -float("inf")
        best_combined_params = None

        for i in range(top_results.height):
            row = top_results.row(i, named=True)
            params = {k: row[k] for k in param_keys}

            # Get stability score
            robustness = _check_flat_region(
                selected_params=params,
                strategy_class=strategy_class,
                train_data=train_data,
                config=config,
                perturbation_pct=stability_cfg.get("perturbation_pct", 0.1),
            )

            stability_score = robustness["flat_region_score"]

            # Normalize Sharpe to 0-1 range (z-score)
            sharpe_normalized = (row["sharpe"] - sharpe_mean) / sharpe_std

            # Combined score: weighted average
            combined_score = (1 - stability_weight) * sharpe_normalized + (
                stability_weight * stability_score
            )

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_combined_params = params

        return (
            best_combined_params
            if best_combined_params
            else {k: all_params[k] for k in param_keys}
        )

    else:
        raise ValueError(
            f"Invalid window selection method: {method}. Must be 'best', 'ivs', 'stability', or 'multi_objective'"
        )


def _get_timeframe_ms(timeframe: str) -> int:
    """Convert timeframe string to milliseconds."""
    timeframe_map = {
        "1m": 60000,
        "3m": 180000,
        "5m": 300000,
        "15m": 900000,
        "30m": 1800000,
        "1h": 3600000,
        "4h": 14400000,
        "1d": 86400000,
    }
    return timeframe_map.get(timeframe, 86400000)


def _calculate_robustness_metrics(windows: list, param_keys: list) -> dict:
    """Calculate comprehensive robustness metrics from walk-forward windows.

    Args:
        windows: List of walk-forward window results with 'oos_sharpe', 'trade_count', 'best_params'
        param_keys: List of parameter names to analyze for stability

    Returns:
        Dict with consistency, performance_distribution, parameter_stability,
        total_windows, and valid_windows metrics. All values are Python floats
        for JSON serialization.

    Raises:
        None - handles all edge cases gracefully
    """
    if not windows:
        return {}

    # Filter windows with trades for OOS Sharpe analysis
    oos_sharpes = np.array([w["oos_sharpe"] for w in windows if w["trade_count"] > 0])

    if len(oos_sharpes) == 0:
        return {"total_windows": len(windows), "valid_windows": 0}

    # Consistency: % of windows with positive OOS Sharpe
    profitable_windows = float(np.sum(oos_sharpes > 0))
    consistency = {
        "profitable_pct": float(profitable_windows / len(oos_sharpes) * 100),
    }

    # Performance distribution statistics
    mean_sharpe = float(np.mean(oos_sharpes))
    std_sharpe = float(np.std(oos_sharpes))

    performance_distribution = {
        "mean": mean_sharpe,
        "std": std_sharpe,
        "min": float(np.min(oos_sharpes)),
        "max": float(np.max(oos_sharpes)),
        "p10": float(np.percentile(oos_sharpes, 10)),
        "p25": float(np.percentile(oos_sharpes, 25)),
        "p50": float(np.percentile(oos_sharpes, 50)),
        "p75": float(np.percentile(oos_sharpes, 75)),
        "p90": float(np.percentile(oos_sharpes, 90)),
        "coefficient_of_variation": (
            float(std_sharpe / abs(mean_sharpe)) if mean_sharpe != 0 else float("inf")
        ),
    }

    # Parameter stability: CV for each parameter across windows
    parameter_stability = {}
    for param in param_keys:
        # Extract parameter values from all windows that have this parameter
        values = np.array(
            [
                w["best_params"][param]
                for w in windows
                if param in w.get("best_params", {})
            ]
        )

        if len(values) == 0:
            continue

        mean_val = float(np.mean(values))
        std_val = float(np.std(values))

        # Calculate CV, handle zero mean case
        if mean_val != 0:
            cv = float(std_val / abs(mean_val))
        else:
            cv = float("inf")

        parameter_stability[param] = {
            "mean": mean_val,
            "std": std_val,
            "cv": cv,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
        }

    return {
        "consistency": consistency,
        "performance_distribution": performance_distribution,
        "parameter_stability": parameter_stability,
        "total_windows": len(windows),
        "valid_windows": len(oos_sharpes),
    }


def _select_params_by_frequency(windows: list, param_keys: list) -> dict:
    """Select parameters that appear most frequently as 'best' across windows.

    Args:
        windows: List of walk-forward window results with 'best_params'
        param_keys: List of parameter names (unused, for interface consistency)

    Returns:
        Dict with selected_params, frequency, confidence_pct, method, all_frequencies
    """
    if not windows:
        return {"selected_params": {}, "method": "frequency"}

    # Count parameter combinations
    param_counts = {}
    for w in windows:
        if "best_params" in w and w["best_params"]:
            param_tuple = tuple(sorted(w["best_params"].items()))
            param_counts[param_tuple] = param_counts.get(param_tuple, 0) + 1

    if not param_counts:
        return {"selected_params": {}, "method": "frequency"}

    # Find most frequent
    most_frequent = max(param_counts.items(), key=lambda x: x[1])
    selected_params = dict(most_frequent[0])
    frequency = most_frequent[1]
    total_windows = len(windows)

    return {
        "selected_params": selected_params,
        "frequency": frequency,
        "confidence_pct": (frequency / total_windows) * 100,
        "method": "frequency",
        "all_frequencies": [
            {"params": dict(k), "count": v}
            for k, v in sorted(param_counts.items(), key=lambda x: x[1], reverse=True)
        ],
    }


def _select_params_by_clustering(
    windows: list, param_keys: list, n_clusters: int = 2
) -> dict:
    """Use K-Means clustering to identify parameter regimes and select robust params.

    Args:
        windows: List of walk-forward window results with 'oos_sharpe', 'trade_count', 'best_params'
        param_keys: List of parameter names to cluster on
        n_clusters: Number of clusters for K-Means (default 2)

    Returns:
        Dict with selected_params, cluster_id, cluster_performance, cluster_stability,
        method, and cluster_analysis. All numeric values are Python floats for JSON serialization.

    Raises:
        None - handles all edge cases gracefully, returns error dict on failure
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {
            "selected_params": {},
            "method": "kmeans",
            "error": "sklearn not installed",
        }

    if not windows or len(windows) < n_clusters * 2:
        return {"selected_params": {}, "method": "kmeans", "error": "insufficient data"}

    # Extract parameter vectors
    param_vectors = []
    window_metrics = []

    for w in windows:
        if "best_params" in w and w["best_params"]:
            vector = [w["best_params"].get(k, 0) for k in param_keys]
            param_vectors.append(vector)
            window_metrics.append(
                {
                    "oos_sharpe": w.get("oos_sharpe", 0),
                    "trade_count": w.get("trade_count", 0),
                }
            )

    if len(param_vectors) < n_clusters:
        return {
            "selected_params": {},
            "method": "kmeans",
            "error": "insufficient windows",
        }

    # Standardize parameters for clustering
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(np.array(param_vectors))

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_vectors)

    # Analyze each cluster
    cluster_analysis = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_params = [param_vectors[idx] for idx in cluster_indices]
        centroid = np.mean(cluster_params, axis=0)

        cluster_sharpes = [window_metrics[idx]["oos_sharpe"] for idx in cluster_indices]
        avg_performance = np.mean(cluster_sharpes)

        # Stability = inverse of CV (higher = more stable)
        param_std = np.std(cluster_params, axis=0)
        param_mean = np.mean(cluster_params, axis=0)
        stability = 1.0 / (np.mean(param_std / (np.abs(param_mean) + 1e-9)) + 1e-9)

        cluster_analysis.append(
            {
                "cluster_id": i,
                "size": len(cluster_indices),
                "centroid": {k: float(v) for k, v in zip(param_keys, centroid)},
                "avg_oos_sharpe": float(avg_performance),
                "stability_score": float(stability),
                "sharpe_std": float(np.std(cluster_sharpes)),
            }
        )

    # Select best cluster
    for cluster in cluster_analysis:
        cluster["selection_score"] = cluster["avg_oos_sharpe"] * np.log1p(
            cluster["stability_score"]
        )

    best_cluster = max(cluster_analysis, key=lambda x: x["selection_score"])

    return {
        "selected_params": best_cluster["centroid"],
        "cluster_id": best_cluster["cluster_id"],
        "cluster_performance": best_cluster["avg_oos_sharpe"],
        "cluster_stability": best_cluster["stability_score"],
        "method": "kmeans",
        "cluster_analysis": cluster_analysis,
    }


def _snap_to_param_space(value: float, param_space: list) -> float:
    """Snap a parameter value to the nearest value in param_space.

    IVS may return mean values (e.g., 62.5) that don't exist in the original space.
    This snaps them to the nearest valid value.

    Args:
        value: Parameter value to snap (may be float)
        param_space: List of valid parameter values

    Returns:
        Nearest valid parameter value
    """
    return min(param_space, key=lambda x: abs(x - value))


def _select_params_by_island_volume(
    windows: list, param_keys: list, param_space: dict
) -> dict:
    """Select parameters using Island Volume Selection across WFO windows.

    Aggregates best parameters from each window and applies IVS algorithm
    to identify the center of the largest plateau of robust performance.

    Args:
        windows: List of walk-forward window results with 'best_params' and 'oos_sharpe'
        param_keys: List of parameter names
        param_space: Parameter space definition (for adjacency calculation)

    Returns:
        Dict with selected_params, method, islands, largest_island
    """
    if not windows:
        return {"selected_params": {}, "method": "island_volume", "islands": []}

    # Build DataFrame from window results
    window_results = []
    for w in windows:
        row = {**w.get("best_params", {}), "sharpe": w.get("oos_sharpe", 0.0)}
        # CRITICAL FIX: Snap float parameter values to nearest valid param_space values
        # Window-level IVS may return mean values (e.g., 62.5) that don't exist in param_space
        # This causes KeyError in IVS adjacency graph building
        for param in param_keys:
            if param in row and param in param_space:
                param_value = row[param]
                if isinstance(param_value, float):
                    valid_space = param_space[param]
                    # Only snap if the value is not already in the valid space
                    if param_value not in valid_space:
                        row[param] = _snap_to_param_space(param_value, valid_space)
        window_results.append(row)

    if not window_results:
        return {"selected_params": {}, "method": "island_volume", "islands": []}

    results_df = pl.DataFrame(window_results)

    # Call optimizer's Island Volume Selection
    ivs_result = select_params_by_island_volume(
        results=results_df,
        param_space=param_space,
        objective="sharpe",
        min_threshold=0.0,
    )

    return ivs_result


def _check_flat_region(
    selected_params: dict,
    strategy_class: type,
    train_data: pl.DataFrame,
    config: dict,
    perturbation_pct: float = 0.1,
) -> dict:
    """Check if selected parameters are in a 'flat region' of performance landscape.

    Tests selected parameters against neighboring values (±10%) to detect
    parameter overfitting. Calculates performance degradation for each
    perturbation and returns a flat region score indicating robustness.

    Args:
        selected_params: Dictionary of parameter names and their selected values
        strategy_class: Strategy class to instantiate for testing
        train_data: Training data DataFrame for evaluation
        config: Configuration dictionary with backtest settings
        perturbation_pct: Percentage to perturb parameters (default 0.1 = 10%)

    Returns:
        Dictionary containing:
            - flat_region_score: float (0-1) - ratio of robust perturbations
            - is_robust: bool - True if flat_region_score >= 0.7
            - total_perturbations: int - total number of perturbation tests
            - robust_perturbations: int - number of perturbations with <20% degradation
            - perturbation_tests: list - detailed results for each test

    Raises:
        None - handles all exceptions gracefully
    """
    from engine.optimizer import _evaluate_params_worker

    perturbation_results = []
    base_performance = None

    # Test base parameters
    try:
        base_result = _evaluate_params_worker(
            (strategy_class, selected_params, train_data, config)
        )
        base_performance = base_result.get("sharpe", 0)
    except Exception:
        return {
            "flat_region_score": 0.0,
            "is_robust": False,
            "total_perturbations": 0,
            "robust_perturbations": 0,
            "perturbation_tests": [],
        }

    # Test perturbations for each parameter
    total_perturbations = 0
    robust_perturbations = 0

    for param, value in selected_params.items():
        if isinstance(value, (int, float)) and value != 0:
            perturbation = abs(value * perturbation_pct)

            for direction in [-1, 1]:
                perturbed_value = value + (perturbation * direction)

                # Round to reasonable precision
                if isinstance(value, int):
                    perturbed_value = int(round(perturbed_value))
                else:
                    perturbed_value = round(perturbed_value, 4)

                if perturbed_value == value:
                    continue

                perturbed_params = selected_params.copy()
                perturbed_params[param] = perturbed_value

                try:
                    perturbed_result = _evaluate_params_worker(
                        (strategy_class, perturbed_params, train_data, config)
                    )
                    perturbed_performance = perturbed_result.get("sharpe", 0)

                    # Calculate degradation
                    if base_performance != 0:
                        degradation = abs(
                            base_performance - perturbed_performance
                        ) / abs(base_performance)
                    else:
                        degradation = abs(base_performance - perturbed_performance)

                    is_robust = degradation < 0.2

                    if is_robust:
                        robust_perturbations += 1
                    total_perturbations += 1

                    perturbation_results.append(
                        {
                            "param": param,
                            "direction": direction,
                            "original_value": value,
                            "perturbed_value": perturbed_value,
                            "base_sharpe": base_performance,
                            "perturbed_sharpe": perturbed_performance,
                            "degradation_pct": degradation * 100,
                            "is_robust": is_robust,
                        }
                    )
                except Exception:
                    total_perturbations += 1
                    continue

    # Calculate flat region score
    if total_perturbations > 0:
        flat_region_score = robust_perturbations / total_perturbations
    else:
        flat_region_score = 0.0

    return {
        "flat_region_score": float(flat_region_score),
        "is_robust": flat_region_score >= 0.7,
        "total_perturbations": total_perturbations,
        "robust_perturbations": robust_perturbations,
        "perturbation_tests": perturbation_results,
    }


def select_final_params(
    windows: List[Dict],
    strategy_class: type,
    train_data: pl.DataFrame,
    config: dict,
    method: str = "auto",
    holdout_data: Optional[pl.DataFrame] = None,
) -> dict:
    """Select final parameters using multiple methods and robustness criteria.

    Methods:
    - 'island_volume': Island Volume Selection (IVS) - identifies parameter plateaus
    - 'frequency': Most frequently winning parameters
    - 'kmeans': Parameters from best cluster
    - 'auto': Automatically select based on data characteristics (prefers island_volume)

    Returns comprehensive selection results with all methods and recommendation.
    """
    if not windows:
        return {"error": "No windows to select from"}

    param_keys = list(windows[0].get("best_params", {}).keys()) if windows else []

    # Calculate robustness metrics
    robustness = _calculate_robustness_metrics(windows, param_keys)

    # Run all selection methods
    selection_results = {"robustness": robustness, "methods": {}}

    # Method 1: Island Volume Selection (primary method per institutional standards)
    ivs_result = _select_params_by_island_volume(
        windows, param_keys, config.get("param_space", {})
    )
    selection_results["methods"]["island_volume"] = ivs_result

    # Method 2: Frequency-based (fallback)
    freq_result = _select_params_by_frequency(windows, param_keys)
    selection_results["methods"]["frequency"] = freq_result

    # Method 3: K-Means clustering (if enough windows)
    if len(windows) >= 6:
        kmeans_result = _select_params_by_clustering(windows, param_keys, n_clusters=2)
        selection_results["methods"]["kmeans"] = kmeans_result

    # Auto-selection logic (prefers island_volume)
    if method == "auto":
        # Priority 1: Island Volume Selection (if it found islands)
        if (
            ivs_result.get("islands")
            and len(ivs_result.get("islands", [])) > 0
            and "fallback" not in ivs_result
        ):
            selection_results["recommended_method"] = "island_volume"
            selection_results["final_params"] = ivs_result["selected_params"]
        # Priority 2: K-Means (if strong performance differentiation)
        elif "kmeans" in selection_results["methods"]:
            kmeans_analysis = selection_results["methods"]["kmeans"].get(
                "cluster_analysis", []
            )
            if len(kmeans_analysis) >= 2:
                performance_gap = abs(
                    kmeans_analysis[0]["avg_oos_sharpe"]
                    - kmeans_analysis[1]["avg_oos_sharpe"]
                )
                if performance_gap > 0.3:
                    selection_results["recommended_method"] = "kmeans"
                    selection_results["final_params"] = selection_results["methods"][
                        "kmeans"
                    ]["selected_params"]
                else:
                    selection_results["recommended_method"] = "frequency"
                    selection_results["final_params"] = selection_results["methods"][
                        "frequency"
                    ]["selected_params"]
            else:
                selection_results["recommended_method"] = "frequency"
                selection_results["final_params"] = selection_results["methods"][
                    "frequency"
                ]["selected_params"]
        # Priority 3: Frequency (fallback)
        else:
            selection_results["recommended_method"] = "frequency"
            selection_results["final_params"] = selection_results["methods"][
                "frequency"
            ]["selected_params"]
    else:
        # Use specified method
        if method in selection_results["methods"]:
            selection_results["recommended_method"] = method
            selection_results["final_params"] = selection_results["methods"][method][
                "selected_params"
            ]
        else:
            selection_results["recommended_method"] = "frequency"
            selection_results["final_params"] = selection_results["methods"][
                "frequency"
            ]["selected_params"]

    if strategy_class is not None:
        flat_region_data = None
        data_source = "none"
        if holdout_data is not None and holdout_data.height > 0:
            flat_region_data = holdout_data
            data_source = "holdout"
        elif train_data is not None and train_data.height > 100:
            split_idx = int(train_data.height * 0.8)
            flat_region_data = train_data[split_idx:]
            data_source = "pseudo_holdout"

        if flat_region_data is not None:
            flat_region = _check_flat_region(
                selection_results["final_params"],
                strategy_class,
                flat_region_data,
                config,
            )
            flat_region["data_source"] = data_source
            selection_results["flat_region_check"] = flat_region

    return selection_results


def run_walk_forward(
    strategy_class: type,
    config: dict,
    full_data: pl.DataFrame,
    train_days: int = 548,
    test_days: int = 90,
    holdout_pct: float = 0.10,
    min_trades_per_window: int = 30,
    param_selection_method: str = "auto",
    robustness_thresholds: dict = None,
) -> dict:
    """Run walk-forward optimization with duration-based windows.

    Uses duration-based windowing per IMPROVEMENT.md institutional standards.
    Window sizing: 548 days train (~1.5 years for crypto), 90 days test (quarterly).

    Args:
        strategy_class: Strategy class to optimize
        config: Configuration dict with backtest settings
        full_data: Complete OHLCV dataset
        train_days: Training window duration in days (default 548)
        test_days: Test window duration in days (default 90)
        holdout_pct: Percentage of data to reserve for final holdout validation (default 0.10)
        min_trades_per_window: Minimum trades required per window (default 30)
        param_selection_method: Parameter selection method (default "auto")

    Returns:
        Dict with windows, final_params, stitched equity curve, and holdout data
    """
    total_bars = len(full_data)
    n_params = len(config.get("param_space", {}))

    # Duration-based window sizing (per IMPROVEMENT.md)
    timeframe_ms = _get_timeframe_ms(config.get("timeframe", "1d"))
    bars_per_day = 86400000 / timeframe_ms
    train_size = int(train_days * bars_per_day)
    test_size = int(test_days * bars_per_day)

    # Validate train/test ratio against parameter space size
    required_ratio = n_params**0.5
    actual_ratio = train_size / test_size
    if actual_ratio < required_ratio:
        console.print(
            f"[yellow]Warning: train/test ratio ({actual_ratio:.2f}) below sqrt(n_params) "
            f"recommendation ({required_ratio:.2f})[/yellow]"
        )

    step_size = test_size

    # Get walk-forward config for window selection
    validation_cfg = config.get("validation", {})
    wf_cfg = validation_cfg.get("walk_forward", {})

    console.print(
        f"[blue]Walk-Forward Setup:[/blue] Train Size: {train_size} bars, "
        f"Test Size: {test_size} bars, Step Size: {step_size} bars"
    )

    holdout_bars = int(total_bars * holdout_pct)
    wfo_data = full_data[: total_bars - holdout_bars]
    holdout_data = full_data[total_bars - holdout_bars :] if holdout_bars > 0 else None

    wfo_bars = len(wfo_data)

    total_windows = (wfo_bars - train_size - test_size) // step_size + 1

    windows = []
    all_oos_trades = []
    all_oos_equity_curves = []

    # Create progress console
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Walk-Forward Optimization",
            total=total_windows,
        )

        window_idx = 0
        while True:
            start_idx = window_idx * step_size
            train_end = start_idx + train_size
            test_end = train_end + test_size

            if test_end > wfo_bars:
                break

            train_data = wfo_data[start_idx:train_end]

            try:
                opt_results = optimize_params(
                    strategy_class=strategy_class,
                    param_space=config.get("param_space", {}),
                    train_data=train_data,
                    method="grid",
                    config=config,
                )

                if opt_results.is_empty():
                    progress.update(task, advance=1)
                    window_idx += 1
                    continue

                # Select parameters using configured method
                window_selection_method = wf_cfg.get("window_selection_method", "best")

                try:
                    best_params = _select_window_params(
                        opt_results=opt_results,
                        strategy_class=strategy_class,
                        train_data=train_data,
                        config=config,
                        method=window_selection_method,
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]Window {window_idx}: {window_selection_method} selection failed, using best: {e}[/yellow]"
                    )
                    # Fallback to best
                    best_row = opt_results.row(0, named=True)
                    param_keys = [
                        k
                        for k in best_row.keys()
                        if k not in ["sharpe", "max_dd", "trade_count", "total_pnl"]
                    ]
                    best_params = {k: best_row[k] for k in param_keys}

                test_data = wfo_data[train_end:test_end].clone()
                strategy = strategy_class(best_params)
                signals = strategy.generate_signals(test_data)

                # Use fixed_amount for WFO to isolate strategy robustness from compounding effects
                wfo_config = config.copy()
                wfo_config["mode"] = "fixed_amount"

                trade_log, equity_curve = run_backtest(signals, wfo_config)

                if trade_log.height == 0 or equity_curve.height < 2:
                    oos_sharpe = 0.0
                    max_dd = 0.0
                else:
                    oos_sharpe = calculate_sharpe(equity_curve)
                    max_dd, _ = calculate_max_drawdown(equity_curve)
                    all_oos_equity_curves.append(equity_curve)

                windows.append(
                    {
                        "window_idx": window_idx,
                        "start_idx": start_idx,
                        "train_end": train_end,
                        "test_end": test_end,
                        "best_params": best_params,
                        "oos_sharpe": oos_sharpe,
                        "oos_max_dd": max_dd,
                        "trade_count": trade_log.height,
                    }
                )

                # Only append if there are trades
                if trade_log.height > 0:
                    all_oos_trades.append(trade_log)

                # Update progress with current metrics
                param_str = ", ".join(
                    [f"{k}={v}" for k, v in list(best_params.items())[:3]]
                )
                if len(best_params) > 3:
                    param_str += "..."
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Walk-Forward: [green]Sharpe={oos_sharpe:.2f} [yellow]DD={max_dd:.2%} [blue]Trades={trade_log.height} [magenta]{param_str}",
                )

            except Exception as e:
                console.print(f"[red]Window {window_idx} failed: {e}[/red]")
                progress.update(task, advance=1)
                window_idx += 1
                continue

            window_idx += 1

    if all_oos_trades:
        combined_trades = pl.concat(all_oos_trades)
    else:
        combined_trades = pl.DataFrame()

    param_keys = list(windows[0].get("best_params", {}).keys()) if windows else []
    robustness_metrics = _calculate_robustness_metrics(windows, param_keys)

    selection_results = select_final_params(
        windows=windows,
        strategy_class=strategy_class,
        train_data=wfo_data,
        config=config,
        method=param_selection_method,
        holdout_data=holdout_data,
    )

    stitched_equity = None
    wfe = None
    if all_oos_equity_curves:
        try:
            stitched_equity = stitch_equity_curves(all_oos_equity_curves)
            if stitched_equity.height >= 2 and len(all_oos_equity_curves) > 0:
                first_equity = all_oos_equity_curves[0]
                wfe = calculate_walk_forward_efficiency(first_equity, stitched_equity)
        except Exception:
            pass

    # Calculate verdict with configurable thresholds
    verdict = _calculate_pass_fail_verdict(
        robustness_metrics,
        selection_results,
        robustness_thresholds=robustness_thresholds,
    )

    diagnostic_results = None
    try:
        validation_cfg = config.get("validation", {})
        checks_config = validation_cfg.get("statistical_checks", {})

        if checks_config.get("enabled", True):
            console.print("[cyan]Running statistical diagnostic checks...[/cyan]")

            from engine.diagnostic_utils import run_all_diagnostics

            diagnostic_results = run_all_diagnostics(
                data=full_data,
                windows=windows,
                train_days=train_days,
                test_days=test_days,
                step_size=step_size,
                timeframe=config.get("timeframe", "1d"),
                config=validation_cfg,
                param_space=config.get("param_space", {}),
                strategy_class=strategy_class,
                train_data=wfo_data,
            )

            summary = diagnostic_results.get("summary", {})
            status = summary.get("overall_status", "UNKNOWN")

            if status == "PASS":
                console.print(
                    f"[green]Diagnostic Checks: {summary.get('overall_message', 'All checks passed')}[/green]"
                )
            elif status == "WARNING":
                console.print(
                    f"[yellow]Diagnostic Checks: {summary.get('overall_message', 'Warnings detected')}[/yellow]"
                )
            elif status == "FAIL":
                console.print(
                    f"[red]Diagnostic Checks: {summary.get('overall_message', 'Critical issues detected')}[/red]"
                )

            for check_name, check_result in diagnostic_results.get(
                "checks", {}
            ).items():
                severity = check_result.get("severity", "INFO")
                if severity in ["WARNING", "ERROR"]:
                    rec = check_result.get("recommendation", "")
                    if severity == "ERROR":
                        console.print(f"  [red]✗ {check_name}: {rec}[/red]")
                    else:
                        console.print(f"  [yellow]⚠ {check_name}: {rec}[/yellow]")

    except Exception as e:
        console.print(
            f"[yellow]Warning: Statistical diagnostic checks failed: {e}[/yellow]"
        )
        diagnostic_results = {"error": str(e), "enabled": True}

    return {
        "windows": windows,
        "final_params": selection_results.get("final_params", {}),
        "combined_oos_trades": combined_trades,
        "stitched_equity_curve": stitched_equity,
        "wfe": wfe,
        "robustness": robustness_metrics,
        "parameter_selection": selection_results,
        "holdout_data": holdout_data,
        "verdict": verdict,
        "diagnostics": diagnostic_results,
    }


def run_combined_monte_carlo(
    trade_log: pl.DataFrame,
    n_simulations: int = 10000,
    skip_pct: float = 0.10,
    slippage_pip: float = 1.0,
    entry_delay_bars: int = 1,
    pip_value: float = 0.1,
    percentile_95_dd_max_pct: float = 25.0,
    dd_increase_fail_pct: float = 25.0,
    dd_increase_marginal_pct: float = 15.0,
    use_block_bootstrap: bool = True,
    block_size: int = 5,
) -> dict:
    """Run combined Monte Carlo stress test with multiple perturbation factors.

    Applies stress factors together in each simulation:
    1. Trade reshuffling (random order) - optionally block bootstrap
    2. Trade skipping (simulating execution failures)
    3. Slippage (1.0 pip per trade)
    4. Entry delay (1-bar lag)

    Args:
        trade_log: Trade log DataFrame with 'pnl' column
        n_simulations: Number of simulations to run (default 10000)
        skip_pct: Percentage of trades to randomly skip (default 0.10 = 10%)
        slippage_pip: Slippage in pips to apply per trade (default 1.0)
        entry_delay_bars: Bars to delay entry (affects compounding, default 1)
        pip_value: Value of 1 pip in quote currency (default 0.0001 for forex/crypto)
        use_block_bootstrap: If True, preserve trade clustering (default True)
        block_size: Size of blocks for bootstrap (default 5 trades)

    Returns:
        Dict with simulation results including robustness grade
    """
    if trade_log.is_empty():
        return {
            "n_simulations": 0,
            "final_returns": [],
            "max_drawdowns": [],
            "robustness_grade": "UNKNOWN",
        }

    trades = trade_log["pnl"].to_numpy()
    n_trades = len(trades)

    initial_capital = 10000.0
    final_returns = []
    max_drawdowns = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Combined Monte Carlo Stress Test",
            total=n_simulations,
        )

        for _ in range(n_simulations):
            if use_block_bootstrap and n_trades >= block_size:
                n_blocks = n_trades // block_size
                blocks = [
                    trades[i * block_size : (i + 1) * block_size]
                    for i in range(n_blocks)
                ]
                shuffled_blocks = np.random.permutation(blocks)
                shuffled = np.concatenate(shuffled_blocks)
                remainder = n_trades % block_size
                if remainder > 0:
                    shuffled = np.concatenate([shuffled, trades[-remainder:]])
            else:
                shuffled = np.random.permutation(trades)

            n_skip = int(n_trades * skip_pct)
            if n_skip > 0:
                skip_indices = np.random.choice(n_trades, size=n_skip, replace=False)
                shuffled[skip_indices] = 0

            slippage_per_trade = slippage_pip * pip_value
            adjusted_pnl = shuffled - slippage_per_trade

            equity_curve = np.zeros(n_trades + entry_delay_bars)
            equity_curve[:entry_delay_bars] = initial_capital
            equity_curve[entry_delay_bars:] = initial_capital + np.cumsum(adjusted_pnl)

            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_dd = np.max(drawdown)

            final_returns.append(equity_curve[-1])
            max_drawdowns.append(max_dd)

            if len(final_returns) % 100 == 0:
                median_return = np.median(final_returns)
                median_dd = np.median(max_drawdowns)
                progress.update(
                    task,
                    advance=100,
                    description=f"[cyan]MC Stress: [green]Median=${median_return:.2f} [yellow]95th DD={np.percentile(max_drawdowns, 95):.2%}",
                )

        progress.update(task, advance=n_simulations % 100)

    original_max_dd = _calculate_original_max_dd(trade_log)
    p95_dd = np.percentile(max_drawdowns, 95)
    median_dd = np.median(max_drawdowns)

    # Calculate new metrics for robust validation
    pct_5_return = np.percentile(final_returns, 5)
    median_return = np.median(final_returns)

    # % of simulations that are profitable (final equity > initial capital)
    profitable_sims = np.sum(np.array(final_returns) > initial_capital)
    pct_profitable_sims = (profitable_sims / n_simulations) * 100

    # Risk of ruin - % of simulations ending at or below zero
    ruin_sims = np.sum(np.array(final_returns) <= 0)
    risk_of_ruin = (ruin_sims / n_simulations) * 100

    # Return/DD ratios at various percentiles
    if p95_dd > 0:
        return_dd_ratio_5 = (
            (pct_5_return - initial_capital) / initial_capital
        ) / p95_dd
        return_dd_ratio_50 = (
            (median_return - initial_capital) / initial_capital
        ) / median_dd
    else:
        return_dd_ratio_5 = float("inf") if median_return > initial_capital else 0.0
        return_dd_ratio_50 = float("inf") if median_return > initial_capital else 0.0

    # Convert percentage threshold to decimal
    max_dd_threshold = percentile_95_dd_max_pct / 100.0

    # ALWAYS calculate DD increase to show actual values to user
    if original_max_dd > 0:
        dd_increase_pct = ((p95_dd - original_max_dd) / original_max_dd) * 100
    else:
        dd_increase_pct = 0.0

    # PROFITABILITY-FIRST grading: A strategy must be PROFITABLE under stress
    # PASS requires ALL criteria met:
    # 1. PRIMARY: 5th percentile return > 0 (must make money in worst case)
    # 2. 95th percentile DD <= 25%
    # 3. Risk of ruin = 0% (no simulation hits zero)
    # 4. At least 80% of simulations profitable

    pct_5_threshold = 0.0  # minimum 5th percentile return (% of initial capital)
    pct_profitable_min = 80.0  # minimum % of profitable simulations
    risk_of_ruin_max = 0.0  # maximum allowed risk of ruin

    p5_return_pct = ((pct_5_return - initial_capital) / initial_capital) * 100

    criteria_passed = {
        "p5_return_positive": pct_5_return > initial_capital,
        "dd_within_limit": p95_dd <= max_dd_threshold,
        "no_ruin": risk_of_ruin <= risk_of_ruin_max,
        "sufficient_profitable": pct_profitable_sims >= pct_profitable_min,
    }

    all_passed = all(criteria_passed.values())

    if all_passed:
        robustness_grade = "PASS"
    else:
        robustness_grade = "FAIL"

    return {
        "n_simulations": n_simulations,
        "final_returns": final_returns,
        "max_drawdowns": max_drawdowns,
        "percentile_5_return": pct_5_return,
        "percentile_95_return": np.percentile(final_returns, 95),
        "median_return": median_return,
        "percentile_95_dd": p95_dd,
        "percentile_50_dd": median_dd,
        "original_max_dd": original_max_dd,
        "dd_increase_pct": dd_increase_pct,
        "mc_max_dd": np.max(max_drawdowns),
        "mc_min_dd": np.min(max_drawdowns),
        "robustness_grade": robustness_grade,
        "use_block_bootstrap": use_block_bootstrap,
        "block_size": block_size,
        "pct_profitable_sims": pct_profitable_sims,
        "risk_of_ruin": risk_of_ruin,
        "return_dd_ratio_5": return_dd_ratio_5,
        "return_dd_ratio_50": return_dd_ratio_50,
        "p5_return_pct": p5_return_pct,
        "criteria_passed": criteria_passed,
    }


def run_full_backtest_monte_carlo(
    strategy_class: type,
    final_params: dict,
    data: pl.DataFrame,
    config: dict,
    n_simulations: int = 10000,
    skip_pct: float = 0.10,
    slippage_pip: float = 1.0,
    entry_delay_bars: int = 1,
    pip_value: float = 0.0001,
    percentile_95_dd_max_pct: float = 25.0,
    dd_increase_fail_pct: float = 25.0,
    dd_increase_marginal_pct: float = 15.0,
) -> dict:
    """Run Monte Carlo stress test on full backtest with selected final parameters.

    This tests whether the FINAL selected strategy (with final_params) is robust
    to execution issues, rather than testing individual window trades.

    Args:
        strategy_class: Strategy class to test
        final_params: Final selected parameters from WFO
        data: Data to run backtest on (typically WFO data)
        config: Configuration dict with backtest settings
        n_simulations: Number of Monte Carlo simulations
        skip_pct: Percentage of trades to randomly skip
        slippage_pip: Slippage in pips per trade
        entry_delay_bars: Bars to delay entry
        pip_value: Value of 1 pip
        percentile_95_dd_max_pct: Max 95th percentile DD threshold
        dd_increase_fail_pct: DD increase threshold for FAIL
        dd_increase_marginal_pct: DD increase threshold for MARGINAL

    Returns:
        Dict with Monte Carlo results including robustness grade
    """
    from engine.backtester import run_backtest

    strategy = strategy_class(final_params)
    signals = strategy.generate_signals(data)

    if signals.is_empty() or "signal" not in signals.columns:
        return {
            "n_simulations": 0,
            "robustness_grade": "UNKNOWN",
            "reason": "No signals generated",
        }

    mc_config = config.copy()
    mc_config["mode"] = "fixed_amount"

    trade_log, equity_curve = run_backtest(signals, mc_config, show_progress=False)

    if trade_log.is_empty():
        return {
            "n_simulations": 0,
            "robustness_grade": "UNKNOWN",
            "reason": "No trades generated",
        }

    mc_results = run_combined_monte_carlo(
        trade_log=trade_log,
        n_simulations=n_simulations,
        skip_pct=skip_pct,
        slippage_pip=slippage_pip,
        entry_delay_bars=entry_delay_bars,
        pip_value=pip_value,
        percentile_95_dd_max_pct=percentile_95_dd_max_pct,
        dd_increase_fail_pct=dd_increase_fail_pct,
        dd_increase_marginal_pct=dd_increase_marginal_pct,
    )

    mc_results["source"] = "full_backtest"
    mc_results["trade_count"] = trade_log.height
    mc_results["final_params"] = final_params

    return mc_results


def _calculate_original_max_dd(trade_log: pl.DataFrame) -> float:
    """Calculate original max drawdown from trade log."""
    if trade_log.is_empty():
        return 0.0

    trades = trade_log["pnl"].to_numpy()
    initial_capital = 10000.0
    equity_curve = initial_capital + np.cumsum(trades)

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak

    return float(np.max(drawdown))


def run_monte_carlo(trade_log: pl.DataFrame, n_simulations: int = 1000) -> dict:
    """Run Monte Carlo simulation on OOS trades.

    Args:
        trade_log: Trade log DataFrame
        n_simulations: Number of simulations to run

    Returns:
        Dict with simulation results
    """
    if trade_log.is_empty():
        return {"n_simulations": 0, "final_returns": [], "max_drawdowns": []}

    trades = trade_log["pnl"].to_numpy()
    n_trades = len(trades)

    initial_capital = 10000.0
    final_returns = []
    max_drawdowns = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Monte Carlo Simulation",
            total=n_simulations,
        )

        for _ in range(n_simulations):
            # Resample with replacement
            resampled = np.random.choice(trades, size=n_trades, replace=True)
            equity_curve = initial_capital + np.cumsum(resampled)

            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_dd = np.max(drawdown)

            final_returns.append(equity_curve[-1])
            max_drawdowns.append(max_dd)

            # Update progress every 10 simulations
            if len(final_returns) % 10 == 0:
                median_return = np.median(final_returns)
                median_dd = np.median(max_drawdowns)
                progress.update(
                    task,
                    advance=10,
                    description=f"[cyan]Monte Carlo: [green]Median Return=${median_return:.2f} [yellow]Median DD={median_dd:.2%}",
                )

        # Final update to ensure completion
        progress.update(task, advance=n_simulations % 10)

    return {
        "n_simulations": n_simulations,
        "final_returns": final_returns,
        "max_drawdowns": max_drawdowns,
        "percentile_5_return": np.percentile(final_returns, 5),
        "percentile_95_return": np.percentile(final_returns, 95),
        "median_return": np.median(final_returns),
    }


def validate_multi_metric_standard(
    equity_curve: pl.DataFrame,
    trade_log: pl.DataFrame,
    wfe: float | None = None,
    cv: float | None = None,
    first_equity: pl.DataFrame | None = None,
    thresholds: dict = None,
) -> dict:
    """Validate strategy against the Multi-Metric Standard.

    The Multi-Metric Standard evaluates strategies on 9 PASS/FAIL metrics
    plus 2 informational metrics:

    PASS/FAIL Metrics (9 total):
    - Performance: Sharpe Ratio, Sortino Ratio, Calmar Ratio
    - Risk: Max Drawdown
    - Trade: Recovery Factor, Profit Factor, Expected Value
    - Stability: WFE, CV

    Informational Metrics (2):
    - CAGR: Annualized return (position-sizing dependent, not a reliable pass/fail indicator)
    - Win Rate: Trade win frequency (strategy-type dependent, not a reliable pass/fail indicator)

    Args:
        equity_curve: Stitched OOS equity curve with 'timestamp' and 'equity' columns
        trade_log: Combined OOS trade log with 'pnl' column
        wfe: Walk Forward Efficiency percentage (if pre-calculated)
        cv: Coefficient of Variation (if pre-calculated)
        first_equity: First window's equity curve for WFE calculation (if wfe not provided)
        thresholds: Configuration dict with all threshold values

    Returns:
        Dict with pass/fail metrics, informational metrics, and overall verdict
    """
    if equity_curve.height < 2:
        return {"overall_verdict": "INSUFFICIENT_DATA", "metrics": {}}

    if trade_log.height == 0:
        return {"overall_verdict": "NO_TRADES", "metrics": {}}

    # Load thresholds from config or use defaults
    if thresholds is None:
        thresholds = {}

    # Performance Metrics
    sharpe_min = thresholds.get("sharpe_min", 1.0)
    sortino_min = thresholds.get("sortino_min", 1.0)
    calmar_min = thresholds.get("calmar_min", 1.0)
    cagr_min_pct = thresholds.get("cagr_min_pct", 10.0)

    # Risk Metrics
    max_dd_max_pct = thresholds.get("max_drawdown_max_pct", 30.0)

    # Trade Metrics
    recovery_factor_min = thresholds.get("recovery_factor_min", 2.0)
    profit_factor_min = thresholds.get("profit_factor_min", 1.5)
    win_rate_min_pct = thresholds.get("win_rate_min_pct", 45.0)
    expectancy_min = thresholds.get("expectancy_min", 0.0)

    # Stability Metrics
    wfe_min_pct = thresholds.get("wfe_min_pct", 60.0)
    cv_max_pct = thresholds.get("cv_max_pct", 20.0)

    # Pass/Fail Criteria
    conditional_min_pct = thresholds.get("conditional_min_pct", 75.0)

    metrics = {}
    checks = []
    informational = []

    # 1. Sharpe Ratio
    try:
        sharpe = calculate_sharpe(equity_curve)
        metrics["sharpe"] = sharpe
        checks.append(("Sharpe Ratio", sharpe, sharpe_min, sharpe > sharpe_min))
    except Exception:
        metrics["sharpe"] = 0.0
        checks.append(("Sharpe Ratio", 0, sharpe_min, False))

    # 2. Sortino Ratio (NEW)
    try:
        sortino = calculate_sortino(equity_curve)
        metrics["sortino"] = sortino
        checks.append(("Sortino Ratio", sortino, sortino_min, sortino > sortino_min))
    except Exception:
        metrics["sortino"] = 0.0
        checks.append(("Sortino Ratio", 0, sortino_min, False))

    # 3. Calmar Ratio (NEW)
    try:
        calmar = calculate_calmar_ratio(equity_curve)
        metrics["calmar"] = calmar
        checks.append(("Calmar Ratio", calmar, calmar_min, calmar > calmar_min))
    except Exception:
        metrics["calmar"] = 0.0
        checks.append(("Calmar Ratio", 0, calmar_min, False))

    # 4. CAGR (INFORMATIONAL - position-sizing dependent)
    try:
        cagr = calculate_cagr(equity_curve)
        cagr_pct = cagr * 100
        metrics["cagr"] = cagr
        informational.append(("CAGR", cagr_pct, cagr_min_pct))
    except Exception:
        metrics["cagr"] = 0.0
        informational.append(("CAGR", 0, cagr_min_pct))

    # 5. Max Drawdown
    try:
        max_dd, _ = calculate_max_drawdown(equity_curve)
        metrics["max_drawdown"] = max_dd
        max_dd_pct = max_dd * 100  # Convert to percentage for display and check
        checks.append(
            ("Max Drawdown", max_dd_pct, max_dd_max_pct, max_dd_pct < max_dd_max_pct)
        )
    except Exception:
        metrics["max_drawdown"] = 1.0
        checks.append(("Max Drawdown", 100, max_dd_max_pct, False))

    # 6. Recovery Factor
    try:
        recovery_factor = calculate_recovery_factor(trade_log)
        metrics["recovery_factor"] = recovery_factor
        checks.append(
            (
                "Recovery Factor",
                recovery_factor,
                recovery_factor_min,
                recovery_factor > recovery_factor_min,
            )
        )
    except Exception:
        metrics["recovery_factor"] = 0.0
        checks.append(("Recovery Factor", 0, recovery_factor_min, False))

    # 7. Profit Factor
    try:
        profit_factor = calculate_profit_factor(trade_log)
        metrics["profit_factor"] = profit_factor
        checks.append(
            (
                "Profit Factor",
                profit_factor,
                profit_factor_min,
                profit_factor > profit_factor_min,
            )
        )
    except Exception:
        metrics["profit_factor"] = 0.0
        checks.append(("Profit Factor", 0, profit_factor_min, False))

    # 8. Win Rate (INFORMATIONAL - strategy-type dependent)
    try:
        win_rate = calculate_win_rate(trade_log)
        metrics["win_rate"] = win_rate
        informational.append(("Win Rate", win_rate, win_rate_min_pct))
    except Exception:
        metrics["win_rate"] = 0.0
        informational.append(("Win Rate", 0, win_rate_min_pct))

    # 9. Expected Value
    try:
        expectancy = calculate_expectancy(trade_log)
        metrics["expectancy"] = expectancy
        checks.append(
            ("Expected Value", expectancy, expectancy_min, expectancy > expectancy_min)
        )
    except Exception:
        metrics["expectancy"] = 0.0
        checks.append(("Expected Value", 0, expectancy_min, False))

    # 10. Walk Forward Efficiency
    if wfe is None and first_equity is not None:
        try:
            wfe = calculate_walk_forward_efficiency(first_equity, equity_curve)
        except Exception:
            pass

    if wfe is not None:
        metrics["wfe"] = wfe
        checks.append(("Walk Forward Efficiency", wfe, wfe_min_pct, wfe > wfe_min_pct))
    else:
        metrics["wfe"] = None
        checks.append(("Walk Forward Efficiency", "N/A", wfe_min_pct, False))

    # 11. Coefficient of Variation
    if cv is not None:
        metrics["cv"] = cv
        checks.append(("Coefficient of Variation", cv, cv_max_pct, cv < cv_max_pct))
    else:
        metrics["cv"] = None
        checks.append(("Coefficient of Variation", "N/A", cv_max_pct, False))

    passed_count = sum(1 for _, _, _, passed in checks if passed)
    total_count = len(checks)

    # Convert conditional percentage to decimal
    conditional_threshold = conditional_min_pct / 100.0

    if passed_count == total_count:
        overall_verdict = "PASS"
    elif passed_count >= total_count * conditional_threshold:
        overall_verdict = "CONDITIONAL"
    else:
        overall_verdict = "FAIL"

    return {
        "overall_verdict": overall_verdict,
        "passed_count": passed_count,
        "total_count": total_count,
        "metrics": metrics,
        "checks": [
            {"name": name, "value": val, "threshold": thresh, "passed": passed}
            for name, val, thresh, passed in checks
        ],
        "informational": [
            {"name": name, "value": val, "threshold": thresh}
            for name, val, thresh in informational
        ],
    }


def run_holdout_validation(
    strategy_class: type,
    final_params: dict,
    holdout_data: pl.DataFrame,
    config: dict,
    sharpe_min: float = 1.0,
    max_drawdown_max_pct: float = 30.0,
) -> dict:
    """Run final validation on reserved holdout data.

    The holdout set is the most recent 10% of data that was NOT used
    in any WFO optimization or testing. This provides the final
    "True Out-of-Sample" verification before live trading.

    Args:
        strategy_class: Strategy class to validate
        final_params: Final selected parameters from WFO
        holdout_data: Reserved holdout OHLCV DataFrame
        config: Configuration dict with backtest settings
        sharpe_min: Minimum Sharpe ratio for holdout period (default 0.0)
        max_drawdown_max_pct: Maximum drawdown for holdout period (default 30.0%)

    Returns:
        Dict with holdout performance metrics and comparison to expectations
    """
    if holdout_data is None or holdout_data.height == 0:
        return {
            "status": "NO_HOLDOUT_DATA",
            "message": "No holdout data provided for validation",
        }

    strategy = strategy_class(final_params)
    signals = strategy.generate_signals(holdout_data)
    trade_log, equity_curve = run_backtest(signals, config)

    if trade_log.height == 0 or equity_curve.height < 2:
        return {
            "status": "NO_TRADES",
            "message": "No trades generated in holdout period",
            "trade_count": 0,
        }

    try:
        sharpe = calculate_sharpe(equity_curve)
    except Exception:
        sharpe = 0.0

    try:
        max_dd, _ = calculate_max_drawdown(equity_curve)
    except Exception:
        max_dd = 0.0

    try:
        profit_factor = calculate_profit_factor(trade_log)
    except Exception:
        profit_factor = 0.0

    try:
        recovery_factor = calculate_recovery_factor(trade_log)
    except Exception:
        recovery_factor = 0.0

    try:
        total_pnl = trade_log["pnl"].sum()
    except Exception:
        total_pnl = 0.0

    # Convert max drawdown percentage to decimal for comparison
    max_dd_threshold = max_drawdown_max_pct / 100.0

    holdout_performance = {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "recovery_factor": recovery_factor,
        "total_pnl": total_pnl,
        "trade_count": trade_log.height,
        "final_equity": equity_curve["equity"][-1],
        "status": "COMPLETE",
    }

    holdout_performance["verdict"] = (
        "PASS"
        if sharpe > sharpe_min and max_dd < max_dd_threshold
        else "MARGINAL" if sharpe > sharpe_min else "FAIL"
    )

    return holdout_performance


def run_final_strategy_evaluation(
    strategy_class: type,
    final_params: dict,
    wfo_data: pl.DataFrame,
    config: dict,
    kelly_fraction: float = 0.25,
) -> dict:
    """Run final strategy evaluation with optimal position sizing.

    This function:
    1. Runs backtest with final params on WFO data (using config's position sizing)
    2. Calculates Kelly fraction from those trades
    3. Re-runs backtest with optimal Kelly sizing
    4. Returns results for multi-metric standard evaluation

    This provides the "actual compounding effects" metric - what you can
    expect when trading live with final parameters and optimal position sizing.

    Args:
        strategy_class: Strategy class to evaluate
        final_params: Final selected parameters from WFO
        wfo_data: Full OHLCV data (WFO portion only, excluding holdout)
        config: Configuration dict with backtest settings
        kelly_fraction: Fraction of Kelly to use (default 0.25 = Quarter Kelly)

    Returns:
        Dict with:
        - trade_log: Trade log from final strategy backtest with optimal sizing
        - equity_curve: Equity curve from final strategy backtest
        - optimal_sizing: Kelly sizing information
        - backtest_metrics: Key metrics from the backtest
    """
    from engine.position_sizing import calculate_kelly_fraction
    from engine.backtester import run_backtest
    from engine.metrics import (
        calculate_sharpe,
        calculate_max_drawdown,
        calculate_profit_factor,
        calculate_recovery_factor,
        calculate_cagr,
        calculate_sortino,
        calculate_calmar_ratio,
        calculate_win_rate,
        calculate_expectancy,
    )

    strategy = strategy_class(final_params)
    signals = strategy.generate_signals(wfo_data)

    if signals.is_empty() or "signal" not in signals.columns:
        return {
            "trade_log": pl.DataFrame(),
            "equity_curve": pl.DataFrame(),
            "optimal_sizing": {},
            "backtest_metrics": {},
            "error": "No signals generated",
        }

    kelly_config = config.copy()
    kelly_config["mode"] = "fixed_amount"

    trade_log_initial, equity_curve_initial = run_backtest(
        signals,
        kelly_config,
        show_progress=False,
    )

    if trade_log_initial.height == 0:
        return {
            "trade_log": pl.DataFrame(),
            "equity_curve": pl.DataFrame(),
            "optimal_sizing": {},
            "backtest_metrics": {},
            "error": "No trades generated",
        }

    kelly_result = calculate_kelly_fraction(
        trade_log_initial,
        fraction=kelly_fraction,
        min_trades=30,
    )

    optimal_fraction = kelly_result["kelly_fraction"]

    final_config = config.copy()
    final_config["mode"] = "percent_equity"
    final_config["percent_equity"] = optimal_fraction

    trade_log, equity_curve = run_backtest(
        signals,
        final_config,
        show_progress=False,
    )

    backtest_metrics = {}
    if trade_log.height > 0 and equity_curve.height > 1:
        try:
            backtest_metrics["sharpe"] = calculate_sharpe(equity_curve)
        except Exception:
            backtest_metrics["sharpe"] = 0.0

        try:
            backtest_metrics["max_drawdown"], _ = calculate_max_drawdown(equity_curve)
        except Exception:
            backtest_metrics["max_drawdown"] = 0.0

        try:
            backtest_metrics["profit_factor"] = calculate_profit_factor(trade_log)
        except Exception:
            backtest_metrics["profit_factor"] = 0.0

        try:
            backtest_metrics["recovery_factor"] = calculate_recovery_factor(trade_log)
        except Exception:
            backtest_metrics["recovery_factor"] = 0.0

        try:
            backtest_metrics["cagr"] = calculate_cagr(equity_curve)
        except Exception:
            backtest_metrics["cagr"] = 0.0

        try:
            backtest_metrics["sortino"] = calculate_sortino(equity_curve)
        except Exception:
            backtest_metrics["sortino"] = 0.0

        try:
            backtest_metrics["calmar"] = calculate_calmar_ratio(equity_curve)
        except Exception:
            backtest_metrics["calmar"] = 0.0

        try:
            backtest_metrics["win_rate"] = calculate_win_rate(trade_log)
        except Exception:
            backtest_metrics["win_rate"] = 0.0

        try:
            backtest_metrics["expectancy"] = calculate_expectancy(trade_log)
        except Exception:
            backtest_metrics["expectancy"] = 0.0

        backtest_metrics["total_trades"] = trade_log.height
        backtest_metrics["total_pnl"] = float(trade_log["pnl"].sum())
        backtest_metrics["final_equity"] = float(equity_curve["equity"][-1])

    return {
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "optimal_sizing": kelly_result,
        "backtest_metrics": backtest_metrics,
        "position_sizing_mode": "percent_equity",
        "position_size_pct": optimal_fraction * 100,
    }


def validate_final_strategy_metrics(
    equity_curve: pl.DataFrame,
    trade_log: pl.DataFrame,
    thresholds: dict = None,
) -> dict:
    """Validate final strategy metrics without WFO stability metrics.

    This is used for the final strategy evaluation (after optimal sizing).
    Unlike validate_multi_metric_standard, this does NOT include:
    - WFE (Walk Forward Efficiency) - not applicable to single backtest
    - CV (Coefficient of Variation) - not applicable to single backtest

    This evaluates only: Performance (Sharpe, Sortino, Calmar, CAGR),
    Risk (Max Drawdown), and Trade metrics (Recovery Factor, Profit Factor,
    Win Rate, Expectancy).

    Args:
        equity_curve: Equity curve with 'timestamp' and 'equity' columns
        trade_log: Trade log with 'pnl' column
        thresholds: Configuration dict with threshold values

    Returns:
        Dict with 9 metrics, pass/fail status, and overall verdict
    """
    if equity_curve.height < 2:
        return {"overall_verdict": "INSUFFICIENT_DATA", "metrics": {}}

    if trade_log.height == 0:
        return {"overall_verdict": "NO_TRADES", "metrics": {}}

    if thresholds is None:
        thresholds = {}

    sharpe_min = thresholds.get("sharpe_min", 1.0)
    sortino_min = thresholds.get("sortino_min", 1.0)
    calmar_min = thresholds.get("calmar_min", 1.0)
    cagr_min_pct = thresholds.get("cagr_min_pct", 10.0)
    max_dd_max_pct = thresholds.get("max_drawdown_max_pct", 30.0)
    recovery_factor_min = thresholds.get("recovery_factor_min", 2.0)
    profit_factor_min = thresholds.get("profit_factor_min", 1.5)
    win_rate_min_pct = thresholds.get("win_rate_min_pct", 45.0)
    expectancy_min = thresholds.get("expectancy_min", 0.0)

    conditional_min_pct = thresholds.get("conditional_min_pct", 75.0)

    checks = []
    informational = []

    try:
        sharpe = calculate_sharpe(equity_curve)
        checks.append(("Sharpe Ratio", sharpe, sharpe_min, sharpe > sharpe_min))
    except Exception:
        checks.append(("Sharpe Ratio", 0, sharpe_min, False))

    try:
        sortino = calculate_sortino(equity_curve)
        checks.append(("Sortino Ratio", sortino, sortino_min, sortino > sortino_min))
    except Exception:
        checks.append(("Sortino Ratio", 0, sortino_min, False))

    try:
        calmar = calculate_calmar_ratio(equity_curve)
        checks.append(("Calmar Ratio", calmar, calmar_min, calmar > calmar_min))
    except Exception:
        checks.append(("Calmar Ratio", 0, calmar_min, False))

    try:
        cagr = calculate_cagr(equity_curve)
        cagr_pct = cagr * 100
        checks.append(("CAGR", cagr_pct, cagr_min_pct, cagr_pct > cagr_min_pct))
    except Exception:
        checks.append(("CAGR", 0, cagr_min_pct, False))

    try:
        max_dd, _ = calculate_max_drawdown(equity_curve)
        max_dd_pct = max_dd * 100
        checks.append(
            ("Max Drawdown", max_dd_pct, max_dd_max_pct, max_dd_pct < max_dd_max_pct)
        )
    except Exception:
        checks.append(("Max Drawdown", 100, max_dd_max_pct, False))

    try:
        recovery_factor = calculate_recovery_factor(trade_log)
        checks.append(
            ("Recovery Factor", recovery_factor, recovery_factor_min, recovery_factor > recovery_factor_min)
        )
    except Exception:
        checks.append(("Recovery Factor", 0, recovery_factor_min, False))

    try:
        profit_factor = calculate_profit_factor(trade_log)
        checks.append(
            ("Profit Factor", profit_factor, profit_factor_min, profit_factor > profit_factor_min)
        )
    except Exception:
        checks.append(("Profit Factor", 0, profit_factor_min, False))

    try:
        win_rate = calculate_win_rate(trade_log)
        informational.append(("Win Rate", win_rate, win_rate_min_pct))
    except Exception:
        informational.append(("Win Rate", 0, win_rate_min_pct))

    try:
        expectancy = calculate_expectancy(trade_log)
        checks.append(
            ("Expectancy", expectancy, expectancy_min, expectancy > expectancy_min)
        )
    except Exception:
        checks.append(("Expectancy", 0, expectancy_min, False))

    passed_count = sum(1 for _, _, _, passed in checks if passed)
    total_count = len(checks)

    conditional_threshold = conditional_min_pct / 100.0

    if passed_count == total_count:
        overall_verdict = "PASS"
    elif passed_count >= total_count * conditional_threshold:
        overall_verdict = "CONDITIONAL"
    else:
        overall_verdict = "FAIL"

    return {
        "overall_verdict": overall_verdict,
        "passed_count": passed_count,
        "total_count": total_count,
        "checks": [
            {"name": name, "value": val, "threshold": thresh, "passed": passed}
            for name, val, thresh, passed in checks
        ],
        "informational": [
            {"name": name, "value": val, "threshold": thresh}
            for name, val, thresh in informational
        ],
    }


# def run_regime_analysis(data: pl.DataFrame, trade_log: pl.DataFrame) -> dict:
#     """Analyze strategy performance across market regimes.

#     Args:
#         data: OHLCV DataFrame
#         trade_log: Trade log DataFrame

#     Returns:
#         Dict with regime breakdown
#     """
#     # Simple regime classification: trending up vs down vs sideways
#     # Using price change over 20-period window
#     price_change = data["close"].pct_change(20).fill_null(0)

#     regimes = price_change.map_elements(
#         lambda x: "trending_up" if x > 0.005 else ("trending_down" if x < -0.005 else "sideways"),
#         return_dtype=pl.Utf8,
#     )

#     return {
#         "regime_counts": regimes.to_pandas().value_counts().to_dict(),
#     }


def _calculate_pass_fail_verdict(
    robustness: dict,
    selection: dict,
    robustness_thresholds: dict = None,
) -> dict:
    """Calculate PASS/FAIL verdict based on robustness criteria with configurable thresholds."""
    # Load thresholds from config or use defaults
    if robustness_thresholds is None:
        robustness_thresholds = {}

    consistency_min_pct = robustness_thresholds.get("consistency_min_pct", 70.0)
    mean_performance_min_sharpe = robustness_thresholds.get(
        "mean_performance_min_sharpe", 0.5
    )
    performance_stability_max_cv = robustness_thresholds.get(
        "performance_stability_max_cv", 1.0
    )
    parameter_stability_max_cv = robustness_thresholds.get(
        "parameter_stability_max_cv", 0.5
    )
    flat_region_min_score = robustness_thresholds.get("flat_region_min_score", 0.7)
    conditional_min_pct = robustness_thresholds.get("conditional_min_pct", 80.0)

    checks = {}

    consistency = robustness.get("consistency", {})
    perf_dist = robustness.get("performance_distribution", {})

    # Check 1: Consistency
    profitable_pct = consistency.get("profitable_pct", 0)
    checks["consistency"] = {
        "metric": f"{profitable_pct:.1f}%",
        "threshold": f">= {consistency_min_pct}%",
        "passed": profitable_pct >= consistency_min_pct,
    }

    # Check 2: Mean Performance
    mean_sharpe = perf_dist.get("mean", 0)
    checks["mean_performance"] = {
        "metric": f"{mean_sharpe:.2f}",
        "threshold": f">= {mean_performance_min_sharpe}",
        "passed": mean_sharpe >= mean_performance_min_sharpe,
    }

    # Check 3: Performance Stability
    cv = perf_dist.get("coefficient_of_variation", float("inf"))
    checks["performance_stability"] = {
        "metric": f"{cv:.2f}",
        "threshold": f"<= {performance_stability_max_cv}",
        "passed": cv <= performance_stability_max_cv,
    }

    # Check 4: Parameter Stability
    param_stability = robustness.get("parameter_stability", {})
    unstable_params = [
        p
        for p, s in param_stability.items()
        if s.get("cv", 0) > parameter_stability_max_cv
    ]
    checks["parameter_stability"] = {
        "metric": f"{len(unstable_params)}",
        "threshold": "0",
        "passed": len(unstable_params) == 0,
    }

    # Check 5: Flat Region
    flat_check = selection.get("flat_region_check", {})
    flat_score = flat_check.get("flat_region_score", 0)
    checks["flat_region"] = {
        "metric": f"{flat_score:.2f}",
        "threshold": f">= {flat_region_min_score}",
        "passed": flat_score >= flat_region_min_score,
    }

    # Overall verdict
    total_checks = len(checks)
    passed_checks = sum(1 for c in checks.values() if c["passed"])

    # Convert conditional percentage to decimal
    conditional_threshold = conditional_min_pct / 100.0

    if passed_checks == total_checks:
        overall = "PASS"
    elif passed_checks >= total_checks * conditional_threshold:
        overall = "CONDITIONAL"
    else:
        overall = "FAIL"

    return {
        "overall": overall,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "checks": checks,
    }


def _generate_recommendations(verdict: dict, robustness: dict, selection: dict) -> list:
    """Generate actionable recommendations."""
    recommendations = []

    overall = verdict.get("overall", "UNKNOWN")

    if overall == "PASS":
        recommendations.append(
            "Strategy shows robust performance across market regimes. Proceed with paper trading."
        )
    elif overall == "CONDITIONAL":
        recommendations.append(
            "Strategy shows promise but has some weaknesses. Review failed checks and consider parameter refinement."
        )
    else:
        recommendations.append(
            "Strategy failed robustness validation. DO NOT trade live. Review strategy logic or parameter space."
        )

    # Check-specific recommendations
    checks = verdict.get("checks", {})

    if not checks.get("consistency", {}).get("passed", True):
        recommendations.append(
            "Low consistency: Strategy only works in specific market conditions. Consider adding regime filters."
        )

    if not checks.get("minimum_performance", {}).get("passed", True):
        recommendations.append(
            "Poor minimum performance: Worst-case scenario is unacceptable. Tighten risk management."
        )

    if not checks.get("parameter_stability", {}).get("passed", True):
        recommendations.append(
            "Unstable parameters: Parameters vary too much across windows. Reduce parameter space or add constraints."
        )

    if not checks.get("flat_region", {}).get("passed", True):
        recommendations.append(
            "Sharp performance peak: Parameters are on a 'knife edge'. Broaden search around selected parameters."
        )

    return recommendations


def _generate_txt_summary(report: dict, output_path: Path) -> None:
    """Generate human-readable summary."""
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("WALK-FORWARD VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Verdict
        verdict = report.get("verdict", {})
        f.write(f"VERDICT: {verdict.get('overall', 'UNKNOWN')}\n")
        f.write(
            f"Passed: {verdict.get('passed_checks', 0)}/{verdict.get('total_checks', 0)} checks\n\n"
        )

        # Individual checks
        f.write("Detailed Checks:\n")
        f.write("-" * 60 + "\n")
        for check_name, check in verdict.get("checks", {}).items():
            status = "PASS" if check["passed"] else "FAIL"
            f.write(
                f"  [{status}] {check['metric']} (threshold: {check['threshold']})\n"
            )
        f.write("\n")

        # Robustness Summary
        robustness = report.get("robustness", {})
        consistency = robustness.get("consistency", {})
        perf_dist = robustness.get("performance_distribution", {})

        f.write("Robustness Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Profitable Windows: {consistency.get('profitable_pct', 0):.1f}%\n")
        f.write(
            f"  Mean OOS Sharpe: {perf_dist.get('mean', 0):.2f} ± {perf_dist.get('std', 0):.2f}\n"
        )
        f.write(
            f"  Min/Max Sharpe: {perf_dist.get('min', 0):.2f} / {perf_dist.get('max', 0):.2f}\n"
        )
        f.write(f"  10th Percentile: {perf_dist.get('p10', 0):.2f}\n")
        f.write(f"  Total Windows: {robustness.get('valid_windows', 0)}\n\n")

        # Parameter Selection
        selection = report.get("parameter_selection", {})
        f.write(
            f"Parameter Selection Method: {selection.get('recommended_method', 'N/A')}\n"
        )
        f.write("-" * 60 + "\n")
        f.write("Final Parameters:\n")
        for param, value in report.get("final_params", {}).items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")

        # Flat Region
        flat = report.get("flat_region_check", {})
        f.write(f"Flat Region Score: {flat.get('flat_region_score', 0):.2f}\n")
        f.write(f"Is Robust: {flat.get('is_robust', False)}\n\n")

        # Recommendations
        f.write("Recommendations:\n")
        f.write("-" * 60 + "\n")
        for rec in report.get("recommendations", []):
            f.write(f"  • {rec}\n")


def generate_validation_report(results: dict, output_path: str) -> None:
    """Generate validation report JSON and TXT files with comprehensive robustness metrics.

    New structure includes:
    - robustness: Full robustness metrics
    - parameter_selection: Results from all methods + recommendation
    - pass_fail_criteria: Explicit pass/fail verdict with reasons
    - recommendations: Actionable advice based on results

    Args:
        results: Validation results dict
        output_path: Strategy name for directory creation
    """
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")
    reports_dir = Path(f"reports/{output_path}/{date_str}")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Extract key metrics
    robustness = results.get("robustness", {})
    selection = results.get("parameter_selection", {})

    # Calculate pass/fail verdict
    verdict = _calculate_pass_fail_verdict(robustness, selection)

    # Build comprehensive report
    report = {
        "generated_at": datetime.now().isoformat(),
        "strategy": output_path,
        "verdict": verdict,
        "robustness": robustness,
        "parameter_selection": selection,
        "final_params": selection.get("final_params", {}),
        "recommended_method": selection.get("recommended_method", "unknown"),
        "flat_region_check": results.get(
            "flat_region_check", selection.get("flat_region_check", {})
        ),
        "windows": results.get("windows", []),
        "monte_carlo": results.get("monte_carlo", {}),
        "recommendations": _generate_recommendations(verdict, robustness, selection),
    }

    # Save JSON
    import json

    with open(reports_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save TXT summary
    _generate_txt_summary(report, reports_dir / "summary.txt")
