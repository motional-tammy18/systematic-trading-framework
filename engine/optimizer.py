"""Parameter optimization module for trading strategies.

Supports grid search and random search with multiprocessing for large parameter spaces.
"""

import itertools
import os
import random
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import polars as pl

from engine.backtester import run_backtest
from engine.metrics import calculate_max_drawdown, calculate_sharpe


def _get_max_workers(config: Optional[Dict] = None) -> int:
    """Get the optimal number of worker processes for multiprocessing.

    Args:
        config: Optional configuration dict. If provided, reads max_workers from
            config.get("system", {}).get("max_workers"). If None or not set,
            defaults to (cpu_count - 1) to leave cores available for other tasks.

    Returns:
        Number of worker processes to use
    """
    if config is not None:
        system_config = config.get("system", {})
        max_workers = system_config.get("max_workers")
        if max_workers is not None:
            return max_workers

    # Default: leave 1 core available (use cpu_count - 1, minimum 1)
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _evaluate_params_worker(args: Tuple[Type, Dict, pl.DataFrame, Dict]) -> Dict[str, Any]:
    """Worker function for multiprocessing - evaluates one parameter combination.

    Args:
        args: Tuple of (strategy_class, param_combo, train_data, config)

    Returns:
        Dict with params, sharpe, max_dd, trade_count, total_pnl
    """
    strategy_class, param_combo, train_data, config = args

    # Instantiate strategy with params
    strategy = strategy_class(param_combo)

    # Generate signals
    signals = strategy.generate_signals(train_data)

    # Run backtest
    trade_log, equity_curve = run_backtest(signals, config)

    # Calculate metrics
    try:
        sharpe = calculate_sharpe(equity_curve)
    except ValueError:
        sharpe = 0.0

    try:
        max_dd_pct, _ = calculate_max_drawdown(equity_curve)
    except ValueError:
        max_dd_pct = 0.0

    trade_count = trade_log.height
    total_pnl = trade_log["pnl"].sum() if trade_count > 0 else 0.0

    # Build result dict with all params + metrics
    result = dict(param_combo)
    result["sharpe"] = sharpe
    result["max_dd"] = max_dd_pct
    result["trade_count"] = trade_count
    result["total_pnl"] = total_pnl

    return result


def _generate_grid_combinations(param_space: Dict[str, List]) -> List[Dict]:
    """Generate all combinations for grid search.

    Args:
        param_space: Dict mapping param names to lists of values

    Returns:
        List of dicts, each dict is one parameter combination
    """
    keys = list(param_space.keys())
    values = [param_space[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def _generate_random_combinations(param_space: Dict[str, List], n: int) -> List[Dict]:
    """Generate N random combinations for random search.

    Args:
        param_space: Dict mapping param names to lists of values
        n: Number of random combinations to generate

    Returns:
        List of dicts, each dict is one parameter combination
    """
    combinations = []
    keys = list(param_space.keys())

    for _ in range(n):
        combo = {key: random.choice(param_space[key]) for key in keys}
        combinations.append(combo)

    return combinations


def optimize_params(
    strategy_class: Type,
    param_space: Dict[str, List],
    train_data: pl.DataFrame,
    method: str = "grid",
    objective: str = "sharpe",
    n_random: int = 100,
    config: Optional[Dict] = None,
) -> pl.DataFrame:
    """Optimize strategy parameters on training data.

    Args:
        strategy_class: Strategy class to optimize (must inherit from BaseStrategy)
        param_space: Dict mapping param names to lists of values to test
        train_data: OHLCV DataFrame for training/optimization
        method: "grid" or "random" search method
        objective: Metric to optimize (default "sharpe")
        n_random: Number of random combinations for random search
        config: Backtest configuration dict (fees, slippage, etc.)

    Returns:
        Polars DataFrame with columns:
        - All param names from param_space
        - sharpe (objective metric)
        - max_dd, trade_count, total_pnl
        Sorted by objective descending (best params first)

    Raises:
        ValueError: If method is not "grid" or "random"
    """
    if config is None:
        config = {}

    if method not in ("grid", "random"):
        raise ValueError(f"Invalid method: {method}. Must be 'grid' or 'random'")

    # Generate parameter combinations
    if method == "grid":
        combinations = _generate_grid_combinations(param_space)
    else:  # random
        combinations = _generate_random_combinations(param_space, n_random)

    n_combos = len(combinations)

    # Prepare worker args
    worker_args = [
        (strategy_class, combo, train_data, config) for combo in combinations
    ]

    # Use multiprocessing for large param spaces (>100 combinations)
    if n_combos > 100:
        max_workers = _get_max_workers(config)
        with Pool(processes=max_workers) as pool:
            results = pool.map(_evaluate_params_worker, worker_args)
    else:
        # Sequential execution for small spaces
        results = [_evaluate_params_worker(args) for args in worker_args]

    # Convert to DataFrame
    if not results:
        # Return empty DataFrame with correct schema
        schema: Dict[str, Any] = {k: pl.Float64 for k in param_space.keys()}
        schema["sharpe"] = pl.Float64
        schema["max_dd"] = pl.Float64
        schema["trade_count"] = pl.Int64
        schema["total_pnl"] = pl.Float64
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(results)

    # Sort by objective descending (best first)
    if objective in df.columns:
        df = df.sort(objective, descending=True)

    return df


def select_params_by_island_volume(
    results: pl.DataFrame,
    param_space: Dict[str, List],
    objective: str = "sharpe",
    min_threshold: float = 0.0,
    adaptive_threshold: bool = True,
    percentile: float = 50.0,
) -> Dict[str, Any]:
    """Select parameters using Island Volume Selection (IVS) algorithm.

    Identifies parameter plateaus (flat regions) in the optimization landscape
    and selects the center of the largest contiguous "island" of profitability.
    Prioritizes broad profit mass over tall profit spikes to avoid overfitting.

    Algorithm:
    1. Filter results above profitability threshold
    2. Build adjacency graph (neighbors differ by 1 parameter step)
    3. Find connected components (islands)
    4. Calculate volume = count × avg_objective for each island
    5. Return center of largest volume island

    Args:
        results: Optimization results DataFrame with param columns and objective column
        param_space: Dict mapping param names to lists of values tested
        objective: Metric column to use for profitability (default "sharpe")
        min_threshold: Minimum objective value to consider profitable (default 0.0)
        adaptive_threshold: If True, use percentile-based threshold (default True)
        percentile: Percentile for adaptive threshold (default 50 = median)

    Returns:
        Dict with:
        - selected_params: Dict of selected parameter values
        - method: "island_volume"
        - islands: List of all discovered islands with stats
        - largest_island: Info about the selected island
    """
    if results.is_empty():
        return {"selected_params": {}, "method": "island_volume", "islands": []}

    param_names = [k for k in param_space.keys() if k in results.columns]

    if not param_names:
        return {"selected_params": {}, "method": "island_volume", "islands": []}

    # Adaptive threshold: use percentile-based approach for more inclusive island detection
    if adaptive_threshold and min_threshold == 0.0:
        objective_values = results[objective].to_numpy()
        profitable_mask = objective_values > 0
        
        if profitable_mask.sum() > 0:
            profitable_values = objective_values[profitable_mask]
            # Use 40th percentile - captures the "heart" of profitable region
            # More inclusive than 80% of max, captures more robust parameters
            min_threshold = float(np.percentile(profitable_values, percentile))
            # Ensure min_threshold is at least 0
            min_threshold = max(0.0, min_threshold)

    profitable = results.filter(pl.col(objective) >= min_threshold)

    if profitable.is_empty():
        best_row = results.row(0, named=True)
        return {
            "selected_params": {k: best_row[k] for k in param_names},
            "method": "island_volume",
            "islands": [],
            "fallback": "no_profitable_combinations",
        }

    profitable_rows = profitable.to_dicts()

    value_to_index = {
        param: {val: idx for idx, val in enumerate(sorted(param_space[param]))}
        for param in param_names
    }

    index_to_value = {
        param: {idx: val for val, idx in value_to_index[param].items()}
        for param in param_names
    }

    visited = set()
    islands = []

    for i, row in enumerate(profitable_rows):
        if i in visited:
            continue

        queue = [i]
        visited.add(i)
        island_indices = []

        while queue:
            current_idx = queue.pop(0)
            island_indices.append(current_idx)
            current_row = profitable_rows[current_idx]

            for neighbor_idx, neighbor_row in enumerate(profitable_rows):
                if neighbor_idx in visited:
                    continue

                adjacent = True
                diff_count = 0

                for param in param_names:
                    current_val = current_row[param]
                    neighbor_val = neighbor_row[param]

                    if current_val != neighbor_val:
                        diff_count += 1
                        current_idx_in_space = value_to_index[param][current_val]
                        neighbor_idx_in_space = value_to_index[param][neighbor_val]

                        if abs(current_idx_in_space - neighbor_idx_in_space) != 1:
                            adjacent = False
                            break

                if adjacent and diff_count == 1:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

        island_rows = [profitable_rows[idx] for idx in island_indices]
        island_objectives = [row[objective] for row in island_rows]

        island_volume = len(island_rows) * sum(island_objectives)

        param_means = {}
        for param in param_names:
            values = [row[param] for row in island_rows]
            param_means[param] = sum(values) / len(values)

        islands.append(
            {
                "size": len(island_rows),
                "volume": island_volume,
                "avg_objective": sum(island_objectives) / len(island_objectives),
                "max_objective": max(island_objectives),
                "center_params": param_means,
                "all_params": island_rows,
            }
        )

    if not islands:
        best_row = results.row(0, named=True)
        return {
            "selected_params": {k: best_row[k] for k in param_names},
            "method": "island_volume",
            "islands": [],
            "fallback": "no_islands_found",
        }

    largest_island = max(islands, key=lambda x: x["volume"])

    rounded_params = {}
    for param, value in largest_island["center_params"].items():
        if isinstance(value, float):
            rounded_params[param] = round(value, 4)
        else:
            rounded_params[param] = value

    return {
        "selected_params": rounded_params,
        "method": "island_volume",
        "islands": islands,
        "largest_island": {
            "size": largest_island["size"],
            "volume": largest_island["volume"],
            "avg_objective": largest_island["avg_objective"],
        },
    }
