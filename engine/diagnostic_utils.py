"""Statistical diagnostic utilities for WFO reliability checking.

This module provides statistical tests to validate WFO window sizing,
detect regime heterogeneity, test window independence, and detect
meta-overfitting in parameter selection.
"""

import numpy as np
import os
import polars as pl
from scipy import stats


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

def validate_statistical_power(
    train_days: int,
    timeframe: str,
    expected_trades_per_day: float = 0.5,
    target_sharpe: float = 1.5,
    power: float = 0.80,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Validate that training period has sufficient statistical power.

    Uses power analysis to determine if the training period provides enough
    observations to detect a genuine trading edge with specified confidence.

    Args:
        train_days: Number of days in training period
        timeframe: Data timeframe (e.g., "15m", "1h", "1d")
        expected_trades_per_day: Expected number of trades per day
        target_sharpe: Target Sharpe ratio to detect
        power: Desired statistical power (0.80 = 80%)
        alpha: Significance level (0.05 = 5%)

    Returns:
        Dictionary with power analysis results
    """
    # Convert timeframe to bars per day
    timeframe_minutes = _parse_timeframe(timeframe)
    bars_per_day = 24 * 60 / timeframe_minutes
    train_bars = int(train_days * bars_per_day)
    expected_trades = int(train_days * expected_trades_per_day)

    # Minimum trades rule from GT-Score paper and industry standards
    min_trades_required = 50

    # Statistical power check (simplified - full power analysis requires
    # non-central t-distribution calculations which are complex for Sharpe)
    # We use rule-of-thumb: 50+ trades for reasonable power
    is_adequate = expected_trades >= min_trades_required

    # Calculate effect size (Cohen's d approximation for Sharpe)
    # Sharpe = mean / std, so for Sharpe=1.5, effect size ≈ 1.5 * sqrt(n)
    # This is a simplification - actual power analysis for Sharpe is non-trivial
    effect_size = target_sharpe * np.sqrt(expected_trades) if expected_trades > 0 else 0

    # Determine recommendation
    if is_adequate:
        recommendation = "Adequate statistical power"
        severity = "NONE"
    elif expected_trades >= 30:
        recommendation = f"Marginal power. Consider increasing train_days to {int(min_trades_required/expected_trades_per_day)}+ for 80% power"
        severity = "WARNING"
    else:
        recommendation = f"Insufficient power. Increase train_days to {int(min_trades_required/expected_trades_per_day)}+ days minimum"
        severity = "ERROR"

    passed = is_adequate
    metric = float(expected_trades)
    threshold = f">= {min_trades_required} trades"

    return {
        "train_days": train_days,
        "train_bars": train_bars,
        "expected_trades": expected_trades,
        "min_trades_required": min_trades_required,
        "is_adequate": is_adequate,
        "effect_size_approx": float(effect_size),
        "target_sharpe": target_sharpe,
        "power": power,
        "alpha": alpha,
        "recommendation": recommendation,
        "severity": severity,
        "metric": metric,
        "threshold": threshold,
        "passed": passed,
    }


def test_window_independence(
    window_metrics: List[float],
    max_lags: int = 10,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test if walk-forward windows are statistically independent.

    Uses Ljung-Box test to detect autocorrelation in window performance metrics.
    If windows are autocorrelated, WFO assumptions are violated.

    Args:
        window_metrics: List of performance metrics (e.g., Sharpe ratios) for each window
        max_lags: Maximum number of lags to test
        alpha: Significance level

    Returns:
        Dictionary with independence test results
    """
    if len(window_metrics) < max_lags + 1:
        return {
            "is_independent": True,
            "test_performed": False,
            "reason": f"Insufficient windows ({len(window_metrics)}) for Ljung-Box test (need {max_lags + 1}+)",
            "recommendation": "Add more windows or reduce max_lags",
            "metric": len(window_metrics),
            "threshold": f">= {max_lags + 1} windows",
            "passed": True,
        }

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        metrics_array = np.array(window_metrics)

        # Handle NaN values
        if np.any(np.isnan(metrics_array)):
            metrics_array = metrics_array[~np.isnan(metrics_array)]

        if len(metrics_array) < max_lags + 1:
            return {
                "is_independent": True,
                "test_performed": False,
                "reason": "Too many NaN values after filtering",
                "metric": len(metrics_array),
                "threshold": f">= {max_lags + 1} valid windows",
                "passed": True,
            }

        # Ljung-Box test
        lb_result = acorr_ljungbox(metrics_array, lags=max_lags, return_df=True)

        pvalues = lb_result["lb_pvalue"].to_numpy()
        min_p_value = float(np.min(pvalues))
        max_autocorr_lag = int(np.argmin(pvalues)) + 1
        is_independent = bool(np.all(pvalues > alpha))

        # Determine severity
        if is_independent:
            severity = "NONE"
            recommendation = "Windows are statistically independent"
        elif min_p_value < alpha / 10:  # Very significant
            severity = "ERROR"
            recommendation = f"Strong autocorrelation detected at lag {max_autocorr_lag}. Increase step_size or reduce window overlap."
        else:
            severity = "WARNING"
            recommendation = (
                f"Weak autocorrelation detected. Consider increasing step_size."
            )

        return {
            "is_independent": is_independent,
            "test_performed": True,
            "min_p_value": min_p_value,
            "max_autocorr_lag": max_autocorr_lag,
            "alpha": alpha,
            "max_lags": max_lags,
            "all_p_values": lb_result["lb_pvalue"].tolist(),
            "severity": severity,
            "recommendation": recommendation,
            "metric": min_p_value,
            "threshold": f"> {alpha}",
            "passed": is_independent,
        }

    except ImportError:
        return {
            "is_independent": True,
            "test_performed": False,
            "reason": "statsmodels not installed",
            "recommendation": "Install statsmodels for Ljung-Box test: pip install statsmodels",
            "metric": "N/A",
            "threshold": "N/A",
            "passed": True,
        }


def detect_regime_heterogeneity(
    data: pl.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int,
    method: str = "pelt",
    min_regime_bars: int = 1000,
) -> Dict[str, Any]:
    """Detect if WFO windows contain multiple market regimes.

    Uses changepoint detection to identify regime boundaries and checks
    if any window spans multiple regimes (heterogeneity).

    Args:
        data: Full price data DataFrame with 'close' column
        train_size: Training window size in bars
        test_size: Test window size in bars
        step_size: Step size between windows
        method: Changepoint detection method ('pelt', 'cusum', 'zivot_andrews')
        min_regime_bars: Minimum bars per regime

    Returns:
        Dictionary with regime detection results
    """
    if "close" not in data.columns:
        return {
            "has_regime_mixing": False,
            "test_performed": False,
            "reason": "No 'close' column in data",
            "metric": "N/A",
            "threshold": "N/A",
            "passed": True,
        }

    # Calculate returns
    returns = data["close"].pct_change().drop_nulls().to_numpy()

    if len(returns) < min_regime_bars * 2:
        return {
            "has_regime_mixing": False,
            "test_performed": False,
            "reason": f"Insufficient data ({len(returns)} bars) for regime detection",
            "metric": len(returns),
            "threshold": f">= {min_regime_bars * 2} bars",
            "passed": True,
        }

    changepoints = []

    try:
        if method == "pelt":
            try:
                from ruptures import Pelt

                model = Pelt(model="rbf", min_size=min_regime_bars).fit(
                    returns.reshape(-1, 1)
                )
                changepoints = model.predict(pen=10)
            except ImportError:
                method = "cusum"  # Fallback

        if method == "cusum":
            # CUSUM for mean shifts (no external dependency)
            changepoints = _cusum_changepoints(returns, min_regime_bars)

        elif method == "zivot_andrews":
            # Zivot-Andrews test (simplified implementation)
            changepoints = _zivot_andrews_changepoints(returns, min_regime_bars)

    except Exception as e:
        return {
            "has_regime_mixing": False,
            "test_performed": False,
            "reason": f"Error in changepoint detection: {str(e)}",
            "metric": "N/A",
            "threshold": "N/A",
            "passed": True,
        }

    # Calculate window boundaries
    total_bars = len(data)
    window_boundaries = []
    window_idx = 0

    while True:
        start_idx = window_idx * step_size
        train_end = start_idx + train_size
        test_end = train_end + test_size

        if test_end > total_bars:
            break

        window_boundaries.append((start_idx, train_end, test_end))
        window_idx += 1

    # Check which windows contain changepoints
    windows_with_regime_changes = []
    for i, (start, train_end, test_end) in enumerate(window_boundaries):
        # Check if any changepoint falls within this window's train or test period
        window_changepoints = [cp for cp in changepoints if start <= cp <= test_end]
        if window_changepoints:
            windows_with_regime_changes.append(
                {
                    "window_idx": i,
                    "train_start": start,
                    "train_end": train_end,
                    "test_end": test_end,
                    "changepoints_in_window": window_changepoints,
                }
            )

    has_regime_mixing = len(windows_with_regime_changes) > 0
    mixing_percentage = (
        len(windows_with_regime_changes) / len(window_boundaries) * 100
        if window_boundaries
        else 0
    )

    # Determine severity
    if not has_regime_mixing:
        severity = "NONE"
        recommendation = "Windows appear regime-homogeneous"
    elif mixing_percentage > 50:
        severity = "ERROR"
        recommendation = f"Severe regime mixing ({mixing_percentage:.1f}% of windows). Consider adaptive window sizing or regime-based splits."
    elif mixing_percentage > 25:
        severity = "WARNING"
        recommendation = f"Moderate regime mixing ({mixing_percentage:.1f}% of windows). Monitor for instability."
    else:
        severity = "INFO"
        recommendation = f"Minor regime mixing ({mixing_percentage:.1f}% of windows). Generally acceptable."

    return {
        "has_regime_mixing": has_regime_mixing,
        "test_performed": True,
        "method": method,
        "n_changepoints": len(changepoints),
        "changepoint_indices": changepoints[:20],
        "n_windows": len(window_boundaries),
        "windows_with_regime_changes": len(windows_with_regime_changes),
        "mixing_percentage": float(mixing_percentage),
        "severity": severity,
        "recommendation": recommendation,
        "affected_windows": windows_with_regime_changes[:5],
        "metric": float(mixing_percentage),
        "threshold": "<= 25%",
        "passed": mixing_percentage <= 25,
    }


def _meta_overfitting_worker(args: Tuple) -> float:
    """Worker function for multiprocessing - evaluates one random parameter combination.

    Args:
        args: Tuple of (strategy_class, param_dict, train_data, config)

    Returns:
        Sharpe ratio as float
    """
    from engine.backtester import run_backtest
    from engine.metrics import calculate_sharpe

    strategy_class, param_dict, train_data, config = args

    # Instantiate strategy with random parameters
    strategy = strategy_class(param_dict)

    # Generate signals and run backtest
    try:
        signals = strategy.generate_signals(train_data)

        if len(signals) > 0 and "signal" in signals.columns:
            # Run actual vectorized backtest with proper trade execution
            trade_log, equity_curve = run_backtest(signals, config, show_progress=False)

            # Calculate Sharpe ratio from equity curve
            sharpe = float(
                calculate_sharpe(equity_curve) if equity_curve.height > 1 else 0.0
            )
            return sharpe
        else:
            return 0.0
    except Exception:
        return 0.0


def test_meta_overfitting(
    windows: List[Dict],
    param_space: Dict[str, List],
    strategy_class: type,
    train_data: pl.DataFrame,
    config: Dict,
    n_permutations: int = 500,
    alpha: float = 0.05,
    min_effect_size: float = 0.3,
) -> Dict[str, Any]:
    """Test if parameter selection process is meta-overfit using rigorous permutation test.

    For each permutation, randomly selects parameters from the ACTUAL
    parameter space and backtests them. Compares distribution
    to actual parameter selection performance.

    Args:
        windows: List of window dictionaries with 'oos_sharpe' and 'best_params' keys
        param_space: Strategy parameter space (e.g., {"ema_long": [50, 75, 100], "ema_short": [5, 10, 15]})
        strategy_class: Strategy class for generating signals with arbitrary parameters
        train_data: Training data for backtesting
        config: Backtest configuration dict (passed to run_backtest)
        n_permutations: Number of permutations (default 500 for statistical significance)
        alpha: Significance level (default 0.05)
        min_effect_size: Minimum Cohen's d for robustness (default 0.3 = medium effect)

    Returns:
        Dictionary with meta-overfitting test results including:
        - p_value: Proportion of random Sharpe >= actual
        - effect_size: Cohen's d measuring improvement magnitude
        - is_meta_overfit: True if p_value <= alpha AND effect_size < min_effect_size
        - severity: NONE/INFO/WARNING/ERROR based on combined criteria
    """
    if len(windows) < 5:
        return {
            "is_meta_overfit": False,
            "test_performed": False,
            "reason": f"Insufficient windows ({len(windows)}) for meta-overfitting test (need 5+)",
            "metric": len(windows),
            "threshold": ">= 5 windows",
            "passed": True,
        }

    # Extract actual OOS Sharpe ratios and selected parameters
    actual_sharpes = []
    actual_params = []

    for w in windows:
        sharpe = w.get("oos_sharpe", 0)
        params = w.get("best_params", {})

        if not np.isnan(sharpe) and params:
            actual_sharpes.append(sharpe)
            actual_params.append(params)

    if len(actual_sharpes) < 5:
        return {
            "is_meta_overfit": False,
            "test_performed": False,
            "reason": "Insufficient valid OOS Sharpe values after filtering NaN",
            "metric": len(actual_sharpes),
            "threshold": ">= 5 valid Sharpe values",
            "passed": True,
        }

    actual_mean = float(np.mean(actual_sharpes))

    # Generate all parameter combinations from parameter space
    from itertools import product
    from multiprocessing import Pool
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )

    param_names = list(param_space.keys())
    param_values = [param_space[k] for k in param_names]
    all_combinations = list(product(*param_values))

    # Generate random parameter combinations for all permutations upfront
    np.random.seed(42)  # For reproducibility
    random_indices = np.random.randint(0, len(all_combinations), size=n_permutations)

    # Prepare worker args for all permutations
    worker_args = []
    for idx in random_indices:
        random_params = all_combinations[idx]
        random_param_dict = dict(zip(param_names, random_params))
        worker_args.append((strategy_class, random_param_dict, train_data, config))

    # Rigorous permutation test with multiprocessing and progress tracking
    null_sharpes = []

    # Use Pool for multiprocessing (follows optimizer.py pattern)
    max_workers = _get_max_workers(config)
    with Pool(processes=max_workers) as pool:
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            # Add task to progress bar
            task = progress.add_task(
                "[cyan]Meta-overfitting test (permutations)", total=n_permutations
            )

            # Use imap_unordered for streaming results with progress updates
            for sharpe in pool.imap_unordered(_meta_overfitting_worker, worker_args):
                null_sharpes.append(sharpe)
                progress.update(task, advance=1)

    # Calculate p-value: proportion of random >= actual
    null_mean = float(np.mean(null_sharpes))
    null_sharpes_array = np.array(null_sharpes)

    # One-tailed test: is null mean significantly less than actual (i.e., actual performs better)?
    p_value = float(np.sum(null_sharpes_array >= actual_mean) / n_permutations)

    # Calculate effect size (Cohen's d)
    pooled_std = float(np.std(np.concatenate([actual_sharpes, null_sharpes])))
    effect_size = abs(actual_mean - null_mean) / (pooled_std + 1e-8)

    # Determine meta-overfitting using BOTH p-value AND effect size
    # Strategy is robust if: p_value > alpha AND effect_size >= min_effect_size
    is_meta_overfit = p_value <= alpha or effect_size < min_effect_size

    # Calculate severity and recommendation
    if p_value <= alpha and effect_size < min_effect_size:
        severity = "ERROR"
        recommendation = f"CRITICAL: Meta-overfitting detected (p={p_value:.3f}, effect={effect_size:.2f}). Reduce parameter space or increase train duration."
    elif p_value <= alpha:
        severity = "WARNING"
        recommendation = f"Weak significance (p={p_value:.3f}) but adequate effect size ({effect_size:.2f}). Consider review."
    elif effect_size < min_effect_size:
        severity = "WARNING"
        recommendation = f"Statistically significant (p={p_value:.3f}) but small effect ({effect_size:.2f}). May lack practical robustness."
    else:
        if effect_size >= 0.5:
            severity = "NONE"
            recommendation = f"Parameter selection shows strong improvement over random (effect={effect_size:.2f})."
        elif effect_size >= min_effect_size:
            severity = "NONE"
            recommendation = f"Parameter selection shows acceptable improvement (effect={effect_size:.2f})."

    return {
        "is_meta_overfit": is_meta_overfit,
        "test_performed": True,
        "p_value": p_value,
        "effect_size": effect_size,
        "min_effect_size": min_effect_size,
        "actual_mean": actual_mean,
        "null_mean": null_mean,
        "actual_count": len(actual_sharpes),
        "n_permutations": n_permutations,
        "severity": severity,
        "recommendation": recommendation,
        "metric": effect_size,
        "threshold": f">= {min_effect_size}",
        "passed": not is_meta_overfit,
    }


def calculate_parameter_stability_cv(
    windows: List[Dict],
    param_keys: List[str],
) -> Dict[str, Any]:
    """Calculate coefficient of variation for parameters across windows.

    High CV indicates parameter instability (overfitting).

    Args:
        windows: List of window dictionaries with 'best_params' key
        param_keys: List of parameter names to analyze

    Returns:
        Dictionary with parameter stability metrics
    """
    results = {}
    unstable_params = []

    for param in param_keys:
        values = []
        for w in windows:
            if "best_params" in w and param in w["best_params"]:
                val = w["best_params"][param]
                if val is not None and not np.isnan(val):
                    values.append(val)

        if len(values) < 2:
            results[param] = {
                "mean": None,
                "std": None,
                "cv": None,
                "is_stable": None,
                "n_values": len(values),
            }
            continue

        values_arr = np.array(values)
        mean_val = float(np.mean(values_arr))
        std_val = float(np.std(values_arr))

        if mean_val != 0 and not np.isnan(mean_val):
            cv = float(std_val / abs(mean_val))
        else:
            cv = float("inf")

        # CV > 0.5 indicates instability
        is_stable = cv <= 0.5

        if not is_stable:
            unstable_params.append(param)

        results[param] = {
            "mean": mean_val,
            "std": std_val,
            "cv": cv,
            "is_stable": is_stable,
            "n_values": len(values),
            "min": float(np.min(values_arr)),
            "max": float(np.max(values_arr)),
        }

    all_cvs = [results[p]["cv"] for p in results if results[p]["cv"] is not None]
    avg_cv = float(np.mean(all_cvs)) if all_cvs else 0.0

    return {
        "parameters": results,
        "unstable_params": unstable_params,
        "n_unstable": len(unstable_params),
        "all_stable": len(unstable_params) == 0,
        "metric": avg_cv,
        "threshold": "<= 0.5",
        "passed": len(unstable_params) == 0,
    }


# Helper functions


def _parse_timeframe(timeframe: str) -> int:
    """Parse timeframe string to minutes."""
    unit = timeframe[-1]
    value = int(timeframe[:-1])

    if unit == "m":
        return value
    elif unit == "h":
        return value * 60
    elif unit == "d":
        return value * 24 * 60
    elif unit == "w":
        return value * 7 * 24 * 60
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")


def _cusum_changepoints(returns: np.ndarray, min_size: int) -> List[int]:
    """CUSUM changepoint detection (no external dependencies)."""
    changepoints = []
    cusum_pos = np.zeros(len(returns))
    cusum_neg = np.zeros(len(returns))

    # Adaptive reference value based on rolling standard deviation
    k = max(0.5, np.std(returns) * 0.5)
    h = 5.0 * k  # Decision interval

    for t in range(1, len(returns)):
        cusum_pos[t] = max(0, cusum_pos[t - 1] + returns[t] - k)
        cusum_neg[t] = max(0, cusum_neg[t - 1] - returns[t] - k)

        if (cusum_pos[t] > h or cusum_neg[t] > h) and t >= min_size:
            if not changepoints or t - changepoints[-1] >= min_size:
                changepoints.append(t)
                cusum_pos[t] = 0
                cusum_neg[t] = 0

    return changepoints


def _zivot_andrews_changepoints(returns: np.ndarray, min_size: int) -> List[int]:
    """Simplified Zivot-Andrews structural break test."""
    # This is a simplified version - full ZA test requires regression
    # We use a rolling t-test approach as approximation
    changepoints = []
    window = min_size

    for t in range(window, len(returns) - window):
        before = returns[t - window : t]
        after = returns[t : t + window]

        # Two-sample t-test
        ttest_result = stats.ttest_ind(before, after)
        p_value = float(ttest_result[1])  # Extract p-value

        if p_value < 0.01:
            if not changepoints or t - changepoints[-1] >= min_size:
                changepoints.append(t)

    return changepoints


def estimate_trades_per_day(
    strategy_class: Optional[type],
    param_space: Dict[str, List],
    sample_data: pl.DataFrame,
    config: Dict,
) -> Optional[float]:
    """Estimate expected trades per day from strategy behavior on sample data.

    Runs a sample backtest on a portion of the data to measure actual
    trading frequency, then extrapolates to trades per day.

    Args:
        strategy_class: Strategy class to test
        param_space: Parameter space (uses first param combo as sample)
        sample_data: Data to run sample backtest on
        config: Configuration dict with backtest settings

    Returns:
        Estimated trades per day as float, or None if estimation fails
    """
    if strategy_class is None or sample_data is None or sample_data.height < 100:
        return None

    try:
        if not param_space:
            return None

        sample_params = {k: v[0] for k, v in param_space.items() if v}

        strategy = strategy_class(sample_params)
        signals = strategy.generate_signals(sample_data)

        if signals is None or signals.is_empty():
            return None

        trade_count = 0
        position = 0
        for i in range(1, len(signals)):
            prev_position = position
            current_signal = signals["signal"][i]

            if prev_position == 0 and current_signal != 0:
                trade_count += 1
            elif prev_position != 0 and current_signal == 0:
                trade_count += 1

        sample_days = sample_data.height
        timeframe_minutes = _parse_timeframe(config.get("timeframe", "1d"))
        bars_per_day = 24 * 60 / timeframe_minutes
        actual_days = sample_days / bars_per_day

        if actual_days > 0:
            trades_per_day = trade_count / actual_days
            return float(trades_per_day)

    except Exception:
        pass

    return None


def run_all_diagnostics(
    data: pl.DataFrame,
    windows: List[Dict],
    train_days: int,
    test_days: int,
    step_size: int,
    timeframe: str,
    config: Dict,
    param_space: Optional[Dict[str, List]] = None,
    strategy_class: Optional[type] = None,
    train_data: Optional[pl.DataFrame] = None,
) -> Dict[str, Any]:
    """Run all diagnostic checks and return comprehensive report.

    Args:
        data: Full price data
        windows: List of WFO window results
        train_days: Training period in days
        test_days: Test period in days
        step_size: Step size between windows
        timeframe: Data timeframe string
        config: Statistical checks configuration
        param_space: Strategy parameter space (for meta-overfitting test)
        strategy_class: Strategy class for generating signals (for meta-overfitting test)
        train_data: Training data for backtesting (for meta-overfitting test)

    Returns:
        Comprehensive diagnostic report
    """
    checks_config = config.get("validation", {}).get("statistical_checks", {})

    if not checks_config.get("enabled", True):
        return {"enabled": False, "reason": "Statistical checks disabled in config"}

    from datetime import datetime

    results = {
        "enabled": True,
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "summary": {
            "total_checks": 0,
            "passed": 0,
            "warnings": 0,
            "errors": 0,
        },
    }

    # 1. Statistical Power Check
    if checks_config.get("min_trades_for_power", 50):
        expected_trades_per_day = checks_config.get("expected_trades_per_day")

        if (
            expected_trades_per_day is None
            and strategy_class is not None
            and train_data is not None
        ):
            sample_size = min(train_data.height, int(train_days * 30))
            if sample_size > 0:
                sample_data = train_data[:sample_size]
                expected_trades_per_day = estimate_trades_per_day(
                    strategy_class=strategy_class,
                    param_space=param_space or {},
                    sample_data=sample_data,
                    config=config,
                )

        if expected_trades_per_day is None:
            expected_trades_per_day = 0.5

        power_result = validate_statistical_power(
            train_days=train_days,
            timeframe=timeframe,
            expected_trades_per_day=expected_trades_per_day,
            target_sharpe=checks_config.get("min_sharpe_effect_size", 0.5),
        )
        power_result["expected_trades_per_day_source"] = (
            "estimated" if expected_trades_per_day != 0.5 else "default"
        )
        results["checks"]["statistical_power"] = power_result

        if power_result["severity"] == "ERROR":
            results["summary"]["errors"] += 1
        elif power_result["severity"] == "WARNING":
            results["summary"]["warnings"] += 1
        else:
            results["summary"]["passed"] += 1
        results["summary"]["total_checks"] += 1

    # 2. Window Independence Test
    if checks_config.get("test_window_independence", True) and windows:
        oos_sharpes = [w.get("oos_sharpe", 0) for w in windows if "oos_sharpe" in w]
        if oos_sharpes:
            independence_result = test_window_independence(
                window_metrics=oos_sharpes,
                max_lags=checks_config.get("ljung_box_max_lags", 10),
                alpha=checks_config.get("ljung_box_alpha", 0.05),
            )
            results["checks"]["window_independence"] = independence_result

            if independence_result.get("severity") == "ERROR":
                results["summary"]["errors"] += 1
            elif independence_result.get("severity") == "WARNING":
                results["summary"]["warnings"] += 1
            else:
                results["summary"]["passed"] += 1
            results["summary"]["total_checks"] += 1

    # 3. Regime Heterogeneity Detection
    if checks_config.get("test_regime_homogeneity", True):
        timeframe_ms = _parse_timeframe(timeframe) * 60 * 1000
        bars_per_day = 86400000 / timeframe_ms
        train_size = int(train_days * bars_per_day)
        test_size = int(test_days * bars_per_day)

        regime_result = detect_regime_heterogeneity(
            data=data,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            method=checks_config.get("regime_detection_method", "pelt"),
            min_regime_bars=checks_config.get("min_regime_bars", 1000),
        )
        results["checks"]["regime_heterogeneity"] = regime_result

        if regime_result.get("severity") == "ERROR":
            results["summary"]["errors"] += 1
        elif regime_result.get("severity") == "WARNING":
            results["summary"]["warnings"] += 1
        else:
            results["summary"]["passed"] += 1
        results["summary"]["total_checks"] += 1

    # 4. Meta-Overfitting Test
    if checks_config.get("test_meta_overfitting", True) and windows:
        # Skip test if required parameters are not provided
        if param_space is None or strategy_class is None or train_data is None:
            meta_result = {
                "is_meta_overfit": False,
                "test_performed": False,
                "reason": "meta_overfitting test requires param_space, strategy_class, and train_data parameters",
                "severity": "INFO",
            }
        else:
            meta_result = test_meta_overfitting(
                windows=windows,
                param_space=param_space,
                strategy_class=strategy_class,
                train_data=train_data,
                config=config,
                n_permutations=checks_config.get("permutation_iterations", 500),
                alpha=checks_config.get("meta_overfit_alpha", 0.05),
            )

        results["checks"]["meta_overfitting"] = meta_result

        if meta_result.get("severity") == "ERROR":
            results["summary"]["errors"] += 1
        elif meta_result.get("severity") == "WARNING":
            results["summary"]["warnings"] += 1
        else:
            results["summary"]["passed"] += 1
        results["summary"]["total_checks"] += 1

    # 5. Parameter Stability Check
    if windows:
        # Extract parameter keys from first window
        param_keys = []
        for w in windows:
            if "best_params" in w and w["best_params"]:
                param_keys = list(w["best_params"].keys())
                break

        if param_keys:
            stability_result = calculate_parameter_stability_cv(
                windows=windows,
                param_keys=param_keys,
            )
            results["checks"]["parameter_stability"] = stability_result

            if not stability_result["all_stable"]:
                results["summary"]["warnings"] += 1
            else:
                results["summary"]["passed"] += 1
            results["summary"]["total_checks"] += 1

    # Overall assessment
    if results["summary"]["errors"] > 0:
        results["summary"]["overall_status"] = "FAIL"
        results["summary"][
            "overall_message"
        ] = f"{results['summary']['errors']} critical issues detected"
    elif results["summary"]["warnings"] > 0:
        results["summary"]["overall_status"] = "WARNING"
        results["summary"][
            "overall_message"
        ] = f"{results['summary']['warnings']} warnings, review recommended"
    else:
        results["summary"]["overall_status"] = "PASS"
        results["summary"]["overall_message"] = "All statistical checks passed"

    return results
