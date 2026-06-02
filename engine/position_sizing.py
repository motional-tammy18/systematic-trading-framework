"""Position sizing optimization module for systematic trading strategies.

Implements Kelly Criterion for optimal position sizing that balances
risk and returns based on historical win rate and win/loss ratio.
"""

from typing import Any, Dict, List, Optional, Tuple
from math import sqrt

import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a potentially None value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_kelly_fraction(
    trades: pl.DataFrame,
    fraction: float = 0.25,
    min_trades: int = 100,
) -> dict:
    """Calculate Kelly Criterion fraction from trade history.

    Formula: Kelly% = W - [(1 - W) / R]
    where W = win_rate, R = win_loss_ratio

    Args:
        trades: DataFrame with 'pnl' column containing trade P&L
        fraction: Fraction of full Kelly to use (0.25 = Quarter Kelly)
        min_trades: Minimum trades required for statistical significance

    Returns:
        Dictionary with:
        - kelly_fraction: Final Kelly fraction (capped at fraction)
        - raw_kelly: Full Kelly fraction before capping
        - win_rate: Win rate (0-1)
        - win_loss_ratio: Average win / average loss
        - avg_win: Average winning trade
        - avg_loss: Average losing trade (absolute value)
        - trade_count: Number of trades
        - confidence: Confidence level based on trade count

    Raises:
        ValueError: If trades DataFrame is empty or missing 'pnl' column
    """
    if trades.height == 0:
        raise ValueError("Trades DataFrame cannot be empty")

    if "pnl" not in trades.columns:
        raise ValueError("Trades DataFrame must have 'pnl' column")

    pnl = trades["pnl"]
    total_trades = trades.height

    if total_trades < min_trades:
        confidence = "low"
    elif total_trades < 100:
        confidence = "medium"
    else:
        confidence = "high"

    winning_trades = pnl.filter(pnl > 0)
    losing_trades = pnl.filter(pnl < 0)

    win_count = winning_trades.len()
    loss_count = losing_trades.len()

    win_rate = win_count / total_trades if total_trades > 0 else 0.0

    avg_win = _safe_float(winning_trades.mean()) if win_count > 0 else 0.0
    avg_loss = _safe_float(losing_trades.mean()) if loss_count > 0 else 0.0

    avg_loss_abs = abs(avg_loss)

    if avg_loss_abs == 0:
        win_loss_ratio = float("inf") if avg_win > 0 else 0.0
    else:
        win_loss_ratio = avg_win / avg_loss_abs

    if win_rate == 0:
        raw_kelly = 0.0
    elif win_loss_ratio == 0:
        raw_kelly = 0.0
    else:
        raw_kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

    raw_kelly = max(0.0, raw_kelly)

    kelly_fraction = min(raw_kelly, fraction)

    return {
        "kelly_fraction": kelly_fraction,
        "raw_kelly": raw_kelly,
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "trade_count": total_trades,
        "confidence": confidence,
    }


def optimize_position_sizing(
    trade_log: pl.DataFrame,
    equity_curve: pl.DataFrame,
    config: dict,
) -> dict:
    """Main optimization function that runs Kelly Criterion sizing and finds optimal parameters.

    Args:
        trade_log: Trade log DataFrame with 'pnl' column
        equity_curve: Equity curve DataFrame with 'equity' and 'drawdown' columns
        config: Configuration dictionary with sizing parameters

    Returns:
        Dictionary with:
        - methods: Dict of results (kelly only)
        - recommendation: Final recommended sizing parameters
        - sensitivity_analysis: Performance at different Kelly fractions
        - metrics: Key metrics (CAGR, Calmar, Sharpe, Max DD)
        - optimal_params: Optimal Kelly parameters
    """
    position_sizing_optimization = config.get("position_sizing_optimization", {})

    kelly_config = position_sizing_optimization.get("kelly", {})
    kelly_enabled = kelly_config.get("enabled", True)
    kelly_fraction = kelly_config.get("default_fraction", 0.25)

    results: Dict[str, Any] = {}
    sensitivity_results: Dict[str, List[dict]] = {}

    # Always run Kelly Criterion analysis
    kelly_result = calculate_kelly_fraction(
        trade_log,
        fraction=kelly_fraction,
        min_trades=100,
    )
    results["kelly"] = kelly_result

    # Sensitivity analysis for different Kelly fractions
    kelly_fractions = [0.125, 0.25, 0.5, 0.75, 1.0]
    sensitivity_analysis: List[dict] = []
    for frac in kelly_fractions:
        adjusted_kelly = min(kelly_result["raw_kelly"], frac)
        sensitivity_analysis.append({
            "fraction": frac,
            "adjusted_kelly": adjusted_kelly,
            "effective_size": adjusted_kelly,
        })
    sensitivity_results["kelly"] = sensitivity_analysis

    metrics: Dict[str, Any] = {}

    if equity_curve.height > 0:
        try:
            from engine.metrics import calculate_cagr, calculate_sharpe, calculate_max_drawdown, calculate_calmar_ratio

            cagr = calculate_cagr(equity_curve)
            sharpe = calculate_sharpe(equity_curve)
            max_dd, _ = calculate_max_drawdown(equity_curve)
            calmar = calculate_calmar_ratio(equity_curve)

            metrics = {
                "cagr": cagr,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "calmar": calmar,
            }
        except (ValueError, ImportError):
            metrics = {
                "cagr": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "calmar": 0.0,
            }
    else:
        metrics = {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }

    recommendation: Dict[str, Any] = {}

    # Kelly is the only method, so it's the best
    recommendation["method"] = "kelly"

    # Use Kelly if raw_kelly > 0, otherwise fall back to percent_equity
    if kelly_result["raw_kelly"] > 0:
        recommendation["method"] = "kelly"
    else:
        recommendation["method"] = "percent_equity"

    recommendation["kelly_fraction"] = kelly_result["kelly_fraction"]

    optimal_params: Dict[str, Any] = {}

    optimal_params["kelly_fraction"] = kelly_result["kelly_fraction"]
    optimal_params["kelly_confidence"] = kelly_result["confidence"]
    optimal_params["kelly_raw"] = kelly_result["raw_kelly"]

    return {
        "methods": results,
        "recommendation": recommendation,
        "sensitivity_analysis": sensitivity_results,
        "metrics": metrics,
        "optimal_params": optimal_params,
    }


def generate_sizing_report(
    optimization_results: dict,
    show_progress: bool = True,
) -> None:
    """Generate rich console report for position sizing optimization results.

    Args:
        optimization_results: Results from optimize_position_sizing()
        show_progress: Whether to display progress (for consistency)

    Returns:
        None (prints to console)
    """
    console.print("\n[bold cyan]Position Sizing Optimization Report[/bold cyan]")
    console.print("=" * 60)

    methods = optimization_results.get("methods", {})
    metrics = optimization_results.get("metrics", {})
    optimal_params = optimization_results.get("optimal_params", {})
    sensitivity_analysis = optimization_results.get("sensitivity_analysis", {})

    if not methods:
        console.print("[yellow]No position sizing methods were run.[/yellow]")
        return

    # Kelly Criterion Analysis (only method)
    if "kelly" in methods:
        kelly_result = methods["kelly"]

        kelly_table = Table(title="Kelly Criterion Analysis", show_header=True)
        kelly_table.add_column("Metric", style="cyan")
        kelly_table.add_column("Value", style="yellow")
        kelly_table.add_column("Description", style="blue")

        kelly_table.add_row(
            "Raw Kelly Fraction",
            f"{kelly_result['raw_kelly']:.4f}",
            "Full Kelly before safety capping",
        )
        kelly_table.add_row(
            "Recommended Kelly",
            f"{kelly_result['kelly_fraction']:.4f}",
            f"Safety capped at {kelly_result['kelly_fraction'] / kelly_result['raw_kelly']:.0%} of raw"
            if kelly_result["raw_kelly"] > 0
            else "No cap applied (raw kelly is 0)",
        )
        kelly_table.add_row(
            "Win Rate",
            f"{kelly_result['win_rate']:.2%}",
            "Percentage of winning trades",
        )
        kelly_table.add_row(
            "Win/Loss Ratio",
            f"{kelly_result['win_loss_ratio']:.2f}",
            "Average win / Average loss",
        )
        kelly_table.add_row(
            "Average Win",
            f"${kelly_result['avg_win']:.2f}",
            "Mean profit on winning trades",
        )
        kelly_table.add_row(
            "Average Loss",
            f"${kelly_result['avg_loss']:.2f}",
            "Mean loss on losing trades (negative)",
        )
        kelly_table.add_row(
            "Trade Count",
            str(kelly_result["trade_count"]),
            "Total trades in sample",
        )
        kelly_table.add_row(
            "Confidence",
            kelly_result["confidence"],
            "Based on trade count",
        )

        console.print(kelly_table)

        confidence_color = "green"
        if kelly_result["confidence"] == "low":
            confidence_color = "red"
        elif kelly_result["confidence"] == "medium":
            confidence_color = "yellow"

        console.print(f"Confidence Level: [{confidence_color}]{kelly_result['confidence']}[/{confidence_color}]")

        if kelly_result["confidence"] == "low":
            console.print("[yellow]Warning: Low trade count may make Kelly estimate unreliable.[/yellow]")

        console.print("")

    console.print("\n[bold cyan]Performance Metrics[/bold cyan]")
    perf_table = Table(show_header=True)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="yellow")

    cagr = metrics.get("cagr", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    calmar = metrics.get("calmar", 0.0)

    perf_table.add_row("CAGR", f"{cagr:.2%}")
    perf_table.add_row("Sharpe Ratio", f"{sharpe:.2f}")
    perf_table.add_row("Max Drawdown", f"{max_dd:.2%}")
    perf_table.add_row("Calmar Ratio", f"{calmar:.2f}")

    console.print(perf_table)
    console.print("")

    console.print("\n[bold cyan]Optimal Parameters Summary[/bold cyan]")
    params_table = Table(show_header=True)
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="yellow")

    for key, value in optimal_params.items():
        key_display = key.replace("_", " ").title()
        if isinstance(value, float):
            value_display = f"{value:.4f}"
        elif isinstance(value, int):
            value_display = str(value)
        else:
            value_display = str(value)
        params_table.add_row(key_display, value_display)

    console.print(params_table)
    console.print("")

    # Kelly Sensitivity Analysis
    if sensitivity_analysis and "kelly" in sensitivity_analysis:
        console.print("\n[bold cyan]Kelly Sensitivity Analysis[/bold cyan]")
        sens_table = Table(show_header=True)
        sens_table.add_column("Kelly Fraction", style="cyan")
        sens_table.add_column("Effective Size", style="yellow")

        for item in sensitivity_analysis["kelly"]:
            sens_table.add_row(
                f"{item['fraction']:.3f}",
                f"{item['effective_size']:.4f}",
            )

        console.print(sens_table)
        console.print("")

    console.print("=" * 60)
    console.print("[bold cyan]Position Sizing Optimization Complete[/bold cyan]")
