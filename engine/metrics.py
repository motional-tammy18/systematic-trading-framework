"""Performance metrics calculation module for algorithmic trading.

All metrics use vectorized polars operations for optimal performance.
Crypto annualization uses 365 days (not 252 like traditional markets).
"""

from datetime import datetime
from math import sqrt
from typing import Any, Tuple

import polars as pl


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a potentially None value to float."""
    if value is None:
        return default
    try:
        return float(value)  # type: ignore
    except (TypeError, ValueError):
        return default


def calculate_sharpe(equity_curve: pl.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe Ratio using CAGR (geometric mean).

    Formula: ((CAGR_daily - risk_free_rate_daily) / std(returns)) * sqrt(365)
    where CAGR_daily = (1 + CAGR_annual)^(1/365) - 1

    Uses geometric mean (CAGR) instead of arithmetic mean for a more
    conservative measure that accounts for compounding effects.

    Args:
        equity_curve: DataFrame with 'timestamp' and 'equity' columns (f64)
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Annualized Sharpe ratio as float

    Raises:
        ValueError: If equity curve has fewer than 2 data points
    """
    if equity_curve.height < 2:
        raise ValueError("Equity curve must have at least 2 data points")

    if "timestamp" not in equity_curve.columns or "equity" not in equity_curve.columns:
        raise ValueError("Equity curve must have 'timestamp' and 'equity' columns")

    equity = equity_curve["equity"]
    returns = equity.pct_change().drop_nulls()

    returns_std = _safe_float(returns.std())
    if returns_std == 0:
        return 0.0

    # Use CAGR for geometric mean (compound annualized return)
    annual_return = calculate_cagr(equity_curve)
    daily_return = (1 + annual_return) ** (1 / 365) - 1

    excess_returns = daily_return - (risk_free_rate / 365)
    sharpe = (excess_returns / returns_std) * sqrt(365)

    return sharpe


def calculate_sortino(equity_curve: pl.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sortino Ratio using CAGR (geometric mean).

    Formula: ((CAGR_daily - risk_free_rate_daily) / downside_std) * sqrt(365)
    where CAGR_daily = (1 + CAGR_annual)^(1/365) - 1
    and downside_std = std of negative returns only

    Uses geometric mean (CAGR) instead of arithmetic mean for consistency
    with Sharpe ratio calculation and a more conservative risk-adjusted measure.

    Args:
        equity_curve: DataFrame with 'timestamp' and 'equity' columns (f64)
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Annualized Sortino ratio as float

    Raises:
        ValueError: If equity curve has fewer than 2 data points
    """
    if equity_curve.height < 2:
        raise ValueError("Equity curve must have at least 2 data points")

    if "timestamp" not in equity_curve.columns or "equity" not in equity_curve.columns:
        raise ValueError("Equity curve must have 'timestamp' and 'equity' columns")

    equity = equity_curve["equity"]
    returns = equity.pct_change().drop_nulls()
    negative_returns = returns.filter(returns < 0)

    if negative_returns.len() == 0:
        return 0.0

    downside_std = _safe_float(negative_returns.std())
    if downside_std == 0:
        return 0.0

    # Use CAGR for geometric mean (compound annualized return)
    annual_return = calculate_cagr(equity_curve)
    daily_return = (1 + annual_return) ** (1 / 365) - 1

    excess_returns = daily_return - (risk_free_rate / 365)
    sortino = (excess_returns / downside_std) * sqrt(365)

    return sortino


def calculate_max_drawdown(equity_curve: pl.DataFrame) -> Tuple[float, int]:
    """Calculate maximum drawdown percentage and duration.

    Drawdown at each point: (peak - current) / peak
    Duration: bars from peak to new peak (recovery)

    Formula:
        - DD% = max((running_max - equity) / running_max)
        - Duration = max bars from peak to when equity exceeds that peak

    Args:
        equity_curve: DataFrame with 'equity' column (f64)

    Returns:
        Tuple of (max_drawdown_pct, max_drawdown_duration_bars)
        DD% is positive (e.g., 0.20 means 20% drawdown)

    Raises:
        ValueError: If equity curve is empty
    """
    if equity_curve.height == 0:
        raise ValueError("Equity curve cannot be empty")

    equity = equity_curve["equity"]
    running_max = equity.cum_max()
    drawdowns = (running_max - equity) / running_max
    max_dd_pct = _safe_float(drawdowns.max())

    in_drawdown = equity < running_max
    max_duration = 0
    current_duration = 0

    for is_in_dd in in_drawdown.to_list():
        if is_in_dd:
            current_duration += 1
        else:
            max_duration = max(max_duration, current_duration)
            current_duration = 0

    max_duration = max(max_duration, current_duration)

    return (max_dd_pct, max_duration)


def calculate_win_rate(trades: pl.DataFrame) -> float:
    """Calculate win rate percentage from trade log.

    Formula: winning_trades / total_trades * 100

    Args:
        trades: DataFrame with 'pnl' column (profit/loss per trade)

    Returns:
        Win rate as percentage (0-100)

    Raises:
        ValueError: If trades DataFrame is empty
    """
    if trades.height == 0:
        raise ValueError("Trades DataFrame cannot be empty")

    if "pnl" not in trades.columns:
        raise ValueError("Trades DataFrame must have 'pnl' column")

    pnl = trades["pnl"]
    total_trades = trades.height
    winning_trades = pnl.filter(pnl > 0).len()

    return (winning_trades / total_trades) * 100.0


def calculate_profit_factor(trades: pl.DataFrame) -> float:
    """Calculate profit factor (gross profit / gross loss).

    Formula: sum(winning_trades) / abs(sum(losing_trades))

    Args:
        trades: DataFrame with 'pnl' column (profit/loss per trade)

    Returns:
        Profit factor as float (1.0 = breakeven, >1 profitable)

    Raises:
        ValueError: If trades DataFrame is empty or has no losing trades
    """
    if trades.height == 0:
        raise ValueError("Trades DataFrame cannot be empty")

    if "pnl" not in trades.columns:
        raise ValueError("Trades DataFrame must have 'pnl' column")

    pnl = trades["pnl"]
    gross_profit = _safe_float(pnl.filter(pnl > 0).sum())
    gross_loss = _safe_float(pnl.filter(pnl < 0).sum())

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / abs(gross_loss)


def calculate_average_trade_pnl(trades: pl.DataFrame) -> float:
    """Calculate average profit/loss per trade.

    Formula: sum(pnl) / number_of_trades

    Args:
        trades: DataFrame with 'pnl' column (profit/loss per trade)

    Returns:
        Average PnL per trade as float

    Raises:
        ValueError: If trades DataFrame is empty
    """
    if trades.height == 0:
        raise ValueError("Trades DataFrame cannot be empty")

    if "pnl" not in trades.columns:
        raise ValueError("Trades DataFrame must have 'pnl' column")

    return _safe_float(trades["pnl"].mean())


def calculate_expectancy(trades: pl.DataFrame) -> float:
    """Calculate trade expectancy (expected value per trade).

    Formula: (avg_win * win_rate) - (avg_loss * loss_rate)
    where win_rate and loss_rate are decimals (0-1)

    Args:
        trades: DataFrame with 'pnl' column (profit/loss per trade)

    Returns:
        Expectancy as float (expected PnL per trade)

    Raises:
        ValueError: If trades DataFrame is empty
    """
    if trades.height == 0:
        raise ValueError("Trades DataFrame cannot be empty")

    if "pnl" not in trades.columns:
        raise ValueError("Trades DataFrame must have 'pnl' column")

    pnl = trades["pnl"]
    total_trades = trades.height

    winning_trades = pnl.filter(pnl > 0)
    losing_trades = pnl.filter(pnl < 0)

    win_rate = winning_trades.len() / total_trades
    loss_rate = losing_trades.len() / total_trades

    avg_win = _safe_float(winning_trades.mean()) if winning_trades.len() > 0 else 0.0
    avg_loss = _safe_float(losing_trades.mean()) if losing_trades.len() > 0 else 0.0

    expectancy = (avg_win * win_rate) - (abs(avg_loss) * loss_rate)

    return expectancy


def calculate_cagr(equity_curve: pl.DataFrame) -> float:
    """Calculate Compound Annual Growth Rate.

    Formula: (ending_value / starting_value)^(1/years) - 1

    For crypto: years = number_of_days / 365

    Args:
        equity_curve: DataFrame with 'timestamp' and 'equity' columns

    Returns:
        CAGR as decimal (e.g., 0.25 means 25% annual growth)

    Raises:
        ValueError: If equity curve has fewer than 2 data points
    """
    if equity_curve.height < 2:
        raise ValueError("Equity curve must have at least 2 data points")

    if "timestamp" not in equity_curve.columns or "equity" not in equity_curve.columns:
        raise ValueError("Equity curve must have 'timestamp' and 'equity' columns")

    start_equity = equity_curve["equity"][0]
    end_equity = equity_curve["equity"][-1]

    if start_equity <= 0:
        raise ValueError("Starting equity must be positive")

    start_time = equity_curve["timestamp"][0]
    end_time = equity_curve["timestamp"][-1]

    total_seconds = _calculate_time_diff_seconds(start_time, end_time)
    years = total_seconds / (365.0 * 24 * 3600)

    if years <= 0:
        return 0.0

    # Handle very short time periods (less than 1 day) to avoid overflow
    if years < 1 / 365:
        return (end_equity / start_equity) - 1.0

    cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0

    return cagr


def _calculate_time_diff_seconds(start_time: Any, end_time: Any) -> float:
    """Calculate time difference in seconds between two timestamps."""
    if isinstance(start_time, datetime) and isinstance(end_time, datetime):
        return (end_time - start_time).total_seconds()

    if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
        return float(end_time) - float(start_time)  # type: ignore

    try:
        start_ts = start_time.timestamp() if hasattr(start_time, "timestamp") else float(start_time)  # type: ignore
        end_ts = end_time.timestamp() if hasattr(end_time, "timestamp") else float(end_time)  # type: ignore
        return end_ts - start_ts
    except (TypeError, AttributeError):
        return (float(end_time) - float(start_time)) * 86400  # type: ignore


def calculate_calmar_ratio(equity_curve: pl.DataFrame) -> float:
    """Calculate Calmar Ratio (CAGR / Max Drawdown).

    Formula: CAGR / max_drawdown_percentage

    Args:
        equity_curve: DataFrame with 'timestamp' and 'equity' columns

    Returns:
        Calmar ratio as float

    Raises:
        ValueError: If max drawdown is zero
    """
    cagr = calculate_cagr(equity_curve)
    max_dd_pct, _ = calculate_max_drawdown(equity_curve)

    if max_dd_pct <= 0:
        return float("inf") if cagr > 0 else 0.0

    return cagr / max_dd_pct


def calculate_recovery_factor(trade_log: pl.DataFrame) -> float:
    """Calculate Recovery Factor (Net Profit / Maximum Drawdown).

    Formula: total_pnl / max_drawdown_in_dollars

    Measures how quickly the strategy recovers from drawdowns.
    Higher values indicate better risk-adjusted performance.

    Args:
        trade_log: DataFrame with 'pnl' column

    Returns:
        Recovery factor as float (higher is better)
        Returns inf if max drawdown is 0 (no drawdown occurred)

    Raises:
        ValueError: If trade_log is empty
    """
    if trade_log.height == 0:
        raise ValueError("Trade log cannot be empty")

    if "pnl" not in trade_log.columns:
        raise ValueError("Trade log must have 'pnl' column")

    total_pnl = _safe_float(trade_log["pnl"].sum())

    # Calculate max drawdown from equity curve derived from trades
    initial_capital = 10000.0  # Standard initial capital for backtests
    if "equity" not in trade_log.columns:
        equity = initial_capital + trade_log["pnl"].cum_sum()
        running_max = equity.cum_max()
        drawdowns = (running_max - equity) / running_max
        max_dd_pct = _safe_float(drawdowns.max())
    else:
        equity = trade_log["equity"]
        running_max = equity.cum_max()
        drawdowns = (running_max - equity) / running_max
        max_dd_pct = _safe_float(drawdowns.max())

    if max_dd_pct == 0:
        return float("inf") if total_pnl > 0 else 0.0

    # Convert DD percentage to dollar amount for proper recovery factor calculation
    max_dd_dollars = max_dd_pct * initial_capital
    return total_pnl / max_dd_dollars


def calculate_walk_forward_efficiency(
    is_equity_curve: pl.DataFrame, oos_equity_curve: pl.DataFrame
) -> float:
    """Calculate Walk Forward Efficiency (WFE).

    Measures how well in-sample performance translates to out-of-sample.
    The "gold standard" metric for assessing whether a strategy captures
    genuine market edge or is overfit to historical noise.

    Formula: (OOS annualized_return / IS annualized_return) * 100

    Interpretation:
    - > 60%: Strategy captures genuine edge (PASS)
    - 50-60%: Marginal, needs investigation
    - < 50%: Likely overfit (FAIL)
    - > 100%: Anomalous, possible data snooping bias

    Args:
        is_equity_curve: In-sample equity curve with 'equity' column
        oos_equity_curve: Out-of-sample equity curve with 'equity' column

    Returns:
        WFE as percentage (e.g., 65.0 means 65% efficiency)

    Raises:
        ValueError: If either equity curve is insufficient
    """
    if is_equity_curve.height < 2:
        raise ValueError("In-sample equity curve must have at least 2 data points")

    if oos_equity_curve.height < 2:
        raise ValueError("Out-of-sample equity curve must have at least 2 data points")

    is_cagr = calculate_cagr(is_equity_curve)
    oos_cagr = calculate_cagr(oos_equity_curve)

    if is_cagr <= 0:
        if oos_cagr > 0:
            return float("inf")
        return 0.0

    wfe = (oos_cagr / is_cagr) * 100.0

    return wfe


def stitch_equity_curves(equity_curves: list[pl.DataFrame]) -> pl.DataFrame:
    """Stitch multiple equity curves into a single chronological curve.

    Takes OOS equity curves from walk-forward windows and combines them
    into a single continuous curve for final performance metrics.
    Resets equity at each window start to avoid discontinuity jumps.

    Args:
        equity_curves: List of equity curve DataFrames with 'timestamp' and 'equity' columns

    Returns:
        Single stitched equity curve DataFrame with 'timestamp' and 'equity' columns

    Raises:
        ValueError: If equity_curves is empty or curves have invalid schema
    """
    if not equity_curves:
        raise ValueError("Cannot stitch empty equity curves list")

    for i, curve in enumerate(equity_curves):
        if curve.height > 0:
            if "timestamp" not in curve.columns or "equity" not in curve.columns:
                raise ValueError(
                    f"Equity curve {i} must have 'timestamp' and 'equity' columns"
                )

    non_empty_curves = [c for c in equity_curves if c.height > 0]

    if not non_empty_curves:
        return pl.DataFrame({"timestamp": [], "equity": []})

    non_empty_curves.sort(key=lambda c: c["timestamp"][0])

    stitched_parts = []
    previous_end_equity = 10000.0

    for curve in non_empty_curves:
        curve = curve.sort("timestamp")
        start_equity = curve["equity"][0]
        equity_array = curve["equity"].to_numpy()
        returns_from_start = equity_array - start_equity
        normalized_equity = previous_end_equity + returns_from_start

        stitched_parts.append(
            pl.DataFrame({"timestamp": curve["timestamp"], "equity": normalized_equity})
        )

        previous_end_equity = normalized_equity[-1]

    if not stitched_parts:
        return pl.DataFrame({"timestamp": [], "equity": []})

    return pl.concat(stitched_parts)
