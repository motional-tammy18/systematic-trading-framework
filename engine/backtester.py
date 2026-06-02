"""Vectorized backtest engine for algorithmic trading strategies."""

from typing import Tuple, Optional
import polars as pl
import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

# Console for rich output
console = Console()


def run_backtest(
    signals: pl.DataFrame,
    config: dict,
    show_progress: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Run vectorized backtest on signal data.

    Args:
        signals: DataFrame with timestamp, open, high, low, close, volume, signal columns
        config: Backtest configuration dict (must contain position_sizing settings)
        show_progress: Whether to display progress bar during backtest

    Returns:
        Tuple of (trade_log, equity_curve) DataFrames
    """
    required_cols = {"timestamp", "open", "high", "low", "close", "volume", "signal"}
    if not required_cols.issubset(set(signals.columns)):
        missing = required_cols - set(signals.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Read position sizing mode from config (strategy config overrides global default)
    position_sizing_mode = config.get(
        "mode", config.get("default_mode", "percent_equity")
    )
    if position_sizing_mode not in ("fixed_amount", "percent_equity"):
        raise ValueError(f"Invalid position_sizing_mode: {position_sizing_mode}")

    # Debug: Log position sizing settings
    if show_progress:
        console.print(
            f"[cyan]Position Sizing:[/cyan] mode={position_sizing_mode}, fixed_amount={config.get('fixed_amount', 'N/A')}, percent_equity={config.get('percent_equity', 'N/A')}"
        )

    # Check if order_management is configured
    use_order_manager = "order_management" in config

    if use_order_manager:
        # Use OrderManager for advanced order lifecycle management
        from engine.order_manager import OrderManager

        manager = OrderManager(config)
        trades_df = manager.process_signals(signals, signals)

        # Calculate PnL with dynamic commission
        return _calculate_pnl_with_trades(
            trades_df, signals, config, position_sizing_mode, show_progress
        )


def _get_commission_rates(
    entry_type: str, exit_reason: Optional[str], config: dict
) -> Tuple[float, float]:
    """Get commission rates based on entry and exit types.

    Commission Logic:
    - Market entry: taker fee (0.055%)
    - Limit entry: maker fee (0.02%)
    - TP/SL exit: taker fee (0.055%)
    - Signal exit: taker fee (0.055%)

    Args:
        entry_type: "market" or "limit"
        exit_reason: "TP", "SL", "SIGNAL", "TRAILING_STOP", or None
        config: Configuration dict with fee settings

    Returns:
        Tuple of (entry_fee_rate, exit_fee_rate)
    """
    maker_fee = config.get("maker_fee", 0.0002)  # 0.02%
    taker_fee = config.get("taker_fee", 0.00055)  # 0.055%

    # Entry fee based on entry type
    if entry_type == "limit":
        entry_rate = maker_fee
    else:  # market or default
        entry_rate = taker_fee

    # Exit fee: all exits are taker (market orders)
    exit_rate = taker_fee

    return entry_rate, exit_rate


def _calculate_pnl_with_trades(
    trades_df: pl.DataFrame,
    signals: pl.DataFrame,
    config: dict,
    position_sizing_mode: str,
    show_progress: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Calculate PnL and equity curve from resolved trades.

    Args:
        trades_df: DataFrame with resolved trades from OrderManager
        signals: Original signals DataFrame for equity tracking
        config: Backtest configuration
        position_sizing_mode: Position sizing mode
        show_progress: Whether to display progress bar during backtest

    Returns:
        Tuple of (trade_log, equity_curve) DataFrames
    """
    initial_capital = config.get("initial_capital", 10000.0)
    slippage_pct = config.get("slippage_pct", 0.0005)
    max_drawdown_pct = config.get("max_drawdown_pct", 0.50)

    fixed_amount = config.get("fixed_amount", 1000.0)
    percent_equity = config.get("percent_equity", 0.10)

    timestamps = signals["timestamp"].to_list()
    opens = signals["open"].to_numpy()
    n_bars = len(signals)

    equity = np.zeros(n_bars)
    equity[0] = initial_capital
    drawdown = np.zeros(n_bars)

    current_equity = initial_capital
    circuit_breaker_triggered = False
    circuit_breaker_bar = -1

    # Lists for trade log
    trades_entry_time = []
    trades_exit_time = []
    trades_entry_price = []
    trades_exit_price = []
    trades_direction = []
    trades_entry_type = []
    trades_exit_reason = []
    trades_pnl = []
    trades_pnl_pct = []
    trades_size = []
    trades_entry_fee = []
    trades_exit_fee = []
    trades_total_fees = []

    # Convert trades to list of dicts for iteration
    if len(trades_df) > 0:
        trades_list = trades_df.to_dicts()
    else:
        trades_list = []

    # Track which bars have trades for equity curve
    trade_idx = 0

    # Progress bar setup
    if show_progress:
        progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        progress_bar.start()
        task = progress_bar.add_task(
            "[cyan]Backtest",
            total=n_bars,
        )
    else:
        progress_bar = None
        task = None

    for i in range(n_bars):
        if i == 0:
            equity[i] = current_equity
            continue

        if circuit_breaker_triggered:
            equity[i] = current_equity
            peak_equity = np.max(equity[:i]) if i > 0 else initial_capital
            drawdown[i] = (
                (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
            )
            if show_progress and progress_bar:
                progress_bar.update(task, advance=1)
            continue

        # Check if there's a trade exiting at this bar
        if trade_idx < len(trades_list):
            trade = trades_list[trade_idx]
            exit_time = trade["exit_time"]

            # Find if this trade exits at current bar
            if exit_time == timestamps[i]:
                # Calculate position size
                entry_price = trade["entry_price"]
                direction = trade["direction"]
                entry_type = trade.get("entry_type", "market")
                exit_reason = trade.get("exit_reason", "SIGNAL")
                exit_price = trade["exit_price"]

                if position_sizing_mode == "fixed_amount":
                    position_size = fixed_amount / entry_price
                else:
                    trade_amount = current_equity * percent_equity
                    position_size = trade_amount / entry_price

                # Calculate gross PnL
                if direction == 1:  # Long
                    gross_pnl = (exit_price - entry_price) * position_size
                else:  # Short
                    gross_pnl = (entry_price - exit_price) * position_size

                # Calculate fees with dynamic rates
                entry_rate, exit_rate = _get_commission_rates(
                    entry_type, exit_reason, config
                )

                entry_value = entry_price * position_size
                exit_value = exit_price * position_size
                entry_fee = entry_value * entry_rate
                exit_fee = exit_value * exit_rate
                total_fees = entry_fee + exit_fee

                net_pnl = gross_pnl - total_fees
                entry_value = entry_price * position_size
                if entry_value > 0:
                    pnl_pct = net_pnl / entry_value * 100
                else:
                    pnl_pct = 0.0

                # Record trade
                # Convert numpy datetime64 to Python datetime for polars compatibility
                entry_time = trade["entry_time"]
                if hasattr(entry_time, "astype"):
                    # numpy datetime64 -> convert to Python datetime
                    if isinstance(entry_time, np.datetime64):
                        entry_time = entry_time.astype("datetime64[s]").astype(object)
                trades_entry_time.append(entry_time)

                exit_time = trade["exit_time"]
                if hasattr(exit_time, "astype"):
                    if isinstance(exit_time, np.datetime64):
                        exit_time = exit_time.astype("datetime64[s]").astype(object)
                trades_exit_time.append(exit_time)

                trades_entry_price.append(entry_price)
                trades_exit_price.append(exit_price)
                trades_direction.append(direction)
                trades_entry_type.append(entry_type)
                trades_exit_reason.append(exit_reason)
                trades_pnl.append(net_pnl)
                trades_pnl_pct.append(pnl_pct)
                trades_size.append(position_size)
                trades_entry_fee.append(entry_fee)
                trades_exit_fee.append(exit_fee)
                trades_total_fees.append(total_fees)

                current_equity += net_pnl
                trade_idx += 1

        equity[i] = current_equity

        peak_equity = np.max(equity[: i + 1])
        if peak_equity > 0:
            drawdown[i] = (peak_equity - current_equity) / peak_equity

        # Update progress every 100 bars
        if show_progress and progress_bar and i % 100 == 0:
            progress_bar.update(
                task,
                advance=100,
                description=f"[cyan]Backtest: [green]Equity=${current_equity:.2f} [yellow]DD={drawdown[i]:.2%} [blue]Trades={len(trades_pnl)}",
            )

        if drawdown[i] >= max_drawdown_pct:
            circuit_breaker_triggered = True
            circuit_breaker_bar = i

    # Close progress bar
    if show_progress and progress_bar:
        progress_bar.update(task, advance=n_bars % 100)
        progress_bar.stop()

    # Create equity curve
    equity_curve = pl.DataFrame(
        {
            "timestamp": timestamps,
            "equity": equity,
            "drawdown": drawdown,
        }
    )

    # Create trade log
    if trades_entry_time:
        trade_log = pl.DataFrame(
            {
                "entry_time": trades_entry_time,
                "exit_time": trades_exit_time,
                "entry_price": trades_entry_price,
                "exit_price": trades_exit_price,
                "direction": trades_direction,
                "entry_type": trades_entry_type,
                "exit_reason": trades_exit_reason,
                "pnl": trades_pnl,
                "pnl_pct": trades_pnl_pct,
                "size": trades_size,
                "entry_fee": trades_entry_fee,
                "exit_fee": trades_exit_fee,
                "total_fees": trades_total_fees,
            }
        )

        if circuit_breaker_triggered:
            trade_log = trade_log.with_columns(
                pl.lit(f"Circuit breaker at bar {circuit_breaker_bar}").alias(
                    "circuit_breaker_reason"
                )
            )
    else:
        trade_log = pl.DataFrame(
            {
                "entry_time": [],
                "exit_time": [],
                "entry_price": [],
                "exit_price": [],
                "direction": [],
                "entry_type": [],
                "exit_reason": [],
                "pnl": [],
                "pnl_pct": [],
                "size": [],
                "entry_fee": [],
                "exit_fee": [],
                "total_fees": [],
            },
            schema={
                "entry_time": pl.Datetime,
                "exit_time": pl.Datetime,
                "entry_price": pl.Float64,
                "exit_price": pl.Float64,
                "direction": pl.Int8,
                "entry_type": pl.Utf8,
                "exit_reason": pl.Utf8,
                "pnl": pl.Float64,
                "pnl_pct": pl.Float64,
                "size": pl.Float64,
                "entry_fee": pl.Float64,
                "exit_fee": pl.Float64,
                "total_fees": pl.Float64,
            },
        )

    return trade_log, equity_curve
