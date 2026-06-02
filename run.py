"""CLI entry point for the systematic trading framework.

Provides command-line interface for running backtests, validation,
and full pipeline execution with progress reporting and exit codes.

Usage:
    python run.py --strategy ema_crossover_rsi --mode full
    python run.py --strategy ema_crossover_rsi --mode backtest
    python run.py --strategy ema_crossover_rsi --mode validate
    python run.py --strategy ema_crossover_rsi --mode full --pairs BTC/USDT:USDT
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console(force_terminal=True, legacy_windows=False)


def _load_global_config(config_path: str = "config/global_config.yaml") -> dict:
    """Load global configuration from YAML file.

    Args:
        config_path: Path to global config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_strategy_config(strategy_name: str) -> dict:
    """Load strategy configuration from YAML file.

    Args:
        strategy_name: Name of the strategy

    Returns:
        Strategy configuration dictionary

    Raises:
        FileNotFoundError: If strategy config doesn't exist
    """
    config_path = Path(f"strategies/{strategy_name}/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Strategy config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_strategy_class(strategy_name: str):
    """Dynamically import and return strategy class.

    Args:
        strategy_name: Name of the strategy

    Returns:
        Strategy class

    Raises:
        ImportError: If strategy module cannot be imported
        AttributeError: If strategy class not found in module
    """
    module_path = f"strategies.{strategy_name}.signals"
    module = importlib.import_module(module_path)

    # Find the strategy class (should be the only class inheriting from BaseStrategy)
    strategy_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and attr_name != "BaseStrategy"
            and hasattr(attr, "generate_signals")
        ):
            strategy_class = attr
            break

    if strategy_class is None:
        raise AttributeError(f"No strategy class found in {module_path}")

    return strategy_class


def _check_and_fetch_data(
    pairs: List[str],
    timeframe: str,
    storage_path: str = "data/raw",
    config: Optional[dict] = None,
) -> None:
    """Ensure OHLCV data is complete from earliest to latest.

    Fetches data from Bybit exchange.

    Args:
        pairs: List of trading pairs
        timeframe: Timeframe string
        storage_path: Path to data storage
        config: Global configuration
    """
    if config is None:
        config = _load_global_config()

    for pair in pairs:
        print(f"  Ensuring complete data coverage for {pair} {timeframe}...")
        _fetch_with_fallback(pair, timeframe, fetch_from="max", storage_path=storage_path, config=config)


def _fetch_with_fallback(
    symbol: str,
    timeframe: str,
    fetch_from: str = "max",
    storage_path: str = "data/raw",
    config: Optional[dict] = None,
) -> None:
    """Fetch OHLCV data from Bybit exchange.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
        timeframe: Timeframe string
        fetch_from: Start point - 'max' for all history
        storage_path: Path to data storage
        config: Global configuration
    """
    from data.fetcher import fetch_ohlcv as bybit_fetch
    bybit_fetch(symbol, timeframe, fetch_from=fetch_from, storage_path=storage_path, config=config)
    print(f"    [Bybit] Successfully fetched data for {symbol}")




def _load_data(
    pairs: List[str],
    timeframe: str,
    storage_path: str = "data/raw",
    days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Any:
    """Load OHLCV data for specified pairs.

    For single pair, returns DataFrame directly.
    For multiple pairs, returns concatenated DataFrame.

    Args:
        pairs: List of trading pairs
        timeframe: Timeframe string
        storage_path: Path to data storage
        days: Optional filter to last N days
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        Polars DataFrame with OHLCV data
    """
    from data.storage import load_ohlcv
    import polars as pl

    dataframes = []
    for pair in pairs:
        df = load_ohlcv(pair, timeframe, storage_path)
        df = df.with_columns(pl.lit(pair).alias("symbol"))
        dataframes.append(df)

    if len(dataframes) == 1:
        data = dataframes[0]
    else:
        # Concatenate multiple pairs
        data = pl.concat(dataframes)

    # Apply date filtering
    data = _filter_data_by_date(data, days, start_date, end_date)

    # Log filtering for user visibility
    if days is not None:
        print(f"  Filtered to last {days} days ({data.height} bars)")
    elif start_date is not None or end_date is not None:
        date_range_str = f"{start_date or 'start'} to {end_date or 'end'}"
        print(f"  Filtered to date range: {date_range_str} ({data.height} bars)")

    return data


def _filter_data_by_date(
    df: pl.DataFrame,
    days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pl.DataFrame:
    """Filter OHLCV DataFrame by date range.

    Priority: absolute dates > --days > no filter (all data)

    Args:
        df: Polars DataFrame with timestamp column
        days: Filter to last N days from latest timestamp
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        Filtered DataFrame

    Raises:
        ValueError: If date format is invalid or dates don't make logical sense
    """
    from datetime import datetime, timedelta

    # Priority 1: Absolute dates (start_date and/or end_date)
    if start_date is not None or end_date is not None:
        if start_date is not None:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(
                    f"Invalid start_date format: '{start_date}'. Use YYYY-MM-DD."
                )
            df = df.filter(pl.col("timestamp") >= start_dt)

        if end_date is not None:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                # Add 1 day to make end_date inclusive (up to end of that day)
                end_dt = end_dt + timedelta(days=1)
            except ValueError:
                raise ValueError(
                    f"Invalid end_date format: '{end_date}'. Use YYYY-MM-DD."
                )
            df = df.filter(pl.col("timestamp") < end_dt)

        return df

    # Priority 2: --days (last N days)
    if days is not None:
        if days <= 0:
            raise ValueError(f"--days must be positive, got {days}")

        # Get the latest timestamp in the data
        latest_ts = df["timestamp"].max()
        if latest_ts is None:
            return df  # Empty dataframe, nothing to filter

        cutoff_ts = latest_ts - timedelta(days=days)
        df = df.filter(pl.col("timestamp") >= cutoff_ts)
        return df

    # Priority 3: No filter - return all data
    return df


def _print_progress(step: int, total: int, message: str) -> None:
    """Print progress indicator in [X/Y] format.

    Args:
        step: Current step number
        total: Total number of steps
        message: Progress message
    """
    print(f"[{step}/{total}] {message}")


def _merge_configs(strategy_config: dict, global_config: dict) -> dict:
    """Merge strategy config with global config.

    Strategy config takes precedence over global config.

    Args:
        strategy_config: Strategy-specific configuration
        global_config: Global framework configuration

    Returns:
        Merged configuration dictionary

    Note:
        - backtest and position_sizing sections are flattened to top-level keys
        - validation and order_management are preserved as nested dicts
        - Strategy config values override global config values
    """
    merged = {}

    # Start with global backtest settings (flattened)
    if "backtest" in global_config:
        merged.update(global_config["backtest"])

    # Add global position sizing (flattened)
    if "position_sizing" in global_config:
        merged.update(global_config["position_sizing"])

    # Add global validation settings (PRESERVE NESTED)
    if "validation" in global_config:
        merged["validation"] = global_config["validation"].copy()

    # Add global order_management (PRESERVE NESTED)
    if "order_management" in global_config:
        merged["order_management"] = global_config["order_management"].copy()

    # Override with strategy-specific position sizing (flattened)
    if "position_sizing" in strategy_config:
        merged.update(strategy_config["position_sizing"])

    # Deep merge validation (strategy overrides specific keys)
    if "validation" in strategy_config:
        if "validation" not in merged:
            merged["validation"] = {}
        # Deep merge: strategy overrides specific sections, inherits others
        for section, values in strategy_config["validation"].items():
            if section in merged["validation"] and isinstance(values, dict):
                merged["validation"][section].update(values)
            else:
                merged["validation"][section] = values

    # Deep merge order_management
    if "order_management" in strategy_config:
        if "order_management" not in merged:
            merged["order_management"] = {}
        merged["order_management"].update(strategy_config["order_management"])

    # Add strategy params and param_space
    merged["params"] = strategy_config.get("params", {})
    merged["param_space"] = strategy_config.get("param_space", {})

    # Add strategy-level fields needed by validator
    merged["timeframe"] = strategy_config.get("timeframe", "15m")

    # Add global position_sizing_optimization (PRESERVE NESTED)
    if "position_sizing_optimization" in global_config:
        if "position_sizing_optimization" not in merged:
            merged["position_sizing_optimization"] = {}
        merged["position_sizing_optimization"] = global_config[
            "position_sizing_optimization"
        ].copy()

    # Override with strategy-specific
    if "position_sizing_optimization" in strategy_config:
        if "position_sizing_optimization" not in merged:
            merged["position_sizing_optimization"] = {}
        # Deep merge strategy overrides
        for section, values in strategy_config["position_sizing_optimization"].items():
            if section in merged["position_sizing_optimization"] and isinstance(
                values, dict
            ):
                merged["position_sizing_optimization"][section].update(values)
            else:
                merged["position_sizing_optimization"][section] = values

    return merged


def _save_backtest_report(
    strategy_name: str,
    pairs: List[str],
    timeframe: str,
    trade_log: pl.DataFrame,
    equity_curve: pl.DataFrame,
    signals: pl.DataFrame,
    config: dict,
) -> None:
    """Save detailed backtest results to files.

    Args:
        strategy_name: Name of the strategy
        pairs: Trading pairs used
        timeframe: Timeframe of the data
        trade_log: DataFrame with trade details
        equity_curve: DataFrame with equity over time
        signals: DataFrame with signals
        config: Configuration used for backtest
    """
    from pathlib import Path
    import json
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports_dir = Path(f"reports/{strategy_name}/{date_str}")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save trade log as CSV
    if trade_log.height > 0:
        trade_csv_path = reports_dir / f"trades_{timestamp_str}.csv"
        trade_log.write_csv(trade_csv_path)

    # Save equity curve as CSV
    equity_csv_path = reports_dir / f"equity_{timestamp_str}.csv"
    equity_curve.write_csv(equity_csv_path)

    # Calculate ALL metrics
    from engine.metrics import (
        calculate_sharpe,
        calculate_max_drawdown,
        calculate_sortino,
        calculate_cagr,
        calculate_calmar_ratio,
        calculate_win_rate,
        calculate_profit_factor,
        calculate_expectancy,
        calculate_recovery_factor,
        calculate_average_trade_pnl,
    )

    # Calculate all performance metrics
    if trade_log.height > 0 and equity_curve.height > 0:
        sharpe = calculate_sharpe(equity_curve)
        sortino = calculate_sortino(equity_curve)
        cagr = calculate_cagr(equity_curve)
        max_dd, max_dd_duration = calculate_max_drawdown(equity_curve)
        calmar = calculate_calmar_ratio(equity_curve)
        win_rate = calculate_win_rate(trade_log)
        profit_factor = calculate_profit_factor(trade_log)
        expectancy = calculate_expectancy(trade_log)
        recovery_factor = calculate_recovery_factor(trade_log)
        avg_trade_pnl = calculate_average_trade_pnl(trade_log)

        total_pnl = trade_log["pnl"].sum()
        total_trades = trade_log.height
        winning_trades = int((trade_log["pnl"] > 0).sum())
        losing_trades = int((trade_log["pnl"] <= 0).sum())

        # Count buy and sell trades
        buy_trades = (
            int((trade_log["direction"] > 0).sum())
            if "direction" in trade_log.columns
            else 0
        )
        sell_trades = (
            int((trade_log["direction"] < 0).sum())
            if "direction" in trade_log.columns
            else 0
        )

        # Average win/loss
        winning_trades_df = trade_log.filter(pl.col("pnl") > 0)
        losing_trades_df = trade_log.filter(pl.col("pnl") < 0)
        avg_win = (
            winning_trades_df["pnl"].mean() if winning_trades_df.height > 0 else 0.0
        )
        avg_loss = (
            losing_trades_df["pnl"].mean() if losing_trades_df.height > 0 else 0.0
        )
        best_trade = float(trade_log["pnl"].max())
        worst_trade = float(trade_log["pnl"].min())

        initial_capital = config.get("initial_capital", 10000)
        final_equity = float(equity_curve["equity"][-1])
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
    else:
        # No trades - set all metrics to zero/None
        sharpe = sortino = cagr = calmar = win_rate = profit_factor = 0.0
        expectancy = avg_trade_pnl = avg_win = avg_loss = recovery_factor = 0.0
        max_dd = max_dd_duration = total_pnl = total_trades = 0
        winning_trades = losing_trades = buy_trades = sell_trades = 0
        best_trade = worst_trade = 0.0
        initial_capital = config.get("initial_capital", 10000)
        final_equity = initial_capital
        total_return = 0.0

    data_start = signals["timestamp"].min()
    data_end = signals["timestamp"].max()
    duration_days = (data_end - data_start).total_seconds() / 86400

    # Build comprehensive summary with ALL metrics
    summary = {
        "strategy": strategy_name,
        "pairs": pairs,
        "timeframe": timeframe,
        "timestamp": timestamp_str,
        "data_start": str(data_start),
        "data_end": str(data_end),
        "duration_days": round(duration_days, 1),
        "metrics": {
            # Performance Metrics
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "cagr": round(cagr, 4),
            "calmar_ratio": round(calmar, 2),
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_duration": max_dd_duration,
            "recovery_factor": round(recovery_factor, 2),
            "profit_factor": round(profit_factor, 2),
            "total_return_pct": round(total_return, 2),
            # Trade Statistics
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "win_rate": round(win_rate, 2),
            "expectancy": round(expectancy, 2),
            "average_trade_pnl": round(avg_trade_pnl, 2),
            "average_win": round(avg_win, 2),
            "average_loss": round(avg_loss, 2),
            "best_trade": round(best_trade, 2),
            "worst_trade": round(worst_trade, 2),
            # PnL
            "total_pnl": round(float(total_pnl), 2),
            "initial_capital": initial_capital,
            "final_equity": round(final_equity, 2),
        },
        "config": {
            "initial_capital": config.get("initial_capital", 10000),
            "taker_fee": config.get("taker_fee", 0.00055),
            "maker_fee": config.get("maker_fee", 0.0002),
            "slippage_pct": config.get("slippage_pct", 0.0005),
            "params": config.get("params", {}),
        },
    }

    # Save summary as JSON
    summary_json_path = reports_dir / f"summary_{timestamp_str}.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save human-readable summary with ALL metrics
    summary_txt_path = reports_dir / f"summary_{timestamp_str}.txt"
    with open(summary_txt_path, "w") as f:
        f.write(f"Backtest Report: {strategy_name}\n")
        f.write(f"Generated: {timestamp_str}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Strategy Configuration:\n")
        f.write(f"  Pairs: {', '.join(pairs)}\n")
        f.write(f"  Timeframe: {timeframe}\n")
        f.write(f"  Data Range: {data_start} to {data_end}\n")
        f.write(
            f"  Duration: {duration_days:.1f} days ({duration_days / 365:.1f} years)\n\n"
        )

        f.write(f"Parameters:\n")
        for key, value in config.get("params", {}).items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\n")

        # Performance Metrics - ALL metrics
        f.write(f"Performance Metrics:\n")
        f.write(f"  Sharpe Ratio: {summary['metrics']['sharpe_ratio']:.2f}\n")
        f.write(f"  Sortino Ratio: {summary['metrics']['sortino_ratio']:.2f}\n")
        f.write(f"  CAGR: {summary['metrics']['cagr']:.2%}\n")
        f.write(f"  Calmar Ratio: {summary['metrics']['calmar_ratio']:.2f}\n")
        f.write(f"  Max Drawdown: {summary['metrics']['max_drawdown']:.2%}\n")
        f.write(f"  Max Drawdown Duration: {summary['metrics']['max_drawdown_duration']} bars\n")
        f.write(f"  Recovery Factor: {summary['metrics']['recovery_factor']:.2f}\n")
        f.write(f"  Profit Factor: {summary['metrics']['profit_factor']:.2f}\n")
        f.write(f"  Total Return: {summary['metrics']['total_return_pct']:.2f}%\n")
        f.write(f"\n")

        # Trade Statistics - ALL metrics
        f.write(f"Trade Statistics:\n")
        f.write(f"  Total Trades: {summary['metrics']['total_trades']}\n")
        f.write(f"  Buy Trades: {summary['metrics']['buy_trades']}\n")
        f.write(f"  Sell Trades: {summary['metrics']['sell_trades']}\n")
        f.write(f"  Winning Trades: {summary['metrics']['winning_trades']}\n")
        f.write(f"  Losing Trades: {summary['metrics']['losing_trades']}\n")
        f.write(f"  Win Rate: {summary['metrics']['win_rate']:.2f}%\n")
        f.write(f"  Expectancy: ${summary['metrics']['expectancy']:,.2f}\n")
        f.write(f"  Avg Trade PnL: ${summary['metrics']['average_trade_pnl']:,.2f}\n")
        f.write(f"  Avg Win: ${summary['metrics']['average_win']:,.2f}\n")
        f.write(f"  Avg Loss: ${summary['metrics']['average_loss']:,.2f}\n")
        f.write(f"  Best Trade: ${summary['metrics']['best_trade']:,.2f}\n")
        f.write(f"  Worst Trade: ${summary['metrics']['worst_trade']:,.2f}\n")
        f.write(f"\n")

        # PnL Summary
        f.write(f"PnL Summary:\n")
        f.write(f"  Initial Capital: ${summary['metrics']['initial_capital']:,.2f}\n")
        f.write(f"  Final Equity: ${summary['metrics']['final_equity']:,.2f}\n")
        f.write(f"  Total PnL: ${summary['metrics']['total_pnl']:,.2f}\n")
        f.write(f"\n")

        # Cost Configuration
        f.write(f"Cost Configuration:\n")
        f.write(f"  Taker Fee: {config.get('taker_fee', 0.00055) * 100:.3f}%\n")
        f.write(f"  Maker Fee: {config.get('maker_fee', 0.0002) * 100:.3f}%\n")
        f.write(f"  Slippage: {config.get('slippage_pct', 0.0005) * 100:.3f}%\n")

    print(f"\nDetailed reports saved to: {reports_dir}/")
    print(f"  - trades_{timestamp_str}.csv")
    print(f"  - equity_{timestamp_str}.csv")
    print(f"  - summary_{timestamp_str}.json")
    print(f"  - summary_{timestamp_str}.txt")


def run_backtest_mode(
    strategy_name: str,
    pairs: Optional[List[str]] = None,
    storage_path: str = "data/raw",
    save_reports: bool = False,
    days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """Run backtest mode with current parameters.

    Args:
        strategy_name: Name of the strategy
        pairs: Optional list of pairs to override config
        storage_path: Path to data storage
        save_reports: If True, save detailed reports to disk
        days: Optional filter to last N days
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        Exit code (0 = success)
    """
    _print_progress(1, 3, "Loading strategy config...")

    strategy_config = _load_strategy_config(strategy_name)
    global_config = _load_global_config()
    merged_config = _merge_configs(strategy_config, global_config)

    # Use provided pairs or from config
    if pairs is None:
        pairs = strategy_config.get("pairs", ["BTC/USDT:USDT"])
    timeframe = strategy_config.get("timeframe", "15m")

    _print_progress(2, 3, f"Fetching data for {pairs[0]} {timeframe}...")
    _check_and_fetch_data(pairs, timeframe, storage_path, global_config)

    _print_progress(3, 3, "Running backtest...")

    # Load data with optional date filtering
    data = _load_data(pairs, timeframe, storage_path, days, start_date, end_date)

    if data.is_empty():
        console.print("[red]Error: No data available after date filtering.[/red]")
        return 1

    if data.height < 100:
        console.print(
            f"[yellow]Warning: Only {data.height} bars after filtering. Results may not be meaningful.[/yellow]"
        )

    # Load strategy and generate signals
    strategy_class = _load_strategy_class(strategy_name)
    strategy = strategy_class(merged_config["params"])
    signals = strategy.generate_signals(data)

    # Run backtest
    from engine.backtester import run_backtest

    trade_log, equity_curve = run_backtest(signals, merged_config, show_progress=True)

    if save_reports:
        _save_backtest_report(
            strategy_name,
            pairs,
            timeframe,
            trade_log,
            equity_curve,
            signals,
            merged_config,
        )

    # Print rich backtest results
    _print_backtest_results_rich(
        strategy_name, pairs, timeframe, trade_log, equity_curve, signals, merged_config
    )

    return 0


def _print_backtest_results_rich(
    strategy_name: str,
    pairs: List[str],
    timeframe: str,
    trade_log: pl.DataFrame,
    equity_curve: pl.DataFrame,
    signals: pl.DataFrame,
    config: dict,
) -> None:
    """Print backtest results with Rich console formatting and detailed metrics.

    Displays:
    - Header panel with basic info
    - Performance metrics table (11 detailed metrics)
    - Trade statistics table (buy/sell breakdown, avg win/loss, expectancy)
    - Drawdown analysis
    - Cost configuration

    Args:
        strategy_name: Strategy name
        pairs: Trading pairs used
        timeframe: Timeframe of the data
        trade_log: DataFrame with trade details
        equity_curve: DataFrame with equity over time
        signals: DataFrame with signals
        config: Configuration used for backtest
    """
    from rich.panel import Panel
    from engine.metrics import (
        calculate_sharpe,
        calculate_max_drawdown,
        calculate_sortino,
        calculate_cagr,
        calculate_calmar_ratio,
        calculate_win_rate,
        calculate_profit_factor,
        calculate_expectancy,
        calculate_recovery_factor,
        calculate_average_trade_pnl,
    )

    console.print("\n[bold cyan]Backtest Results[/bold cyan]")
    console.print("=" * 60)

    # Header Panel
    data_start = signals["timestamp"].min()
    data_end = signals["timestamp"].max()
    duration_days = (data_end - data_start).total_seconds() / 86400
    years = duration_days / 365.0

    header_content = f"""\
[b]Strategy:[/b] {strategy_name}
[b]Pairs:[/b] {', '.join(pairs)}
[b]Timeframe:[/b] {timeframe}
[b]Data Range:[/b] {data_start.strftime('%Y-%m-%d %H:%M:%S')} to {data_end.strftime('%Y-%m-%d %H:%M:%S')}
[b]Duration:[/b] {duration_days:.1f} days ({years:.1f} years)"""

    console.print(
        Panel(
            header_content,
            title="[bold cyan]Backtest Summary[/bold cyan]",
            border_style="blue",
        )
    )

    # Calculate all metrics
    if trade_log.height > 0 and equity_curve.height > 0:
        sharpe = calculate_sharpe(equity_curve)
        sortino = calculate_sortino(equity_curve)
        cagr = calculate_cagr(equity_curve)
        max_dd, max_dd_duration = calculate_max_drawdown(equity_curve)
        calmar = calculate_calmar_ratio(equity_curve)
        win_rate = calculate_win_rate(trade_log)
        profit_factor = calculate_profit_factor(trade_log)
        expectancy = calculate_expectancy(trade_log)
        recovery_factor = calculate_recovery_factor(trade_log)
        avg_trade_pnl = calculate_average_trade_pnl(trade_log)

        total_pnl = trade_log["pnl"].sum()
        total_trades = trade_log.height
        winning_trades = (trade_log["pnl"] > 0).sum()
        losing_trades = (trade_log["pnl"] < 0).sum()

        # Count buy and sell trades (direction: 1 = long/buy, -1 = short/sell)
        buy_trades = (
            (trade_log["direction"] > 0).sum()
            if "direction" in trade_log.columns
            else 0
        )
        sell_trades = (
            (trade_log["direction"] < 0).sum()
            if "direction" in trade_log.columns
            else 0
        )

        # Average win/loss
        winning_trades_df = trade_log.filter(pl.col("pnl") > 0)
        losing_trades_df = trade_log.filter(pl.col("pnl") < 0)
        avg_win = (
            winning_trades_df["pnl"].mean() if winning_trades_df.height > 0 else 0.0
        )
        avg_loss = (
            losing_trades_df["pnl"].mean() if losing_trades_df.height > 0 else 0.0
        )
        best_trade = trade_log["pnl"].max()
        worst_trade = trade_log["pnl"].min()

        initial_capital = config.get("initial_capital", 10000)
        final_equity = equity_curve["equity"][-1]
        total_return = ((final_equity - initial_capital) / initial_capital) * 100

        # Performance Metrics Table
        console.print("\n[bold cyan]Performance Metrics[/bold cyan]")
        perf_table = Table(show_header=True, header_style="bold cyan")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="yellow")
        perf_table.add_column("Description", style="blue")

        perf_table.add_row(
            "Sharpe Ratio", f"{sharpe:.2f}", "Risk-adjusted return (volatility)"
        )
        perf_table.add_row(
            "Sortino Ratio", f"{sortino:.2f}", "Risk-adjusted return (downside only)"
        )
        perf_table.add_row("CAGR", f"{cagr:.2%}", "Compound annual growth rate")
        perf_table.add_row("Calmar Ratio", f"{calmar:.2f}", "CAGR / Max Drawdown")
        perf_table.add_row(
            "Max Drawdown",
            f"{max_dd:.2%}",
            f"Peak to trough decline ({max_dd_duration} bars)",
        )
        perf_table.add_row(
            "Recovery Factor", f"{recovery_factor:.2f}", "Net profit / Max DD"
        )
        perf_table.add_row(
            "Profit Factor", f"{profit_factor:.2f}", "Gross profit / Gross loss"
        )
        perf_table.add_row(
            "Win Rate", f"{win_rate:.1f}%", "Percentage of profitable trades"
        )
        perf_table.add_row(
            "Expectancy", f"${expectancy:.2f}", "Average profit/loss per trade"
        )
        perf_table.add_row(
            "Total Return", f"{total_return:.2f}%", "Total return on investment"
        )
        perf_table.add_row("Total PnL", f"${total_pnl:,.2f}", "Net profit/loss")

        console.print(perf_table)

        # Trade Statistics Table
        console.print("\n[bold cyan]Trade Statistics[/bold cyan]")
        trade_table = Table(show_header=True, header_style="bold cyan")
        trade_table.add_column("Metric", style="cyan")
        trade_table.add_column("Value", style="yellow")
        trade_table.add_column("Details", style="blue")

        trade_table.add_row("Total Trades", str(total_trades), f"All executed trades")
        trade_table.add_row("Buy Trades", str(buy_trades), "Long position entries")
        trade_table.add_row("Sell Trades", str(sell_trades), "Short position entries")
        trade_table.add_row(
            "Winning Trades", str(winning_trades), f"{win_rate:.1f}% of total"
        )
        trade_table.add_row(
            "Losing Trades", str(losing_trades), f"{100 - win_rate:.1f}% of total"
        )
        trade_table.add_row(
            "Avg Trade PnL", f"${avg_trade_pnl:.2f}", "Average profit/loss per trade"
        )
        trade_table.add_row("Avg Win", f"${avg_win:.2f}", "Average profitable trade")
        trade_table.add_row(
            "Avg Loss", f"${avg_loss:.2f}", "Average losing trade (negative)"
        )
        trade_table.add_row(
            "Best Trade", f"${best_trade:.2f}", "Highest single trade profit"
        )
        trade_table.add_row(
            "Worst Trade", f"${worst_trade:.2f}", "Lowest single trade loss"
        )

        console.print(trade_table)

        # Initial and Final Equity Panel
        net_profit_str = f"${total_pnl:,.2f}"
        equity_content = f"""\
[b]Initial Capital:[/b] [yellow]${initial_capital:,.2f}[/yellow]
[b]Final Equity:[/b] [yellow]${final_equity:,.2f}[/yellow]
[b]Net Profit:[/b] [cyan]{net_profit_str}[/cyan] [green]({total_return:+.2f}%)[/green]"""

        console.print(
            Panel(
                equity_content, title="[bold]Equity Summary[/bold]", border_style="cyan"
            )
        )

        # Cost Configuration
        console.print("\n[bold cyan]Cost Configuration[/bold cyan]")
        cost_table = Table(show_header=True, header_style="bold cyan")
        cost_table.add_column("Parameter", style="cyan")
        cost_table.add_column("Value", style="yellow")

        cost_table.add_row(
            "Taker Fee",
            f"{config.get('taker_fee', 0.00055) * 100:.3f}%",
            "Market order commission",
        )
        cost_table.add_row(
            "Maker Fee",
            f"{config.get('maker_fee', 0.0002) * 100:.3f}%",
            "Limit order commission",
        )
        cost_table.add_row(
            "Slippage",
            f"{config.get('slippage_pct', 0.0005) * 100:.3f}%",
            "Price execution assumption",
        )

        console.print(cost_table)

        # Strategy Parameters
        if config.get("params"):
            console.print("\n[bold cyan]Strategy Parameters[/bold cyan]")
            params_table = Table(show_header=True, header_style="bold cyan")
            params_table.add_column("Parameter", style="cyan")
            params_table.add_column("Value", style="yellow")

            for key, value in config.get("params", {}).items():
                params_table.add_row(key, str(value))

            console.print(params_table)

        # Explanations
        explanations = """\
[yellow]Metric Explanations:[/yellow]
• [cyan]Sharpe Ratio[/cyan]: Risk-adjusted return using volatility as risk measure (higher is better)
• [cyan]Sortino Ratio[/cyan]: Risk-adjusted return penalizing only downside volatility (higher is better)
• [cyan]CAGR[/cyan]: Compound annual growth rate - annualized return with compounding (higher is better)
• [cyan]Calmar Ratio[/cyan]: Return relative to maximum drawdown - reward-to-risk ratio (higher is better)
• [cyan]Max Drawdown[/cyan]: Largest peak-to-trough decline (lower is better)
• [cyan]Recovery Factor[/cyan]: Net profit divided by max drawdown (higher = better recovery)
• [cyan]Profit Factor[/cyan]: Gross wins divided by gross losses (above 1.0 = profitable)
• [cyan]Win Rate[/cyan]: Percentage of trades that were profitable (higher is better)
• [cyan]Expectancy[/cyan]: Average profit/loss per trade - positive means profitable per trade"""

        console.print(
            Panel(explanations, title="[bold]Glossary[/bold]", border_style="blue")
        )

    else:
        console.print("[yellow]No trades executed during backtest.[/yellow]")
        console.print(
            "The strategy did not generate any signals or signals did not result in trades."
        )

    console.print("\n" + "=" * 60)


def _print_header_panel(strategy_name, pairs, timeframe, data, wf_results):
    """Print header panel with basic info."""
    data_start = data["timestamp"].min()
    data_end = data["timestamp"].max()
    duration_days = (data_end - data_start).total_seconds() / 86400
    wfe = wf_results.get("wfe")

    content = f"""\
[b]Strategy:[/b] {strategy_name}
[b]Pairs:[/b] {', '.join(pairs)}
[b]Timeframe:[/b] {timeframe}
[b]Data Range:[/b] {data_start.strftime('%Y-%m-%d %H:%M:%S')} to {data_end.strftime('%Y-%m-%d %H:%M:%S')}
[b]Duration:[/b] {duration_days:.1f} days ({duration_days / 365:.1f} years)
[b]WFO Windows:[/b] {len(wf_results.get('windows', []))}
[b]Walk Forward Efficiency:[/b] {f'{wfe:.1f}%' if wfe is not None else 'N/A'}
"""
    console.print(
        Panel(
            content,
            title="[bold cyan]Validation Summary[/bold cyan]",
            border_style="blue",
        )
    )


def _print_wfo_panel(wf_results):
    """Print WFO robustness checks panel with additional metadata."""
    verdict = wf_results.get("verdict", {})
    checks = verdict.get("checks", {})
    robustness = wf_results.get("robustness", {})
    windows = wf_results.get("windows", [])

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_column("Threshold", style="blue")
    table.add_column("Status", style="bold")

    check_display_mapping = {
        "consistency": "Consistency (Profitable Windows)",
        "mean_performance": "Mean Performance",
        "performance_stability": "Performance Stability (CV)",
        "parameter_stability": "Parameter Stability",
        "flat_region": "Flat Region Check",
    }

    for check_key, check_data in checks.items():
        display_name = check_display_mapping.get(
            check_key, check_key.replace("_", " ").title()
        )
        metric_display = check_data.get("metric", "N/A")
        threshold = check_data.get("threshold", "N/A")
        passed = check_data.get("passed", False)

        status = "[green][OK] PASS[/green]" if passed else "[red][XX] FAIL[/red]"

        table.add_row(display_name, metric_display, threshold, status)

    # Get per-window distribution stats
    perf_dist = robustness.get("performance_distribution", {})
    window_sharpes = perf_dist.get("oos_sharpes", [])
    window_dds = perf_dist.get("max_drawdowns", [])

    flat_region_check = wf_results.get("parameter_selection", {}).get(
        "flat_region_check", {}
    )
    flat_data_source = flat_region_check.get("data_source", "unknown")

    if flat_data_source == "holdout":
        flat_source_note = " (verified on holdout data)"
    elif flat_data_source == "pseudo_holdout":
        flat_source_note = " (verified on pseudo-holdout)"
    else:
        flat_source_note = ""

    explanations = f"""\
[yellow]Explanations:[/yellow]
• [cyan]Profitable Windows[/cyan]: % of WFO windows with positive OOS Sharpe
• [cyan]Mean OOS Sharpe[/cyan]: Average out-of-sample Sharpe across windows
• [cyan]Sharpe Consistency[/cyan]: Coefficient of variation (lower = more consistent)
• [cyan]Param Stability[/cyan]: % of parameters stable across windows
• [cyan]Flat Region Check[/cyan]: Sensitivity to parameter perturbation{flat_source_note}"""

    wfo_verdict = verdict.get("overall", "UNKNOWN")
    border_color = (
        "green"
        if wfo_verdict == "PASS"
        else "yellow" if wfo_verdict == "CONDITIONAL" else "red"
    )

    passed_count = verdict.get("passed_checks", 0)
    total_count = verdict.get("total_checks", 0)

    # Additional metadata table
    meta_table = Table(show_header=True, header_style="bold cyan")
    meta_table.add_column("Info", style="cyan")
    meta_table.add_column("Value", style="yellow")

    total_windows = len(windows)
    meta_table.add_row("Total Windows", str(total_windows))

    # Per-window distribution
    if window_sharpes:
        min_sharpe = min(window_sharpes) if window_sharpes else 0
        max_sharpe = max(window_sharpes) if window_sharpes else 0
        mean_sharpe = sum(window_sharpes) / len(window_sharpes) if window_sharpes else 0
        meta_table.add_row("Min Window Sharpe", f"{min_sharpe:.2f}")
        meta_table.add_row("Max Window Sharpe", f"{max_sharpe:.2f}")
        meta_table.add_row("Mean Window Sharpe", f"{mean_sharpe:.2f}")

    if window_dds:
        min_dd = min(window_dds) if window_dds else 0
        max_dd = max(window_dds) if window_dds else 0
        meta_table.add_row("Min Window DD", f"{min_dd:.2%}")
        meta_table.add_row("Max Window DD", f"{max_dd:.2%}")

    panel_content = Table.grid()
    panel_content.add_column()
    panel_content.add_row(table)
    panel_content.add_row("")
    panel_content.add_row("[bold cyan]═══ WFO Metadata ═══[/bold cyan]")
    panel_content.add_row(meta_table)
    panel_content.add_row("")
    panel_content.add_row(f"Passed: {passed_count}/{total_count} checks")
    panel_content.add_row("")
    panel_content.add_row(explanations)

    console.print(
        Panel(
            panel_content,
            title="[bold]Walk-Forward Robustness Checks[/bold]",
            border_style=border_color,
        )
    )


def _print_multi_metric_panel(metric_results):
    """Print Multi-Metric Standard panel with pass/fail and informational metrics."""
    metric_verdict = metric_results.get("overall_verdict", "UNKNOWN")
    checks = metric_results.get("checks", [])
    informational = metric_results.get("informational", [])

    if not checks:
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_column("Threshold", style="blue")
    table.add_column("Status", style="bold")

    for check in checks:
        name = check["name"]
        value = check["value"]
        threshold = check["threshold"]
        passed = check["passed"]

        if isinstance(value, float):
            if "Ratio" in name or "Factor" in name:
                value_str = f"{value:.2f}"
            elif "Drawdown" in name:
                value_str = f"{value:.2f}%"
            elif "Rate" in name:
                value_str = f"{value:.1f}%"
            elif "Efficiency" in name or "Variation" in name:
                value_str = f"{value:.1f}%"
            else:
                value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        if isinstance(threshold, float):
            if "Ratio" in name or "Factor" in name:
                thresh_str = f">= {threshold}"
            elif "Drawdown" in name:
                thresh_str = f"<= {threshold:.1f}%"
            elif "Rate" in name:
                thresh_str = f">= {threshold:.0f}%"
            elif "Efficiency" in name:
                thresh_str = f">= {threshold:.0f}%"
            elif "Variation" in name:
                thresh_str = f"<= {threshold:.0f}%"
            else:
                thresh_str = f">= {threshold}"
        else:
            thresh_str = str(threshold)

        status = "[green][OK] PASS[/green]" if passed else "[red][XX] FAIL[/red]"
        table.add_row(name, value_str, thresh_str, status)

    # Add informational metrics section (CAGR, Win Rate - not used for verdict)
    info_table = Table(show_header=True, header_style="bold blue")
    info_table.add_column("Informational Metric", style="blue")
    info_table.add_column("Value", style="yellow")
    info_table.add_column("Note", style="dim")

    for info in informational:
        name = info["name"]
        value = info["value"]
        threshold = info["threshold"]

        if isinstance(value, float):
            if "Rate" in name:
                value_str = f"{value:.1f}%"
            elif "CAGR" in name:
                value_str = f"{value:.1f}%"
            else:
                value_str = f"{value:.2f}"
        else:
            value_str = str(value)

        if "CAGR" in name:
            note = "Position-sizing dependent"
        elif "Win Rate" in name:
            note = "Strategy-type dependent"
        else:
            note = "Informational only"

        info_table.add_row(name, value_str, note)

    explanations = """\
[yellow]Pass/Fail Metrics:[/yellow]
• [cyan]Sharpe[/cyan]: Risk-adjusted return (>= 1.0)
• [cyan]Sortino[/cyan]: Downside risk-adjusted (>= 1.0)
• [cyan]Calmar[/cyan]: CAGR/MaxDD (>= 1.0)
• [cyan]Max Drawdown[/cyan]: Peak-to-trough (<= 30%)
• [cyan]Recovery Factor[/cyan]: Net profit/MaxDD (>= 2.0)
• [cyan]Profit Factor[/cyan]: Gross profit/loss (>= 1.5)
• [cyan]Expectancy[/cyan]: Avg $/trade (> 0)
• [cyan]WFE[/cyan]: IS→OOS translation (>= 50%)
• [cyan]CV[/cyan]: Consistency (<= 20%)

[yellow]Informational Metrics (not used for verdict):[/yellow]
• [cyan]CAGR[/cyan]: Annualized return - position-sizing dependent
• [cyan]Win Rate[/cyan]: Trade win % - strategy-type dependent"""

    border_color = (
        "green"
        if metric_verdict == "PASS"
        else "yellow" if metric_verdict == "CONDITIONAL" else "red"
    )

    passed_count = metric_results.get("passed_count", 0)
    total_count = metric_results.get("total_count", 0)

    panel_content = Table.grid()
    panel_content.add_column()
    panel_content.add_row(table)
    if informational:
        panel_content.add_row("")
        panel_content.add_row("[bold blue]═══ Informational Metrics ═══[/bold blue]")
        panel_content.add_row(info_table)
    panel_content.add_row("")
    panel_content.add_row(f"Passed: {passed_count}/{total_count} metrics")
    panel_content.add_row("")
    panel_content.add_row(explanations)

    console.print(
        Panel(
            panel_content,
            title="[bold]Multi-Metric Standard[/bold]",
            border_style=border_color,
        )
    )


def _print_mc_panel(mc_results):
    """Print Monte Carlo stress test panel with OOS and Full Backtest results."""
    mc_grade = mc_results.get("robustness_grade", "UNKNOWN")
    n_sims = mc_results.get("n_simulations", 0)

    # Get all new metrics
    pct_5_return = mc_results.get("percentile_5_return")
    median_return = mc_results.get("median_return")
    pct_95_return = mc_results.get("percentile_95_return")
    pct_95_dd = mc_results.get("percentile_95_dd")
    pct_profitable_sims = mc_results.get("pct_profitable_sims")
    risk_of_ruin = mc_results.get("risk_of_ruin")
    dd_increase = mc_results.get("dd_increase_pct")
    original_dd = mc_results.get("original_max_dd")

    initial_capital = 10000.0

    # Table 1: OOS Trades Monte Carlo Results
    table1 = Table(show_header=True, header_style="bold cyan")
    table1.add_column("Test", style="cyan")
    table1.add_column("Metric", style="cyan")
    table1.add_column("Value", style="yellow")
    table1.add_column("Threshold", style="blue")
    table1.add_column("Status", style="bold")

    if mc_grade == "PASS":
        status = "[green][OK] PASS[/green]"
    else:
        status = "[red][XX] FAIL[/red]"

    table1.add_row("OOS Trades", "Robustness Grade", mc_grade, "PASS", status)

    # PRIMARY CRITERIA: Profitability
    # 1. 5th Percentile Return > 0 (PRIMARY)
    if pct_5_return is not None:
        p5_passed = pct_5_return > initial_capital
        p5_status = "[green][OK] PASS[/green]" if p5_passed else "[red][XX] FAIL[/red]"
        p5_pct = ((pct_5_return - initial_capital) / initial_capital) * 100
        table1.add_row(
            "OOS Trades",
            "5th Pctl Return > 0%",
            f"{pct_5_return:,.0f} ({p5_pct:+.1f}%)",
            "> $10k",
            p5_status,
        )

    # 2. % Profitable Simulations >= 80%
    if pct_profitable_sims is not None:
        profitable_passed = pct_profitable_sims >= 80.0
        profitable_status = (
            "[green][OK] PASS[/green]" if profitable_passed else "[red][XX] FAIL[/red]"
        )
        table1.add_row(
            "OOS Trades",
            "Profitable Sims >= 80%",
            f"{pct_profitable_sims:.1f}%",
            ">= 80%",
            profitable_status,
        )

    # 3. Risk of Ruin = 0%
    if risk_of_ruin is not None:
        ruin_passed = risk_of_ruin == 0.0
        ruin_status = (
            "[green][OK] PASS[/green]" if ruin_passed else "[red][XX] FAIL[/red]"
        )
        table1.add_row(
            "OOS Trades",
            "Risk of Ruin = 0%",
            f"{risk_of_ruin:.1f}%",
            "= 0%",
            ruin_status,
        )

    # SECONDARY CRITERIA: Risk Management
    # 4. Worst Case DD <= 25%
    if pct_95_dd is not None:
        dd_passed = pct_95_dd <= 0.25
        dd_status = "[green][OK] PASS[/green]" if dd_passed else "[red][XX] FAIL[/red]"
        table1.add_row(
            "OOS Trades",
            "Worst Case DD <= 25%",
            f"{pct_95_dd:.2%}",
            "<= 25%",
            dd_status,
        )

    # Additional info
    if dd_increase is not None:
        table1.add_row(
            "OOS Trades",
            "DD Increase (Info)",
            f"{dd_increase:+.1f}%",
            "observed",
            "[blue]INFO[/blue]",
        )
    if original_dd is not None:
        table1.add_row(
            "OOS Trades",
            "Original Max DD",
            f"{original_dd:.2%}",
            "baseline",
            "[blue]INFO[/blue]",
        )

    table1.add_row(
        "OOS Trades", "Simulations", str(n_sims), "10,000", "[blue]INFO[/blue]"
    )

    # Additional return metrics
    if median_return is not None:
        median_pct = ((median_return - initial_capital) / initial_capital) * 100
        table1.add_row(
            "OOS Trades",
            "Median Return",
            f"${median_return:,.0f} ({median_pct:+.1f}%)",
            "typical",
            "[blue]INFO[/blue]",
        )
    if pct_95_return is not None:
        p95_pct = ((pct_95_return - initial_capital) / initial_capital) * 100
        table1.add_row(
            "OOS Trades",
            "95th Pctl Return",
            f"${pct_95_return:,.0f} ({p95_pct:+.1f}%)",
            "best-case",
            "[blue]INFO[/blue]",
        )

    # Block bootstrap info
    use_block = mc_results.get("use_block_bootstrap")
    block_size = mc_results.get("block_size")
    if use_block is not None:
        table1.add_row(
            "Method",
            "Block Bootstrap",
            "Enabled" if use_block else "Disabled",
            "optional",
            "[blue]INFO[/blue]",
        )
    if block_size is not None:
        table1.add_row(
            "Method", "Block Size", str(block_size), "N/A", "[blue]INFO[/blue]"
        )

    # Table 2: Full Backtest Monte Carlo Results
    full_backtest = mc_results.get("full_backtest")

    if full_backtest:
        table2 = Table(show_header=True, header_style="bold cyan")
        table2.add_column("Test", style="cyan")
        table2.add_column("Metric", style="cyan")
        table2.add_column("Value", style="yellow")
        table2.add_column("Threshold", style="blue")
        table2.add_column("Status", style="bold")

        fb_grade = full_backtest.get("robustness_grade", "UNKNOWN")
        fb_trade_count = full_backtest.get("trade_count", 0)
        fb_n_sims = full_backtest.get("n_simulations", 0)

        fb_pct_5_return = full_backtest.get("percentile_5_return")
        fb_median_return = full_backtest.get("median_return")
        fb_pct_95_return = full_backtest.get("percentile_95_return")
        fb_pct_95_dd = full_backtest.get("percentile_95_dd")
        fb_pct_profitable = full_backtest.get("pct_profitable_sims")
        fb_risk_of_ruin = full_backtest.get("risk_of_ruin")
        fb_dd_increase = full_backtest.get("dd_increase_pct")
        fb_original_dd = full_backtest.get("original_max_dd")

        if fb_grade == "PASS":
            fb_status = "[green][OK] PASS[/green]"
        else:
            fb_status = "[red][XX] FAIL[/red]"

        table2.add_row("Full Backtest", "Robustness Grade", fb_grade, "PASS", fb_status)

        # PRIMARY: 5th percentile return
        if fb_pct_5_return is not None:
            fb_p5_passed = fb_pct_5_return > initial_capital
            fb_p5_status = (
                "[green][OK] PASS[/green]" if fb_p5_passed else "[red][XX] FAIL[/red]"
            )
            fb_p5_pct = ((fb_pct_5_return - initial_capital) / initial_capital) * 100
            table2.add_row(
                "Full Backtest",
                "5th Pctl Return > 0%",
                f"{fb_pct_5_return:,.0f} ({fb_p5_pct:+.1f}%)",
                "> $10k",
                fb_p5_status,
            )

        # % profitable
        if fb_pct_profitable is not None:
            fb_prof_passed = fb_pct_profitable >= 80.0
            fb_prof_status = (
                "[green][OK] PASS[/green]" if fb_prof_passed else "[red][XX] FAIL[/red]"
            )
            table2.add_row(
                "Full Backtest",
                "Profitable Sims >= 80%",
                f"{fb_pct_profitable:.1f}%",
                ">= 80%",
                fb_prof_status,
            )

        # Risk of ruin
        if fb_risk_of_ruin is not None:
            fb_ruin_passed = fb_risk_of_ruin == 0.0
            fb_ruin_status = (
                "[green][OK] PASS[/green]" if fb_ruin_passed else "[red][XX] FAIL[/red]"
            )
            table2.add_row(
                "Full Backtest",
                "Risk of Ruin = 0%",
                f"{fb_risk_of_ruin:.1f}%",
                "= 0%",
                fb_ruin_status,
            )

        # Worst case DD
        if fb_pct_95_dd is not None:
            fb_dd_passed = fb_pct_95_dd <= 0.25
            fb_dd_status = (
                "[green][OK] PASS[/green]" if fb_dd_passed else "[red][XX] FAIL[/red]"
            )
            table2.add_row(
                "Full Backtest",
                "Worst Case DD <= 25%",
                f"{fb_pct_95_dd:.2%}",
                "<= 25%",
                fb_dd_status,
            )

        # Additional info
        if fb_dd_increase is not None:
            table2.add_row(
                "Full Backtest",
                "DD Increase (Info)",
                f"{fb_dd_increase:+.1f}%",
                "observed",
                "[blue]INFO[/blue]",
            )
        if fb_original_dd is not None:
            table2.add_row(
                "Full Backtest",
                "Original Max DD",
                f"{fb_original_dd:.2%}",
                "baseline",
                "[blue]INFO[/blue]",
            )

        table2.add_row(
            "Full Backtest",
            "Trade Count",
            str(fb_trade_count),
            "observed",
            "[blue]INFO[/blue]",
        )
        table2.add_row(
            "Full Backtest",
            "Simulations",
            str(fb_n_sims),
            "10,000",
            "[blue]INFO[/blue]",
        )

        # Additional return metrics
        if fb_median_return is not None:
            fb_median_pct = (
                (fb_median_return - initial_capital) / initial_capital
            ) * 100
            table2.add_row(
                "Full Backtest",
                "Median Return",
                f"${fb_median_return:,.0f} ({fb_median_pct:+.1f}%)",
                "typical",
                "[blue]INFO[/blue]",
            )
        if fb_pct_95_return is not None:
            fb_p95_pct = ((fb_pct_95_return - initial_capital) / initial_capital) * 100
            table2.add_row(
                "Full Backtest",
                "95th Pctl Return",
                f"${fb_pct_95_return:,.0f} ({fb_p95_pct:+.1f}%)",
                "best-case",
                "[blue]INFO[/blue]",
            )

    # Determine overall grade
    overall_grade = mc_grade
    if full_backtest:
        fb_grade = full_backtest.get("robustness_grade", "UNKNOWN")
        if fb_grade == "FAIL" or mc_grade == "FAIL":
            overall_grade = "FAIL"

    border_color = "green" if overall_grade == "PASS" else "red"

    explanations = """\
[yellow]Explanations:[/yellow]
• [cyan]5th Pctl Return[/cyan]: Worst-case return at 95% confidence - PRIMARY criterion (> $10k = PASS)
• [cyan]Profitable Sims[/cyan]: % of simulations ending in profit - must be >= 80%
• [cyan]Risk of Ruin[/cyan]: % hitting zero - must be 0%
• [cyan]Worst Case DD[/cyan]: 95th percentile drawdown - must be <= 25%

[yellow]GRADE LOGIC:[/yellow]
Strategy PASSES only if ALL 4 criteria pass:
1. 5th percentile return > initial capital (PROFITABLE)
2. At least 80% of simulations are profitable
3. Risk of ruin = 0% (no simulation hits zero)
4. 95th percentile DD <= 25% (acceptable risk)

[yellow]Simulates:[/yellow]
Trade reshuffling | 10% trade skip | 1 pip slippage | 1-bar entry delay"""

    explanations = """\
[yellow]Explanations:[/yellow]
• [cyan]OOS Trades[/cyan]: MC on combined out-of-sample trades from all WFO windows
• [cyan]Full Backtest[/cyan]: MC on full backtest with selected final parameters
• [cyan]Robustness Grade[/cyan]: Overall grade from 10,000 Monte Carlo simulations
• [cyan]Worst Case DD[/cyan]: 95th percentile max drawdown - primary pass/fail criteria (<= 25%)
• [cyan]DD Increase (Info)[/cyan]: % increase in DD vs original - shown as additional info only
• [cyan]Percentile Returns[/cyan]: 5th (worst), 50th (typical), 95th (best) simulated returns
• [cyan]Percentile DD[/cyan]: 95th percentile max drawdown - measures tail risk

Simulates random trade sequences with:
- Trade reshuffling (random order) | 10% trade skipping (execution failures)
- 1 pip slippage per trade | 1-bar entry delay"""

    panel_content = Table.grid()
    panel_content.add_column()
    panel_content.add_row("[bold cyan]═══ OOS Trades Monte Carlo ═══[/bold cyan]")
    panel_content.add_row(table1)

    if full_backtest:
        panel_content.add_row("")
        panel_content.add_row(
            "[bold cyan]═══ Full Backtest Monte Carlo ═══[/bold cyan]"
        )
        panel_content.add_row(table2)

    panel_content.add_row("")
    panel_content.add_row(explanations)

    console.print(
        Panel(
            panel_content,
            title="[bold]Monte Carlo Stress Test[/bold]",
            border_style=border_color,
        )
    )


def _print_holdout_panel(holdout_results):
    """Print holdout validation panel with additional metadata."""
    if not holdout_results:
        return

    holdout_verdict = holdout_results.get("verdict", "UNKNOWN")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_column("Threshold", style="blue")
    table.add_column("Status", style="bold")

    sharpe = holdout_results.get("sharpe", 0)
    sharpe_passed = sharpe >= 1.0
    sharpe_status = (
        "[green][OK] PASS[/green]" if sharpe_passed else "[red][XX] FAIL[/red]"
    )
    table.add_row("Holdout Sharpe", f"{sharpe:.2f}", ">= 1.0", sharpe_status)

    max_dd = holdout_results.get("max_drawdown", 0)
    dd_passed = max_dd < 0.30
    dd_status = "[green][OK] PASS[/green]" if dd_passed else "[red][XX] FAIL[/red]"
    table.add_row("Holdout Max DD", f"{max_dd:.2%}", "< 30%", dd_status)

    trades = holdout_results.get("trade_count", 0)
    table.add_row("Holdout Trades", str(trades), "N/A", "[blue]INFO[/blue]")

    # Additional metrics
    profit_factor = holdout_results.get("profit_factor", 0)
    if profit_factor:
        table.add_row(
            "Holdout Profit Factor",
            f"{profit_factor:.2f}",
            "> 1.0",
            "[blue]INFO[/blue]",
        )

    recovery_factor = holdout_results.get("recovery_factor", 0)
    if recovery_factor:
        table.add_row(
            "Holdout Recovery Factor",
            f"{recovery_factor:.2f}",
            "> 1.0",
            "[blue]INFO[/blue]",
        )

    total_pnl = holdout_results.get("total_pnl", 0)
    initial_capital_holdout = 10000.0
    if total_pnl:
        if isinstance(total_pnl, (int, float)):
            pnl_pct = (total_pnl / initial_capital_holdout) * 100
            pnl_str = f"${total_pnl:,.2f} ({pnl_pct:+.1f}%)"
        else:
            pnl_str = str(total_pnl)
        table.add_row("Holdout Total PnL", pnl_str, "positive", "[blue]INFO[/blue]")

    final_equity = holdout_results.get("final_equity", 0)
    if final_equity:
        final_pct = (
            (final_equity - initial_capital_holdout) / initial_capital_holdout
        ) * 100
        table.add_row(
            "Holdout Final Equity",
            f"${final_equity:,.2f} ({final_pct:+.1f}%)",
            "observed",
            "[blue]INFO[/blue]",
        )

    # Status from holdout results
    status = holdout_results.get("status", "UNKNOWN")
    if status and status != "UNKNOWN":
        table.add_row("Holdout Status", status, "COMPLETE", "[blue]INFO[/blue]")

    explanations = """\
[yellow]Explanation:[/yellow]
• Tests selected parameters on [cyan]true holdout data[/cyan] (never used during WFO)
• Validates that final parameters work on completely unseen, out-of-sample data
• Acts as a final sanity check before considering a strategy viable
• Profit Factor: Gross profit / gross loss (higher is better)
• Recovery Factor: Net profit / max drawdown (higher is better)"""

    border_color = (
        "green"
        if holdout_verdict == "PASS"
        else "yellow" if holdout_verdict == "MARGINAL" else "red"
    )

    panel_content = Table.grid()
    panel_content.add_column()
    panel_content.add_row(table)
    panel_content.add_row("")
    panel_content.add_row(explanations)

    console.print(
        Panel(
            panel_content,
            title="[bold]Holdout Validation[/bold]",
            border_style=border_color,
        )
    )


def _print_final_strategy_panel(final_strategy_results, final_metric_results):
    """Print final strategy evaluation panel with optimal position sizing.

    This shows the actual compounding effects from using:
    - Final selected parameters from WFO
    - Optimal Kelly position sizing

    This represents what you can expect when trading live.
    """
    if not final_strategy_results or not final_metric_results:
        return

    optimal_sizing = final_strategy_results.get("optimal_sizing", {})
    backtest_metrics = final_strategy_results.get("backtest_metrics", {})
    position_size_pct = final_strategy_results.get("position_size_pct", 0)

    metric_verdict = final_metric_results.get("overall_verdict", "UNKNOWN")
    checks = final_metric_results.get("checks", [])
    informational = final_metric_results.get("informational", [])

    if not checks:
        return

    sizing_info_table = Table(show_header=True, header_style="bold cyan")
    sizing_info_table.add_column("Parameter", style="cyan")
    sizing_info_table.add_column("Value", style="yellow")
    sizing_info_table.add_column("Description", style="blue")

    sizing_info_table.add_row(
        "Position Sizing Mode", "percent_equity", "Fixed percentage of equity per trade"
    )
    sizing_info_table.add_row(
        "Kelly Fraction Used",
        f"{position_size_pct:.2f}%",
        "Quarter Kelly from OOS trades",
    )

    raw_kelly = optimal_sizing.get("raw_kelly", 0)
    sizing_info_table.add_row(
        "Raw Kelly", f"{raw_kelly:.4f}", "Full Kelly before safety capping"
    )

    win_rate = optimal_sizing.get("win_rate", 0)
    sizing_info_table.add_row(
        "OOS Win Rate", f"{win_rate:.2%}", "From combined OOS trades"
    )

    win_loss_ratio = optimal_sizing.get("win_loss_ratio", 0)
    sizing_info_table.add_row(
        "OOS Win/Loss Ratio", f"{win_loss_ratio:.2f}", "Average win / Average loss"
    )

    metric_table = Table(show_header=True, header_style="bold cyan")
    metric_table.add_column("Metric", style="cyan")
    metric_table.add_column("Value", style="yellow")
    metric_table.add_column("Threshold", style="blue")
    metric_table.add_column("Status", style="bold")

    for check in checks:
        name = check["name"]
        value = check["value"]
        threshold = check["threshold"]
        passed = check["passed"]

        if isinstance(value, float):
            if "Ratio" in name or "Factor" in name:
                value_str = f"{value:.2f}"
            elif "Drawdown" in name:
                value_str = f"{value:.2f}%"
            elif "Rate" in name:
                value_str = f"{value:.1f}%"
            elif "CAGR" in name:
                value_str = f"{value:.2f}%"
            else:
                value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        if isinstance(threshold, float):
            if "Ratio" in name or "Factor" in name:
                thresh_str = f">= {threshold}"
            elif "Drawdown" in name:
                thresh_str = f"<= {threshold:.1f}%"
            elif "Rate" in name:
                thresh_str = f">= {threshold:.0f}%"
            elif "CAGR" in name:
                thresh_str = f">= {threshold:.0f}%"
            else:
                thresh_str = f">= {threshold}"
        else:
            thresh_str = str(threshold)

        status = "[green][OK] PASS[/green]" if passed else "[red][XX] FAIL[/red]"
        metric_table.add_row(name, value_str, thresh_str, status)

    initial_capital = 10000.0
    final_equity = backtest_metrics.get("final_equity", initial_capital)
    total_return = (
        ((final_equity - initial_capital) / initial_capital) * 100
        if initial_capital > 0
        else 0
    )

    equity_info_table = Table(show_header=True, header_style="bold cyan")
    equity_info_table.add_column("Metric", style="cyan")
    equity_info_table.add_column("Value", style="yellow")

    equity_info_table.add_row("Initial Capital", f"${initial_capital:,.2f}")
    equity_info_table.add_row("Final Equity", f"${final_equity:,.2f}")
    equity_info_table.add_row("Total Return", f"{total_return:+.2f}%")

    total_trades = backtest_metrics.get("total_trades", 0)
    equity_info_table.add_row("Total Trades", str(total_trades))

    win_rate = backtest_metrics.get("win_rate", 0)
    equity_info_table.add_row("Win Rate", f"{win_rate:.1f}%")

    total_pnl = backtest_metrics.get("total_pnl", 0)
    equity_info_table.add_row("Total PnL", f"${total_pnl:,.2f}")

    border_color = (
        "green"
        if metric_verdict == "PASS"
        else "yellow" if metric_verdict == "CONDITIONAL" else "red"
    )

    passed_count = final_metric_results.get("passed_count", 0)
    total_count = final_metric_results.get("total_count", 0)

    info_table = None
    if informational:
        info_table = Table(show_header=True, header_style="bold blue")
        info_table.add_column("Informational Metric", style="blue")
        info_table.add_column("Value", style="yellow")
        info_table.add_column("Note", style="dim")

        for info in informational:
            name = info["name"]
            value = info["value"]
            threshold = info["threshold"]

            if isinstance(value, float):
                if "Rate" in name:
                    value_str = f"{value:.1f}%"
                elif "CAGR" in name:
                    value_str = f"{value:.1f}%"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)

            if "Win Rate" in name:
                note = "Strategy-type dependent"
            else:
                note = "Informational only"

            info_table.add_row(name, value_str, note)

    explanations = """\
[yellow]Pass/Fail Metrics:[/yellow]
• [cyan]Sharpe[/cyan]: Risk-adjusted return (>= 1.0)
• [cyan]Sortino[/cyan]: Downside risk-adjusted (>= 1.0)
• [cyan]Calmar[/cyan]: CAGR/MaxDD (>= 1.0)
• [cyan]CAGR[/cyan]: Annualized return (>= 10%)
• [cyan]Max Drawdown[/cyan]: Peak-to-trough (<= 30%)
• [cyan]Recovery Factor[/cyan]: Net profit/MaxDD (>= 2.0)
• [cyan]Profit Factor[/cyan]: Gross profit/loss (>= 1.5)
• [cyan]Expectancy[/cyan]: Avg $/trade (> 0)

[yellow]Informational Metrics:[/yellow]
• [cyan]Win Rate[/cyan]: Trade win % - strategy-type dependent"""

    panel_content = Table.grid()
    panel_content.add_column()
    panel_content.add_row(
        "[bold cyan]═══ Position Sizing Configuration ═══[/bold cyan]"
    )
    panel_content.add_row(sizing_info_table)
    panel_content.add_row("")
    panel_content.add_row(
        "[bold cyan]═══ Final Strategy Metrics (with Optimal Sizing) ═══[/bold cyan]"
    )
    panel_content.add_row(metric_table)
    if info_table:
        panel_content.add_row("")
        panel_content.add_row("[bold blue]═══ Informational Metrics ═══[/bold blue]")
        panel_content.add_row(info_table)
    panel_content.add_row("")
    panel_content.add_row("[bold cyan]═══ Equity Summary ═══[/bold cyan]")
    panel_content.add_row(equity_info_table)
    panel_content.add_row("")
    panel_content.add_row(f"Passed: {passed_count}/{total_count} metrics")
    panel_content.add_row("")
    panel_content.add_row(explanations)

    console.print(
        Panel(
            panel_content,
            title="[bold]Final Strategy Multi-Metric Standard[/bold]",
            border_style=border_color,
        )
    )


def _print_final_verdict_panel(wf_results, metric_results, mc_results, holdout_results):
    """Print unified final verdict panel."""
    wfo_verdict = wf_results.get("verdict", {}).get("overall", "UNKNOWN")
    metric_verdict = metric_results.get("overall_verdict", "UNKNOWN")
    mc_grade = mc_results.get("robustness_grade", "UNKNOWN")
    holdout_verdict = (
        holdout_results.get("verdict", "UNKNOWN") if holdout_results else "UNKNOWN"
    )

    all_pass = (
        wfo_verdict == "PASS"
        and metric_verdict == "PASS"
        and mc_grade == "PASS"
        and holdout_verdict in ("PASS", "MARGINAL")
    )

    any_fail = (
        wfo_verdict == "FAIL"
        or metric_verdict == "FAIL"
        or mc_grade == "FAIL"
        or holdout_verdict == "FAIL"
    )

    if all_pass:
        final_verdict = "PASS"
        final_color = "green"
        final_emoji = "[OK]"
        summary = (
            "Strategy demonstrates robust performance across all validation dimensions."
        )
    elif any_fail:
        final_verdict = "FAIL"
        final_color = "red"
        final_emoji = "[XX]"
        summary = (
            "Strategy fails validation criteria. Review metrics and consider redesign."
        )
    else:
        final_verdict = "CONDITIONAL"
        final_color = "yellow"
        final_emoji = "[!]"
        summary = (
            "Strategy shows promise but has areas needing attention before deployment."
        )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Component", style="cyan")
    table.add_column("Verdict", style="bold")
    table.add_column("Checks", style="blue")
    table.add_column("Explanation", style="white")  # NEW

    # Explanations mapping
    component_explanations = {
        "WFO Robustness": "Walk-forward optimization consistency checks",
        "Multi-Metric Standard": "11-metric institutional validation",
        "Monte Carlo": "10,000 simulation stress test",
        "Holdout": "True out-of-sample verification",
    }

    wfo_passed = wf_results.get("verdict", {}).get("passed_checks", 0)
    wfo_total = wf_results.get("verdict", {}).get("total_checks", 0)
    wfo_status = (
        f"[green]{wfo_verdict}[/green]"
        if wfo_verdict == "PASS"
        else (
            f"[red]{wfo_verdict}[/red]"
            if wfo_verdict == "FAIL"
            else f"[yellow]{wfo_verdict}[/yellow]"
        )
    )
    explanation = component_explanations.get(
        "WFO Robustness", "Walk-forward optimization consistency"
    )
    table.add_row(
        "WFO Robustness", wfo_status, f"{wfo_passed}/{wfo_total}", explanation
    )

    metric_passed = metric_results.get("passed_count", 0)
    metric_total = metric_results.get("total_count", 0)
    metric_status = (
        f"[green]{metric_verdict}[/green]"
        if metric_verdict == "PASS"
        else (
            f"[red]{metric_verdict}[/red]"
            if metric_verdict == "FAIL"
            else f"[yellow]{metric_verdict}[/yellow]"
        )
    )
    explanation = component_explanations.get(
        "Multi-Metric Standard", "11-metric institutional validation"
    )
    table.add_row(
        "Multi-Metric Standard",
        metric_status,
        f"{metric_passed}/{metric_total}",
        explanation,
    )

    mc_status = (
        f"[green]{mc_grade}[/green]"
        if mc_grade == "PASS"
        else (
            f"[red]{mc_grade}[/red]"
            if mc_grade == "FAIL"
            else f"[yellow]{mc_grade}[/yellow]"
        )
    )
    explanation = component_explanations.get(
        "Monte Carlo", "10,000 simulation stress test"
    )
    table.add_row("Monte Carlo", mc_status, "Grade", explanation)

    holdout_status = (
        f"[green]{holdout_verdict}[/green]"
        if holdout_verdict == "PASS"
        else (
            f"[red]{holdout_verdict}[/red]"
            if holdout_verdict == "FAIL"
            else f"[yellow]{holdout_verdict}[/yellow]"
        )
    )
    explanation = component_explanations.get(
        "Holdout", "True out-of-sample verification"
    )
    table.add_row("Holdout", holdout_status, "3 metrics", explanation)

    verdict_text = f"[b {final_color}]{final_emoji} FINAL VERDICT: {final_verdict}[/b {final_color}]"
    criteria_text = """\
[yellow]Pass Criteria:[/yellow]
 • All components must PASS or be MARGINAL
 • No component may FAIL
 • Strategy must demonstrate consistency across time periods"""
    panel_content = Table.grid()
    panel_content.add_column()
    panel_content.add_row(table)
    panel_content.add_row("")
    panel_content.add_row(verdict_text)
    panel_content.add_row("")
    panel_content.add_row(summary)
    panel_content.add_row("")
    panel_content.add_row(criteria_text)

    console.print(
        Panel(
            panel_content,
            title=f"[bold {final_color}]Final Verdict[/bold {final_color}]",
            border_style=final_color,
            padding=(1, 2),
        )
    )


def _print_param_selection_panel(wf_results):
    """Print parameter selection panel (moved from validator.py)."""
    selection = wf_results.get("parameter_selection", {})
    method = selection.get("recommended_method", "unknown")
    final_params = selection.get("final_params", {})

    params_text = "\n".join(
        [
            f"  • [cyan]{param}:[/cyan] [yellow]{value}[/yellow]"
            for param, value in final_params.items()
        ]
    )

    methods_details = []
    for method_name, method_result in selection.get("methods", {}).items():
        if method_name == "island_volume":
            islands = method_result.get("islands", [])
            if islands:
                volume = method_result.get("largest_island", {}).get("volume", 0)
                methods_details.append(
                    f"• Island Volume: {len(islands)} islands, largest: {volume:.1f}"
                )
        elif method_name == "frequency":
            conf = method_result.get("confidence_pct", 0)
            methods_details.append(f"• Frequency: {conf:.1f}% confidence")
        elif method_name == "kmeans":
            perf = method_result.get("cluster_performance", 0)
            methods_details.append(f"• K-Means: Best cluster Sharpe {perf:.2f}")

    methods_text = (
        "\n".join(methods_details)
        if methods_details
        else "Auto-selected based on robustness criteria"
    )

    content = f"""\n\
[b]Selection Method:[/b] [green]{method}[/green]
[b]Final Parameters:[/b]
{params_text}
[b]Method Details:[/b]
[blue]{methods_text}[/blue]
"""

    console.print(
        Panel(content, title="[bold]Parameter Selection[/bold]", border_style="blue")
    )


def _print_diagnostic_panel(diagnostic_results: Optional[Dict]) -> None:
    """Print diagnostic checks panel with inline explanations.

    Shows statistical diagnostic results with:
    - Check name, value, threshold, status
    - Inline explanation for each metric
    - Color-coded status indicators
    - Detailed breakdown for meta_overfitting (p_value, effect_size, etc.)

    Args:
        diagnostic_results: Optional dict with diagnostic test results
    """
    if not diagnostic_results:
        return

    summary = diagnostic_results.get("summary", {})
    checks = diagnostic_results.get("checks", {})

    if not checks:
        return

    # Create table with 5 columns: Diagnostic, Value, Threshold, Status, Explanation
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Diagnostic", style="cyan", width=28)
    table.add_column("Value", style="yellow", width=14)
    table.add_column("Threshold", style="blue", width=14)
    table.add_column("Status", style="bold", width=10)
    table.add_column("Explanation", style="white")

    # Diagnostic check explanations mapping
    diagnostic_explanations = {
        "shapiro_normality": "Tests if returns follow normal distribution (p > 0.05)",
        "adf_stationarity": "Tests if time series is stationary (p < 0.05)",
        "ljung_box": "Tests for autocorrelation in returns (p > 0.05)",
        "arch_lm": "Tests for heteroscedasticity (p < 0.05)",
        "jarque_bera": "Tests for normality using skew/kurtosis (p > 0.05)",
        "augmented_dickey": "Alternative stationarity test (p < 0.05)",
        "meta_overfitting": "Tests if parameter selection is overfitted to in-sample data",
        "statistical_power": "Tests if sample size provides sufficient statistical power",
        "window_independence": "Tests if WFO windows are statistically independent",
        "regime_heterogeneity": "Tests if performance varies across market regimes",
        "parameter_stability": "Tests if selected parameters remain stable across windows",
    }

    # Track if we've added expanded rows for meta_overfitting
    meta_overfitting_expanded = False

    # Process meta_overfitting FIRST to ensure proper ordering
    if "meta_overfitting" in checks:
        check_result = checks["meta_overfitting"]
        if check_result.get("test_performed", False):
            severity = check_result.get("severity", "INFO")
            passed = check_result.get("passed", False)

            if severity == "ERROR":
                status = "[red][XX] FAIL[/red]"
            elif severity == "WARNING":
                status = "[yellow][!] WARN[/yellow]"
            elif passed:
                status = "[green][OK] PASS[/green]"
            else:
                status = "[blue][ℹ] INFO[/blue]"

            recommendation = check_result.get(
                "recommendation", "Statistical validation test"
            )
            table.add_row(
                "[b]Meta Overfitting[/b]",
                "See details below",
                "combined",
                status,
                recommendation,
            )

            # Row 1: p_value
            p_value = check_result.get("p_value", "N/A")
            p_threshold = f"<= {check_result.get('alpha', 0.05)}"
            p_passed = (
                p_value <= check_result.get("alpha", 0.05)
                if isinstance(p_value, (int, float))
                else False
            )
            if isinstance(p_value, float):
                p_value_str = f"{p_value:.3f}"
            else:
                p_value_str = str(p_value)
            p_status = (
                "[green][OK] PASS[/green]" if p_passed else "[yellow][!] WARN[/yellow]"
            )

            table.add_row(
                "  └─ p-value (Significance)",
                p_value_str,
                p_threshold,
                p_status,
                "Proportion of random params >= actual",
            )

            # Row 2: effect_size (Cohen's d)
            effect_size = check_result.get("effect_size", "N/A")
            min_effect = check_result.get("min_effect_size", 0.3)
            effect_passed = (
                effect_size >= min_effect
                if isinstance(effect_size, (int, float))
                else False
            )
            if isinstance(effect_size, float):
                effect_str = f"{effect_size:.4f}"
            else:
                effect_str = str(effect_size)

            effect_status = (
                "[green][OK] PASS[/green]"
                if effect_passed
                else "[yellow][!] WARN[/yellow]"
            )

            table.add_row(
                "  └─ Effect Size (Cohen's d)",
                effect_str,
                f">= {min_effect}",
                effect_status,
                "Magnitude of actual vs random performance gap",
            )

            # Row 3: actual_mean (actual OOS Sharpe)
            actual_mean = check_result.get("actual_mean", "N/A")
            if isinstance(actual_mean, float):
                actual_str = f"{actual_mean:.4f}"
            else:
                actual_str = str(actual_mean)
            table.add_row(
                "  └─ Actual Mean OOS Sharpe",
                actual_str,
                "observed",
                "[blue]INFO[/blue]",
                "Mean out-of-sample Sharpe from selected params",
            )

            # Row 4: null_mean (random params Sharpe)
            null_mean = check_result.get("null_mean", "N/A")
            if isinstance(null_mean, float):
                null_str = f"{null_mean:.4f}"
            else:
                null_str = str(null_mean)
            table.add_row(
                "  └─ Null Mean (Random Sharpe)",
                null_str,
                "observed",
                "[blue]INFO[/blue]",
                "Mean Sharpe from random parameter selection",
            )

            # Row 5: actual_count (windows)
            actual_count = check_result.get("actual_count", "N/A")
            table.add_row(
                "  └─ Windows Tested",
                str(actual_count),
                ">= 5",
                "[blue]INFO[/blue]",
                "Number of WFO windows used in test",
            )

            # Row 6: n_permutations
            n_perms = check_result.get("n_permutations", "N/A")
            table.add_row(
                "  └─ Permutations",
                str(n_perms),
                "500",
                "[blue]INFO[/blue]",
                "Number of random parameter permutations tested",
            )

            meta_overfitting_expanded = True

    for check_name, check_result in checks.items():
        # Skip meta_overfitting as we already processed it
        if check_name == "meta_overfitting":
            continue

        # Standard processing for other checks
        metric = check_result.get("metric", "N/A")
        threshold = check_result.get("threshold", "N/A")
        passed = check_result.get("passed", False)
        severity = check_result.get("severity", "INFO")

        # Determine status based on severity and explicit threshold comparison for numeric values
        # If severity is ERROR/WARNING, use that; otherwise check threshold comparison
        if severity == "ERROR":
            status = "[red][XX] FAIL[/red]"
        elif severity == "WARNING":
            status = "[yellow][!] WARN[/yellow]"
        else:
            # For INFO/NONE severity, explicitly compare metric vs threshold for numeric values
            if isinstance(metric, (int, float)) and threshold != "N/A":
                # Parse threshold to get comparison type and value
                thresh_str = str(threshold)
                if thresh_str.startswith(">="):
                    try:
                        thresh_val = float(thresh_str.replace(">=", "").strip())
                        passed = metric >= thresh_val
                    except ValueError:
                        passed = check_result.get("passed", False)
                elif thresh_str.startswith("<="):
                    try:
                        thresh_val = float(thresh_str.replace("<=", "").strip())
                        passed = metric <= thresh_val
                    except ValueError:
                        passed = check_result.get("passed", False)
                else:
                    passed = check_result.get("passed", False)
            else:
                passed = check_result.get("passed", False)

            status = "[green][OK] PASS[/green]" if passed else "[red][XX] FAIL[/red]"

        # Format metric value nicely
        if isinstance(metric, float):
            if abs(metric) < 0.01:
                metric_str = f"{metric:.6f}"
            elif abs(metric) < 1:
                metric_str = f"{metric:.4f}"
            else:
                metric_str = f"{metric:.2f}"
        else:
            metric_str = str(metric)

        display_name = check_name.replace("_", " ").title()
        explanation = diagnostic_explanations.get(
            check_name,
            check_result.get("recommendation", "Statistical validation test"),
        )

        if check_name == "statistical_power":
            source = check_result.get("expected_trades_per_day_source", "")
            if source:
                explanation = f"{explanation} ({source})"

        table.add_row(display_name, metric_str, str(threshold), status, explanation)

    # Overall status
    overall = summary.get("overall_status", "UNKNOWN")
    border_color = (
        "green" if overall == "PASS" else "yellow" if overall == "WARNING" else "red"
    )

    passed_count = sum(1 for c in checks.values() if c.get("passed", False))
    total_count = len(checks)

    panel_content = Table.grid()
    panel_content.add_column()
    panel_content.add_row(table)
    panel_content.add_row("")
    panel_content.add_row(f"Passed: {passed_count}/{total_count} checks")

    console.print(
        Panel(
            panel_content,
            title="[bold]Statistical Diagnostics[/bold]",
            border_style=border_color,
        )
    )


def _print_recommendations_panel(
    wf_results, metric_results, mc_results, holdout_results
):
    """Print recommendations panel (moved from validator.py)."""
    recommendations = []

    verdict_checks = wf_results.get("verdict", {}).get("checks", {})

    consistency_check = verdict_checks.get("consistency", {})
    if not consistency_check.get("passed", True):
        recommendations.append(
            "WFO: Low consistency - strategy may be overfitted or market-dependent"
        )

    mean_perf_check = verdict_checks.get("mean_performance", {})
    if not mean_perf_check.get("passed", True):
        recommendations.append(
            "WFO: Mean OOS Sharpe below threshold - consider strategy refinement"
        )

    param_stability_check = verdict_checks.get("parameter_stability", {})
    if not param_stability_check.get("passed", True):
        recommendations.append(
            "WFO: Parameter instability - reduce parameter space or add constraints"
        )

    flat_check = verdict_checks.get("flat_region", {})
    if not flat_check.get("passed", True):
        recommendations.append(
            "WFO: Parameters not in flat region - small changes cause large performance swings"
        )

    for check in metric_results.get("checks", []):
        if not check.get("passed", True):
            name = check["name"]
            if "Sharpe" in name:
                recommendations.append(
                    f"Multi-Metric: Low Sharpe ratio - improve risk-adjusted returns"
                )
            elif "Sortino" in name:
                recommendations.append(
                    f"Multi-Metric: Low Sortino ratio - reduce downside volatility"
                )
            elif "Calmar" in name:
                recommendations.append(
                    f"Multi-Metric: Low Calmar ratio - improve return/drawdown ratio"
                )
            elif "CAGR" in name:
                recommendations.append(
                    f"Multi-Metric: Low CAGR - improve annualized growth rate"
                )
            elif "Drawdown" in name:
                recommendations.append(
                    f"Multi-Metric: High drawdown - implement stricter risk management"
                )
            elif "Win Rate" in name:
                recommendations.append(
                    f"Multi-Metric: Low win rate - review entry/exit logic"
                )
            elif "Expectancy" in name:
                recommendations.append(
                    f"Multi-Metric: Negative expectancy - strategy loses money on average"
                )

    if mc_results.get("robustness_grade") == "FAIL":
        recommendations.append(
            "MC: Strategy fragile under stress - reduce position size or add filters"
        )

    dd_increase = mc_results.get("dd_increase_pct", 0)
    if dd_increase and dd_increase > 25:
        recommendations.append(
            f"MC: Drawdown increases {dd_increase:.1f}% under stress - high tail risk"
        )

    if holdout_results:
        if holdout_results.get("verdict") == "FAIL":
            recommendations.append(
                "Holdout: Failed on holdout data - likely overfitted to WFO period"
            )
        elif holdout_results.get("verdict") == "MARGINAL":
            recommendations.append(
                "Holdout: Marginal performance - monitor closely in live trading"
            )

    if not recommendations:
        recommendations.append(
            "Strategy shows strong robustness across all validation layers"
        )

    content = "\n".join([f"• {rec}" for rec in recommendations])

    console.print(
        Panel(content, title="[bold]Recommendations[/bold]", border_style="yellow")
    )


def _print_validation_results_unified(
    strategy_name: str,
    pairs: List[str],
    timeframe: str,
    data: pl.DataFrame,
    wf_results: Dict,
    metric_results: Dict,
    mc_results: Dict,
    holdout_results: Dict,
    diagnostic_results: Optional[Dict] = None,
    final_strategy_results: Optional[Dict] = None,
    final_metric_results: Optional[Dict] = None,
) -> None:
    """Print unified validation results using Rich Panels.

    Consolidates all validation metrics into a cohesive display:
    - WFO Robustness (6 checks from validator.py verdict)
    - 8-Metric Standard
    - Monte Carlo stress test
    - Holdout validation
    - Final Strategy Multi-Metric (with optimal sizing)
    - Single unified verdict
    - Parameter selection info
    - Recommendations
    """
    from rich.panel import Panel

    _print_header_panel(strategy_name, pairs, timeframe, data, wf_results)
    _print_wfo_panel(wf_results)
    _print_diagnostic_panel(diagnostic_results)
    _print_multi_metric_panel(metric_results)
    _print_mc_panel(mc_results)
    _print_holdout_panel(holdout_results)
    _print_final_strategy_panel(final_strategy_results, final_metric_results)
    _print_final_verdict_panel(wf_results, metric_results, mc_results, holdout_results)
    _print_param_selection_panel(wf_results)
    _print_recommendations_panel(
        wf_results, metric_results, mc_results, holdout_results
    )


def run_validate_mode(
    strategy_name: str,
    pairs: Optional[List[str]] = None,
    storage_path: str = "data/raw",
    save_reports: bool = False,
    auto_size: bool = False,
    days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    quick_test: bool = False,
) -> int:
    """Run validation mode with walk-forward and Monte Carlo.

    Args:
        strategy_name: Name of the strategy
        pairs: Optional list of pairs to override config
        storage_path: Path to data storage
        save_reports: If True, save detailed reports to disk
        auto_size: If True, run position sizing optimization after validation passes
        days: Optional filter to last N days
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        quick_test: If True, use smaller validation windows (180 train / 30 test / 365 lookback)

    Returns:
        Exit code (0 = pass, 1 = fail)
    """
    _print_progress(1, 5, "Loading strategy config...")

    strategy_config = _load_strategy_config(strategy_name)
    global_config = _load_global_config()
    merged_config = _merge_configs(strategy_config, global_config)

    # Use provided pairs or from config
    if pairs is None:
        pairs = strategy_config.get("pairs", ["BTC/USDT:USDT"])
    timeframe = strategy_config.get("timeframe", "15m")

    _print_progress(2, 5, f"Fetching data for {pairs[0]} {timeframe}...")
    _check_and_fetch_data(pairs, timeframe, storage_path, global_config)

    # Load data with optional date filtering
    data = _load_data(pairs, timeframe, storage_path, days, start_date, end_date)

    if data.is_empty():
        console.print("[red]Error: No data available after date filtering.[/red]")
        return 1

    if data.height < 100:
        console.print(
            f"[yellow]Warning: Only {data.height} bars after filtering. Validation results may not be meaningful.[/yellow]"
        )

    # Load strategy class
    strategy_class = _load_strategy_class(strategy_name)

    _print_progress(3, 5, "Running walk-forward optimization...")

    from engine.validator import (
        run_combined_monte_carlo,
        run_full_backtest_monte_carlo,
        run_holdout_validation,
        run_walk_forward,
        validate_multi_metric_standard,
        validate_final_strategy_metrics,
        generate_validation_report,
        run_final_strategy_evaluation,
    )

    validation_config = merged_config.get("validation", {})
    wf_config = validation_config.get("walk_forward", {})
    wfo_robustness_config = validation_config.get("wfo_robustness", {})

    # Quick test mode: override with smaller windows for faster validation
    if quick_test:
        wf_config = wf_config.copy() if wf_config else {}
        wf_config["train_days"] = 30
        wf_config["test_days"] = 7
        wf_config["lookback_days"] = 90
        # Override days filter for quick test
        if days is None:
            days = 90
        console.print(
            f"[yellow]Quick test mode: using smaller validation windows ({wf_config['train_days']} train / {wf_config['test_days']} test / {wf_config['lookback_days']} lookback)[/yellow]"
        )

    wf_results = run_walk_forward(
        strategy_class=strategy_class,
        config=merged_config,
        full_data=data,
        train_days=wf_config.get("train_days", 548),
        test_days=wf_config.get("test_days", 90),
        holdout_pct=wf_config.get("holdout_pct", 0.10),
        min_trades_per_window=wf_config.get("min_trades_per_window", 30),
        param_selection_method=wf_config.get("param_selection_method", "auto"),
        robustness_thresholds=wfo_robustness_config,
    )

    _print_progress(4, 5, "Running combined Monte Carlo stress test...")

    validation_config = merged_config.get("validation", {})
    mc_config = validation_config.get("monte_carlo", {})
    combined_oos_trades = wf_results.get("combined_oos_trades", pl.DataFrame())

    if not combined_oos_trades.is_empty():
        mc_results = run_combined_monte_carlo(
            combined_oos_trades,
            n_simulations=mc_config.get("n_simulations", 10000),
            skip_pct=mc_config.get("skip_pct", 0.10),
            slippage_pip=mc_config.get("slippage_pip", 1.0),
            entry_delay_bars=mc_config.get("entry_delay_bars", 1),
            pip_value=mc_config.get("pip_value", 0.0001),
            percentile_95_dd_max_pct=mc_config.get("percentile_95_dd_max_pct", 25.0),
            dd_increase_fail_pct=mc_config.get("dd_increase_fail_pct", 25.0),
            dd_increase_marginal_pct=mc_config.get("dd_increase_marginal_pct", 15.0),
        )
    else:
        mc_results = {"n_simulations": 0, "robustness_grade": "UNKNOWN"}

    mc_results["oos_trades"] = mc_results.copy()
    wfo_data = data[: int(len(data) * 0.9)] if data is not None else None
    final_params = wf_results.get("final_params", {})

    if wfo_data is not None and final_params:
        mc_full_results = run_full_backtest_monte_carlo(
            strategy_class=strategy_class,
            final_params=final_params,
            data=wfo_data,
            config=merged_config,
            n_simulations=mc_config.get("n_simulations", 10000),
            skip_pct=mc_config.get("skip_pct", 0.10),
            slippage_pip=mc_config.get("slippage_pip", 1.0),
            entry_delay_bars=mc_config.get("entry_delay_bars", 1),
            pip_value=mc_config.get("pip_value", 0.0001),
            percentile_95_dd_max_pct=mc_config.get("percentile_95_dd_max_pct", 25.0),
            dd_increase_fail_pct=mc_config.get("dd_increase_fail_pct", 25.0),
            dd_increase_marginal_pct=mc_config.get("dd_increase_marginal_pct", 15.0),
        )
        mc_results["full_backtest"] = mc_full_results

    _print_progress(5, 5, "Validating against Multi-Metric Standard...")

    stitched_equity = wf_results.get("stitched_equity_curve", pl.DataFrame())
    wfe = wf_results.get("wfe")
    cv = (
        wf_results.get("robustness", {})
        .get("performance_distribution", {})
        .get("coefficient_of_variation")
    )

    metric_config = validation_config.get("multi_metric_standard", {})
    metric_results = validate_multi_metric_standard(
        equity_curve=stitched_equity,
        trade_log=combined_oos_trades,
        wfe=wfe,
        cv=cv,
        thresholds=metric_config,
    )

    holdout_results = {}
    holdout_data = wf_results.get("holdout_data")

    if holdout_data is not None and final_params:
        holdout_config = validation_config.get("holdout", {})
        holdout_results = run_holdout_validation(
            strategy_class=strategy_class,
            final_params=final_params,
            holdout_data=holdout_data,
            config=merged_config,
            sharpe_min=holdout_config.get("sharpe_min", 1.0),
            max_drawdown_max_pct=holdout_config.get("max_drawdown_max_pct", 30.0),
        )

    diagnostic_results = wf_results.get("diagnostics", {})

    # Run final strategy evaluation with optimal position sizing
    final_strategy_results = {}
    final_metric_results = {}

    if final_params and wfo_data is not None:
        console.print(
            "[cyan]Running final strategy evaluation with optimal sizing...[/cyan]"
        )

        final_strategy_results = run_final_strategy_evaluation(
            strategy_class=strategy_class,
            final_params=final_params,
            wfo_data=wfo_data,
            config=merged_config,
            kelly_fraction=0.25,
        )

        # Run final strategy metrics (without WFE/CV - not applicable to single backtest)
        if (
            final_strategy_results.get("trade_log", pl.DataFrame()).height > 0
            and final_strategy_results.get("equity_curve", pl.DataFrame()).height > 1
        ):
            final_metric_results = validate_final_strategy_metrics(
                equity_curve=final_strategy_results["equity_curve"],
                trade_log=final_strategy_results["trade_log"],
                thresholds=metric_config,
            )
        else:
            final_metric_results = {
                "overall_verdict": "NO_TRADES",
                "passed_count": 0,
                "total_count": 0,
                "metrics": {},
                "checks": [],
            }

    results = {
        "strategy": strategy_name,
        "pairs": pairs,
        "timeframe": timeframe,
        "walk_forward": wf_results,
        "monte_carlo": mc_results,
        "multi_metric_standard": metric_results,
        "holdout": holdout_results,
        "diagnostics": diagnostic_results,
        "final_strategy": final_strategy_results,
        "final_metric_standard": final_metric_results,
    }

    if save_reports:
        generate_validation_report(results, strategy_name)

    # Print detailed validation results with Rich formatting
    _print_validation_results_unified(
        strategy_name=strategy_name,
        pairs=pairs,
        timeframe=timeframe,
        data=data,
        wf_results=wf_results,
        metric_results=metric_results,
        mc_results=mc_results,
        holdout_results=holdout_results,
        diagnostic_results=diagnostic_results,
        final_strategy_results=final_strategy_results,
        final_metric_results=final_metric_results,
    )

    # Calculate final pass/fail
    metric_verdict = metric_results.get("overall_verdict", "UNKNOWN")
    mc_grade = mc_results.get("robustness_grade", "UNKNOWN")
    holdout_verdict = holdout_results.get("verdict", "UNKNOWN")

    passed = (
        metric_verdict == "PASS"
        and mc_grade == "PASS"
        and holdout_verdict in ("PASS", "MARGINAL")
    )

    # Auto-run position sizing optimization if requested
    if auto_size and passed:
        console.print("\n[cyan]Running position sizing optimization...[/cyan]")
        _run_auto_position_sizing(strategy_name, strategy_config, global_config, data)

    return 0 if passed else 1


def run_optimize_sizing_mode(
    strategy_name: str,
    pairs: Optional[List[str]] = None,
    storage_path: str = "data/raw",
    apply: bool = False,
    use_fresh_backtest: bool = False,
    days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """Run position sizing optimization mode.

    Finds optimal position sizing parameters using Kelly Criterion.

    Args:
        strategy_name: Name of the strategy
        pairs: Optional list of pairs to override config
        storage_path: Path to data storage
        apply: If True, apply optimal sizing to strategy config.yaml
        use_fresh_backtest: If True, run fresh backtest instead of using validation results
        days: Optional filter to last N days
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        Exit code (0 = success)
    """
    from engine.position_sizing import optimize_position_sizing, generate_sizing_report

    _print_progress(1, 4, "Loading strategy config...")

    strategy_config = _load_strategy_config(strategy_name)
    global_config = _load_global_config()
    merged_config = _merge_configs(strategy_config, global_config)

    # Use provided pairs or from config
    if pairs is None:
        pairs = strategy_config.get("pairs", ["BTC/USDT:USDT"])
    timeframe = strategy_config.get("timeframe", "15m")

    _print_progress(2, 4, f"Fetching data for {pairs[0]} {timeframe}...")
    _check_and_fetch_data(pairs, timeframe, storage_path, global_config)

    # Get trade history (either from fresh backtest or validation results)
    if use_fresh_backtest:
        _print_progress(3, 4, "Running backtest for trade history...")

        data = _load_data(pairs, timeframe, storage_path, days, start_date, end_date)

        if data.is_empty():
            console.print("[red]Error: No data available after date filtering.[/red]")
            return 1

        if data.height < 100:
            console.print(
                f"[yellow]Warning: Only {data.height} bars after filtering. Results may not be meaningful.[/yellow]"
            )

        strategy_class = _load_strategy_class(strategy_name)
        strategy = strategy_class(merged_config["params"])
        signals = strategy.generate_signals(data)

        from engine.backtester import run_backtest

        trade_log, equity_curve = run_backtest(
            signals, merged_config, show_progress=True
        )
    else:
        _print_progress(3, 4, "Loading validation results...")

        # Try to load from latest validation report
        import json
        from pathlib import Path

        reports_dir = Path(f"reports/{strategy_name}")
        if not reports_dir.exists():
            console.print(
                "[red]No validation reports found. Run --mode validate first or use --backtest flag.[/red]"
            )
            return 1

        # Find most recent validation report
        validation_reports = sorted(
            reports_dir.glob("*/validation_report.json"), reverse=True
        )

        if not validation_reports:
            console.print(
                "[red]No validation_report.json found. Run --mode validate first or use --backtest flag.[/red]"
            )
            return 1

        # Load validation results
        with open(validation_reports[0], "r") as f:
            validation_results = json.load(f)

        # Extract OOS trade log and equity curve from validation
        # Note: This requires validation to have saved these. If not available, fall back to backtest.
        if "oos_trades" in validation_results and "oos_equity" in validation_results:
            import polars as pl

            trade_log = pl.DataFrame(validation_results["oos_trades"])
            equity_curve = pl.DataFrame(validation_results["oos_equity"])
        else:
            console.print(
                "[yellow]Validation report does not contain trade data. Running fresh backtest...[/yellow]"
            )
            use_fresh_backtest = True

            data = _load_data(
                pairs, timeframe, storage_path, days, start_date, end_date
            )

            if data.is_empty():
                console.print(
                    "[red]Error: No data available after date filtering.[/red]"
                )
                return 1

            strategy_class = _load_strategy_class(strategy_name)
            strategy = strategy_class(merged_config["params"])
            signals = strategy.generate_signals(data)

            from engine.backtester import run_backtest

            trade_log, equity_curve = run_backtest(
                signals, merged_config, show_progress=True
            )

    _print_progress(4, 4, "Running position sizing optimization...")

    # Run optimization
    optimization_results = optimize_position_sizing(
        trade_log=trade_log,
        equity_curve=equity_curve,
        config=merged_config,
    )

    # Generate report
    generate_sizing_report(optimization_results)

    # Apply to config if requested
    if apply:
        _apply_optimal_sizing(strategy_name, optimization_results)

    return 0


def _apply_optimal_sizing(strategy_name: str, optimization_results: dict) -> None:
    """Apply optimal position sizing parameters to strategy config.yaml.

    Args:
        strategy_name: Name of the strategy
        optimization_results: Results from optimize_position_sizing()
    """
    from pathlib import Path
    import yaml
    from datetime import datetime

    config_path = Path(f"strategies/{strategy_name}/config.yaml")

    # Backup existing config
    backup_path = config_path.with_suffix(
        f".yaml.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    import shutil

    shutil.copy(config_path, backup_path)
    console.print(f"[cyan]Backed up config to: {backup_path}[/cyan]")

    # Load current config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply optimal parameters
    optimal_params = optimization_results.get("optimal_params", {})

    if "position_sizing" not in config:
        config["position_sizing"] = {}

    # Update based on recommended method
    recommendation = optimization_results.get("recommendation", {})
    method = recommendation.get("method", "kelly")

    config["position_sizing"]["mode"] = "percent_equity"

    if method == "kelly":
        config["position_sizing"]["percent_equity"] = optimal_params.get(
            "kelly_fraction", 0.10
        )
    elif method == "volatility_adjusted":
        config["position_sizing"]["volatility_adjusted"] = {
            "enabled": True,
            "atr_period": optimal_params.get("atr_period", 14),
            "target_risk_pct": optimal_params.get("target_risk_pct", 0.02),
            "atr_multiplier": optimal_params.get("atr_multiplier", 2.0),
        }
    elif method == "antimartingale":
        config["position_sizing"]["antimartingale"] = {
            "enabled": True,
            "max_drawdown_threshold": optimal_params.get(
                "max_drawdown_threshold", 0.20
            ),
        }
    else:
        # For 'percent_equity' or any other mode, apply Kelly fraction
        config["position_sizing"]["percent_equity"] = optimal_params.get(
            "kelly_fraction", 0.10
        )

    # Write updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print(f"[green]Applied optimal sizing to: {config_path}[/green]")


def _run_auto_position_sizing(
    strategy_name: str,
    strategy_config: dict,
    global_config: dict,
    data: pl.DataFrame,
) -> None:
    """Run position sizing optimization automatically after validation.

    Called when --auto-size flag is used with validation mode.

    Args:
        strategy_name: Name of the strategy
        strategy_config: Strategy configuration
        global_config: Global configuration
        data: OHLCV data used for validation
    """
    from engine.position_sizing import optimize_position_sizing, generate_sizing_report

    merged_config = _merge_configs(strategy_config, global_config)

    # Run backtest with validated parameters to get trade history
    strategy_class = _load_strategy_class(strategy_name)
    strategy = strategy_class(merged_config["params"])
    signals = strategy.generate_signals(data)

    from engine.backtester import run_backtest

    trade_log, equity_curve = run_backtest(signals, merged_config, show_progress=False)

    # Run optimization
    optimization_results = optimize_position_sizing(
        trade_log=trade_log,
        equity_curve=equity_curve,
        config=merged_config,
    )

    # Generate report
    generate_sizing_report(optimization_results)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Systematic Trading Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
   python run.py --strategy ema_crossover_rsi --mode full
   python run.py --strategy ema_crossover_rsi --mode backtest
   python run.py --strategy ema_crossover_rsi --mode validate
   python run.py --strategy ema_crossover_rsi --mode validate --quick-test
   python run.py --strategy ema_crossover_rsi --mode full --pairs BTC/USDT:USDT
   python run.py --strategy ema_crossover_rsi --mode backtest --days 30
   python run.py --strategy ema_crossover_rsi --mode validate --start-date 2024-01-01 --end-date 2024-06-30
   python run.py --strategy ema_crossover_rsi --mode optimize_sizing --backtest
          """,
    )

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy name (must match directory in strategies/)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["full", "backtest", "validate", "optimize_sizing"],
        help="Execution mode: full/validate (walk-forward + Monte Carlo), backtest (single run), optimize_sizing (position sizing optimization)",
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Override trading pairs (e.g., BTC/USDT:USDT ETH/USDT:USDT)",
    )

    parser.add_argument(
        "--storage-path",
        type=str,
        default="data/raw",
        help="Path to data storage (default: data/raw)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/global_config.yaml",
        help="Path to global config (default: config/global_config.yaml)",
    )

    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Save detailed backtest/validation reports to reports/ directory",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply optimal position sizing parameters to strategy config.yaml",
    )

    parser.add_argument(
        "--auto-size",
        action="store_true",
        help="Automatically run position sizing optimization after validation passes",
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run fresh backtest for sizing optimization instead of using validation results",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Filter data to last N days (e.g., 7, 30, 365). Ignored if --start-date or --end-date is specified.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for data filtering (YYYY-MM-DD format). Overrides --days if specified.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data filtering (YYYY-MM-DD format, inclusive). Overrides --days if specified.",
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use smaller validation windows for faster testing (180 train / 30 test / 365 lookback). Revert to production settings before final validation.",
    )

    args = parser.parse_args()

    try:
        if args.mode == "backtest":
            exit_code = run_backtest_mode(
                strategy_name=args.strategy,
                pairs=args.pairs,
                storage_path=args.storage_path,
                save_reports=args.save_reports,
                days=args.days,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        elif args.mode in ("validate", "full"):
            exit_code = run_validate_mode(
                strategy_name=args.strategy,
                pairs=args.pairs,
                storage_path=args.storage_path,
                save_reports=args.save_reports,
                auto_size=args.auto_size,
                days=args.days,
                start_date=args.start_date,
                end_date=args.end_date,
                quick_test=args.quick_test,
            )
        elif args.mode == "optimize_sizing":
            exit_code = run_optimize_sizing_mode(
                strategy_name=args.strategy,
                pairs=args.pairs,
                storage_path=args.storage_path,
                apply=args.apply,
                use_fresh_backtest=args.backtest,
                days=args.days,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        else:
            print(f"Unknown mode: {args.mode}")
            exit_code = 1

        sys.exit(exit_code)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"Error importing strategy: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
