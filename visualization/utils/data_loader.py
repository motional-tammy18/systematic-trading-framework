"""Data loader utilities for backtest reports.

Provides functions to discover and load backtest results from the reports directory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import polars as pl


def discover_reports(reports_dir: str = "reports") -> List[Dict[str, Any]]:
    """Scan reports directory and return list of available backtest reports.

    Args:
        reports_dir: Path to reports directory (default: "reports")

    Returns:
        List of dicts with keys: strategy, date, timestamp, path
        Sorted by timestamp descending (most recent first)
    """
    reports_path = Path(reports_dir)
    
    if not reports_path.exists():
        return []
    
    reports = []
    
    # Iterate through strategy directories
    for strategy_dir in reports_path.iterdir():
        if not strategy_dir.is_dir():
            continue
        
        strategy_name = strategy_dir.name
        
        # Iterate through date directories
        for date_dir in strategy_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            date_str = date_dir.name
            
            for csv_file in date_dir.glob("trades_*.csv"):
                filename = csv_file.stem  # e.g., trades_20250217_143052
                parts = filename.split("_")
                if len(parts) >= 3:
                    timestamp_str = f"{parts[1]}_{parts[2]}"  # e.g., 20250217_143052
                    try:
                        ts = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        
                        reports.append({
                            "strategy": strategy_name,
                            "date": date_str,
                            "timestamp": ts,
                            "timestamp_str": timestamp_str,
                            "path": str(date_dir),
                        })
                    except ValueError:
                        continue
    
    # Sort by timestamp descending (most recent first)
    reports.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return reports


def load_latest_backtest(
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """Load the most recent backtest results.

    Args:
        reports_dir: Path to reports directory (default: "reports")

    Returns:
        Dict with keys:
        - equity_df: Polars DataFrame with equity curve
        - trades_df: Polars DataFrame with trade log
        - summary_dict: Dict with summary metrics
        - report_info: Dict with strategy, date, timestamp info

    Raises:
        FileNotFoundError: If no reports found
    """
    reports = discover_reports(reports_dir)
    
    if not reports:
        raise FileNotFoundError(
            "No reports found. Run a backtest with --save-reports first."
        )
    
    # Get most recent report
    latest = reports[0]
    report_path = Path(latest["path"])
    timestamp_str = latest["timestamp_str"]
    
    return load_backtest_by_path(report_path, timestamp_str)


def load_backtest_by_path(
    report_path: Path | str,
    timestamp_str: str,
) -> Dict[str, Any]:
    """Load backtest results from a specific report path.

    Args:
        report_path: Path to report directory (e.g., reports/ema_crossover/2025-02-17/)
        timestamp_str: Timestamp string (e.g., "20250217_143052")

    Returns:
        Dict with keys: equity_df, trades_df, summary_dict, report_info

    Raises:
        FileNotFoundError: If required files not found
        ValueError: If files have unexpected format
    """
    report_path = Path(report_path)
    
    # Build file paths
    trades_file = report_path / f"trades_{timestamp_str}.csv"
    equity_file = report_path / f"equity_{timestamp_str}.csv"
    summary_file = report_path / f"summary_{timestamp_str}.json"
    
    # Check for required files
    if not trades_file.exists():
        raise FileNotFoundError(f"Trade log not found: {trades_file}")
    if not equity_file.exists():
        raise FileNotFoundError(f"Equity curve not found: {equity_file}")
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary not found: {summary_file}")
    
    # Load equity curve
    try:
        equity_df = pl.read_csv(
            equity_file,
            try_parse_dates=True,
        )
    except Exception as e:
        raise ValueError(f"Failed to parse equity curve: {e}")
    
    # Validate equity columns
    required_equity_cols = {"timestamp", "equity", "drawdown"}
    if not required_equity_cols.issubset(set(equity_df.columns)):
        missing = required_equity_cols - set(equity_df.columns)
        raise ValueError(f"Equity curve missing columns: {missing}")
    
    # Load trades
    try:
        trades_df = pl.read_csv(
            trades_file,
            try_parse_dates=True,
        )
    except Exception as e:
        raise ValueError(f"Failed to parse trade log: {e}")
    
    # Validate trade columns - check for essential columns
    # Some columns may not exist depending on backtest type
    essential_trade_cols = {"entry_time", "exit_time", "pnl"}
    if not essential_trade_cols.issubset(set(trades_df.columns)):
        missing = essential_trade_cols - set(trades_df.columns)
        raise ValueError(f"Trade log has unexpected format. Missing: {missing}")
    
    # Load summary
    import json
    try:
        with open(summary_file, "r") as f:
            summary_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid summary file format: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load summary: {e}")
    
    # Extract report info from path
    parts = report_path.parts
    strategy_name = parts[-2] if len(parts) >= 2 else "unknown"
    date_str = parts[-1] if len(parts) >= 1 else "unknown"
    
    return {
        "equity_df": equity_df,
        "trades_df": trades_df,
        "summary_dict": summary_dict,
        "report_info": {
            "strategy": strategy_name,
            "date": date_str,
            "timestamp": timestamp_str,
        },
    }


def get_report_info(reports_dir: str = "reports") -> List[Dict[str, Any]]:
    """Get summary info of all available reports.

    Args:
        reports_dir: Path to reports directory

    Returns:
        List of dicts with strategy, date, timestamp for each report
    """
    reports = discover_reports(reports_dir)
    
    return [
        {
            "strategy": r["strategy"],
            "date": r["date"],
            "timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        }
        for r in reports
    ]


def load_ohlc_data(
    symbol: str,
    timeframe: str,
    data_dir: str = "../data/raw",
) -> Optional[pl.DataFrame]:
    """Load OHLC data from parquet file.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT:USDT")
        timeframe: Timeframe (e.g., "15m", "1h", "4h")
        data_dir: Directory containing raw OHLC data files

    Returns:
        Polars DataFrame with OHLC data or None if file not found
    """
    symbol_map = {
        "BTC/USDT:USDT": "BTC_USDT_USDT",
        "ETH/USDT:USDT": "ETH_USDT_USDT",
    }
    
    file_symbol = symbol_map.get(symbol, symbol.replace("/", "_").replace(":", "_"))
    filename = f"{file_symbol}_{timeframe}.parquet"
    filepath = Path(data_dir) / filename
    
    if not filepath.exists():
        return None
    
    try:
        df = pl.read_parquet(filepath)
        
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            return None
        
        df = df.sort("timestamp")
        
        return df
    except Exception:
        return None
