"""
Storage module for OHLCV data using Polars and Parquet format.

Provides functions to save, load, and query OHLCV (Open, High, Low, Close, Volume)
data for trading symbols in a standardized Parquet format.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl


def _get_file_path(symbol: str, timeframe: str, storage_path: str = "data/raw") -> Path:
    """
    Generate standardized file path for OHLCV data.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT', 'ETH/USD')
        timeframe: Timeframe string (e.g., '15m', '1h', '1d')
        storage_path: Base directory for data storage

    Returns:
        Path object pointing to the Parquet file
    """
    clean_symbol = (
        symbol.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )
    filename = f"{clean_symbol}_{timeframe}.parquet"
    return Path(storage_path) / filename


def save_ohlcv(
    symbol: str,
    timeframe: str,
    df: pl.DataFrame,
    storage_path: str = "data/raw"
) -> None:
    """
    Save OHLCV DataFrame to Parquet file.

    Creates the storage directory if it doesn't exist.
    If the file already exists, appends new data (deduplicates by timestamp).

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT', 'ETH/USD')
        timeframe: Timeframe string (e.g., '15m', '1h', '1d')
        df: Polars DataFrame with columns: timestamp, open, high, low, close, volume
        storage_path: Base directory for data storage

    Raises:
        ValueError: If DataFrame doesn't contain required columns
    """
    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime)
        )

    file_path = _get_file_path(symbol, timeframe, storage_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        existing_df = pl.read_parquet(file_path)
        combined_df = pl.concat([existing_df, df])
        combined_df = combined_df.unique(subset=["timestamp"], keep="last")
        combined_df = combined_df.sort("timestamp")
        combined_df.write_parquet(file_path)
    else:
        df.sort("timestamp").write_parquet(file_path)


def load_ohlcv(
    symbol: str,
    timeframe: str,
    storage_path: str = "data/raw"
) -> pl.DataFrame:
    """
    Load OHLCV DataFrame from Parquet file.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT', 'ETH/USD')
        timeframe: Timeframe string (e.g., '15m', '1h', '1d')
        storage_path: Base directory for data storage

    Returns:
        Polars DataFrame with columns: timestamp, open, high, low, close, volume

    Raises:
        FileNotFoundError: If the Parquet file doesn't exist
    """
    file_path = _get_file_path(symbol, timeframe, storage_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"No data found for {symbol} {timeframe} at {file_path}"
        )

    df = pl.read_parquet(file_path)
    return df.sort("timestamp")


def get_latest_timestamp(
    symbol: str,
    timeframe: str,
    storage_path: str = "data/raw"
) -> Optional[datetime]:
    """
    Get the most recent timestamp in the OHLCV data.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT', 'ETH/USD')
        timeframe: Timeframe string (e.g., '15m', '1h', '1d')
        storage_path: Base directory for data storage

    Returns:
        Latest datetime timestamp, or None if file doesn't exist or is empty
    """
    file_path = _get_file_path(symbol, timeframe, storage_path)

    if not file_path.exists():
        return None

    try:
        df = pl.read_parquet(file_path)
        if df.is_empty():
            return None

        latest = df["timestamp"].max()

        if latest is not None:
            return latest  # type: ignore
        return None
    except Exception:
        return None
