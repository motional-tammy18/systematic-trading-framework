"""
Data inspection utility for OHLCV Parquet files.

This script helps visualize and validate the data stored in Parquet format,
showing head/tail samples, date ranges, and data quality metrics.
"""

from pathlib import Path
from typing import Optional

import polars as pl


def inspect_parquet(
    symbol: str,
    timeframe: str,
    storage_path: str = "data/raw",
    head_n: int = 5,
    tail_n: int = 5,
    show_sample_bars: bool = True,
) -> None:
    """Inspect a Parquet file and display comprehensive information.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
        timeframe: Timeframe string (e.g., '15m', '1h', '1d')
        storage_path: Base directory for data storage
        head_n: Number of rows to show from the start
        tail_n: Number of rows to show from the end
        show_sample_bars: Whether to show OHLCV bars
    """
    # Clean symbol for file path
    clean_symbol = symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
    file_path = Path(storage_path) / f"{clean_symbol}_{timeframe}.parquet"

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    # Load data
    print(f"\n[*] Loading: {file_path}")
    df = pl.read_parquet(file_path).sort("timestamp")

    if df.is_empty():
        print("[!] DataFrame is empty!")
        return

    # Basic info
    print("\n[#] Data Overview")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Total Bars: {df.height:,}")

    # Date range
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    duration_days = (end_date - start_date).total_seconds() / 86400

    # Calculate expected vs actual bars
    if timeframe == "15m":
        expected_bars_per_day = 96
    elif timeframe == "30m":
        expected_bars_per_day = 48
    elif timeframe == "1h":
        expected_bars_per_day = 24
    elif timeframe == "4h":
        expected_bars_per_day = 6
    elif timeframe == "1d":
        expected_bars_per_day = 1
    else:
        expected_bars_per_day = None

    actual_bars = df.height
    if expected_bars_per_day and duration_days > 0:
        expected_total_bars = int(duration_days * expected_bars_per_day)
        completeness_pct = (actual_bars / expected_total_bars) * 100
    else:
        expected_total_bars = actual_bars
        completeness_pct = 100.0

    print(f"\n[Calendar] Date Range")
    print(f"   Start: {start_date}")
    print(f"   End: {end_date}")
    print(f"   Duration: {duration_days:.1f} days ({duration_days / 365:.1f} years)")

    print(f"\n[Data Completeness]")
    if expected_bars_per_day:
        print(f"   Actual bars: {actual_bars:,}")
        print(f"   Expected bars (~{expected_bars_per_day}/day): {expected_total_bars:,}")
        print(f"   Completeness: {completeness_pct:.1f}%")
        if completeness_pct < 90:
            print(f"   [!] WARNING: Data is {100 - completeness_pct:.1f}% INCOMPLETE!")
    else:
        print(f"   Total bars: {actual_bars:,}")

    # Data quality checks
    print(f"\n[?] Data Quality")

    # Check for nulls
    null_counts = df.null_count().row(0)
    has_nulls = any(null_counts)
    if has_nulls:
        print("[!] Null values detected:")
        for col, count in zip(df.columns, null_counts):
            if count > 0:
                print(f"   - {col}: {count:,} nulls")
    else:
        print("[OK] No null values")

    # Check for duplicates
    duplicates = df.is_duplicated().sum()
    if duplicates > 0:
        print(f"[!] Duplicate timestamps: {duplicates:,}")
    else:
        print("[OK] No duplicate timestamps")

    # Show sample data
    if show_sample_bars:
        print(f"\n[Chart] First {head_n} Bars")
        print("-" * 100)
        print(f"{'Timestamp':<25} {'Open':>12} {'High':>12} {'Low':>12} {'Close':>12} {'Volume':>15}")
        print("-" * 100)

        for row in df.head(head_n).iter_rows(named=True):
            print(f"{str(row['timestamp']):<25} {row['open']:>12.2f} {row['high']:>12.2f} {row['low']:>12.2f} {row['close']:>12.2f} {row['volume']:>15,.0f}")

        print(f"\n[Chart] Last {tail_n} Bars")
        print("-" * 100)
        print(f"{'Timestamp':<25} {'Open':>12} {'High':>12} {'Low':>12} {'Close':>12} {'Volume':>15}")
        print("-" * 100)

        for row in df.tail(tail_n).iter_rows(named=True):
            print(f"{str(row['timestamp']):<25} {row['open']:>12.2f} {row['high']:>12.2f} {row['low']:>12.2f} {row['close']:>12.2f} {row['volume']:>15,.0f}")

    # Price statistics
    print(f"\n[$] Price Statistics")
    print(f"   Min Close: ${df['close'].min():,.2f}")
    print(f"   Max Close: ${df['close'].max():,.2f}")
    print(f"   Avg Close: ${df['close'].mean():,.2f}")

    # Volume statistics
    print(f"\n[Bar] Volume Statistics")
    print(f"   Min Volume: {df['volume'].min():,.0f}")
    print(f"   Max Volume: {df['volume'].max():,.0f}")
    print(f"   Avg Volume: {df['volume'].mean():,.0f}")

    # File size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"\n[Floppy] File Size")
    print(f"   {file_size_mb:.2f} MB")

    print("\n[OK] Inspection complete!\n")


def list_all_data(storage_path: str = "data/raw") -> None:
    """List all available Parquet files with basic info.

    Args:
        storage_path: Base directory for data storage
    """
    data_dir = Path(storage_path)

    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return

    parquet_files = list(data_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"[!] No Parquet files found in {data_dir}")
        return

    print(f"\n[Folder] Available Data Files\n")
    print("-" * 120)
    print(f"{'File':<35} {'Bars':>15} {'Date Range':<45} {'Size (MB)':>12}")
    print("-" * 120)

    for file_path in sorted(parquet_files):
        try:
            df = pl.read_parquet(file_path).sort("timestamp")

            if df.is_empty():
                continue

            start_date = df["timestamp"].min()
            end_date = df["timestamp"].max()
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            print(f"{file_path.name:<35} {df.height:>15,} {str(start_date) + ' to ' + str(end_date):<45} {file_size_mb:>12.2f}")
        except Exception as e:
            print(f"{file_path.name:<35} {'ERROR':>15} {str(e):<45} {'-':>12}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect OHLCV data in Parquet format")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available data files",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT:USDT",
        help="Trading symbol (default: BTC/USDT:USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        help="Timeframe (default: 15m)",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="data/raw",
        help="Path to data storage directory (default: data/raw)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to show from start (default: 5)",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=5,
        help="Number of rows to show from end (default: 5)",
    )
    parser.add_argument(
        "--no-bars",
        action="store_true",
        help="Don't show OHLCV bar samples",
    )

    args = parser.parse_args()

    if args.list:
        list_all_data(args.storage_path)
    else:
        inspect_parquet(
            symbol=args.symbol,
            timeframe=args.timeframe,
            storage_path=args.storage_path,
            head_n=args.head,
            tail_n=args.tail,
            show_sample_bars=not args.no_bars,
        )
