"""
Bybit API-based OHLCV data fetcher.

Provides functions to fetch historical candlestick data with support for:
- Incremental updates (fetch only new data)
- Gap detection and filling
- Complete data coverage from earliest to latest
- Pagination (1000 candles per call)
- Single or multiple symbol fetching
- CLI interface for manual data retrieval
"""

import argparse
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Any, Tuple, cast

import requests
from requests.adapters import HTTPAdapter, Retry
import polars as pl
import yaml

# Module-level session with retry for robust HTTP requests
_requests_session: Optional[requests.Session] = None

def _get_session() -> requests.Session:
    """Get or create a requests Session with retry configuration.

    Uses a module-level singleton session to reuse connection pooling
    while providing robust retry behavior for transient failures.
    """
    global _requests_session
    if _requests_session is None:
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _requests_session = session
    return _requests_session

try:
    from data.storage import get_latest_timestamp, save_ohlcv, load_ohlcv, _get_file_path
except ImportError:
    from storage import get_latest_timestamp, save_ohlcv, load_ohlcv, _get_file_path


BYBIT_BASE_URL = "https://api.bybit.com"
BYBIT_KLINE_ENDPOINT = "/v5/market/kline"


def _load_config(config_path: str = "config/global_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _convert_symbol_for_bybit(symbol: str) -> str:
    """
    Convert symbol format from ccxt style to Bybit format.

    Args:
        symbol: Trading symbol in ccxt format (e.g., 'BTC/USDT:USDT')

    Returns:
        Symbol in Bybit format (e.g., 'BTCUSDT')

    Examples:
        >>> _convert_symbol_for_bybit('BTC/USDT:USDT')
        'BTCUSDT'
        >>> _convert_symbol_for_bybit('ETH/USDT:USDT')
        'ETHUSDT'
    """
    return symbol.replace("/", "").split(":")[0]


def _convert_timeframe_for_bybit(timeframe: str) -> str:
    """
    Convert timeframe format to Bybit API format.

    Args:
        timeframe: Timeframe string (e.g., '15m', '1h', '4h', '1d')

    Returns:
        Bybit interval string (e.g., '15', '60', '240', 'D')

    Examples:
        >>> _convert_timeframe_for_bybit('15m')
        '15'
        >>> _convert_timeframe_for_bybit('1h')
        '60'
        >>> _convert_timeframe_for_bybit('4h')
        '240'
        >>> _convert_timeframe_for_bybit('1d')
        'D'
    """
    timeframe_map = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }

    if timeframe in timeframe_map:
        return timeframe_map[timeframe]

    if timeframe.isdigit():
        return timeframe

    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _get_timeframe_ms(timeframe: str) -> int:
    """
    Convert timeframe string to milliseconds.

    Args:
        timeframe: Timeframe string (e.g., '15m', '1h', '4h', '1d')

    Returns:
        Timeframe in milliseconds

    Raises:
        ValueError: If timeframe is not supported
    """
    timeframe_ms_map = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000,
    }

    if timeframe in timeframe_ms_map:
        return timeframe_ms_map[timeframe]

    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _get_buffer_time_ms(timeframe: str) -> int:
    """
    Get buffer time in milliseconds to avoid fetching incomplete candles.

    Different timeframes need different buffers before current time.

    Args:
        timeframe: Timeframe string (e.g., '15m', '1h', '4h', '1d')

    Returns:
        Buffer time in milliseconds
    """
    buffer_map = {
        "1m": 60 * 60 * 1000,  # 1 hour
        "3m": 2 * 60 * 60 * 1000,  # 2 hours
        "5m": 3 * 60 * 60 * 1000,  # 3 hours
        "15m": 4 * 60 * 60 * 1000,  # 4 hours
        "30m": 6 * 60 * 60 * 1000,  # 6 hours
        "1h": 24 * 60 * 60 * 1000,  # 1 day
        "2h": 2 * 24 * 60 * 60 * 1000,  # 2 days
        "4h": 2 * 24 * 60 * 60 * 1000,  # 2 days
        "6h": 3 * 24 * 60 * 60 * 1000,  # 3 days
        "12h": 4 * 24 * 60 * 60 * 1000,  # 4 days
        "1d": 7 * 24 * 60 * 60 * 1000,  # 7 days
        "1w": 30 * 24 * 60 * 60 * 1000,  # 30 days
        "1M": 30 * 24 * 60 * 60 * 1000,  # 30 days
    }

    return buffer_map.get(timeframe, 4 * 60 * 60 * 1000)


def _detect_missing_periods(
    df_existing: Optional[pl.DataFrame],
    from_date: datetime,
    to_date: datetime,
    timeframe_ms: int,
) -> List[Tuple[int, int]]:
    """
    Detect missing periods in existing data.

    Args:
        df_existing: Existing OHLCV DataFrame (can be None)
        from_date: Desired start date
        to_date: Desired end date
        timeframe_ms: Timeframe in milliseconds

    Returns:
        List of (start_ms, end_ms) tuples for missing periods
    """
    from_timestamp = int(from_date.timestamp() * 1000)
    to_timestamp = int(to_date.timestamp() * 1000)

    periods_to_fetch = []

    if df_existing is None or df_existing.is_empty():
        return [(from_timestamp, to_timestamp)]

    min_ts_val = cast(datetime, df_existing["timestamp"].min())
    max_ts_val = cast(datetime, df_existing["timestamp"].max())
    min_timestamp = int(min_ts_val.timestamp() * 1000)
    max_timestamp = int(max_ts_val.timestamp() * 1000)

    if min_timestamp <= from_timestamp and max_timestamp >= to_timestamp:
        return []

    if min_timestamp > from_timestamp:
        periods_to_fetch.append((from_timestamp, min_timestamp - 1))

    if max_timestamp < to_timestamp:
        periods_to_fetch.append((max_timestamp + 1, to_timestamp))

    return periods_to_fetch


def _load_existing_data(
    symbol: str,
    timeframe: str,
    storage_path: str,
) -> Optional[pl.DataFrame]:
    """
    Load existing data for gap detection.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        storage_path: Base directory for data storage

    Returns:
        Existing DataFrame or None if file doesn't exist
    """
    try:
        return load_ohlcv(symbol, timeframe, storage_path)
    except FileNotFoundError:
        return None


def _fetch_kline_data(
    symbol: str,
    interval: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 1000,
    category: str = "linear",
    base_url: str = BYBIT_BASE_URL,
) -> List[List[str]]:
    """
    Fetch kline data from Bybit API v5.

    Args:
        symbol: Trading symbol in Bybit format (e.g., 'BTCUSDT')
        interval: Bybit interval string (e.g., '15', '60', 'D')
        start_ms: Start timestamp in milliseconds
        end_ms: Optional end timestamp in milliseconds
        limit: Number of candles to fetch (1-1000)
        category: Product type ('linear', 'spot', 'inverse')
        base_url: Bybit API base URL

    Returns:
        List of candle data arrays [timestamp, open, high, low, close, volume, turnover]

    Raises:
        requests.HTTPError: If API call fails
        ValueError: If API returns an error
    """
    url = f"{base_url}{BYBIT_KLINE_ENDPOINT}"

    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "start": start_ms,
        "limit": limit,
    }

    if end_ms is not None:
        params["end"] = end_ms

    session = _get_session()
    last_error: Optional[Exception] = None

    # Retry loop for SSL/connection errors (not covered by urllib3 Retry)
    for attempt in range(4):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()

            data = response.json()

            if data["retCode"] != 0:
                raise ValueError(f"Bybit API error: {data['retMsg']} (code: {data['retCode']})")

            return data["result"]["list"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < 3:
                sleep_sec = (2 ** attempt) + random.random()
                print(f"  Connection error (attempt {attempt + 1}/3), retrying in {sleep_sec:.1f}s...")
                time.sleep(sleep_sec)
                # Force new connection on SSL errors
                session.close()
                _requests_session = None
                session = _get_session()
                continue
            raise

    raise last_error  # type: ignore[misc]  # last_error is set if we exit the loop


def _validate_and_deduplicate_ohlcv_data(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """
    Validate OHLCV data for quality issues and remove duplicates.

    Args:
        df: Polars DataFrame with OHLCV data
        symbol: Trading symbol for error messages

    Returns:
        Deduplicated and validated DataFrame

    Raises:
        ValueError: If data validation fails
    """
    if df.is_empty():
        return df

    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{symbol}: Missing columns: {missing_cols}")

    for col in ["open", "high", "low", "close", "volume"]:
        null_count = df[col].is_null().sum()
        if null_count > 0:
            raise ValueError(f"{symbol}: Found {null_count} NaN values in {col}")

    duplicate_count = df["timestamp"].is_duplicated().sum()
    if duplicate_count > 0:
        print(f"  Warning: {symbol}: Found {duplicate_count} duplicate timestamps, keeping last occurrence")

    df = df.unique(subset=["timestamp"], keep="last")
    df = df.sort("timestamp")

    timestamps = df["timestamp"].to_list()
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            raise ValueError(f"{symbol}: Data not in chronological order at index {i}")

    return df


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    fetch_from: str = "max",
    storage_path: str = "data/raw",
    config: Optional[dict] = None,
    rate_limit_ms: int = 150,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
) -> pl.DataFrame:
    """
    Fetch OHLCV data from Bybit with gap detection and complete coverage.

    This function ensures data coverage from the earliest available (2018 or
    exchange start) to the latest (current time) by detecting and filling gaps
    in existing data.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
        timeframe: Timeframe string (e.g., '15m', '1h', '1d')
        fetch_from: Start point - 'max' for all history, '1d' for 1 day, etc.
        storage_path: Base directory for data storage
        config: Configuration dictionary. If None, loads from default path.
        rate_limit_ms: Delay between API calls in milliseconds
        from_date: Optional explicit start date (overrides fetch_from)
        to_date: Optional explicit end date (defaults to now with buffer)

    Returns:
        Polars DataFrame with columns: timestamp, open, high, low, close, volume

    Raises:
        ValueError: If data validation fails or API returns error
        requests.HTTPError: If HTTP request fails
    """
    if config is None:
        config = _load_config()

    bybit_symbol = _convert_symbol_for_bybit(symbol)
    bybit_interval = _convert_timeframe_for_bybit(timeframe)
    timeframe_ms = _get_timeframe_ms(timeframe)
    buffer_ms = _get_buffer_time_ms(timeframe)

    max_history_days = config.get("data", {}).get("max_history_days", 3650)

    now = datetime.now()
    now_ms = int(now.timestamp() * 1000)

    if to_date is None:
        to_date = now
    else:
        to_date = min(to_date, now)

    if from_date is None:
        if fetch_from == "max":
            from_date = now - timedelta(days=max_history_days)
        else:
            unit = fetch_from[-1]
            value = int(fetch_from[:-1])
            if unit == "d":
                from_date = now - timedelta(days=value)
            elif unit == "h":
                from_date = now - timedelta(hours=value)
            elif unit == "m":
                from_date = now - timedelta(minutes=value)
            else:
                raise ValueError(f"Invalid fetch_from format: {fetch_from}")

    df_existing = _load_existing_data(symbol, timeframe, storage_path)

    fetch_all_available = fetch_from == "max" or (
        fetch_from.endswith("d") and int(fetch_from[:-1]) >= 365 * 5
    )

    if fetch_all_available:
        to_timestamp = int(to_date.timestamp() * 1000)
        
        if df_existing is None or df_existing.is_empty():
            # Use a very early start date for backward pagination
            from_timestamp = int(datetime(2015, 1, 1).timestamp() * 1000)
            periods_to_fetch = [(from_timestamp, to_timestamp)]
            df_existing = None
            from_timestamp = int((to_date - timedelta(days=365 * 3)).timestamp() * 1000)
            periods_to_fetch = [(from_timestamp, to_timestamp)]
            df_existing = None
        else:
            max_history_days = config.get("data", {}).get("max_history_days", 3650)
            from_timestamp = int((to_date - timedelta(days=max_history_days)).timestamp() * 1000)
            periods_to_fetch = _detect_missing_periods(df_existing, from_date, to_date, timeframe_ms)
    else:
        periods_to_fetch = _detect_missing_periods(df_existing, from_date, to_date, timeframe_ms)

    if not periods_to_fetch:
        print(f"Data complete for {symbol} {timeframe} from {from_date} to {to_date}")
        if df_existing is not None:
            return df_existing.filter(
                (pl.col("timestamp") >= from_date) & (pl.col("timestamp") <= to_date)
            )
        return pl.DataFrame()

    all_candles: List[List[str]] = []
    total_fetched = 0
    first_timestamp: Optional[int] = None
    last_timestamp: Optional[int] = None

    base_url = config.get("exchange", {}).get("base_url", BYBIT_BASE_URL)
    category = config.get("exchange", {}).get("category", "linear")

    for start_ts, end_ts in periods_to_fetch:
        print(f"Fetching {symbol} {timeframe} from {datetime.fromtimestamp(start_ts / 1000)} to {datetime.fromtimestamp(end_ts / 1000)}")

        since_ms = start_ts
        target_ms = end_ts

        if fetch_all_available and df_existing is None:
            # Use backward pagination with end parameter - Bybit returns newest first
            discovered_batches = []
            search_ts = target_ms  # Start from most recent
            batch_num = 0
            consecutive_empty = 0
            
            while True:
                batch_num += 1
                try:
                    # Use end only - returns data up to that point (newest first)
                    batch = _fetch_kline_data(
                        symbol=bybit_symbol,
                        interval=bybit_interval,
                        end_ms=search_ts,
                        limit=1000,
                        category=category,
                        base_url=base_url,
                    )
                except (requests.HTTPError, ValueError):
                    break

                if not batch:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        break
                    continue

                consecutive_empty = 0
                discovered_batches.extend(batch)
                oldest = int(batch[-1][0])  # timestamp is first element of each candle
                print(f"  Batch {batch_num}: got {len(batch)} candles, oldest: {datetime.fromtimestamp(oldest/1000)}")

                if len(batch) < 1000:
                    break

                # Move backwards by 1 candle before oldest
                search_ts = oldest - timeframe_ms
                time.sleep(rate_limit_ms / 1000)

            if discovered_batches:
                discovered_batches.reverse()  # Now oldest first
                all_candles = discovered_batches
                total_fetched = len(discovered_batches)
                first_timestamp = int(discovered_batches[0][0])  # timestamp is first element of each candle
                last_timestamp = int(discovered_batches[-1][0])
                continue
            discovered_batches = []
            search_start = start_ts
            search_end = target_ms
            batch_num = 0
            consecutive_small_batches = 0
            
            while True:
                batch_num += 1
                try:
                    batch = _fetch_kline_data(
                        symbol=bybit_symbol,
                        interval=bybit_interval,
                        start_ms=search_start,
                        end_ms=search_end,
                        limit=1000,
                        category=category,
                        base_url=base_url,
                    )
                except (requests.HTTPError, ValueError):
                    break
                
                if not batch:
                    break
                
                discovered_batches.extend(batch)
                oldest = int(batch[-1][0])
                print(f"  Batch {batch_num}: got {len(batch)} candles, oldest: {datetime.fromtimestamp(oldest/1000)}")
                
                if len(batch) < 1000:
                    consecutive_small_batches += 1
                    if consecutive_small_batches >= 3:
                        break
                else:
                    consecutive_small_batches = 0
                
                search_start = oldest - (366 * 24 * 60 * 60 * 1000)
                search_end = oldest - 1
                time.sleep(rate_limit_ms / 1000)
            
            if discovered_batches:
                discovered_batches.reverse()
                all_candles = discovered_batches
                total_fetched = len(discovered_batches)
                first_timestamp = int(discovered_batches[-1][0])
                last_timestamp = int(discovered_batches[0][0])
                print(f"Fetched {total_fetched} candles from {datetime.fromtimestamp(first_timestamp / 1000)} to {datetime.fromtimestamp(last_timestamp / 1000)}")
                continue

        while since_ms <= target_ms:
            try:
                candles = _fetch_kline_data(
                    symbol=bybit_symbol,
                    interval=bybit_interval,
                    start_ms=since_ms,
                    end_ms=min(since_ms + (1000 * timeframe_ms), target_ms) if since_ms + (1000 * timeframe_ms) < target_ms else target_ms,
                    limit=1000,
                    category=category,
                    base_url=base_url,
                )
            except requests.HTTPError as e:
                raise requests.HTTPError(f"HTTP error fetching {symbol}: {e}")
            except ValueError as e:
                raise ValueError(f"API error fetching {symbol}: {e}")

            if not candles:
                print(f"  No more data from {datetime.fromtimestamp(since_ms / 1000)}")
                break

            candles.reverse()
            all_candles.extend(candles)
            total_fetched += len(candles)

            batch_first_ts = int(candles[0][0])
            batch_last_ts = int(candles[-1][0])

            if first_timestamp is None or batch_first_ts < first_timestamp:
                first_timestamp = batch_first_ts
            if last_timestamp is None or batch_last_ts > last_timestamp:
                last_timestamp = batch_last_ts

            if batch_last_ts >= target_ms:
                break

            if len(candles) < 1000:
                break

            since_ms = batch_last_ts + timeframe_ms
            time.sleep(rate_limit_ms / 1000)

    if not all_candles:
        if df_existing is not None and not df_existing.is_empty():
            print(f"No new data for {symbol} {timeframe} (up to date)")
            return df_existing.filter(
                (pl.col("timestamp") >= from_date) & (pl.col("timestamp") <= to_date)
            )
        raise ValueError(f"No data returned for {symbol} {timeframe}")

    df_new = pl.DataFrame({
        "timestamp": [datetime.fromtimestamp(int(c[0]) / 1000) for c in all_candles],
        "open": [float(c[1]) for c in all_candles],
        "high": [float(c[2]) for c in all_candles],
        "low": [float(c[3]) for c in all_candles],
        "close": [float(c[4]) for c in all_candles],
        "volume": [float(c[5]) for c in all_candles],
    })

    df_new = _validate_and_deduplicate_ohlcv_data(df_new, symbol)

    df_combined = pl.concat([df_existing, df_new]) if df_existing is not None else df_new
    df_combined = df_combined.unique(subset=["timestamp"], keep="last")
    df_combined = df_combined.sort("timestamp")

    save_ohlcv(symbol, timeframe, df_combined, storage_path)

    if total_fetched > 0:
        first_dt = datetime.fromtimestamp(int(first_timestamp) / 1000) if first_timestamp else from_date
        last_dt = datetime.fromtimestamp(int(last_timestamp) / 1000) if last_timestamp else to_date
        print(f"Fetched {total_fetched} candles for {symbol} {timeframe} from {first_dt} to {last_dt}")

    return df_combined.filter(
        (pl.col("timestamp") >= from_date) & (pl.col("timestamp") <= to_date)
    )


def fetch_multiple(
    symbols: list[str],
    timeframe: str,
    fetch_from: str = "max",
    storage_path: str = "data/raw",
    config: Optional[dict] = None,
    rate_limit_ms: int = 150,
) -> None:
    """
    Fetch OHLCV data for multiple symbols.

    Args:
        symbols: List of trading symbols
        timeframe: Timeframe string (e.g., '15m', '1h', '1d')
        fetch_from: Start point - 'max' for all history, '1d' for 1 day, etc.
        storage_path: Base directory for data storage
        config: Configuration dictionary. If None, loads from default path.
        rate_limit_ms: Delay between API calls in milliseconds
    """
    if config is None:
        config = _load_config()

    print(f"Fetching data for {len(symbols)} symbols on {timeframe} timeframe")

    for i, symbol in enumerate(symbols):
        try:
            fetch_ohlcv(
                symbol, timeframe, fetch_from, storage_path, config, rate_limit_ms
            )
            print(f"  [{i + 1}/{len(symbols)}] {symbol} - OK")
        except Exception as e:
            print(f"  [{i + 1}/{len(symbols)}] {symbol} - ERROR: {e}")

        if i < len(symbols) - 1:
            time.sleep(rate_limit_ms / 1000)

    print("Fetch complete")


def main():
    """CLI entry point for data fetching."""
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV data from Bybit exchange",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/fetcher.py --symbol BTC/USDT:USDT --timeframe 15m
  python data/fetcher.py --symbol BTC/USDT:USDT --timeframe 1h --days 7
  python data/fetcher.py --pairs BTC/USDT:USDT ETH/USDT:USDT --timeframe 1h
        """,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        help="Single trading symbol (e.g., BTC/USDT:USDT)",
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Multiple trading symbols (e.g., BTC/USDT:USDT ETH/USDT:USDT)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        required=True,
        help="Timeframe (e.g., 15m, 1h, 4h, 1d)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to fetch (overrides incremental fetch)",
    )

    parser.add_argument(
        "--storage-path",
        type=str,
        default="data/raw",
        help="Path to store Parquet files (default: data/raw)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/global_config.yaml",
        help="Path to config file (default: config/global_config.yaml)",
    )

    parser.add_argument(
        "--rate-limit",
        type=int,
        default=150,
        help="Rate limit delay in ms between calls (default: 150)",
    )

    args = parser.parse_args()

    if not args.symbol and not args.pairs:
        parser.error("Must specify either --symbol or --pairs")

    if args.symbol and args.pairs:
        parser.error("Cannot specify both --symbol and --pairs")

    config = _load_config(args.config)

    if args.days is not None:
        fetch_from = f"{args.days}d"
    else:
        fetch_from = "max"

    if args.symbol:
        fetch_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            fetch_from=fetch_from,
            storage_path=args.storage_path,
            config=config,
            rate_limit_ms=args.rate_limit,
        )
    else:
        fetch_multiple(
            symbols=args.pairs,
            timeframe=args.timeframe,
            fetch_from=fetch_from,
            storage_path=args.storage_path,
            config=config,
            rate_limit_ms=args.rate_limit,
        )


if __name__ == "__main__":
    main()
