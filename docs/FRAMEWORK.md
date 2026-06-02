# Framework Development Guide

This document provides a comprehensive guide for modifying and extending the systematic trading framework. It covers architecture, inter-module contracts, module responsibilities, and best practices for making changes safely. Read this document before modifying any core framework components.

## 1. Architecture Overview

The framework follows a modular architecture with clear separation of concerns. Each module has a specific responsibility, and data flows through the system in a well-defined pipeline. Understanding this architecture is essential before making any modifications, as changes to one module can impact others through the data contracts they share.

### 1.1 Directory Structure

The framework is organized into three main directories plus configuration and documentation:

| Directory     | Purpose                                                 |
| ------------- | ------------------------------------------------------- |
| `data/`       | Data fetching and storage modules                       |
| `engine/`     | Core backtesting, optimization, and metrics computation |
| `strategies/` | Trading strategy implementations                        |
| `config/`     | Global configuration settings                           |
| `docs/`       | Documentation including this guide                      |

The `data/` directory handles all interactions with external data sources and persistent storage. The `engine/` directory contains the computational core that processes data and generates results. The `strategies/` directory is where trading strategies live and is the primary extension point for adding new strategies. The `config/` directory holds global settings that apply across all strategies.

### 1.2 Data Flow Pipeline

Data flows through the system in a sequential pipeline, with each module transforming data and passing results to the next module. Understanding this flow is critical for debugging and for ensuring any modifications maintain compatibility.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   fetcher    │───▶│   storage    │───▶│  backtester  │───▶│   optimizer   │───▶│  validator   │
│(gap detect)  │    │   (load)     │    │   (simulate) │    │   (optimize) │    │   (validate) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

The pipeline begins with the **fetcher** module, which retrieves OHLCV (Open, High, Low, Close, Volume) data from the Bybit exchange API. The fetcher handles rate limiting, pagination, automatic gap detection to ensure complete data coverage, and deduplication. It intelligently detects missing periods in existing data and fetches only what's needed, making incremental updates fast and efficient. After fetching, data passes to the **storage** module, which saves OHLCV data to Parquet files for efficient subsequent access.

When running a backtest or optimization, the **backtester** loads historical data from storage and processes it with a strategy. The backtester simulates trading by applying strategy signals to historical prices, computing trades, and building equity curves. The **optimizer** wraps the backtester and systematically tests different parameter combinations to find optimal values. Finally, the **validator** applies statistical tests including walk-forward analysis and Monte Carlo simulation to assess strategy robustness.

The `run.py` script orchestrates this entire pipeline, loading configuration, instantiating the appropriate strategy, and coordinating the fetch-backtest-optimize-validate flow.

### 1.3 Module Dependencies

Modules depend on each other through well-defined interfaces. These dependencies form a directed acyclic graph where data flows in one direction and configuration flows downward from global settings to individual modules.

```
run.py
├── config/global_config.yaml
├── strategies/*/config.yaml
├── strategies/*/signals.py
│   └── engine/base_strategy.py
├── data/fetcher.py
│   └── data/storage.py
├── engine/backtester.py
├── engine/optimizer.py
│   └── engine/backtester.py
│   └── engine/metrics.py
└── engine/metrics.py
```

The `base_strategy.py` module is a leaf dependency that all strategies inherit from but does not depend on other engine components. The `backtester.py` and `optimizer.py` modules depend on each other through function calls. The `metrics.py` module is a utility library with no dependencies on other framework components.

## 2. Inter-Module Contracts

Modules communicate through strict data contracts. These contracts define the exact schema of DataFrames that pass between modules. Violating these contracts causes failures that may not be immediately obvious, so understanding and maintaining these contracts is essential for framework stability.

### 2.1 OHLCV Data Contract

The OHLCV data contract defines the format of price data that flows through the system. This is the foundational contract that all other modules depend on.

**Source**: `data/fetcher.py` produces, `data/storage.py` stores and retrieves

**Schema**:

| Column    | Data Type   | Description                                    |
| --------- | ----------- | ---------------------------------------------- |
| timestamp | pl.Datetime | Candle open time in UTC, millisecond precision |
| open      | pl.Float64  | Opening price for the period                   |
| high      | pl.Float64  | Highest price during the period                |
| low       | pl.Float64  | Lowest price during the period                 |
| close     | pl.Float64  | Closing price for the period                   |
| volume    | pl.Float64  | Trading volume during the period               |

**Example**:

```python
# Valid OHLCV DataFrame
df = pl.DataFrame({
    "timestamp": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 15)],
    "open": [50000.0, 50100.0],
    "high": [50200.0, 50300.0],
    "low": [49950.0, 50050.0],
    "close": [50150.0, 50200.0],
    "volume": [1250.5, 980.3],
})
```

**Validation**: The `fetcher.py` module validates and cleans OHLCV data before storage:

- Checks for required columns (timestamp, open, high, low, close, volume)
- Validates no null values in price/volume columns
- Removes duplicate timestamps (keeping last occurrence, with warning)
- Ensures chronological order
- Any critical validation failure raises a `ValueError` with a descriptive message

**Data Completeness**: The fetcher ensures near-complete data coverage:

- Automatic gap detection identifies missing periods
- Only missing periods are fetched (efficient incremental updates)
- Target completeness: 99.6%+ (typical for cryptocurrency exchanges)
- Small gaps are normal due to exchange maintenance, API rate limits, or no trades during low liquidity periods

### 2.2 Signal Data Contract

The signal contract defines the output format of strategy signal generation. This is the interface between strategies and the backtesting engine.

**Source**: Strategies (inheriting from `BaseStrategy`) produce, `engine/backtester.py` consumes

**Schema**:

| Column      | Data Type             | Description                               |
| ----------- | --------------------- | ----------------------------------------- |
| timestamp   | pl.Datetime           | Candle timestamp (from input OHLCV)       |
| open        | pl.Float64            | Opening price (from input OHLCV)          |
| high        | pl.Float64            | High price (from input OHLCV)             |
| low         | pl.Float64            | Low price (from input OHLCV)              |
| close       | pl.Float64            | Close price (from input OHLCV)            |
| volume      | pl.Float64            | Volume (from input OHLCV)                 |
| signal      | pl.Int8               | Position signal: 1=long, -1=short, 0=flat |
| entry_price | pl.Float64 (optional) | Limit entry price for order management    |
| tp_price    | pl.Float64 (optional) | Take profit price for order management    |
| sl_price    | pl.Float64 (optional) | Stop loss price for order management      |

**Signal Values**:

- **1 (Long)**: Enter or hold a long position. The engine opens or adds to a long position.
- **-1 (Short)**: Enter or hold a short position. The engine opens or adds to a short position.
- **0 (Flat)**: No position. The engine closes any existing position.

**Example**:

```python
# Valid signal DataFrame
signals = df.with_columns([
    pl.lit(1, dtype=pl.Int8).alias("signal")  # Long signal
])
```

**Validation**: The backtester validates signals before processing, checking that all required OHLCV columns exist and that the signal column contains only valid values (1, -1, 0). Missing columns raise `ValueError` with the list of missing columns.

### 2.3 Trade Log Contract

The trade log contract defines the output format of trade simulation. This is the backtester's output that feeds into metrics calculations.

**Source**: `engine/backtester.py` produces, `engine/metrics.py` consumes

**Schema**:

| Column                 | Data Type             | Description                                    |
| ---------------------- | --------------------- | ---------------------------------------------- |
| entry_time             | pl.Datetime           | Timestamp when the trade was entered           |
| exit_time              | pl.Datetime           | Timestamp when the trade was exited            |
| entry_price            | pl.Float64            | Price at which the position was entered        |
| exit_price             | pl.Float64            | Price at which the position was exited         |
| direction              | pl.Int8               | Trade direction: 1=long, -1=short              |
| pnl                    | pl.Float64            | Net profit/loss in quote currency (after fees) |
| pnl_pct                | pl.Float64            | Net PnL as percentage of entry value           |
| size                   | pl.Float64            | Position size in base currency                 |
| entry_type             | pl.Utf8 (optional)    | Entry execution type: market, limit, stop      |
| exit_reason            | pl.Utf8 (optional)    | Exit trigger: signal, tp, sl, circuit_breaker  |
| tp_price               | pl.Float64 (optional) | Take profit price level                        |
| sl_price               | pl.Float64 (optional) | Stop loss price level                          |
| max_price              | pl.Float64 (optional) | Highest price during trade (for trailing SL)   |
| bars_held              | pl.Int64 (optional)   | Number of bars the position was held           |
| entry_fee              | pl.Float64 (optional) | Fee paid on entry                              |
| exit_fee               | pl.Float64 (optional) | Fee paid on exit                               |
| total_fees             | pl.Float64 (optional) | Sum of entry and exit fees                     |
| circuit_breaker_reason | pl.Utf8 (optional)    | Reason for circuit breaker trigger             |

**Example**:

```python
# Valid trade log
trade_log = pl.DataFrame({
    "entry_time": [datetime(2024, 1, 1, 0, 0)],
    "exit_time": [datetime(2024, 1, 1, 1, 0)],
    "entry_price": [50000.0],
    "exit_price": [50500.0],
    "direction": [1],
    "pnl": [490.25],  # After fees
    "pnl_pct": [0.98],
    "size": [0.1],
})
```

**Empty Trade Log**: When no trades occur, the backtester returns an empty DataFrame with the correct schema. This is important for metrics calculations that iterate over trades.

### 2.4 Equity Curve Contract

The equity curve contract defines the format of portfolio value over time. This is the primary output for performance analysis.

**Source**: `engine/backtester.py` produces, `engine/metrics.py` consumes

**Schema**:

| Column    | Data Type   | Description                                 |
| --------- | ----------- | ------------------------------------------- |
| timestamp | pl.Datetime | Timestamp for each bar                      |
| equity    | pl.Float64  | Portfolio equity value at this bar          |
| drawdown  | pl.Float64  | Drawdown percentage: (peak - equity) / peak |

**Example**:

```python
# Valid equity curve
equity_curve = pl.DataFrame({
    "timestamp": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 15)],
    "equity": [10000.0, 10050.0],
    "drawdown": [0.0, 0.0],
})
```

**Validation**: The metrics module validates equity curves before calculation, checking for minimum data points and required columns. An equity curve with fewer than 2 data points raises `ValueError` for metrics that require returns calculation.

### 2.5 Optimization Result Contract

The optimization result contract defines the output format of parameter optimization. This is the optimizer's output used for selecting best parameters.

**Source**: `engine/optimizer.py` produces, `run.py` consumes

**Schema**:

| Column       | Data Type  | Description                                          |
| ------------ | ---------- | ---------------------------------------------------- |
| \*param_name | pl.Float64 | Each parameter from param_space                      |
| sharpe       | pl.Float64 | Annualized Sharpe ratio of the parameter combination |
| max_dd       | pl.Float64 | Maximum drawdown percentage                          |
| trade_count  | pl.Int64   | Number of trades in the backtest                     |
| total_pnl    | pl.Float64 | Total profit/loss in quote currency                  |

**Example**:

```python
# Valid optimization results
results = pl.DataFrame({
    "ema_short": [10.0, 20.0, 30.0],
    "ema_long": [50.0, 50.0, 50.0],
    "sharpe": [1.2, 1.5, 1.1],
    "max_dd": [0.15, 0.12, 0.18],
    "trade_count": [150, 180, 140],
    "total_pnl": [2500.0, 3200.0, 2100.0],
})
```

**Sorting**: Results are sorted by the objective metric (typically Sharpe ratio) in descending order, so the best parameters appear first.

### 2.6 Strategy Config Contract

The strategy config contract defines the YAML structure for strategy configuration. This is the primary configuration interface for strategies.

**Schema**:

```yaml
name: "strategy_identifier" # Required: unique strategy name
description: "Brief description" # Required: human-readable description
pairs: ["BTC/USDT:USDT"] # Required: trading pairs
timeframe: "15m" # Required: candle timeframe

params: # Required: current parameter values
  param1: 20
  param2: 50

param_space: # Required: optimization ranges
  param1: [10, 15, 20, 25, 30]
  param2: [40, 50, 60]

position_sizing: # Required: risk management
  mode: "percent_equity" # "fixed_amount" or "percent_equity"
  percent_equity: 0.10 # Fraction of equity per trade
  max_position_value: 50000 # Maximum position size

optimization: # Optional: optimization settings
  method: "grid" # "grid" or "random"
  objective: "sharpe" # Metric to optimize

validation: # Optional: validation settings
  walk_forward: # Walk-forward analysis (duration-based)
    train_days: 548 # Training window duration (~1.5 years for crypto)
    test_days: 90 # Test window duration (quarterly re-optimization)
    holdout_pct: 0.10 # 10% True Holdout for final verification
    min_trades_per_window: 30
    param_selection_method: "auto" # auto, frequency, kmeans, island_volume

  # Monte Carlo stress test settings
  monte_carlo:
    n_simulations: 10000 # Statistical significance threshold
    skip_pct: 0.10 # 10% execution failure simulation
    slippage_pip: 1.0 # 1 pip slippage stress
    entry_delay_bars: 1 # 1-bar execution delay
    pip_value: 0.0001
    max_dd_increase_pct: 25.0 # Robustness threshold

  # 8-Metric Standard validation thresholds
  eight_metric_standard:
    wfe_min_pct: 60.0
    cv_max_pct: 20.0
    sharpe_min: 1.5
    recovery_factor_min: 2.0
    profit_factor_min: 1.5
    max_drawdown_max_pct: 25.0
    win_rate_min_pct: 45.0
    expectancy_min: 0.0

order_management: # Optional: order management
  entry:
    type: "market"
  take_profit:
    type: "atr_multiple"
    value: 2.0
  stop_loss:
    type: "atr_multiple"
    value: 1.0
```

**Validation**: The framework validates config structure and values at startup. Missing required fields raise errors. Invalid values (e.g., negative capital) are caught during backtest initialization.

### 2.7 Order Manager Contracts

The order manager handles the complete lifecycle of trading orders including entry, take-profit, and stop-loss execution. These contracts define the interfaces between order management, backtesting, and strategy components.

#### 2.7.1 Signal to Order Manager Contract

**Source**: Strategies (with order management enabled) produce, `engine/order_manager.py` consumes

**Purpose**: Convert strategy signals with optional order parameters into structured orders for execution

**Schema**:

| Column      | Data Type             | Description                                       |
| ----------- | --------------------- | ------------------------------------------------- |
| timestamp   | pl.Datetime           | Candle timestamp from OHLCV data                  |
| signal      | pl.Int8               | Position signal: 1=long, -1=short, 0=flat         |
| entry_price | pl.Float64 (optional) | Limit entry price, null for market orders         |
| tp_price    | pl.Float64 (optional) | Take profit price level, null if disabled         |
| sl_price    | pl.Float64 (optional) | Stop loss price level, null if disabled           |
| size        | pl.Float64 (optional) | Position size override, null for auto-calculation |

**Order Types**:

- **Market Entry**: `entry_price` is null. Entry executes at next bar's open price.
- **Limit Entry**: `entry_price` is specified. Order fills when price touches or crosses the level.
- **Stop Entry**: `entry_price` below current price for long (above for short). Triggers market entry.

**Example**:

```python
# Signal with limit entry and TP/SL orders
signals = df.with_columns([
    pl.lit(1, dtype=pl.Int8).alias("signal"),
    pl.lit(50100.0).alias("entry_price"),  # Limit buy at 50100
    pl.lit(50500.0).alias("tp_price"),     # Take profit at 50500
    pl.lit(49800.0).alias("sl_price"),     # Stop loss at 49800
])
```

#### 2.7.2 Order Manager Output Contract

**Source**: `engine/order_manager.py` produces, `engine/backtester.py` consumes

**Purpose**: Enhanced trade log with complete order execution details

**Schema**:

| Column              | Data Type   | Description                                   |
| ------------------- | ----------- | --------------------------------------------- |
| entry_time          | pl.Datetime | Timestamp when entry order was filled         |
| exit_time           | pl.Datetime | Timestamp when exit order was filled          |
| entry_price         | pl.Float64  | Actual fill price for entry                   |
| exit_price          | pl.Float64  | Actual fill price for exit                    |
| direction           | pl.Int8     | Trade direction: 1=long, -1=short             |
| pnl                 | pl.Float64  | Net profit/loss in quote currency             |
| pnl_pct             | pl.Float64  | Net PnL as percentage of entry value          |
| size                | pl.Float64  | Position size in base currency                |
| entry_type          | pl.Utf8     | Execution type: market, limit, stop           |
| exit_reason         | pl.Utf8     | Exit trigger: signal, tp, sl, circuit_breaker |
| tp_price            | pl.Float64  | Original take profit price level              |
| sl_price            | pl.Float64  | Original stop loss price level                |
| max_price           | pl.Float64  | Highest price reached (for trailing SL)       |
| min_price           | pl.Float64  | Lowest price reached (for trailing SL)        |
| bars_held           | pl.Int64    | Number of bars position was active            |
| entry_fee           | pl.Float64  | Fee paid on entry execution                   |
| exit_fee            | pl.Float64  | Fee paid on exit execution                    |
| total_fees          | pl.Float64  | Combined entry and exit fees                  |
| filled_entry_orders | pl.List     | List of filled entry orders with prices/sizes |
| filled_exit_orders  | pl.List     | List of filled exit orders with prices/sizes  |

**Exit Reasons**:

- **signal**: Strategy generated exit signal (position closed)
- **tp**: Take profit order was hit
- **sl**: Stop loss order was hit
- **circuit_breaker**: Max drawdown threshold exceeded

**Example**:

```python
# Order manager output with complete execution details
trade_log = pl.DataFrame({
    "entry_time": [datetime(2024, 1, 1, 0, 0)],
    "exit_time": [datetime(2024, 1, 1, 2, 0)],
    "entry_price": [50000.0],
    "exit_price": [49800.0],
    "direction": [1],
    "pnl": [-195.5],  # After all fees
    "pnl_pct": [-0.39],
    "size": [0.1],
    "entry_type": ["limit"],
    "exit_reason": ["sl"],
    "tp_price": [50500.0],
    "sl_price": [49800.0],
    "max_price": [50450.0],
    "min_price": [49700.0],
    "bars_held": [8],
    "entry_fee": [2.75],
    "exit_fee": [2.74],
    "total_fees": [5.49],
})
```

#### 2.7.3 Order Manager Position Events Contract

**Source**: `engine/order_manager.py` produces, `engine/backtester.py` consumes

**Purpose**: Real-time position state updates for backtester synchronization

**Event Types**:

| Event             | Description                               |
| ----------------- | ----------------------------------------- |
| position_opened   | New position initiated                    |
| position_modified | Existing position modified (TP/SL update) |
| position_closed   | Position fully closed                     |
| order_filled      | Individual order executed                 |
| order_cancelled   | Pending order cancelled                   |

**Schema**:

| Column         | Data Type             | Description                          |
| -------------- | --------------------- | ------------------------------------ |
| timestamp      | pl.Datetime           | Event timestamp                      |
| event_type     | pl.Utf8               | Type of position event               |
| position_id    | pl.Utf8               | Unique identifier for position       |
| direction      | pl.Int8               | Position direction: 1=long, -1=short |
| size           | pl.Float64            | Current position size                |
| entry_price    | pl.Float64            | Average entry price                  |
| tp_price       | pl.Float64 (optional) | Current take profit level            |
| sl_price       | pl.Float64 (optional) | Current stop loss level              |
| unrealized_pnl | pl.Float64            | Current unrealized PnL               |
| order_id       | pl.Utf8 (optional)    | Related order ID for order events    |

**Example**:

```python
# Position event stream
events = pl.DataFrame({
    "timestamp": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 15)],
    "event_type": ["position_opened", "order_filled"],
    "position_id": ["pos_001", "pos_001"],
    "direction": [1, 1],
    "size": [0.1, 0.1],
    "entry_price": [50000.0, 50000.0],
    "tp_price": [50500.0, 50500.0],
    "sl_price": [49800.0, 49800.0],
    "unrealized_pnl": [0.0, 25.0],
    "order_id": [None, "order_entry_001"],
})
```

## 3. Module Responsibilities

Each module has a clearly defined responsibility. Understanding these boundaries helps identify which module to modify when adding features or fixing bugs.

### 3.1 data/fetcher.py

**Responsibility**: Retrieve OHLCV data from Bybit exchange via direct API calls.

This module handles all interaction with the external data source. It implements rate limiting to respect exchange limits, pagination to fetch large datasets in chunks, automatic gap detection to ensure complete data coverage, and intelligent duplicate handling. The fetcher validates data quality before passing to storage, ensuring no corrupted or malformed data enters the system.

**Key Features**:

- **Automatic Gap Detection**: Scans existing data for missing periods and fetches only what's missing
- **Complete Coverage**: Ensures data from earliest available (2018 or configured start) to current time
- **Duplicate Handling**: Gracefully handles duplicate timestamps returned by the API (deduplicates with warning)
- **Timeframe-Aware Buffers**: Adjusts buffer before current time based on timeframe to avoid incomplete candles
- **Incremental Updates**: On subsequent runs, fetches only new data since last stored timestamp

**Key Functions**:

- `fetch_ohlcv(symbol, timeframe, fetch_from, storage_path, config, from_date, to_date)`: Fetch data for a single symbol with gap detection
- `fetch_multiple(symbols, timeframe, fetch_from, storage_path, config)`: Fetch data for multiple symbols
- `_detect_missing_periods(df_existing, from_date, to_date, timeframe_ms)`: Detect gaps in existing data
- `_get_timeframe_ms(timeframe)`: Convert timeframe string to milliseconds
- `_get_buffer_time_ms(timeframe)`: Get timeframe-aware buffer for avoiding incomplete candles
- `_validate_and_deduplicate_ohlcv_data(df, symbol)`: Validate data and remove duplicates

**Data Completeness**:

- Target completeness: 99.6%+ (typical for free cryptocurrency data sources)
- Small gaps (<1%) are normal due to exchange maintenance, API limits, or low liquidity periods
- Gaps are automatically filled on next fetch run
- Use `data/inspect.py` to validate data quality and completeness

**Output**: OHLCV DataFrame matching the OHLCV contract

**Side Effects**: Creates or updates Parquet files in the storage directory

### 3.2 data/storage.py

**Responsibility**: Persist and retrieve OHLCV data in Parquet format.

This module provides efficient binary storage for time-series data. Parquet provides columnar compression that reduces storage size and speeds up subsequent reads. The storage module handles deduplication (by timestamp) when appending new data, ensuring no duplicate candles exist in storage.

**Key Functions**:

- `save_ohlcv(symbol, timeframe, df, storage_path)`: Save OHLCV data to Parquet
- `load_ohlcv(symbol, timeframe, storage_path)`: Load OHLCV data from Parquet
- `get_latest_timestamp(symbol, timeframe, storage_path)`: Get the most recent timestamp

**Input**: OHLCV DataFrame (from fetcher)
**Output**: OHLCV DataFrame (to backtester/optimizer)

### 3.3 data/inspect.py

**Responsibility**: Provide data quality inspection and validation tools.

This module offers utilities to inspect Parquet files, validate data quality, and check completeness. It helps identify gaps, duplicates, and data quality issues before running backtests.

**Key Functions**:

- `inspect_parquet(symbol, timeframe, storage_path)`: Comprehensive data inspection with quality metrics
- `list_all_data(storage_path)`: List all available data files with summary statistics

**Usage**:

```bash
# Inspect specific symbol and timeframe
python data/inspect.py --symbol BTC/USDT:USDT --timeframe 15m

# List all available data files
python data/inspect.py --list
```

**Validation Checks**:

- Data completeness percentage (expected vs actual bars)
- Null value detection
- Duplicate timestamp detection
- Date range and duration analysis
- Price and volume statistics

### 3.4 engine/base_strategy.py

**Responsibility**: Define the abstract base class that all strategies must inherit from.

This module establishes the contract that all strategies must follow. It enforces implementation of the `generate_signals()` method and provides no default implementation—all signal logic must be provided by concrete strategy classes. The base class is intentionally minimal to avoid constraining strategy implementations.

**Key Classes**:

- `BaseStrategy`: Abstract base class

**Key Methods**:

- `__init__(params)`: Initialize strategy with parameters
- `generate_signals(data) -> pl.DataFrame`: Abstract method for signal generation

**Input**: OHLCV DataFrame
**Output**: Signal DataFrame (OHLCV + signal column)

### 3.5 engine/backtester.py

**Responsibility**: Simulate trading based on signals and compute trade logs and equity curves.

This module is the computational core of the framework. It takes strategy signals and simulates order execution, including slippage, fees, and position management. The backtester maintains position state across bars and generates a complete record of all trades along with a continuous equity curve. The module implements circuit breaker logic to halt trading if drawdown exceeds the configured threshold.

**Key Functions**:

- `run_backtest(signals, config, position_sizing_mode) -> Tuple[pl.DataFrame, pl.DataFrame]`: Run backtest simulation

**Input**: Signal DataFrame (OHLCV + signal column), config dict
**Output**: Tuple of (trade_log, equity_curve)

**Configuration Parameters**:

- `initial_capital`: Starting portfolio value (default: 10000)
- `taker_fee`: Trading fee percentage (default: 0.00055)
- `slippage_pct`: Slippage percentage (default: 0.0005)
- `max_drawdown_pct`: Circuit breaker threshold (default: 0.50)

### 3.6 engine/optimizer.py

**Responsibility**: Systematically search parameter space for optimal strategy parameters.

This module wraps the backtester and tests multiple parameter combinations. It supports both grid search (exhaustive testing of all combinations) and random search (random sampling). For large parameter spaces (>100 combinations), it uses multiprocessing to parallelize evaluation. The optimizer returns results sorted by the objective metric for easy selection of best parameters.

**Key Functions**:

- `optimize_params(strategy_class, param_space, train_data, method, objective, n_random, config) -> pl.DataFrame`: Optimize strategy parameters
- `select_params_by_island_volume(results, param_keys, min_profitable_count=3) -> dict`: Island Volume Selection for robust parameter choice
- `select_params_by_frequency(results, param_keys) -> dict`: Frequency-based parameter selection
- `select_params_by_kmeans(results, param_keys, n_clusters=3) -> dict`: K-Means clustering parameter selection

**Input**: Strategy class, parameter space dict, training OHLCV data, config
**Output**: Optimization result DataFrame

**Optimization Methods**:

- `grid`: Exhaustive search through all parameter combinations
- `random`: Random sampling of parameter combinations

**Parameter Selection Methods:**

- **island_volume** (recommended): Identifies broad parameter plateaus to avoid overfitting (institutional standard)
- **frequency**: Selects most commonly occurring optimal parameters
- **kmeans**: Clusters similar parameter combinations and selects cluster centroid
- **auto**: Prefers island_volume, falls back to kmeans or frequency based on data characteristics

### 3.7 engine/metrics.py

**Responsibility**: Calculate performance metrics from trade logs and equity curves.

This module provides pure computation functions for evaluating strategy performance. All functions use vectorized polars operations for optimal performance. The module includes risk-adjusted metrics (Sharpe, Sortino, Calmar), trade statistics (win rate, profit factor, expectancy), growth metrics (CAGR), and robustness metrics (Walk Forward Efficiency, Recovery Factor).

**Enhanced Metrics Functions:**

- `calculate_sharpe(equity_curve, risk_free_rate) -> float`: Annualized risk-adjusted returns using volatility
- `calculate_sortino(equity_curve, risk_free_rate) -> float`: Downside risk-adjusted returns (penalizes only downside volatility)
- `calculate_max_drawdown(equity_curve) -> Tuple[float, int]`: Maximum peak-to-trough decline and duration
- `calculate_win_rate(trades) -> float`: Percentage of profitable trades
- `calculate_profit_factor(trades) -> float`: Gross wins divided by gross losses
- `calculate_average_trade_pnl(trades) -> float`: Average profit/loss per trade
- `calculate_expectancy(trades) -> float`: Expected value per trade
- `calculate_cagr(equity_curve) -> float`: Compound annual growth rate
- `calculate_calmar_ratio(equity_curve) -> float`: CAGR adjusted for maximum drawdown
- `calculate_recovery_factor(trades, initial_capital) -> float`: Total profit relative to maximum drawdown
- `calculate_walk_forward_efficiency(is_equity, oos_equity) -> float`: Ratio of OOS to IS Sharpe ratio
- `stitch_equity_curves(equity_curves) -> pl.DataFrame`: Combine multiple equity curves chronologically

**Input**: Equity curve or trade log DataFrame
**Output**: Float metric values

### 3.8 engine/validator.py

**Responsibility**: Run walk-forward optimization, Monte Carlo simulation, and regime analysis.

This module implements the complete validation framework with duration-based walk-forward windows, parameter selection methods, and comprehensive robustness metrics. It provides rich console output with progress bars, tables, and color-coded pass/fail indicators.

**Key Functions**:

- `run_walk_forward(strategy_class, data, config) -> dict`: Duration-based walk-forward optimization with holdout
- `run_monte_carlo(trades, config) -> dict`: Monte Carlo simulation on OOS trades
- `_calculate_robustness_metrics(windows, param_keys) -> dict`: Calculate consistency, performance distribution, parameter stability
- `_analyze_regime(data, signals) -> dict`: Performance breakdown by market regime (trending up/down, sideways)

**Validation Features:**

- **Duration-based windows**: 548 days train, 90 days test (institutional standard for crypto)
- **True holdout**: 10% most recent data reserved for final verification
- **Stitched equity curve**: Chronological collation of all OOS windows for final metrics
- **8-Metric Standard**: All metrics must pass for validation approval
- **Robustness metrics**: Consistency, performance distribution, parameter stability across windows
- **Rich console output**: Progress bars with time estimates, rich tables, UTF-8 support for Windows

**Input**: Strategy class, OHLCV data, configuration dict
**Output**: Validation results dict with walk-forward windows, Monte Carlo distributions, regime analysis, robustness metrics

## 4. How to Safely Make Changes

Making changes to the framework requires careful consideration of downstream impacts. Follow these guidelines to ensure changes are safe and maintain backward compatibility.

### 4.1 Before Making Changes

Always verify the current behavior before modifying any code. Run existing tests or manual backtests to establish a baseline. Document the expected behavior and how your change will modify it. Identify all modules that depend on the component you're changing and consider how they will be affected.

Check the inter-module contracts to understand what inputs and outputs are expected. Any change that modifies these contracts requires updating all consuming modules. Consider whether the change is additive (adding new functionality without modifying existing contracts) or breaking (modifying existing contracts).

### 4.2 Modifying Data Contracts

Data contracts are the most critical part of the framework. Changes to contracts cascade through the entire system and can cause silent failures if not properly managed.

**Safe changes**: Adding new optional columns that don't affect existing processing is safe. The downstream modules simply ignore columns they don't recognize. However, be cautious about adding columns that should be required but are marked as optional—this creates implicit dependencies.

**Unsafe changes**: Renaming, removing, or changing the data type of existing columns breaks downstream modules. If you must make such changes, you must update all modules that produce or consume the affected data. Consider creating a migration path that supports both old and new formats during a transition period.

**When modifying contracts**:

1. Update the producing module to generate the new format
2. Update all consuming modules to handle the new format
3. If backward compatibility is needed, maintain code paths for both formats
4. Test all affected modules with realistic data

### 4.3 Modifying Module Behavior

When modifying internal module behavior (not contracts), follow these steps:

1. **Isolate the change**: Make the smallest possible modification to achieve your goal
2. **Test in isolation**: Create a minimal test case that exercises the changed behavior
3. **Verify downstream effects**: Run backtests and optimizations to ensure no regressions
4. **Update documentation**: Document any changes to behavior, configuration, or contracts

### 4.4 Adding New Dependencies

The framework has strict constraints on dependencies. Before adding a new dependency:

1. Check if existing dependencies can achieve the same goal
2. Consider the maintenance burden of adding another dependency
3. Verify the dependency is compatible with the existing tech stack (polars, no pandas)
4. Get approval before adding new dependencies (as per AGENTS.md constraints)

If you must add a dependency, update `requirements.txt` or `pyproject.toml` and document the rationale in the decision log.

### 4.5 Testing Changes

Always test changes thoroughly before considering them complete:

**⚠️ CRITICAL: Do NOT Run Full Validation During Feature Development**

When fixing bugs or adding features:

- **ONLY use `--mode backtest`** for quick validation
- **NEVER run `--mode full` or `--mode validate`** unless explicitly requested by the user
- These validation modes are **extremely time-intensive**:
  - Walk-forward optimization with multiple train/test windows
  - 10,000 Monte Carlo simulations
  - Can take **hours** to complete
- Only run full validation when:
  - User explicitly requests validation
  - Finalizing a complete strategy implementation
  - User asks to verify strategy robustness

**🧪 Quick Validation Mode:** For faster validation testing during development:

```bash
# Quick validation test (1 year, 180 train / 30 test windows)
python run.py --strategy ema_crossover_rsi --mode validate --quick-test

# Or manually override in config:
# validation.walk_forward.train_days: 180
# validation.walk_forward.test_days: 30
# validation.walk_forward.lookback_days: 365
```

**Always revert to production settings after testing:**

- Production: 548 train / 90 test / ~2 years lookback
- Quick test: 180 train / 30 test / 365 lookback

**Development Testing Workflow:**

1. **Unit tests**: Test the changed function in isolation with various inputs
2. **Quick integration test**: Run `python run.py --strategy <name> --mode backtest`
3. **Regression tests**: Compare backtest results before and after the change
4. **Edge case tests**: Test with empty data, single-row data, extreme values
5. **Full validation**: ONLY when user explicitly requests it

For any change that affects calculation logic, maintain a known dataset and expected results to verify the change produces correct output.

## 5. Adding New Metrics

The metrics module provides performance analysis functions. Adding new metrics follows a consistent pattern.

### 5.1 Metric Function Pattern

All metric functions follow the same pattern for consistency:

```python
def calculate_new_metric(input_data: pl.DataFrame, ...) -> float:
    """Calculate the new metric.

    Args:
        input_data: DataFrame with required columns
        ...: Additional parameters

    Returns:
        Metric value as float

    Raises:
        ValueError: If input data is invalid or insufficient
    """
    # Validate input
    if input_data.height == 0:
        raise ValueError("Input data cannot be empty")

    if "required_column" not in input_data.columns:
        raise ValueError("Input data must have 'required_column' column")

    # Calculate metric using vectorized operations
    result = input_data["required_column"].some_operation()

    return float(result)
```

### 5.2 Steps to Add a New Metric

To add a new performance metric to `engine/metrics.py`:

1. **Implement the function**: Follow the pattern above with proper validation and vectorized operations
2. **Add docstring**: Document the formula, inputs, outputs, and any edge cases
3. **Update imports**: If the function needs new imports, add them at the top of the file
4. **Add helper functions**: For complex calculations, add helper functions with `_` prefix for internal use
5. **Test the metric**: Verify correct output with known inputs

### 5.3 Metric Implementation Guidelines

**Use vectorized operations**: Always use polars expressions rather than iterating through rows. This is critical for performance when processing large datasets.

**Handle edge cases**: Consider what happens with empty data, single data points, division by zero, and other edge cases. Raise `ValueError` with descriptive messages for invalid inputs.

**Return float values**: All metrics should return Python float values for consistency with other metrics and for easy use in sorting and comparison.

**Use helper functions**: For complex calculations, extract common operations into helper functions like `_safe_float()` that already exists in the metrics module.

### 5.4 Example: Adding a Simple Metric

Here's how to add a simple metric like `calculate_total_return`:

```python
def calculate_total_return(equity_curve: pl.DataFrame) -> float:
    """Calculate total return percentage from equity curve.

    Formula: (ending_equity - starting_equity) / starting_equity * 100

    Args:
        equity_curve: DataFrame with 'equity' column (f64)

    Returns:
        Total return as percentage (e.g., 25.0 means 25% gain)

    Raises:
        ValueError: If equity curve is empty or has no equity column
    """
    if equity_curve.height == 0:
        raise ValueError("Equity curve cannot be empty")

    if "equity" not in equity_curve.columns:
        raise ValueError("Equity curve must have 'equity' column")

    start_equity = equity_curve["equity"][0]
    end_equity = equity_curve["equity"][-1]

    if start_equity <= 0:
        raise ValueError("Starting equity must be positive")

    return ((end_equity - start_equity) / start_equity) * 100.0
```

## 6. Modifying Validation Logic

Validation logic determines whether a strategy meets minimum quality standards. The framework implements the 8-Metric Standard for institutional-grade validation. Changes to validation can significantly impact which strategies are accepted.

### 6.1 Validation Threshold Locations

Validation thresholds are defined in multiple locations:

| Location                    | Purpose                                                              |
| --------------------------- | -------------------------------------------------------------------- |
| `strategies/*/config.yaml`  | Strategy-specific validation settings (8-Metric Standard thresholds) |
| `config/global_config.yaml` | Global validation defaults                                           |
| `engine/validator.py`       | Validation logic and robustness metrics calculation                  |
| `engine/metrics.py`         | Metric calculation functions                                         |

### 6.2 The 8-Metric Standard

All strategies must pass ALL 8 metrics simultaneously for validation approval:

1. **Walk Forward Efficiency (WFE)** ≥ 60%
   - Measures IS→OOS performance translation
   - Values <50% indicate overfitting
   - Function: `calculate_walk_forward_efficiency(is_equity, oos_equity)`

2. **Coefficient of Variation (CV)** ≤ 20%
   - Performance consistency across walk-forward windows
   - Lower = more consistent performance
   - Calculated in `_calculate_robustness_metrics()`

3. **Sharpe Ratio** ≥ 1.5
   - Annualized risk-adjusted returns using volatility
   - Function: `calculate_sharpe(equity_curve, risk_free_rate)`

4. **Recovery Factor** ≥ 2.0
   - Total profit relative to maximum drawdown
   - Higher = better recovery from losses
   - Function: `calculate_recovery_factor(trades, initial_capital)`

5. **Profit Factor** ≥ 1.5
   - Gross wins divided by gross losses
   - Values >1.0 indicate profitable trading
   - Function: `calculate_profit_factor(trades)`

6. **Max Drawdown** ≤ 25%
   - Largest peak-to-trough equity decline
   - Lower = better capital preservation
   - Function: `calculate_max_drawdown(equity_curve)`

7. **Win Rate** ≥ 45%
   - Percentage of profitable trades
   - Higher = more frequent wins
   - Function: `calculate_win_rate(trades)`

8. **Expectancy** ≥ $0
   - Average profit/loss per trade
   - Positive = profitable on a per-trade basis
   - Function: `calculate_expectancy(trades)`

### 6.3 Modifying Global Thresholds

To modify validation thresholds, update the `validation.eight_metric_standard` section in the strategy's `config.yaml`:

```yaml
validation:
  eight_metric_standard:
    wfe_min_pct: 60.0 # Walk Forward Efficiency minimum
    cv_max_pct: 20.0 # Coefficient of Variation maximum
    sharpe_min: 1.5 # Sharpe Ratio minimum
    recovery_factor_min: 2.0 # Recovery Factor minimum
    profit_factor_min: 1.5 # Profit Factor minimum
    max_drawdown_max_pct: 25.0 # Max Drawdown maximum
    win_rate_min_pct: 45.0 # Win Rate minimum
    expectancy_min: 0.0 # Expectancy minimum
```

**Caution**: Thresholds in `config/global_config.yaml` affect all strategies. Before modifying:

1. Understand the impact: Lower thresholds accept more strategies, higher thresholds reject more
2. Test with existing strategies: Run validation before and after to see impact
3. Consider the market: Thresholds appropriate for crypto may not work for other assets
4. Document the change: Record the rationale for the new threshold

### 6.4 Modifying Walk-Forward Validation

Walk-forward validation uses duration-based windows consistent with institutional practices:

| Parameter                | Description                                 | Impact                                                         |
| ------------------------ | ------------------------------------------- | -------------------------------------------------------------- |
| `train_days`             | Training window duration in days            | Higher = more data for optimization (default 548)              |
| `test_days`              | Test window duration in days                | Higher = more reliable OOS testing (default 90)                |
| `holdout_pct`            | Fraction of data reserved for final holdout | Higher = more stringent verification (default 0.10)            |
| `min_trades_per_window`  | Minimum trades per window                   | Higher = more statistical significance                         |
| `param_selection_method` | Parameter selection algorithm               | auto (prefers island_volume), frequency, kmeans, island_volume |

**Auto Selection Priority:** When `param_selection_method: "auto"`, the system:

1. **Prefers Island Volume Selection** - identifies parameter plateaus to avoid overfitting
2. Falls back to K-Means if strong performance differentiation between clusters
3. Falls back to Frequency-based selection as final default

**Island Volume Selection (IVS)** is the institutional standard because it:

- Identifies broad parameter plateaus rather than sharp peaks
- Selects the center of the largest contiguous "island" of profitability
- Prioritizes robust, generalizable parameters over curve-fitted spikes

To modify walk-forward parameters, update the `validation.walk_forward` section in the strategy's `config.yaml`.

### Walk Forward Efficiency (WFE) Optimization

> **⚠️ IMPORTANT:** As of 2026-02, the framework now defaults to **Island Volume Selection (IVS)** for window-level parameter selection. This change improves WFE from ~48% to 73.7% on average. Existing strategies will automatically benefit from this default.

The framework supports **four window-level parameter selection methods** to maximize WFE:

#### 1. Island Volume Selection (DEFAULT - Recommended)

- Identifies broad parameter plateaus instead of isolated peaks
- Selects center of largest "island" of profitable parameters
- Avoids overfitting by prioritizing robust regions
- **Recommended for most strategies**
- Config: `window_selection_method: "ivs"`
  - `ivs.min_threshold`: Min Sharpe to consider (0 = dynamic top 80%)

#### 2. Best (Legacy)

- Selects parameters with highest in-sample Sharpe for each window
- Fastest method
- Backward compatible with existing strategies
- May overfit to IS noise
- Not recommended for production use

#### 3. Stability-Based Selection

- Tests top N parameters against ±10% perturbations
- Selects params with lowest performance degradation
- Uses `_check_flat_region()` for robustness testing
- Config: `window_selection_method: "stability"`
  - `stability.top_n`: Number of params to test (default: 5)
  - `stability.perturbation_pct`: Perturbation percentage (default: 0.1)

#### 4. Multi-Objective Optimization

- Combines Sharpe ratio with stability score
- Formula: `(1 - weight) × Sharpe_normalized + weight × Stability`
- Config: `window_selection_method: "multi_objective"`
  - `multi_objective.stability_weight`: Stability importance (0-1, default: 0.3)

#### Configuration Example

```yaml
validation:
  walk_forward:
    window_selection_method: "ivs" # Options: best, ivs, stability, multi_objective

    ivs:
      min_threshold: 0.0 # Dynamic threshold if 0

    stability:
      top_n: 5
      perturbation_pct: 0.1

    multi_objective:
      stability_weight: 0.3
```

#### Performance Impact

| Method            | WFE       | Status             |
| ----------------- | --------- | ------------------ |
| Best (legacy)     | 48%       | Not recommended    |
| **IVS (default)** | **73.7%** | **✅ Recommended** |

Real-world results (EMA Crossover strategy, 2-year test):

- **Before (Best method)**: WFE = 48%
- **After (IVS method)**: WFE = 73.7%
- **Improvement**: +53.5%

All other metrics improved:

- 100% profitable windows (was lower)
- Zero unstable parameters
- Perfect flat region score (1.00)

#### Method Selection Guide

| Use Case                      | Recommended Method                 |
| ----------------------------- | ---------------------------------- |
| **Most strategies (default)** | **`ivs`** (already set as default) |
| Fast testing / legacy         | `best`                             |
| High parameter sensitivity    | `stability`                        |
| Balanced approach             | `multi_objective`                  |

### 6.4 Modifying Monte Carlo Settings

Monte Carlo simulation tests strategy robustness by randomizing trade order:

| Parameter          | Description           | Impact                                         |
| ------------------ | --------------------- | ---------------------------------------------- |
| `n_simulations`    | Number of simulations | More = more reliable results, slower execution |
| `confidence_level` | Confidence threshold  | Higher = stricter acceptance criteria          |

To modify Monte Carlo settings, update the `validation.monte_carlo` section in the strategy's `config.yaml`.

### 6.5 Adding New Validation Tests

To add a new validation test:

1. Identify where the test belongs (in the validator module or run.py)
2. Implement the test function with clear pass/fail criteria
3. Add the test to the validation pipeline in the appropriate location
4. Add configuration parameters if the test has tunable settings
5. Update this documentation with the new test description

## 7. Common Pitfalls

This section documents common mistakes and how to avoid them. These pitfalls have been encountered in practice and are listed here to help you avoid repeating them.

### 7.1 Using Pandas Instead of Polars

**Pitfall**: Using pandas for data manipulation instead of polars.

**Symptoms**: Code runs correctly on small datasets but becomes extremely slow with larger datasets. Memory usage grows unbounded. Results may differ between runs due to unpredictable iteration order.

**Cause**: Pandas uses row-oriented iteration by default and doesn't optimize operations automatically. Polars uses columnar storage with lazy evaluation, enabling aggressive optimization and parallelization.

**Solution**: Never import or use pandas. Use polars exclusively. Translate any pandas code you find to polars equivalents using the patterns shown in the template strategy.

**Example of wrong code**:

```python
import pandas as pd  # WRONG
df = pd.DataFrame(...)
for i in range(len(df)):  # WRONG: row iteration
    df.loc[i, 'new_col'] = ...
```

**Correct code**:

```python
import polars as pl  # CORRECT
df = pl.DataFrame(...)
df = df.with_columns([
    pl.col('close').diff().alias('close_diff')  # CORRECT: vectorized
])
```

### 7.2 Row Iteration in Backtesting

**Pitfall**: Using row iteration instead of vectorized operations in the backtester or strategy.

**Symptoms**: Backtests take an unreasonably long time. CPU usage is high but throughput is low. Memory usage is unpredictable.

**Cause**: Row iteration in Python is extremely slow compared to vectorized operations. A backtest that processes 100,000 bars using row iteration can take minutes or hours, while a vectorized version takes seconds.

**Solution**: The backtester currently uses row iteration for position state management (see backtester.py lines 69-168). This is a known limitation. When adding new backtesting logic, use numpy arrays for state management and avoid Python loops over rows.

**Current workaround**: The backtester uses numpy arrays for equity, drawdown, and trade tracking, which provides acceptable performance. However, position state is tracked with Python loops. Future improvements should aim to eliminate this iteration.

### 7.3 Breaking Data Contracts

**Pitfall**: Modifying function signatures or DataFrame schemas without updating all consuming modules.

**Symptoms**: `ValueError` exceptions about missing columns. Incorrect or missing data in outputs. Silent failures where wrong values are computed but no error is raised.

**Cause**: Modules depend on specific input and output formats. When one module changes its output format without updating consumers, those consumers receive unexpected data.

**Solution**: Before modifying any function signature or DataFrame schema:

1. Identify all modules that produce and consume the affected data
2. Update all producing modules to generate the new format
3. Update all consuming modules to handle the new format
4. Test the full pipeline to ensure all modules work together

**Example**: If you rename the `signal` column to `position_signal` in the strategy output, you must update the backtester to expect `position_signal` instead of `signal`.

### 7.4 Hardcoding Values

**Pitfall**: Hardcoding parameter values, thresholds, or magic numbers in strategy or engine code.

**Symptoms**: Parameters cannot be optimized. Different strategies use inconsistent settings. Changing a value requires modifying code.

**Cause**: Hardcoded values are invisible to the configuration system and cannot be systematically varied during optimization.

**Solution**: All tunable values must be defined in YAML configuration files and accessed via the config system. This includes:

- Strategy parameters (moving average periods, indicator thresholds)
- Backtest settings (initial capital, fees, slippage)
- Validation thresholds (minimum Sharpe, minimum trades)

**Example of wrong code**:

```python
ema_period = 20  # WRONG: hardcoded
rsi_threshold = 70  # WRONG: hardcoded
```

**Correct code**:

```python
ema_period = self.params.get("ema_short", 20)  # From config
rsi_threshold = self.params.get("rsi_overbought", 70)  # From config
```

### 7.5 Non-Deterministic Behavior

**Pitfall**: Using random numbers, system time, or external data sources that vary between runs.

**Symptoms**: Backtest results differ between runs with the same parameters. Optimization results are not reproducible. Debugging is extremely difficult because behavior is unpredictable.

**Cause**: Systematic trading requires reproducibility. If the same input produces different outputs, results cannot be trusted.

**Solution**: Never use randomness without seeding. Never use system time for calculations. Never fetch external data during backtesting (only use historical data from storage).

**Example of wrong code**:

```python
import random
value = random.random()  # WRONG: non-deterministic
```

**Correct code**:

```python
import random
random.seed(42)  # Set seed at start of backtest
value = random.random()  # Now deterministic
```

### 7.6 State Management in Strategies

**Pitfall**: Strategies maintaining position state or other internal state between calls to `generate_signals()`.

**Symptoms**: Backtest results depend on execution order. Optimization produces inconsistent results. The same parameters produce different signals on different runs.

**Cause**: Strategies should be stateless functions. The same input data with the same parameters should always produce the same signals. State in strategies breaks this property.

**Solution**: Strategies should compute signals based solely on the current input data. If information from previous bars is needed, carry it forward in the DataFrame rather than storing it in the strategy instance.

**Example of wrong code**:

```python
class MyStrategy(BaseStrategy):
    def __init__(self, params):
        self.current_position = 0  # WRONG: stateful

    def generate_signals(self, data):
        if self.current_position == 0:  # WRONG: depends on state
            ...
```

**Correct code**:

```python
class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # All logic based on data columns only
        df = data.with_columns([
            pl.col('close').shift(1).alias('prev_close')
        ])
        ...
```

### 7.7 Incomplete Error Handling

**Pitfall**: Not handling edge cases or providing unclear error messages.

**Symptoms**: Cryptic error messages that don't indicate the root cause. Crashes on edge cases that should be handled gracefully. Silent failures where wrong values are used.

**Cause**: Error handling is added after bugs are discovered rather than being designed in from the start.

**Solution**: Always validate inputs and provide descriptive error messages. Handle edge cases explicitly. Use try/except blocks only when recovery is possible; otherwise, let errors propagate with clear messages.

**Example of inadequate error handling**:

```python
def calculate_sharpe(equity_curve):
    equity = equity_curve["equity"]  # CRYPIC error if column missing
    returns = equity.pct_change()
    ...
```

**Better error handling**:

```python
def calculate_sharpe(equity_curve, risk_free_rate=0.0):
    if equity_curve.height < 2:
        raise ValueError("Equity curve must have at least 2 data points")

    if "equity" not in equity_curve.columns:
        raise ValueError("Equity curve must have 'equity' column")

    equity = equity_curve["equity"]
    ...
```

### 7.8 Memory Leaks in Loops

**Pitfall**: Allocating new arrays or DataFrames in loops without cleanup.

**Symptoms**: Memory usage grows continuously during backtests. Backtests slow down over time. Eventually, the process runs out of memory.

**Cause**: Each loop iteration allocates new memory that isn't released. Over thousands of iterations, this accumulates.

**Solution**: Pre-allocate arrays when possible. Use in-place operations. Clear references to large objects when they're no longer needed.

**In the backtester**, arrays are pre-allocated:

```python
equity = np.zeros(n_bars)  # Pre-allocated
```

**When adding new loops**, pre-allocate outputs and use in-place operations where possible.

### 7.9 Ignoring Data Quality Issues

**Pitfall**: Processing data without validating quality first.

**Symptoms**: Incorrect calculations due to NaN values. Duplicate timestamps causing unexpected behavior. Data not in chronological order causing look-ahead bias.

**Cause**: Data from exchanges may have gaps, duplicates, or ordering issues.

**Solution**: Always validate data quality before processing. The fetcher module includes validation that should catch most issues. When loading data for backtesting, consider adding additional validation if the data source is untrusted.

**The fetcher validates**:

- Required columns exist
- No null values in price/volume columns
- No duplicate timestamps
- Chronological order

If you modify data loading or add new data sources, ensure equivalent validation is performed.

### 7.10 Modifying Contracts Without Documentation

**Pitfall**: Changing data contracts or API signatures without updating documentation.

**Symptoms**: Other developers don't know about the change. Old code breaks silently. Integration issues that are hard to debug.

**Cause**: Documentation is updated after development is complete, or not at all.

**Solution**: Update documentation as part of the change. This includes:

- Updating this FRAMEWORK.md file for contract changes
- Updating docstrings for API changes
- Adding notes to CHANGELOG or migration guide

Any change that affects how other modules interact with your code should be documented before the change is merged.

## 8. Advanced Validation Framework (IMPROVEMENT.md)

The framework implements institutional-grade validation methodology based on Walk Forward Optimization (WFO), parameter plateau analysis, and comprehensive stress testing.

### 8.1 Duration-Based Walk Forward Optimization

**Function**: `run_walk_forward()` in `engine/validator.py`

**Key Features**:

- **Duration-based windows**: 548 days train, 90 days test (per IMPROVEMENT.md institutional standards)
- **Holdout reservation**: 10% most recent data reserved for final validation
- **Stitched equity curve**: Chronological collation of all OOS windows for final metrics
- **Parameter plateau selection**: Island Volume algorithm for robust parameter selection

**Configuration** (`config/global_config.yaml`):

```yaml
validation:
  walk_forward:
    train_days: 548 # ~1.5 years of crypto data for regime capture
    test_days: 90 # Quarterly re-optimization cycle
    holdout_pct: 0.10 # 10% True Holdout for final verification
    param_selection_method: "auto" # auto, frequency, kmeans, island_volume
```

**WFE (Walk Forward Efficiency)**: Measures how well IS performance translates to OOS

- `calculate_walk_forward_efficiency(is_equity, oos_equity)` in `engine/metrics.py`
- **Threshold**: > 60% indicates genuine edge, < 50% suggests overfitting

### 8.2 Island Volume Selection

**Function**: `select_params_by_island_volume()` in `engine/optimizer.py`

**Algorithm**:

1. Filter profitable parameter combinations (Sharpe > threshold)
2. Build adjacency graph (neighbors differ by 1 parameter step)
3. Find connected components ("islands" of profitability)
4. Calculate volume = count × avg_objective for each island
5. Return center of largest volume island

**Benefits**:

- Avoids "isolated peaks" (single lucky parameters)
- Selects parameters from broad profit plateaus
- More robust to parameter drift and regime changes

### 8.3 Combined Monte Carlo Stress Test

**Function**: `run_combined_monte_carlo()` in `engine/validator.py`

**Stress Factors** (applied together in each simulation):

1. **Trade reshuffling**: Random order
2. **Trade skipping**: 10% random removal (execution failures)
3. **Slippage**: 1.0 pip per trade
4. **Entry delay**: 1-bar lag

**Default**: 10,000 iterations (vs. 1,000 legacy)

**Robustness Grading**:

- **PASS**: 95th percentile DD < 25% worse than original
- **MARGINAL**: DD increase 15-25%
- **FAIL**: DD increase > 25%

### 8.4 8-Metric Standard Validation

**Function**: `validate_8_metric_standard()` in `engine/validator.py`

**Metrics and Thresholds**:
| Metric | Threshold | Description |
|--------|-----------|-------------|
| WFE | > 60% | Walk Forward Efficiency |
| CV | < 20% | Coefficient of Variation |
| Sharpe | > 1.5 | Annualized Sharpe Ratio |
| Recovery Factor | > 2.0 | Net Profit / Max DD |
| Profit Factor | > 1.5 | Gross Profit / Gross Loss |
| Max Drawdown | < 25% | Maximum peak-to-trough decline |
| Win Rate | > 45% | Percentage of winning trades |
| Expected Value | > 0 | Average profit per trade |

**Verdict**:

- **PASS**: All 8 metrics pass
- **CONDITIONAL**: ≥ 75% pass (6/8)
- **FAIL**: < 75% pass

### 8.5 True Holdout Verification

**Function**: `run_holdout_validation()` in `engine/validator.py`

**Purpose**: Final validation on data never seen during optimization

**Process**:

1. Reserve 10% most recent data during WFO setup
2. Do NOT use in any optimization or testing
3. After all validation passes, run backtest on holdout
4. Compare holdout performance to OOS expectations

**Configuration**:

```yaml
validation:
  eight_metric_standard:
    wfe_min_pct: 60.0
    cv_max_pct: 20.0
    sharpe_min: 1.5
    recovery_factor_min: 2.0
    profit_factor_min: 1.5
    max_drawdown_max_pct: 25.0
    win_rate_min_pct: 45.0
    expectancy_min: 0.0
```

### 8.6 New Metrics Functions

**Added to `engine/metrics.py`**:

- `calculate_walk_forward_efficiency()`: WFE percentage
- `calculate_recovery_factor()`: Recovery Factor from trade log
- `stitch_equity_curves()`: Combine multiple OOS equity curves

### 8.7 Validation Orchestration

**Updated in `run.py`**:

1. Run duration-based WFO with holdout reservation
2. Run combined Monte Carlo stress test (10k iterations)
3. Validate against 8-Metric Standard
4. Run final holdout verification
5. Generate comprehensive report

**Exit Codes**:

- `0`: PASS (all validation levels pass)
- `1`: FAIL (any validation level fails)

## Summary

The systematic trading framework follows strict architectural principles that ensure reliability, performance, and maintainability. Key takeaways:

**Use polars exclusively**: Never use pandas or other data manipulation libraries.

**Maintain contracts**: Data contracts between modules must be preserved. Changes to contracts require updating all producing and consuming modules.

**Be stateless**: Strategies and validation logic should be pure functions without internal state.

**Configure everything**: All tunable values should be in configuration files, not hardcoded in code.

**Validate rigorously**: Validate inputs, handle edge cases, and provide clear error messages.

**Document changes**: Update documentation when modifying contracts, APIs, or behavior.

Following these guidelines ensures the framework remains stable and maintainable as it grows to support more strategies and use cases.
