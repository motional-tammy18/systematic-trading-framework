# Systematic Trading Framework

[![GitHub](https://img.shields.io/badge/github-dikibagast/systematic--trading--framework-blue)](https://github.com/dikibagast/systematic-trading-framework)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source backtesting framework for systematic trading strategies. Built for performance, accuracy, and institutional-grade validation through walk-forward optimization, Monte Carlo simulation, Island Volume Selection, and a Multi-Metric Standard.

This framework is designed for:

- **Quantitative researchers** developing and testing trading hypotheses with rigorous statistical validation
- **Systematic traders** seeking robust out-of-sample performance before committing capital
- **AI agents** automating strategy development through structured mode routing and deterministic code contracts
- **Developers** extending the framework with custom strategies, indicators, or validation methods

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Framework Pipeline](#framework-pipeline)
- [Validation Methodology](#validation-methodology)
- [Performance Optimizations](#performance-optimizations)
- [AI Agent Guide](#ai-agent-guide)
- [Developer Guide](#developer-guide)
- [Configurable and Customization](#configurable-and-customization)
- [Interactive Visualization Dashboard](#interactive-visualization-dashboard)
- [Adding a New Strategy](#adding-a-new-strategy)
- [Quick Reference](#quick-reference)
- [Documentation](#documentation)

---

## Overview

This framework provides a complete pipeline for developing, testing, and validating trading strategies. It handles the entire lifecycle from data acquisition through backtesting, optimization, and statistical validation. The framework is designed specifically for Bybit perpetual futures and implements rigorous out-of-sample testing to prevent overfitting.

```
Data Acquisition -> Signal Generation -> Backtest -> Walk-Forward -> Monte Carlo -> Multi-Metric Pass/Fail
```

### Core philosophy

The framework centers on robust validation. Rather than optimizing parameters on the full historical dataset and trusting those results, it uses walk-forward optimization to test whether optimized parameters generalize to unseen data across rolling windows. Combined with Monte Carlo simulation, Island Volume Selection, and a Multi-Metric Standard, this approach gives you confidence that a strategy has genuine edge rather than being curve-fitted to historical noise.

### What this framework provides

The framework delivers six core capabilities:

1. **Data acquisition**: Fetches OHLCV data from Bybit public endpoints with automatic incremental updates. Stores everything in efficient Parquet format for fast repeated access. No API keys or authentication required.

2. **Vectorized backtest engine**: Processes thousands of bars in seconds using columnar operations rather than row-by-row loops. Applies realistic commission models, slippage assumptions, and configurable bar-fill logic.

3. **Parameter optimization**: Performs systematic parameter searches using grid or random methods with multiprocessing support across all available CPU cores.

4. **Walk-forward validation**: Tests parameters on out-of-sample data across multiple time windows. Uses Island Volume Selection (IVS) to identify robust parameter plateaus rather than isolated performance peaks.

5. **Monte Carlo simulation**: Resamples trade sequences to assess strategy robustness under thousands of alternative market scenarios, revealing the full distribution of possible outcomes.

6. **Position sizing optimization**: Calculates optimal position sizes using Kelly Criterion to maximize long-term compounding while balancing risk and returns.

### Performance

The framework is built for speed. All backtesting and signal generation logic uses vectorized operations on Polars DataFrames rather than iterating over individual candles or rows. This means:

- A strategy that generates signals across 50,000+ candles completes in milliseconds, not minutes
- Parameter grid searches evaluate thousands of combinations in seconds
- The full validation pipeline (WFO + Monte Carlo) finishes in minutes to hours on commodity hardware, not days

The optimizer parallelizes across CPU cores, and data storage in Apache Parquet format enables fast columnar reads with compression.

### What this framework does not do

This is a backtesting-only framework. It does not connect to live exchanges for real trading, execute orders, or manage portfolio positions in real-time. All trading logic is simulated using historical data. The framework also supports only Bybit perpetual futures -- not spot trading, options, or other exchanges. If you need live trading capabilities, build a separate system that imports your strategy logic from this framework.

---

## Quick Start

### Create a Strategy by Describing It to an AI Agent

You don't need to write code from scratch. Describe your trading idea in plain language to an AI agent and it will generate the strategy files for you. The agent will:

1. Create the strategy directory with `config.yaml` and `signals.py`
2. Implement your entry and exit logic using the framework's signal contract
3. Fetch historical data and run a backtest
4. Iterate on parameters until validation passes

**Example prompt for an AI agent:**

> "Create a mean reversion strategy for BTC/USDT on a 15m timeframe. I want to enter a long position when the price drops below the lower Bollinger Band (period 20, std dev 2) AND RSI(14) is below 30. Exit when price returns above the middle band (SMA 20). For shorts: enter when price goes above upper band AND RSI above 70, exit when price falls below middle band. Use Half Kelly position sizing."

The agent will understand the framework conventions, write the signal logic in Polars, configure the parameter space for optimization, and run the validation pipeline. You review the results and iterate on the conversation.

---

### Step 1: Install Dependencies

```bash
git clone https://github.com/dikibagast/systematic-trading-framework.git
cd systematic-trading
pip install -r requirements.txt
```

### Step 2: Fetch Historical Data

```bash
python data/fetcher.py --symbol BTC/USDT:USDT --timeframe 15m
```

This fetches all available historical candle data from Bybit's public API and saves it to `data/raw/BTCUSDT_15m.parquet`. Subsequent runs perform incremental updates -- only new candles are fetched. The fetcher handles backward pagination for full history retrieval, gap detection, and automatic retry on transient SSL failures.

Available timeframes: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`

### Step 3: Run a Backtest

```bash
python run.py --strategy ema_crossover_rsi --mode backtest
```

Runs a single vectorized backtest using parameters defined in the strategy's `config.yaml`. The backtester simulates trade execution with configurable fees, slippage, and bar-fill assumptions. Output includes: total trades, win rate, Sharpe ratio, Sortino ratio, maximum drawdown, profit factor, CAGR, and equity curve summary.

### Step 4: Run Validation

```bash
python run.py --strategy ema_crossover_rsi --mode validate
```

Runs the full validation pipeline:

1. Walk-forward optimization across multiple rolling train/test windows
2. Island Volume Selection for robust parameter selection
3. Monte Carlo simulation with 10,000 resampling iterations
4. Multi-Metric Standard pass/fail assessment
5. Statistical reliability checks with diagnostic warnings

Outputs a structured validation report showing pass/fail status for every metric.

> **Resource warning:** Full validation (`--mode validate`) uses all available CPU cores by default to parallelize the optimization and simulation workload. If you have not limited `max_workers` in `config/global_config.yaml`, the process will consume 100% of CPU, which can make your system laggy or unresponsive during execution. To limit resource usage, set `max_workers` to a specific number (e.g., `max_workers: 2`) in your global config. For routine development, use `--mode backtest` for instant feedback. Reserve `--mode validate` for final verification when you are ready to assess a strategy thoroughly.

### Step 5: Optimize Position Sizing

```bash
python run.py --strategy ema_crossover_rsi --mode optimize_sizing
```

Calculates the optimal position size using Kelly Criterion based on the strategy's trade statistics. Apply the result to your strategy config:

```bash
python run.py --strategy ema_crossover_rsi --mode optimize_sizing --apply
```

---

## Framework Pipeline

The framework processes data through a series of well-defined stages. Each stage produces a specific contract output that feeds into the next.

### Stage 1: Data Acquisition

**Entry point:** `data/fetcher.py`

Historical OHLCV (Open, High, Low, Close, Volume) data is fetched from Bybit's public REST API. The fetcher employs backward pagination to retrieve the full available history for any symbol/timeframe combination, starting from the present and moving backward in time.

**Data contract:**

```python
# Polars DataFrame schema
{
    "timestamp": pl.Datetime,  # UTC timestamp of the candle
    "open":      pl.Float64,   # Opening price
    "high":      pl.Float64,   # Highest price
    "low":       pl.Float64,   # Lowest price
    "close":     pl.Float64,   # Closing price
    "volume":    pl.Float64,   # Trading volume
}
```

Data is stored in Apache Parquet format (`data/raw/{symbol}_{timeframe}.parquet`) for efficient columnar storage, compression, and fast read performance. Each subsequent fetch call performs incremental updates, only retrieving candles newer than the latest stored timestamp.

**Incremental update logic:**

1. Load existing data from Parquet file
2. Query the API for the latest candle timestamps
3. Calculate the gap between the last stored candle and now
4. Fetch only the missing candles in batches
5. Append, deduplicate, and sort before saving

### Stage 2: Signal Generation

**Entry point:** `strategies/{name}/signals.py`

Each strategy defines a class that inherits from `BaseStrategy` and implements `generate_signals(data: pl.DataFrame) -> pl.DataFrame`. This method receives raw OHLCV data and must return a DataFrame with a `signal` column:

- `1` = Long position
- `-1` = Short position
- `0` = Flat (no position)

**Signal contract:**

```python
{
    "timestamp": pl.Datetime,  # Passed through from input
    "open":      pl.Float64,   # Passed through from input
    "high":      pl.Float64,   # Passed through from input
    "low":       pl.Float64,   # Passed through from input
    "close":     pl.Float64,   # Passed through from input
    "volume":    pl.Float64,   # Passed through from input
    "signal":    pl.Int8,      # -1, 0, or 1
}
```

Strategies use Polars for all calculations -- the framework enforces vectorized operations over iterating row-by-row, critical for performance when processing hundreds of thousands of candles.

**Lookahead bias prevention:** Signals are automatically shifted forward by one bar before backtesting, ensuring the signal at bar `t` is based only on data available at bar `t-1`.

### Stage 3: Backtesting Engine

**Entry point:** `engine/backtester.py`

The backtester takes signals and simulates trade execution using vectorized operations. It works in three phases:

**Phase 1 -- Trade Detection:**
The engine identifies position changes (flat-to-long, flat-to-short, long-to-flat, short-to-flat) by comparing adjacent signal values. Entry and exit points are timestamped.

**Phase 2 -- Execution Simulation:**
Each trade is executed with configurable assumptions:

- **Fees**: Maker (0.02%) and taker (0.055%) fee rates
- **Slippage**: Configurable basis-point slippage per trade
- **Bar-fill assumption**: Pessimistic (enter at high/low), optimistic (enter at open), or random within the bar
- **Circuit breaker**: If drawdown exceeds the configured maximum, all positions are closed

**Phase 3 -- Equity Curve Construction:**
The engine tracks equity over time, accounting for:

- Entry and exit prices (with slippage)
- Fee deductions per trade
- Compounding position sizing
- Drawdown from peak equity

**Output contracts:**

```python
# Trade log -- one row per trade
{
    "entry_time":    pl.Datetime,
    "exit_time":     pl.Datetime,
    "direction":     pl.Int8,     # 1 = long, -1 = short
    "entry_price":   pl.Float64,
    "exit_price":    pl.Float64,
    "pnl":           pl.Float64,  # Gross profit/loss
    "pnl_pct":       pl.Float64,  # Percentage return
    "fees_paid":     pl.Float64,
    "bars_held":     pl.Int32,
}

# Equity curve -- one row per bar
{
    "timestamp":     pl.Datetime,
    "equity":        pl.Float64,
    "peak_equity":   pl.Float64,
    "drawdown_pct":  pl.Float64,
}
```

### Stage 4: Performance Metrics

**Entry point:** `engine/metrics.py`

The metrics module calculates the full set of performance indicators from the trade log and equity curve:

| Metric          | Formula                                              | What It Measures                       |
| --------------- | ---------------------------------------------------- | -------------------------------------- |
| Sharpe Ratio    | (mean(returns) - rf) / std(returns) \* sqrt(periods) | Return per unit of total risk          |
| Sortino Ratio   | (mean(returns) - rf) / downside_std \* sqrt(periods) | Return per unit of downside risk       |
| Calmar Ratio    | CAGR / Max Drawdown                                  | Return-to-drawdown efficiency          |
| CAGR            | (end_equity / start_equity)^(1/years) - 1            | Annualized growth rate                 |
| Max Drawdown    | max(peak - trough) / peak                            | Worst peak-to-trough decline           |
| Recovery Factor | total_profit / max_drawdown                          | How well strategy recovers             |
| Profit Factor   | gross_wins / gross_losses                            | Win-to-loss magnitude ratio            |
| Win Rate        | winning_trades / total_trades                        | Percentage of profitable trades        |
| Expectancy      | mean(pnl_per_trade)                                  | Average profit per trade               |
| WFE             | mean(OOS_sharpe) / mean(IS_sharpe)                   | In-sample to out-of-sample translation |
| CV              | std(returns) / mean(returns)                         | Performance consistency                |

---

## Validation Methodology

The validation pipeline is the core differentiator of this framework. It combines multiple statistical techniques to evaluate whether a strategy has genuine predictive power or is simply overfitted to historical data.

### Walk-Forward Optimization (WFO)

Walk-forward optimization is the gold standard for testing strategy robustness. Instead of optimizing once on the full dataset and testing on a separate holdout period, WFO repeatedly optimizes on rolling windows and tests each window's out-of-sample performance.

**How it works:**

```
Window 1: [Train 548 days][Test 90 days] ...
Window 2:   ... [Train 548 days][Test 90 days] ...
Window 3:       ... [Train 548 days][Test 90 days] ...
...
Final:                                         [Holdout 10%]
```

For each window:

1. Parameters are optimized on the training period using grid search over the strategy's parameter space
2. The optimal parameters are frozen and applied to the unseen test period
3. Performance on the test period (out-of-sample) is recorded
4. The window slides forward by the test period length and repeats

**Window sizing rationale:**

- **Training window: 548 days (~1.5 years)** -- Long enough to capture multiple market regimes but short enough to remain relevant. Based on institutional quant research for cryptocurrency markets.
- **Test window: 90 days (quarterly)** -- Balances statistical significance (enough trades for reliable metrics) with frequent re-optimization to adapt to changing market conditions.
- **Holdout: 10%** -- Reserved for final verification. Never used during optimization or walk-forward. Provides an honest assessment of how the strategy might perform on truly unseen data.

**Robustness checks after WFO:**

| Check                 | Threshold                    | Purpose                                                        |
| --------------------- | ---------------------------- | -------------------------------------------------------------- |
| Consistency           | >= 70% of windows profitable | Strategy should work in most market conditions, not just a few |
| Mean OOS Sharpe       | >= 1.0                       | Average out-of-sample risk-adjusted returns                    |
| Performance Stability | Sharpe CV <= 1.0             | Performance should not swing wildly between windows            |
| Parameter Stability   | Parameter CV <= 0.5          | Optimal parameters should not change drastically each window   |

### Island Volume Selection (IVS)

Traditional optimization selects the single best parameter set from each window (the peak performer). This is problematic because the peak is often an artifact of noise -- a tiny change in parameters can produce a large change in performance, suggesting instability.

Island Volume Selection (IVS) addresses this by identifying **parameter plateaus** -- regions in the parameter space where many nearby parameter sets all perform well. These plateaus represent robust strategy behavior rather than isolated lucky fits.

**How IVS works:**

1. **Map the performance surface**: For each parameter combination in the grid search, record the Sharpe ratio
2. **Threshold to find "profitable" regions**: Identify parameter sets above a minimum Sharpe threshold
3. **Cluster into islands**: Group connected parameter sets (parameter sets that differ by one step) into islands
4. **Rank by volume**: Score each island by its size (number of parameter sets) weighted by their collective performance
5. **Select from the largest island**: Choose parameters from the largest, highest-performing island rather than the single best peak

**Why IVS works:**

- Parameter plateaus are more likely to represent genuine strategy logic rather than noise
- Small changes in real-world market conditions won't push a plateau-based strategy off a performance cliff
- Strategies selected via IVS consistently outperform peak-selected strategies in out-of-sample testing

**Alternative methods available in the framework:**

| Method                    | Description                                                       | When to Use                                                    |
| ------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------- |
| `island_volume` (default) | Selects from largest robust parameter plateau                     | Most cases -- best balance of robustness and performance       |
| `frequency`               | Selects most frequently optimal parameter set across windows      | When consistency across windows is critical                    |
| `stability`               | Selects parameters with lowest variance under small perturbations | When deployment robustness is the primary concern              |
| `multi_objective`         | Optimizes for both Sharpe ratio and parameter stability           | When you want to explicitly balance performance and robustness |

### Monte Carlo Simulation

Even after WFO, a single historical path is just one possible outcome. Monte Carlo simulation resamples the out-of-sample trade sequence to estimate the distribution of possible outcomes the strategy might produce.

**How it works:**

1. Collect all out-of-sample trades from the walk-forward process
2. Resample with replacement to create 10,000 alternative trade sequences of the same length
3. Simulate the equity curve for each resampled sequence
4. Analyze the distribution of outcomes

**Stress parameters:**

- **Skip rate (10%)**: Randomly drop 10% of trades to simulate execution failures
- **Slippage stress (5 pip)**: Add extra slippage to simulate adverse execution conditions
- **Entry delay (1 bar)**: Delay all entries by one bar to simulate slow execution
- **Drawdown stress**: 95th percentile drawdown must not exceed 25% of equity

**What Monte Carlo reveals:**

```
Return Distribution (10,000 iterations):
  P5:   -15.2%    (worst 5% of outcomes)
  P25:    5.8%
  P50:   18.3%    (median outcome)
  P75:   32.1%
  P95:   52.7%    (best 5% of outcomes)
```

- A robust strategy has positive P5 (or at minimum, P5 drawdown is contained)
- A fragile strategy shows extreme variance across percentiles
- The median approximates the "true" expected return, removing the luck factor from the single historical path

### Multi-Metric Standard

A strategy must pass **all** of the following metrics to receive a PASS validation verdict:

| Category      | Metric          | Threshold | Purpose                                                      |
| ------------- | --------------- | --------- | ------------------------------------------------------------ |
| Performance   | Sharpe Ratio    | >= 1.0    | Risk-adjusted returns must justify the risk taken            |
| Performance   | Sortino Ratio   | >= 1.0    | Focus on downside risk only (upside volatility is desirable) |
| Performance   | Calmar Ratio    | >= 1.0    | Annualized return should exceed maximum drawdown             |
| Performance   | CAGR            | >= 10%    | Strategy must generate meaningful returns                    |
| Risk          | Max Drawdown    | <= 30%    | Drawdowns must be survivable                                 |
| Trade Quality | Recovery Factor | >= 2.0    | Strategy should recover at least 2x its worst drawdown       |
| Trade Quality | Profit Factor   | >= 1.5    | Wins should meaningfully exceed losses in magnitude          |
| Trade Quality | Win Rate        | >= 45%    | At least 45% of trades should be profitable                  |
| Trade Quality | Expectancy      | >= $0     | Average trade must be profitable                             |

**Informational metrics** (reported but not used for pass/fail):

- **WFE >= 50%**: At least half of in-sample performance should translate to out-of-sample
- **CV <= 20%**: Performance should be consistent, not volatile across windows

**Verdict system:**

- **PASS**: All metrics pass their thresholds
- **CONDITIONAL**: At least 75% of metrics pass (reported but not recommended for deployment)
- **FAIL**: Below CONDITIONAL threshold

### Statistical Reliability Checks

Beyond the core metrics, the validator runs diagnostic statistical tests that detect subtle issues in the validation process:

| Check                  | Method                                                 | Detects                                                                   |
| ---------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| Statistical Power      | Sample size analysis vs. Cohen's d effect size         | Insufficient trades (min 50 for 80% power at Sharpe=1.5)                  |
| Regime Heterogeneity   | PELT (Pruned Exact Linear Time) change point detection | Multiple market regimes mixed in training data that shouldn't be combined |
| Window Autocorrelation | Ljung-Box test                                         | Walk-forward windows that are not truly independent                       |
| Parameter Instability  | Coefficient of variation across windows                | Parameters that change too much across windows                            |
| Meta-Overfitting       | Permutation test (100 iterations)                      | Whether performance could be explained by random label shuffling          |

When these checks detect problems, they generate warnings in the validation report. If thresholds are exceeded, they can cause a FAIL verdict (configurable).

### Kelly Criterion Position Sizing

The position sizing module computes optimal bet sizes to maximize long-term compounding growth:

```
Kelly % = (W - (1-W) / R) * fraction
```

Where:

- **W** = Win rate (probability of winning)
- **R** = Win/loss ratio (average win / average loss)
- **fraction** = Safety multiplier (default: 0.25 = Quarter Kelly)

**Why fractional Kelly:**
Full Kelly maximizes growth rate but produces extreme volatility (~50% drawdown risk). Fractional Kelly trades some growth for reduced volatility:

| Fraction                | Growth Captured | Volatility vs Full Kelly |
| ----------------------- | --------------- | ------------------------ |
| 1.0 (Full)              | 100%            | 100%                     |
| 0.5 (Half)              | 75%             | 50%                      |
| 0.25 (Quarter, default) | ~70%            | 25%                      |

**Optimization targets:**

| Target             | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `calmar` (default) | Maximize CAGR / Max Drawdown ratio -- balances growth and safety  |
| `cagr`             | Maximize raw compound annual growth rate -- ignores drawdown risk |
| `kelly`            | Use pure Kelly-optimal fraction without safety constraints        |

**Constraints:**

- Maximum position: 25% of equity per trade
- Maximum total exposure: 100% of equity
- Minimum trades for reliable estimation: 100

---

## Performance Optimizations

The framework is designed for performance from the ground up, making it practical to run full validations (WFO + Monte Carlo) on commodity hardware.

### Vectorized computation

All backtesting logic operates on Polars DataFrames using vectorized column operations rather than row iteration. This means:

- A single backtest on 50,000+ candles completes in milliseconds
- Parameter grid searches run in seconds, not minutes
- The entire validation pipeline completes in minutes or hours (depending on parameter space size) rather than days

### Parallel parameter search

The optimizer (`engine/optimizer.py`) uses Python's multiprocessing to parallelize grid and random search across available CPU cores. By default it uses `cpu_count - 1` workers, leaving one core free for system responsiveness.

### Efficient data storage

OHLCV data is stored in Apache Parquet format, providing:

- Columnar storage (read only the columns you need)
- Compression (parquet files are typically 60-80% smaller than CSV)
- Fast predicate pushdown (load only the date range you need)
- Schema enforcement (type safety across reads and writes)

### Incremental data updates

The fetcher performs incremental updates instead of re-fetching the entire history. Each run:

1. Checks the timestamp of the most recent stored candle
2. Fetches only candles newer than that timestamp
3. Appends and deduplicates in seconds

This makes daily data updates practical even with years of historical data.

### SSL retry resilience

The fetcher includes a requests.Session with HTTPAdapter and Retry logic (5 retries with exponential backoff) for transient network failures. An additional inner retry loop handles SSL connection resets (observed on Python 3.14+), forcing fresh SSL connections on each attempt.

---

## AI Agent Guide

AI agents can use this framework effectively by following its structured conventions. The codebase is designed with deterministic contracts and clear mode routing so agents can work autonomously without ambiguity.

### Before starting

Read `AGENTS.md`. This file is the canonical entry point for any agent working on this codebase. It defines:

- **Mode routing**: Whether a task falls under strategy development or framework modification
- **Code style**: Import organization, type hints, Polars usage, docstring format -- all mandatory
- **Hard rules**: Constraints that prevent common mistakes like using pandas in engine code

### The two modes

**Mode A -- Strategy Development:**
Scope: `strategies/{name}/` directory only.

- Read `docs/STRATEGY.md` for the signal contract, config schema, indicator reference, and validation thresholds
- Follow the template in `strategies/_template/`
- Run `python run.py --strategy {name} --mode backtest` for rapid iteration

**Mode B -- Framework Development:**
Scope: `engine/`, `data/`, `run.py`, `config/`.

- Read `docs/FRAMEWORK.md` for inter-module contracts, safe modification guidelines, and architecture docs
- Changes affect all strategies -- test with the included example strategy before deploying

### Agent workflow

1. Read `AGENTS.md` to determine which mode applies
2. Read the relevant docs guide (`STRATEGY.md` or `FRAMEWORK.md`)
3. Follow existing patterns in the codebase (every module has at least one concrete example)
4. Run a backtest to verify: `python run.py --strategy {name} --mode backtest`
5. Iterate until validation passes

### Resource considerations for AI agents

Full validation (`--mode validate` or `--mode full`) uses all available CPU cores by default to parallelize walk-forward optimization and Monte Carlo simulations. This workload can consume 100% of CPU for minutes to hours, depending on the parameter space size and data range.

**Do not command an AI agent to run full validation within its own session.** The resource spike may freeze or crash the agent's session due to overload. Instead:

- Use `--mode backtest` within the AI session for rapid iteration on signal logic
- Use `--mode validate --quick-test` if you need a lighter validation check during development
- Run `--mode validate` or `--mode full` separately in a terminal, as a background process, outside the AI agent session

To control CPU usage, set `max_workers` in your global config:

```yaml
# config/global_config.yaml
system:
  max_workers: 2 # Limit to 2 cores instead of using all available
```

This applies whether you are running validation manually or through an AI agent. The default (`null`) uses all available CPU cores minus one.

### Why agents work well with this framework

- **Deterministic contracts**: Every module has clear input/output schemas (Polars DataFrames with specific columns)
- **Configuration-driven**: Parameters live in YAML, not in code -- agents can modify configs without touching implementation
- **No authentication**: No API keys, .env files, or external services to configure
- **Instant feedback**: Backtest completes in seconds -- agents can iterate rapidly
- **Structured rules**: The hard rules in AGENTS.md remove ambiguity about what is and isnt allowed

---

## Developer Guide

### Code style (mandatory)

**Import organization:**

```python
# 1. Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 2. Third-party
import polars as pl
import yaml

# 3. Local imports
from engine.base_strategy import BaseStrategy
```

**Type hints (mandatory on all functions):**

```python
def run_backtest(
    signals: pl.DataFrame,
    config: dict,
    position_sizing_mode: str = "percent_equity",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Run vectorized backtest on signal data."""
    ...
```

**Docstrings (Google style):**

```python
def calculate_sharpe(equity_curve: pl.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe Ratio.

    Args:
        equity_curve: DataFrame with 'equity' column (f64)
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Annualized Sharpe ratio as float

    Raises:
        ValueError: If equity curve has fewer than 2 data points
    """
```

**Polars usage (mandatory -- no pandas in strategy or engine code):**

```python
# Correct
df = df.with_columns([
    pl.col("close").ewm_mean(alpha=0.1).alias("ema")
])
signals = pl.when(condition).then(1).otherwise(-1)

# Wrong -- never use pandas in strategy or engine code
import pandas as pd
```

Exception: The `visualization/` directory may use pandas for Streamlit/Plotly display purposes.

**Naming conventions:**

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Private functions: `_leading_underscore`
- Constants: `UPPER_SNAKE_CASE`

**Configuration management:**
All strategy parameters MUST be in `config.yaml`. Access via `self.params.get("key", default)`. Never hardcode values.

### Project structure

```
systematic-trading/
  AGENTS.md                # AI agent router -- read first
  README.md                # This file
  run.py                   # Main CLI entry point
  requirements.txt         # Python dependencies
  config/
    global_config.yaml          # Global configuration with all settings
  data/
    fetcher.py             # Bybit OHLCV fetcher with incremental updates
    storage.py             # Parquet read/write
    inspect.py             # Data quality inspection
  engine/
    base_strategy.py       # Abstract base class for strategies
    backtester.py          # Vectorized backtest engine
    metrics.py             # Performance metrics calculation
    optimizer.py           # Parameter grid/random search
    validator.py           # WFO, Monte Carlo, Multi-Metric Standard
    position_sizing.py     # Kelly Criterion optimization
    order_manager.py       # Order execution simulation (SL/TP/trailing)
  strategies/
    _template/             # Template for new strategies
    ema_crossover_rsi/     # Example strategy (EMA crossover + RSI)
  docs/
    STRATEGY.md            # Strategy development guide
    FRAMEWORK.md           # Framework modification guide
    POSITION_SIZING.md     # Position sizing guide
  visualization/
    dashboard.py           # Streamlit interactive dashboard
```

---

## Configurable and Customization

Every trader has a different risk appetite, time horizon, and confidence threshold. This framework is designed to be configured to match your specific risk profile -- not to enforce a one-size-fits-all standard.

All customization is done through YAML configuration files, so you can tune the validation pipeline, risk parameters, and optimization goals without modifying any code.

### Validation thresholds

The Multi-Metric Standard thresholds in `config/global_config.yaml` are preset to institutional-grade defaults, but you can adjust every single threshold to match your own standards:

```yaml
multi_metric_standard:
  sharpe_min: 1.0 # Raise to 1.5 for stricter risk-adjusted returns
  sortino_min: 1.0 # Raise if downside volatility is your primary concern
  calmar_min: 1.0 # Raise for strategies prioritizing drawdown recovery
  cagr_min_pct: 10.0 # Raise for higher return targets
  max_drawdown_max_pct: 30.0 # Lower to 20% for conservative profiles
  recovery_factor_min: 2.0 # Raise for faster recovery requirements
  profit_factor_min: 1.5 # Raise if win magnitude is critical
  win_rate_min_pct: 30.0 # Raise for higher-confidence entries
  expectancy_min: 0.0 # Raise to require meaningful edge per trade
```

For example, a conservative trader might set `max_drawdown_max_pct: 15.0` and `sharpe_min: 1.5`. A high-frequency strategy might require `win_rate_min_pct: 50.0` and `profit_factor_min: 2.0`.

You can also adjust the **pass/fail criteria** -- the `conditional_min_pct` controls what percentage of metrics must pass for a CONDITIONAL verdict (default 75%). Raise it to 90% for stricter requirements.

### Walk-forward window sizing

The walk-forward window sizes control how frequently your strategy is re-optimized and how much data each optimization uses:

```yaml
walk_forward:
  train_days: 548 # Training window length
  test_days: 90 # Out-of-sample test window length
  holdout_pct: 0.10 # Percentage of data held back for final verification
```

- **Shorter train windows** (e.g., 365 days): Respond faster to regime changes but less statistical power
- **Longer train windows** (e.g., 730 days): More statistical power but may include stale regimes
- **Shorter test windows** (e.g., 30 days): More frequent re-optimization but fewer trades per evaluation
- **Longer test windows** (e.g., 180 days): More robust evaluation but slower to adapt

Adjust these based on your strategy's holding period and the market you are trading.

### Parameter selection method

You can choose how the framework selects parameters for out-of-sample testing:

```yaml
window_selection_method: "ivs" # Options: 'best', 'ivs', 'stability', 'multi_objective'
```

- `ivs` (island volume selection, default): Selects from robust parameter plateaus -- best for avoiding overfitting
- `best`: Selects the single best parameter set from each window -- aggressive, higher variance
- `stability`: Selects parameters with lowest variance under small perturbations -- most conservative
- `multi_objective`: Balances Sharpe ratio and parameter stability with a configurable weight

### Monte Carlo stress levels

Simulate different levels of market friction and execution risk:

```yaml
monte_carlo:
  n_simulations: 10000 # More = tighter confidence intervals
  skip_pct: 0.10 # Simulate 10-20% order failures in stressed conditions
  slippage_pip: 5.0 # Higher = more conservative execution assumptions
  entry_delay_bars: 1 # Simulate 0-3 bar entry delays
  percentile_95_dd_max_pct: 25.0 # Maximum acceptable 95th percentile drawdown
```

Conservative traders can increase `skip_pct` to 0.20, `slippage_pip` to 10.0, and lower `percentile_95_dd_max_pct` to 15.0.

### Kelly Criterion constraints

Position sizing is fully configurable to match your risk tolerance:

```yaml
kelly:
  default_fraction: 0.25 # Quarter Kelly (conservative)
  max_fraction: 0.5 # Half Kelly maximum

optimization:
  objective: "calmar" # calmar, cagr, or kelly

constraints:
  max_position_pct: 0.25 # Max 25% of equity per position
  min_trades: 100 # Minimum trades for reliable Kelly estimation
  max_total_exposure: 1.0 # 100% max total exposure
```

- **Quarter Kelly** (0.25): Captures ~70% of growth with ~25% of full-Kelly volatility
- **Half Kelly** (0.5): Captures ~75% of growth with ~50% of full-Kelly volatility
- **Full Kelly** (1.0): Maximum growth rate but ~50% drawdown risk

### Fee and slippage assumptions

Realistic execution cost modeling is critical for accurate backtests:

```yaml
backtest:
  maker_fee: 0.0002 # 0.02% -- Bybit maker fee
  taker_fee: 0.00055 # 0.055% -- Bybit taker fee
  slippage_pct: 0.00005 # 0.005% additional slippage buffer
  max_drawdown_pct: 0.25 # 25% circuit breaker
```

Adjust these to match your broker or exchange fee schedule. For more conservative estimates, increase `slippage_pct` or use `taker_fee` for all trades.

### Position sizing modes

```yaml
position_sizing:
  default_mode: "percent_equity" # "percent_equity" or "fixed_amount"
  percent_equity: 0.10 # 10% of equity per trade
  fixed_amount: 1000 # Fixed $1000 per trade (when using fixed_amount mode)
```

### Bar-fill assumption

Controls how entry/exit prices are simulated within a candle:

```yaml
order_management:
  bar_fill_assumption: "pessimistic" # pessimistic, optimistic, or random
```

- `pessimistic`: Longs fill at bar high, shorts fill at bar low (worst case)
- `optimistic`: Longs fill at bar low, shorts fill at bar high (best case)
- `random`: Random fill within the bar range (neutral)

### Putting it together

A conservative configuration might look like:

```yaml
multi_metric_standard:
  sharpe_min: 1.5
  max_drawdown_max_pct: 15.0
  win_rate_min_pct: 40.0

walk_forward:
  train_days: 730 # 2 years of training data
  test_days: 180 # Semi-annual re-optimization

monte_carlo:
  skip_pct: 0.20
  slippage_pip: 10.0
  percentile_95_dd_max_pct: 15.0

kelly:
  default_fraction: 0.25
  max_fraction: 0.25 # Quarter Kelly only, never higher
```

An aggressive configuration might use:

```yaml
multi_metric_standard:
  sharpe_min: 0.8
  max_drawdown_max_pct: 40.0

walk_forward:
  train_days: 365
  test_days: 30

window_selection_method: "best"

kelly:
  default_fraction: 0.5
  max_fraction: 1.0
```

The framework gives you full control. The defaults represent a balanced starting point, but every parameter exists to be adjusted to fit your strategy's characteristics and your personal risk profile.

---

## Interactive Visualization Dashboard

The framework includes a Streamlit-based interactive dashboard for exploring backtest results. It provides a visual interface to review equity curves, performance metrics, trade logs, and price charts with trade markers -- all without running commands.

### Prerequisites

To use the dashboard, you need:

1. **Backtest reports saved to disk**: Run a backtest with the `--save-reports` flag to generate report files.
2. **OHLCV data available locally**: The fetcher should have previously downloaded data for the symbol you backtested (used for the price chart with trade markers).

### Launching the dashboard

```bash
streamlit run visualization/dashboard.py
```

This starts a local web server (typically at `http://localhost:8501`). The dashboard automatically scans the `reports/` directory for saved backtest results.

### Generating reports

If you have not saved any reports yet:

```bash
# Run a backtest and save the full report
python run.py --strategy ema_crossover_rsi --mode backtest --save-reports
```

This creates report files in `reports/{strategy_name}/{date}/`:

- `equity_{timestamp}.csv` -- Equity curve data
- `trades_{timestamp}.csv` -- Trade log with entry/exit times, prices, PnL
- `summary_{timestamp}.json` -- Summary metrics (Sharpe, Sortino, drawdown, etc.)

Each time you run with `--save-reports`, a new timestamped report set is created, preserving your history.

### Dashboard features

**Sidebar -- Report Selector:**

- Select a strategy from all available report directories
- Choose a date (each day of backtesting creates a dated folder)
- Pick a specific run timestamp (useful when you ran multiple backtests on the same day)
- Click "Refresh Data" to reload reports without restarting the dashboard

**Performance Metrics Panel:**
Displays key metrics in a dark-themed card layout:

| Metric        | Description                                |
| ------------- | ------------------------------------------ |
| Total Return  | Percentage return over the backtest period |
| Sharpe Ratio  | Risk-adjusted returns                      |
| Sortino Ratio | Downside risk-adjusted returns             |
| Max Drawdown  | Largest peak-to-trough decline             |
| Win Rate      | Percentage of profitable trades            |
| Profit Factor | Gross wins / Gross losses                  |
| Total Trades  | Number of round-trip trades executed       |
| CAGR          | Compound annual growth rate                |

**Equity Curve Chart:**
An interactive Plotly chart showing:

- Equity progression over time (green line)
- Drawdown from peak equity (filled area below)
- Hover tooltips showing exact values at any point
- Zoom and pan for detailed inspection

**Price Chart with Trades:**
An OHLCV candlestick chart overlaid with trade markers:

- Green upward triangles = Long entries
- Red downward triangles = Short entries
- Square markers = Exits
- Configurable date range via the date picker expander (default shows last 7 days)
- Loads OHLC data from your local `data/raw/` parquet files

**Trade Log Table:**
A sortable, paginated table showing every trade:

- Entry and exit timestamps
- Direction (Long/Short)
- Entry and exit prices
- PnL (absolute and percentage)
- Bars held
- Fees paid
- Color-coded rows (green for profit, red for loss)

### Dashboard layout

```
+------------------------------------------+
|  Backtest Results                         |
|  Strategy | Date | Timeframe              |
+----------+-------------------------------+
| SIDER    | Performance Metrics Panel      |
|  BAR     | [Cards: Return Sharpe DD ...]  |
|          +-------------------------------+
| Strategy | Equity Curve Chart             |
| Date     | [Plotly interactive chart]     |
| Time     |                               |
|          +-------------------------------+
| Refresh  | OHLC Price Chart with Trades   |
|          | [Candlestick + trade markers]  |
|          +-------------------------------+
|          | Trade Log Table                |
|          | [Sortable paginated table]     |
+----------+-------------------------------+
```

### Troubleshooting

- **"No reports found"**: Backtest with `--save-reports` first
- **"OHLC data not available"**: Run `python data/fetcher.py --symbol X --timeframe Y` to download data for the symbol used in your backtest
- **Dashboard is slow**: The OHLC chart loads all available candle data -- use the date range filter to limit the visible window

---

## Adding a New Strategy

### Step 1: Copy the Template

```bash
cp -r strategies/_template strategies/my_new_strategy
```

### Step 2: Configure Parameters

Edit `strategies/my_new_strategy/config.yaml`:

```yaml
name: "my_new_strategy"
description: "Brief description of your strategy logic"

pairs: ["BTC/USDT:USDT"]
timeframe: "15m"

params:
  ema_short: 20
  ema_long: 50
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70

param_space:
  ema_short: [10, 15, 20, 25, 30]
  ema_long: [40, 50, 60, 70]
  rsi_period: [10, 14, 20]
```

### Step 3: Implement Signals

Edit `strategies/my_new_strategy/signals.py`:

```python
"""My new strategy description."""

import polars as pl
from engine.base_strategy import BaseStrategy


class MyNewStrategy(BaseStrategy):
    """Describe entry and exit conditions."""

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals.

        Args:
            data: OHLCV DataFrame (timestamp, open, high, low, close, volume)

        Returns:
            DataFrame with 'signal' column: 1=long, -1=short, 0=flat
        """
        params = self.params

        # Calculate indicators using polars expressions
        # ...

        return data.with_columns(
            pl.col("signal").cast(pl.Int8)
        )
```

### Step 4: Run and Iterate

```bash
python run.py --strategy my_new_strategy --mode backtest    # Quick feedback
python run.py --strategy my_new_strategy --mode validate    # Full validation
```

### Important Rules

- Never hardcode parameter values -- use config.yaml
- Use Polars for all calculations (pandas allowed only in visualization/)
- Keep strategies stateless -- no global variables or persistent state
- Do not include live trading code in strategy files
- Strategy files should contain only signal generation logic

---

## Quick Reference

| Command                                                         | Description                                |
| --------------------------------------------------------------- | ------------------------------------------ |
| `python run.py --strategy X --mode backtest`                    | Run single backtest                        |
| `python run.py --strategy X --mode backtest --save-reports`     | Backtest with report generation            |
| `python run.py --strategy X --mode validate`                    | Full validation (WFO + Monte Carlo)        |
| `python run.py --strategy X --mode validate --quick-test`       | Quick validation (fewer windows, dev only) |
| `python run.py --strategy X --mode full`                        | Same as validate                           |
| `python run.py --strategy X --mode optimize_sizing`             | Kelly Criterion sizing optimization        |
| `python run.py --strategy X --mode optimize_sizing --apply`     | Apply optimal sizing to config             |
| `python run.py --strategy X --mode validate --auto-size`        | Validate then auto-optimize sizing         |
| `python data/fetcher.py --symbol BTC/USDT:USDT --timeframe 15m` | Fetch historical data                      |
| `python data/inspect.py --symbol BTC/USDT:USDT --timeframe 15m` | Inspect data quality                       |
| `python data/inspect.py --list`                                 | List all locally available data            |
| `streamlit run visualization/dashboard.py`                      | Launch interactive dashboard               |

### CLI options

```bash
# Custom date range
python run.py --strategy ema_crossover_rsi --mode backtest --start-date 2023-01-01 --end-date 2024-01-01

# Relative date range (days back from today)
python run.py --strategy ema_crossover_rsi --mode backtest --days 365

# Multiple pairs
python run.py --strategy ema_crossover_rsi --mode backtest --pairs BTC/USDT:USDT ETH/USDT:USDT

# Multiple timeframes (fetch)
python data/fetcher.py --symbol BTC/USDT:USDT --timeframe 15m 1h 4h
```

---

## Documentation

| Document                  | Purpose                                                                        | Read When                         |
| ------------------------- | ------------------------------------------------------------------------------ | --------------------------------- |
| `AGENTS.md`               | AI agent router -- mode routing, code style, hard rules                        | First -- always                   |
| `docs/STRATEGY.md`        | Strategy development -- signal contract, config schema, indicator reference    | Creating or modifying strategies  |
| `docs/FRAMEWORK.md`       | Framework architecture -- inter-module contracts, safe modification guidelines | Modifying engine, data, or config |
| `docs/POSITION_SIZING.md` | Kelly Criterion -- formula, fractional Kelly, methodology, best practices      | Optimizing position sizing        |

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## Next Steps

1. Run the included example strategy to verify your environment works
2. Read `docs/STRATEGY.md` for a comprehensive guide to strategy development
3. Copy `strategies/_template/` and implement your first custom strategy
4. Run validation and analyze the detailed report
5. Iterate until you achieve consistent out-of-sample performance

The purpose of this framework is not to find parameters that maximize historical returns. It is to identify strategies that demonstrate genuine, repeatable edge across varying market conditions. The rigorous validation pipeline protects you from the false confidence that comes from curve-fitted backtests.
