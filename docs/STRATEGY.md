# Strategy Development Guide

This document provides a comprehensive guide for creating new trading strategies in the systematic trading framework. Follow these guidelines to ensure your strategy integrates seamlessly with the backtesting engine, optimization system, and validation framework.

## 1. How to Add a New Strategy

Creating a new strategy involves three main steps: copying the template, configuring parameters, and implementing signal generation. The framework is designed to be opinionated about structure but flexible about logic, ensuring consistency while allowing creativity in strategy development.

### Step 1: Copy the Template

Begin by creating a new directory in the `strategies/` folder using the template as a starting point. Copy the entire `_template` directory and rename it to reflect your strategy's name, such as `ema_rsi_trend` or `bollinger_squeeze`. This preserves the required file structure while giving you a clean slate for implementation. The template provides working examples of all required components, so you can incrementally modify each section rather than building from scratch.

```bash
cp -r strategies/_template strategies/my_new_strategy
```

After copying, rename the class in `signals.py` to match your strategy name and update the `name` field in `config.yaml`. These two names must match and should be descriptive enough to indicate the strategy's approach. Consider including the primary indicators or logic in the name for quick identification.

### Step 2: Configure config.yaml

The configuration file controls all runtime parameters, optimization ranges, and risk management settings. Each section serves a specific purpose and must be properly configured for your strategy to function correctly. The config is divided into discrete sections that the engine parses and applies at different stages of the backtest.

The `params` section defines the current values used during a standard backtest. These are the values that will be used when running the strategy without optimization. Set these to reasonable default values that you believe will perform well based on historical analysis or parameter sweeping. The `param_space` section defines the ranges to explore during optimization, allowing the system to systematically test different combinations.

The `position_sizing` section controls how much capital is allocated to each trade. The `validation` section configures walk-forward analysis and Monte Carlo simulation parameters. The `order_management` section defines entry, exit, and risk management rules. Each of these sections is documented in detail below.

### Step 3: Implement generate_signals()

The signal generation function is the core of your strategy. It receives a polars DataFrame containing price data and must return the same DataFrame with an additional `signal` column. The signal column contains integer values that indicate the desired position direction: 1 for long, -1 for short, and 0 for flat (no position).

Your implementation should use polars expressions for all calculations to ensure vectorized performance. Avoid iterating through rows or using Python loops, as these will be extremely slow for large datasets. Instead, leverage polars' expression API to compute indicators across the entire DataFrame simultaneously. The template provides helper methods for common calculations that you can use or adapt for your own indicators.

## 2. Signal Contract

The signal contract defines the interface between your strategy and the backtesting engine. Following this contract exactly is critical—violations will cause errors or incorrect behavior. The framework enforces this contract strictly to ensure consistent behavior across all strategies.

### Input Requirements

Your `generate_signals()` method must accept a single argument: a polars DataFrame containing OHLCV data. The DataFrame must have the following columns with these exact names and data types. The framework will validate these requirements before passing data to your strategy, but you should also ensure your implementation handles them correctly.

| Column Name | Data Type   | Description                                  |
| ----------- | ----------- | -------------------------------------------- |
| timestamp   | pl.Datetime | Unix timestamp in UTC, millisecond precision |
| open        | pl.Float64  | Opening price for the period                 |
| high        | pl.Float64  | Highest price during the period              |
| low         | pl.Float64  | Lowest price during the period               |
| close       | pl.Float64  | Closing price for the period                 |
| volume      | pl.Float64  | Trading volume during the period             |

The framework expects these columns to be present and properly typed. If your data source provides different column names, transform them in a preprocessing step before passing to the strategy. The backtesting engine handles data fetching and normalization, so this transformation typically happens before your strategy receives the data.

### Output Requirements

Your `generate_signals()` method must return a polars DataFrame with the same input columns plus an additional `signal` column. The signal column must be of type `pl.Int8` and contain only the values 1, -1, or 0. These values represent the following position states:

- **1 (Long)**: Enter or hold a long position. The engine will open a long position if flat or add to an existing long position.
- **-1 (Short)**: Enter or hold a short position. The engine will open a short position if flat or add to an existing short position.
- **0 (Flat)**: No position. The engine will close any existing long or short position at the next opportunity.

The signal is evaluated at each bar and represents the desired position state at the close of that bar. The order management system handles actual order execution, including partial fills, slippage, and commission. Your strategy should focus purely on signal generation without concern for execution details.

### Signal Generation Best Practices

Generate signals as a single vectorized operation rather than iterating through rows. This approach is orders of magnitude faster and allows the framework to efficiently process large datasets. Use polars' `when().then().otherwise()` pattern for conditional logic, which compiles to efficient native code.

**Position-Aware Exits (Recommended Pattern)**

To implement exit signals that only trigger when a corresponding position is open, use forward-fill tracking with transition-based exits. This is the CORRECT approach—do NOT use cumsum which causes position accumulation bugs.

```python
# STEP 1: Create entry signals (1 for long, -1 for short, null otherwise)
# Use null for no entry signal, not 0 (0 means flat/exit)
entry_signal = (
    pl.when(long_entry)
    .then(pl.lit(1, dtype=pl.Int8))
    .when(short_entry)
    .then(pl.lit(-1, dtype=pl.Int8))
    .otherwise(None)  # None = no entry trigger
)

df = df.with_columns([
    entry_signal.alias("entry_signal"),
    long_exit.alias("long_exit"),
    short_exit.alias("short_exit"),
])

# STEP 2: Forward-fill entry signals to track position
# This is the CORRECT approach: forward_fill NOT cumsum
# Each entry persists until explicitly exited
df = df.with_columns([
    pl.col("entry_signal")
    .forward_fill()
    .fill_null(0)
    .alias("position_from_entries")
])

# STEP 3: Get previous position to determine if we should exit
df = df.with_columns([
    pl.col("position_from_entries")
    .shift(1)
    .fill_null(0)
    .alias("prev_position")
])

# STEP 4: Determine exit signals based on previous position + current exit condition
df = df.with_columns([
    ((pl.col("prev_position") == 1) & pl.col("long_exit")).alias("should_exit_long"),
    ((pl.col("prev_position") == -1) & pl.col("short_exit")).alias("should_exit_short"),
])

# STEP 5: Combine - use entry if present, exit if triggered, otherwise hold
df = df.with_columns([
    pl.when(pl.col("entry_signal").is_not_null())
    .then(pl.col("entry_signal"))
    .when(pl.col("should_exit_long") | pl.col("should_exit_short"))
    .then(pl.lit(0, dtype=pl.Int8))
    .otherwise(None)
    .alias("combined_signal")
])

# STEP 6: Forward-fill combined signal (hold position until changed)
df = df.with_columns([
    pl.col("combined_signal")
    .shift(1)
    .forward_fill()
    .fill_null(0)
    .alias("signal")
])
```

**Why forward_fill is CORRECT and cumsum is WRONG:**

| Aspect           | forward_fill (CORRECT)               | cumsum (WRONG)                           |
| ---------------- | ------------------------------------ | ---------------------------------------- |
| Position state   | 1 = in long, -1 = in short, 0 = flat | Accumulates: 1, 2, 3... or -1, -2, -3... |
| Multiple entries | Stays at 1 (doesnt double up)        | Grows: 1→2→3 (wrong!)                    |
| Long/short ratio | Balanced                             | Skewed by accumulation                   |
| Exit behavior    | Resets to 0 cleanly                  | Doesnt reset properly                    |

**Critical Implementation Details:**

- Use `None` (not 0) for no entry signal in entry_signal column
- Use `forward_fill()` to persist position until explicit exit
- Check `prev_position` (shifted) with current exit condition
- After exit, position resets to 0 correctly

**Alternative: Simple Entry Signals**

For strategies without complex exit conditions, you can generate simple entry signals and let the backtester handle exits:

```python
df = df.with_columns([
    pl.when(long_entry).then(1)
    .when(short_entry).then(-1)
    .otherwise(0)
    .alias("signal")
])
```

The backtester will close positions when signal changes to 0 or opposite direction.

**Stateless Requirement:**

Strategies must be deterministic—the same input data with the same parameters must always produce the same signals. Both approaches above satisfy this requirement. Do not use Python variables to track position state across function calls, as this breaks reproducibility.

## 3. Config Schema

The configuration file uses YAML format and is divided into several sections, each controlling a different aspect of strategy execution. All fields are required unless marked as optional. The configuration is parsed once at startup and validated before any backtesting begins.

### Top-Level Fields

```yaml
name: "strategy_identifier"
description: "Brief description of the strategy approach"
```

The `name` field uniquely identifies the strategy and must match the strategy directory name. Use lowercase letters, numbers, and underscores only. The `description` field provides a human-readable summary that appears in reports and logs.

### Pairs and Timeframe

```yaml
pairs: ["BTC/USDT:USDT"]
timeframe: "15m"
```

The `pairs` field specifies which trading pairs to include in the backtest. Each pair should use the format `SYMBOL:QUOTE` with the exchange suffix for futures contracts. The `timeframe` field specifies the bar interval using exchange-standard format (1m, 5m, 15m, 1h, 4h, 1d, etc.).

### Parameters Section (params)

The `params` section defines the current parameter values used during backtesting. These values are applied directly when running without optimization. Parameters should be organized logically, with related parameters grouped together. All parameter values must be valid for the indicator calculations—they should not cause division by zero, negative periods, or other mathematical errors.

```yaml
params:
  ema_short: 20
  ema_long: 50
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70
```

### Parameter Space Section (param_space)

The `param_space` section defines the search ranges for optimization. Each parameter in `params` should have a corresponding entry in `param_space` unless you intend to fix that parameter during optimization. The optimization engine will systematically test all combinations of values across the parameter space.

```yaml
param_space:
  ema_short: [10, 15, 20, 25, 30]
  ema_long: [40, 50, 60, 70]
  rsi_period: [10, 14, 20]
  rsi_oversold: [25, 30, 35]
  rsi_overbought: [65, 70, 75]
```

Grid optimization iterates through every combination, so keep parameter spaces reasonably sized. For example, five parameters with five values each creates 3125 combinations. Larger spaces may require significant computation time. Consider using coarser ranges for initial exploration and finer ranges for refinement.

### Position Sizing Section (position_sizing)

The `position_sizing` section controls how capital is allocated to each trade. Position sizing significantly impacts risk and return characteristics and should be configured carefully. The framework supports multiple modes, with `percent_equity` being the most common for systematic strategies.

```yaml
position_sizing:
  mode: "percent_equity"
  percent_equity: 0.10
  max_position_value: 50000
```

- `mode`: Currently supports `"percent_equity"` which allocates a fixed percentage of portfolio equity per trade.
- `percent_equity`: The fraction of portfolio equity to risk per trade (0.10 = 10%).
- `max_position_value`: Absolute maximum position size in quote currency, providing a cap to prevent overexposure.

### Optimization Section (optimization)

The `optimization` section configures the parameter search algorithm. Grid search is currently the only supported method, which exhaustively tests all parameter combinations. Future versions may add Bayesian optimization and genetic algorithms.

```yaml
optimization:
  method: "grid"
  objective: "sharpe"
```

- `method`: Search algorithm, currently only `"grid"` is supported.
- `objective`: Metric to optimize. Common options include `"sharpe"` (Sharpe ratio), `"sortino"` (Sortino ratio), `"profit_factor"`, and `"total_return"`.

### Validation Section (validation)

The `validation` section configures walk-forward analysis and statistical validation. Walk-forward analysis tests whether optimized parameters generalize to out-of-sample data, helping identify curve-fitting and over-optimization.

```yaml
validation:
  walk_forward:
    train_days: 548 # ~1.5 years for crypto regime capture
    test_days: 90 # Quarterly re-optimization cycle
    holdout_pct: 0.10 # 10% True Holdout for final verification
    min_trades_per_window: 30
    param_selection_method: "auto" # auto (prefers island_volume), frequency, kmeans, island_volume

  # Monte Carlo stress test settings
  monte_carlo:
    n_simulations: 10000
    skip_pct: 0.10
    slippage_pip: 1.0
    entry_delay_bars: 1
    pip_value: 0.0001
    max_dd_increase_pct: 25.0

  # Multi-Metric Standard thresholds (11 metrics)
  multi_metric_standard:
    # Performance Metrics
    sharpe_min: 1.0
    sortino_min: 1.0
    calmar_min: 1.0
    cagr_min_pct: 10.0

    # Risk Metrics
    max_drawdown_max_pct: 30.0

    # Trade Metrics
    recovery_factor_min: 2.0
    profit_factor_min: 1.5
    win_rate_min_pct: 30.0
    expectancy_min: 0.0

    # Stability Metrics
    wfe_min_pct: 50.0
    cv_max_pct: 20.0

    # Pass/Fail Criteria
    conditional_min_pct: 75.0
```

**Walk-Forward Configuration:**

- `train_days`: Training window duration in days (default 548 for ~1.5 years of crypto data).
- `test_days`: Test window duration in days (default 90 for quarterly re-optimization).
- `holdout_pct`: Fraction of most recent data reserved for final holdout validation (default 0.10 = 10%).
- `min_trades_per_window`: Minimum trades required in each window for results to be considered valid.
- `param_selection_method`: Parameter selection algorithm:
  - **auto** (recommended): Prefers Island Volume Selection, falls back to K-Means or Frequency-based
  - **island_volume**: Identifies broad parameter plateaus (institutional standard)
  - **kmeans**: Clusters similar parameter combinations
  - **frequency**: Selects most commonly occurring optimal parameters

**Monte Carlo Configuration:**

- `n_simulations`: Number of Monte Carlo simulations (default 10,000 for statistical significance)
- `skip_pct`: Percentage of trades to randomly skip (simulates execution failures)
- `slippage_pip`: Slippage in pips applied to each trade
- `entry_delay_bars`: Number of bars delay before order execution
- `pip_value`: Value of one pip for slippage calculations
- `max_dd_increase_pct`: Maximum allowable drawdown increase vs baseline

**Multi-Metric Standard (Institutional-Grade Validation):**
The framework requires strategies to pass the Multi-Metric Standard validation with 11 metrics:

**Performance Metrics:**

1. **Sharpe Ratio**: Risk-adjusted returns using volatility
   - Minimum: 1.0 (higher = better risk-adjusted performance)

2. **Sortino Ratio**: Downside risk-adjusted return measure
   - Minimum: 1.0 (penalizes only downside volatility)

3. **Calmar Ratio**: CAGR adjusted for maximum drawdown
   - Minimum: 1.0 (higher = better risk-adjusted growth)

4. **CAGR**: Compound annual growth rate
   - Minimum: 10% (annualized growth rate)

**Risk Metrics:** 5. **Max Drawdown**: Largest peak-to-trough decline

- Maximum: 30% (lower = better capital preservation)

**Trade Metrics:** 6. **Recovery Factor**: Returns relative to max drawdown

- Minimum: 2.0 (higher = better recovery from losses)

7. **Profit Factor**: Gross wins divided by gross losses
   - Minimum: 1.5 (values >1.0 indicate profitable trading)

8. **Win Rate**: Percentage of profitable trades
   - Minimum: 30% (higher = more frequent wins)

9. **Expectancy**: Average profit/loss per trade
   - Minimum: $0 (positive = profitable per trade)

**Stability Metrics:** 10. **Walk Forward Efficiency (WFE)**: Measures IS→OOS performance translation - Minimum: 50% (indicates genuine edge, <50% suggests overfitting)

11. **Coefficient of Variation (CV)**: Performance consistency across windows
    - Maximum: 20% (lower = more consistent performance)

### Order Management Section (order_management)

The `order_management` section defines entry, exit, and risk management rules. These settings apply to all trades generated by the strategy and ensure consistent risk management across different parameter sets.

```yaml
order_management:
  entry:
    type: "market"
  take_profit:
    type: "atr_multiple"
    value: 2.0
  stop_loss:
    type: "atr_multiple"
    value: 1.0
  trailing_stop:
    enabled: false
```

- `entry.type`: Order type for entries, currently only `"market"` is supported.
- `take_profit.type`: Take profit calculation method. `"atr_multiple"` uses a multiple of ATR for dynamic targets.
- `take_profit.value`: The multiple or absolute value for take profit calculation.
- `stop_loss.type`: Stop loss calculation method, mirrors take profit options.
- `stop_loss.value`: The multiple or absolute value for stop loss calculation.
- `trailing_stop.enabled`: Whether to enable trailing stop loss for additional downside protection.

## 4. What NOT to Do

The following practices are prohibited or strongly discouraged. Violations may cause errors, incorrect results, or strategies that fail validation. These rules exist to ensure consistency, performance, and reproducibility across all strategies in the framework.

### No Pandas

Never use pandas for calculations. The framework uses polars exclusively for all data manipulation. Pandas and polars have fundamentally different memory models and API designs—mixing them causes confusion and performance issues. If you encounter pandas code in examples or legacy code, translate it to polars equivalents using the patterns shown in this guide.

Pandas is prohibited because it uses row-oriented iteration by default, causing catastrophic performance degradation on large datasets. Polars uses column-oriented storage with lazy evaluation, enabling aggressive optimization and parallelization. Even simple pandas operations that appear to work correctly may silently produce incorrect results due to different handling of missing data, alignment, and data types.

### No Hardcoded Values

Never hardcode parameter values, thresholds, or magic numbers in the signals.py file. All tunable values must be defined in config.yaml and accessed via `self.params`. This requirement ensures that optimization can systematically explore the parameter space and that the same logic can be tested with different configurations.

Hardcoded values create hidden parameters that are never tested during optimization, potentially masking overfitting. They also make it difficult to compare strategies fairly because the same logic with different hardcoded values produces different results. By requiring all parameters to be in the config, every value is explicit, documented, and optimizable.

### No Live Trading Code

The framework is strictly for backtesting only. Never include live trading functionality, API keys, exchange connections, or order execution logic in strategy files. Strategies should contain only signal generation logic—they are pure mathematical functions that map price data to position signals.

Live trading requires fundamentally different architecture including risk management, order management, error handling, and state persistence that is outside the scope of backtesting. Mixing live trading code with backtesting code creates confusion and increases the risk of accidental live trades during testing. If you need live trading functionality, build a separate system that imports your strategy logic.

### No Non-Deterministic Behavior

Strategies must be deterministic—the same input data with the same parameters must always produce the same signals. Do not use random numbers, system time, or external data sources that may vary between runs. This requirement ensures that backtest results are reproducible and that optimization meaningfully explores parameter space.

If your strategy requires randomness (for example, in a regime-detection algorithm), seed the random number generator with a constant value and make the seed a configurable parameter. This allows systematic testing of different random seeds while maintaining reproducibility.

### No Position State Management (Python Variables)

Do not use Python instance variables or external state to track positions across `generate_signals()` calls. Strategies must be pure functions—the same input DataFrame with the same parameters must always produce the same output signals.

**Stateless approaches using DataFrame columns are allowed and encouraged:**

- Use `pl.col("signal").shift(1)` to reference previous signals
- Use `forward_fill()` to carry forward position state (see "Position-Aware Exits" above)

**Stateful approaches that break reproducibility (PROHIBITED):**

```python
# ❌ WRONG - Uses Python state (breaks optimization)
self.current_position = 0  # Instance variable

def generate_signals(self, data):
    if self.current_position == 1 and exit_condition:
        signal = 0
    # ...
    self.current_position = signal  # Mutates state!
```

**Correct approach using forward_fill:**

```python
# ✅ CORRECT - Pure function using forward_fill
entry_signal = (
    pl.when(long_entry)
    .then(pl.lit(1, dtype=pl.Int8))
    .when(short_entry)
    .then(pl.lit(-1, dtype=pl.Int8))
    .otherwise(None)
)

df = df.with_columns([
    pl.col("entry_signal")
    .forward_fill()
    .fill_null(0)
    .alias("signal")
])
```

The key difference: The stateless approach produces the same output for the same input, enabling reproducible backtesting and valid optimization. The stateful approach produces different outputs depending on call order, making optimization meaningless.

### No Same-Bar Entry (Data Leakage)

Never generate a signal on candle N and enter on the same candle N. This constitutes data leakage and produces unrealistic backtest results. When a signal is detected based on candle N's data (close price, indicators calculated from candle N), the entry must occur on candle N+1's open price, not candle N.

**Why This Is Data Leakage:**

To determine whether an entry condition is met on candle N, you need candle N's complete data (open, high, low, close). At the time candle N's open price is available, you don't yet know the close or the indicator values that depend on the close. By the time you know the signal, candle N has already closed and you've missed that entry opportunity. The earliest realistic entry is candle N+1's open.

**Correct Signal Interpretation:**

The `signal` column on row N represents: "Based on all information available at the close of candle N, I want to be in this position starting from candle N+1's open."

**Example of the Anti-Pattern (WRONG):**

```python
# ❌ WRONG - Signal on bar N, assumed entry on bar N (data leakage!)
# This implicitly assumes you can enter at bar N's open AFTER seeing bar N's close
df = df.with_columns([
    pl.when(long_entry).then(1)
    .when(short_entry).then(-1)
    .otherwise(0)
    .alias("signal")
])
# If backtester enters at bar N's open when signal[N]=1, that's using future info!
```

**Correct Pattern:**

```python
# ✅ CORRECT - Signal on bar N means "enter at bar N+1's open"
# The backtester should handle this: when signal[N] = 1, enter at open[N+1]
df = df.with_columns([
    pl.when(long_entry).then(1)
    .when(short_entry).then(-1)
    .otherwise(0)
    .alias("signal")
])
    - Use `forward_fill()` to carry forward position state (see "Position-Aware Exits" above)
```

**How the Backtester Handles This:**

The backtester correctly implements realistic timing:

- When `signal[N] = 1` (long) and previous signal was 0 or -1, the backtester enters a long position at `open[N+1]`
- When `signal[N] = 0` (exit) and previous signal was 1 or -1, the backtester exits at `open[N+1]`
- This ensures no information from candle N is used before it becomes available

**Verification:**

To verify your strategy is not leaking data:

1. Check that entry prices in backtest results match the open price of candles AFTER the signal candle
2. For a signal generated at row N (timestamp T), the entry should occur at row N+1 (timestamp T+1)
3. If you see entries at the same timestamp as signal generation, you have data leakage

**Consequences of Violation:**

**Correct approach using forward_fill:**

```python
# ✅ CORRECT - Pure function using forward_fill
entry_signal = (
    pl.when(long_entry)
    .then(pl.lit(1, dtype=pl.Int8))
    .when(short_entry)
    .then(pl.lit(-1, dtype=pl.Int8))
    .otherwise(None)
)

df = df.with_columns([
    pl.col("entry_signal")
    .forward_fill()
    .fill_null(0)
    .alias("signal")
])
```

- Produce unrealistically high returns (you're "trading with tomorrow's newspaper")
- Fail validation when the backtester correctly delays entry to the next bar
- Give false confidence that won't translate to live trading

## 5. Common Indicators Reference

This section provides polars implementations of commonly used technical indicators. Each implementation follows the framework's conventions: using polars expressions, accepting parameters via config, and returning properly named columns. Copy and adapt these patterns for your own strategies.

### Exponential Moving Average (EMA)

The Exponential Moving Average gives more weight to recent prices, making it more responsive to price changes than a simple moving average. EMA is calculated using a smoothing factor derived from the period: multiplier = 2 / (period + 1). This implementation uses polars' built-in `ewm_mean` function with the correct alpha value.

```python
def _calculate_ema(self, df: pl.DataFrame, period: int) -> pl.Expr:
    """Calculate Exponential Moving Average.

    EMA = (close * multiplier) + (previous_ema * (1 - multiplier))
    where multiplier = 2 / (period + 1)
    """
    multiplier = 2.0 / (period + 1)
    return (
        pl.col("close")
        .ewm_mean(alpha=multiplier, adjust=False)
        .alias(f"ema_{period}")
    )
```

To use this in your strategy, call it with the desired period and add the result to your DataFrame. The `adjust=False` parameter uses the traditional EMA formula without the adjusted weighting that some implementations use. This matches the behavior expected by most trading frameworks and trading education materials.

### Relative Strength Index (RSI)

RSI measures the speed and magnitude of price changes to identify overbought or oversold conditions. RSI oscillates between 0 and 100, with readings above 70 typically considered overbought and readings below 30 considered oversought. This implementation uses exponential moving averages for the gain and loss calculations, which is the Welles Wilder method.

```python
def _calculate_rsi(self, df: pl.DataFrame, period: int) -> pl.Expr:
    """Calculate Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    where RS = average_gain / average_loss
    """
    delta = pl.col("close").diff()

    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.alias(f"rsi_{period}")
```

The first RSI value is unreliable because the EWM calculation requires historical data to stabilize. Most strategies either discard the first N periods or use simple moving averages for the initial calculation. For practical purposes, RSI values are meaningful after approximately one period of data, but best results come from longer lookback periods.

### Average True Range (ATR)

ATR measures market volatility by decomposing the entire range of an asset price for that period. The true range is the maximum of: current high minus current low, absolute value of current high minus previous close, and absolute value of current low minus previous close. ATR is the exponential moving average of the true range.

```python
def _calculate_atr(self, df: pl.DataFrame, period: int) -> pl.Expr:
    """Calculate Average True Range.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = EWM of True Range
    """
    high = pl.col("high")
    low = pl.col("low")
    prev_close = pl.col("close").shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pl.concat([tr1, tr2, tr3]).max(axis=1)

    atr = true_range.ewm_mean(alpha=1.0 / period, adjust=False)

    return atr.alias(f"atr_{period}")
```

ATR is essential for position sizing and risk management because it adapts to current market volatility. Fixed percentage stops may be too tight in volatile markets or too loose in calm markets. Using ATR multiples for stops and targets ensures consistent risk across different market conditions.

### Moving Average Convergence Divergence (MACD)

MACD is a trend-following momentum indicator that shows the relationship between two exponential moving averages. The MACD line is the difference between the 12-period EMA and the 26-period EMA. The signal line is a 9-period EMA of the MACD line. The histogram is the difference between the MACD line and the signal line.

```python
def _calculate_macd(self, df: pl.DataFrame, fast_period: int = 12,
                    slow_period: int = 26, signal_period: int = 9) -> list[pl.Expr]:
    """Calculate MACD components.

    Returns list of expressions for macd_line, signal_line, and histogram.
    """
    fast_ema = pl.col("close").ewm_mean(alpha=2.0 / (fast_period + 1), adjust=False)
    slow_ema = pl.col("close").ewm_mean(alpha=2.0 / (slow_period + 1), adjust=False)

    macd_line = (fast_ema - slow_ema).alias("macd_line")

    signal_line = macd_line.ewm_mean(alpha=2.0 / (signal_period + 1), adjust=False).alias("signal_line")

    histogram = (pl.col("macd_line") - pl.col("signal_line")).alias("macd_histogram")

    return [macd_line, signal_line, histogram]
```

MACD signals include the MACD line crossing above or below the signal line (momentum changes), MACD crossing zero (trend changes), and divergences between MACD and price (reversal signals). The histogram helps visualize the momentum and can be used for early entry signals before the lines cross.

### Bollinger Bands

Bollinger Bands consist of a middle band (N-period SMA) and two outer bands at K standard deviations above and below the middle band. The bands expand and contract with market volatility, providing a dynamic envelope around price. They are useful for identifying mean reversion opportunities and measuring relative value.

```python
def _calculate_bollinger_bands(self, df: pl.DataFrame, period: int = 20,
                               std_dev: float = 2.0) -> list[pl.Expr]:
    """Calculate Bollinger Bands.

    Middle Band = SMA(close, period)
    Upper Band = Middle Band + (K * std_dev)
    Lower Band = Middle Band - (K * std_dev)
    """
    middle_band = pl.col("close").rolling_mean(window_size=period).alias("bb_middle")

    rolling_std = pl.col("close").rolling_std(window_size=period)

    upper_band = (middle_band + (std_dev * rolling_std)).alias("bb_upper")
    lower_band = (middle_band - (std_dev * rolling_std)).alias("bb_lower")

    bandwidth = ((upper_band - lower_band) / middle_band).alias("bb_bandwidth")
    percent_b = ((pl.col("close") - lower_band) / (upper_band - lower_band)).alias("bb_percent_b")

    return [middle_band, upper_band, lower_band, bandwidth, percent_b]
```

Common Bollinger Band strategies include mean reversion (price touching lower band suggests buy, upper band suggests sell), trend following (band compression followed by expansion signals breakout), and regime identification (narrow bands indicate low volatility, wide bands indicate high volatility). The bandwidth and percent_b indicators help quantify these observations.

## 6. Validation Thresholds

The framework enforces institutional-grade quality thresholds through the Multi-Metric Standard (11 metrics). Strategies must pass sufficient metrics to achieve validation approval (75% = at least 8 of 11 metrics). These thresholds prevent overfitted strategies from advancing and ensure that accepted strategies meet rigorous institutional standards.

### Multi-Metric Standard (All Required)

| Metric                       | Threshold | Description                                            | Why It Matters                                                 |
| ---------------------------- | --------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| **Walk Forward Efficiency**  | ≥ 50%     | Ratio of OOS to IS Sharpe ratio                        | Measures parameter generalization. <50% indicates overfitting. |
| **Coefficient of Variation** | ≤ 20%     | Std dev of OOS Sharpe across windows / Mean OOS Sharpe | Lower = more consistent performance across market conditions.  |
| **Sharpe Ratio**             | ≥ 1.0     | Annualized risk-adjusted returns                       | Higher = better returns per unit of risk taken.                |
| **Sortino Ratio**            | ≥ 1.0     | Downside risk-adjusted returns                         | Similar to Sharpe but only penalizes downside volatility.      |
| **Calmar Ratio**             | ≥ 1.0     | CAGR / Maximum Drawdown                                | Measures growth relative to worst-case loss.                   |
| **CAGR**                     | ≥ 10%     | Compound annual growth rate                            | Annualized growth rate assuming compounding.                   |
| **Recovery Factor**          | ≥ 2.0     | Total profit / Maximum drawdown                        | Higher = better recovery from losses.                          |
| **Profit Factor**            | ≥ 1.5     | Gross wins / Gross losses                              | Values >1.0 indicate profitable trading overall.               |
| **Max Drawdown**             | ≤ 30%     | Largest peak-to-trough equity decline                  | Lower = better capital preservation during losses.             |
| **Win Rate**                 | ≥ 30%     | Percentage of profitable trades                        | Higher = more frequent winning trades.                         |
| **Expectancy**               | ≥ $0      | Average profit/loss per trade                          | Positive = profitable on a per-trade basis.                    |

**Institutional Validation Philosophy:**

- Strategies must pass the Multi-Metric Standard validation (11 metrics)
- Each metric captures a different dimension of strategy quality
- This approach prevents strategies that excel in one area but fail in others
- Consistent with institutional trading standards for robust strategy validation
- Pass/Fail determined by conditional_min_pct (75% = must pass at least 8 of 11 metrics)

### Walk-Forward Requirements

Walk-forward validation uses duration-based windows consistent with institutional practices:

- **Training window**: 548 days (~1.5 years for crypto regime capture)
- **Test window**: 90 days (quarterly re-optimization cycle)
- **Holdout**: 10% most recent data reserved for final verification
- **Stitched equity curve**: Chronological collation of all OOS windows

**🧪 Quick Validation Mode:** For faster validation during strategy development:

```bash
# Quick validation test (1 year, 180 train / 30 test windows)
python run.py --strategy my_strategy --mode validate --quick-test

# Or manually override in config:
# validation.walk_forward.train_days: 180
# validation.walk_forward.test_days: 30
# validation.walk_forward.lookback_days: 365
```

**Always revert to production settings before final validation:**

- Production: 548 train / 90 test / ~2 years lookback
- Quick test: 180 train / 30 test / 365 lookback

**Parameter Selection Methods:**

The framework supports multiple parameter selection algorithms with **auto** mode recommended:

| Method            | Description                                              | When to Use                                           |
| ----------------- | -------------------------------------------------------- | ----------------------------------------------------- |
| **auto**          | Prefers Island Volume, falls back to K-Means/Frequency   | **Default** - best for most strategies                |
| **island_volume** | Identifies broad parameter plateaus to avoid overfitting | **Institutional standard** - use for final validation |
| **kmeans**        | Clusters similar parameter combinations                  | When parameters form clear groups                     |
| **frequency**     | Selects most commonly occurring optimal parameters       | Quick exploration of parameter space                  |

**Island Volume Selection (IVS)** is the institutional standard because it:

- Identifies broad parameter plateaus rather than sharp peaks
- Selects the center of the largest contiguous "island" of profitability
- Prioritizes robust, generalizable parameters over curve-fitted spikes
- Reduces overfitting risk by avoiding parameter edge cases

**Minimum Requirements:**

- Each window must contain at least `min_trades_per_window` trades (default: 30)
- Windows with insufficient trades are excluded from validation
- At least 50% of windows must have valid results for meaningful validation

### Monte Carlo Requirements

Monte Carlo simulation tests strategy robustness by randomizing trade order and simulating various market scenarios. The framework runs thousands of simulations and checks whether the strategy remains profitable under adverse conditions. The minimum acceptable Monte Carlo Sharpe ratio at the 5th percentile is 0.5, meaning the strategy should remain profitable even in unlucky scenarios.

Equity curve stability is assessed by checking the maximum drawdown distribution. The strategy should not experience catastrophic losses in more than 5% of simulations. If the strategy has a significant chance of losing more than 50% of equity, it fails validation regardless of average performance.

### Enhanced Metrics Reference

The framework now includes institutional-grade metrics beyond traditional performance measures:

**Walk Forward Efficiency (WFE)**

- Measures how well in-sample (IS) performance translates to out-of-sample (OOS)
- Formula: (OOS Sharpe / IS Sharpe) × 100
- Threshold: ≥60% indicates genuine edge
- Values <50% suggest overfitting to historical data

**Recovery Factor**

- Measures total profit relative to maximum drawdown
- Formula: Total Profit / Maximum Drawdown
- Threshold: ≥2.0 indicates good recovery from losses
- Higher values mean the strategy recovers well from drawdowns

**Sortino Ratio**

- Downside risk-adjusted return measure
- Similar to Sharpe but only penalizes downside volatility
- Formula: (Return - RiskFreeRate) / DownsideDeviation
- Better than Sharpe for strategies with asymmetric returns

**Calmar Ratio**

- CAGR adjusted for maximum drawdown
- Formula: CAGR / Maximum Drawdown
- Measures growth relative to worst-case loss
- Higher values indicate better risk-adjusted growth

**CAGR (Compound Annual Growth Rate)**

- Annualized growth rate assuming compounding
- Formula: (EndingEquity / StartingEquity)^(1/Years) - 1
- More meaningful than total return for multi-year periods

**Coefficient of Variation (CV)**

- Performance consistency measure across walk-forward windows
- Formula: (Std Dev of OOS Sharpe / Mean OOS Sharpe) × 100
- Lower values indicate more consistent performance
- Threshold: ≤20% for institutional approval

### Continuous Improvement

Validation thresholds are minimum requirements, not targets. A strategy that barely passes validation is not a good candidate for live trading. The goal should be strategies that significantly exceed these thresholds, demonstrating robust edge and consistent performance across various market conditions.

When validation fails, analyze the specific failure mode. Poor Sharpe ratio suggests the strategy may not have a genuine edge. Excessive walk-forward degradation suggests overfitting to historical data. Monte Carlo failures indicate sensitivity to execution order or catastrophic loss scenarios. Each failure mode has different remediation strategies, so understanding the specific failure is essential for improvement.
