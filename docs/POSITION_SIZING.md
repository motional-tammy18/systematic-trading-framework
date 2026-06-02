# Position Sizing Optimization

Complete guide to Kelly Criterion position sizing optimization in the systematic trading framework.

## Table of Contents

- [Overview](#overview)
- [What is Position Sizing?](#what-is-position-sizing)
- [Why Optimize Position Sizing?](#why-optimize-position-sizing)
- [Kelly Criterion](#kelly-criterion)
- [Configuration](#configuration)
- [Usage](#usage)
- [Methodology](#methodology)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [FAQ](#faq)

## Overview

The position sizing optimization feature helps you find the optimal Kelly Criterion position size for your validated trading strategies. Kelly Criterion determines the theoretically optimal bet size based on your expected edge and the odds offered.

## What is Position Sizing?

Position sizing determines HOW MUCH to trade. It's the most critical risk management decision in systematic trading. A strategy with a great edge will fail if you risk too much per trade, while a mediocre strategy can become profitable with proper position sizing.

Position sizing affects your results in multiple ways:

- **Risk**: The dollar amount you're exposed to per trade
- **Drawdowns**: How severe your losses can become
- **Growth**: Your compound annual growth rate (CAGR)
- **Psychology**: How well you can handle losing streaks

### Three Key Questions Position Sizing Answers

1. **What fraction of capital should I risk on each trade?**
2. **How much should I bet to maximize long-term growth?**
3. **What fraction of Kelly is safe to use?**

### Common Mistakes

**Over-Risking**: Betting 5% or more per trade leads to catastrophic drawdowns. Professional traders typically risk 0.5-2% per trade.

**Fixed Size**: Using a fixed dollar amount ignores the statistical edge of your strategy.

**Ignoring Edge**: Not matching position size to your statistical advantage (Kelly Criterion).

**Using Full Kelly**: Using the raw Kelly percentage can be dangerous due to estimation errors.

## Why Optimize Position Sizing?

Optimizing position sizing transforms a good strategy into a great one by balancing risk and return.

### Key Benefits

**Maximize Risk-Adjusted Returns**: Correct sizing maximizes your Sharpe ratio and CAGR while controlling drawdowns.

**Protect Against Ruin**: Proper sizing prevents catastrophic losses from variance.

**Compound Growth**: Kelly Criterion provides mathematical maximum long-term growth rate.

**Psychological Comfort**: Knowing you're not over-exposed reduces stress during drawdowns.

**Backtest Reality**: Position sizing significantly impacts backtest results. A strategy with 2% risk might lose 30% of capital, but with 0.5% risk might only lose 8%.

### The Growth-Risk Tradeoff

```python
# Example: 10% per trade vs 1% per trade
# Assuming 60% win rate, 2:1 reward/risk ratio

# 10% per trade:
# - Expected monthly return: ~6%
# - Max drawdown: ~40%

# 1% per trade:
# - Expected monthly return: ~3%
# - Max drawdown: ~15%

# Both strategies have the same edge, but different risk profiles.
# The 1% sizing gives you 2x growth with 2.5x lower drawdown.
```

## Kelly Criterion

The Kelly Criterion is a mathematical formula for determining the optimal bet size based on your expected edge and the odds offered. It's the foundation of modern position sizing theory.

### The Formula

**Kelly% = W - [(1 - W) / R]**

Where:

- `W` = Win rate (probability of winning)
- `R` = Win/Loss ratio (average win / average loss)

### Understanding the Components

**Win Rate (W)**: The probability of a winning trade. This can be estimated from your strategy's win rate on out-of-sample data.

**Win/Loss Ratio (R)**: The average winning trade size divided by the average losing trade size. This reflects your strategy's average risk/reward profile.

**Example Calculation**:

```python
# Win rate: 55%
# Average win: $250
# Average loss: $150
# Win/loss ratio: 1.67

# Full Kelly = 0.55 - (0.45 / 1.67) = 0.55 - 0.27 = 0.28 (28%)
```

### Why Fractional Kelly?

Full Kelly is "edge of ruin" - any estimation error leads to bankruptcy. Fractional Kelly provides safer tradeoffs between growth and volatility.

| Kelly Fraction | Growth Capture | Volatility Capture |
| -------------- | -------------- | ------------------ |
| Full Kelly     | 100%           | 100%               |
| Half Kelly     | 50%            | 50%                |
| Quarter Kelly  | 75%            | 25%                |
| 1/8 Kelly      | 12.5%          | 12.5%              |

**Key Insight**: You can capture 75% of maximum growth with only 25% of volatility using Quarter Kelly. Half Kelly captures 50% of growth with 50% of volatility.

### When to Use Kelly Criterion

**Use Kelly Criterion when:**

- Strategy has > 100 trades with consistent edge
- Win rate and win/loss ratio are stable
- You want maximum long-term compounding
- Strategy passes validation with strong metrics
- You understand fractional Kelly and risk tolerance

**When NOT to Use Kelly Criterion:**

- Trade count is too low (< 50)
- Edge is uncertain or changing rapidly
- You can't handle large drawdowns
- Strategy doesn't have a statistical edge (Kelly is negative)
- You prefer fixed percentage sizing regardless of edge

**What if Kelly is Negative?**

Negative Kelly means your strategy has no statistical edge. Do NOT trade it. Re-optimize strategy parameters first or consider a different strategy. Position sizing cannot fix a strategy with negative expectancy.

## Configuration

Add to `config/global_config.yaml`:

```yaml
position_sizing_optimization:
  enabled: true

  kelly:
    enabled: true
    default_fraction: 0.25 # Quarter Kelly
    max_fraction: 0.5 # Half Kelly max

  optimization:
    objective: "kelly" # kelly, calmar
    risk_tolerance: 1.0

  constraints:
    max_position_pct: 0.25
    min_trades: 100
```

### Configuration Parameters

**`position_sizing_optimization.enabled`**

- Type: boolean
- Description: Enable position sizing optimization feature

**`kelly.enabled`**

- Type: boolean
- Description: Enable Kelly Criterion method
- Default: true

**`kelly.default_fraction`**

- Type: float (0.0 to 1.0)
- Description: Base fraction of Kelly to use
- Default: 0.25 (Quarter Kelly)
- Recommended: 0.25 to 0.5

**`kelly.max_fraction`**

- Type: float (0.0 to 1.0)
- Description: Maximum fraction of Kelly to use (safety cap)
- Default: 0.5 (Half Kelly)
- Recommended: 0.5

**`optimization.objective`**

- Type: string
- Description: Optimization target metric
- Options: "kelly", "calmar"
- Default: "kelly"
- Description: Higher = better risk-adjusted growth

**`optimization.risk_tolerance`**

- Type: float (positive)
- Description: Risk tolerance multiplier
- Default: 1.0
- Higher = more aggressive, Lower = more conservative

**`constraints.max_position_pct`**

- Type: float (0.0 to 1.0)
- Description: Maximum position size as percentage of equity
- Default: 0.25 (25%)
- Recommended: 0.10 to 0.25

**`constraints.min_trades`**

- Type: integer
- Description: Minimum trades required for Kelly optimization
- Default: 100
- Recommended: 50-200 depending on strategy frequency

## Usage

### Basic Usage

```bash
# Run sizing optimization (uses validation results)
python run.py --strategy ema_crossover_rsi --mode optimize_sizing

# With fresh backtest instead
python run.py --strategy ema_crossover_rsi --mode optimize_sizing --backtest

# Apply optimal sizing to config
python run.py --strategy ema_crossover_rsi --mode optimize_sizing --apply

# Auto-run after validation
python run.py --strategy ema_crossover_rsi --mode validate --auto-size

# Quick validation test (during development)
python run.py --strategy ema_crossover_rsi --mode validate --quick-test
```

### Reading the Report

The optimization report shows Kelly Criterion performance and provides recommendations.

**Kelly Criterion Table**:

| Metric           | Value  | Description                |
| ---------------- | ------ | -------------------------- |
| Raw Kelly        | 0.28   | Full Kelly percentage      |
| Applied Fraction | 0.25   | Quarter Kelly (0.25 × raw) |
| Position Size    | $7,000 | Size for $100,000 capital  |
| Win Rate         | 55%    | Statistical edge           |
| Win/Loss Ratio   | 1.67   | Average win / average loss |
| Trade Count      | 312    | Number of trades analyzed  |
| Confidence       | High   | Statistical significance   |

**Sensitivity Analysis**:

| Kelly Fraction | CAGR | Max DD | Sharpe | Calmar |
| -------------- | ---- | ------ | ------ | ------ |
| 0.125 (1/8)    | 18%  | 12%    | 1.8    | 1.5    |
| 0.250 (1/4)    | 24%  | 15%    | 2.1    | 1.6    |
| 0.500 (1/2)    | 30%  | 22%    | 2.4    | 1.4    |
| 1.000 (Full)   | 38%  | 35%    | 2.7    | 1.1    |

**Final Recommendation**:

```
RECOMMENDATION: Quarter Kelly (25% fraction)

Why:
  • Strong statistical edge with 55% win rate
  • 1.67:1 reward/risk ratio provides good compensation
  • 312 trades provides high confidence
  • Quarter Kelly balances growth and risk well

Calculated Position Size:
  • Raw Kelly: 28% of equity
  • Applied Fraction: 25% (Quarter Kelly)
  • Final Position Size: $7,000 (7% of $100,000)
  • Risk Amount: $2,100 per trade

Expected Performance:
  • Expected monthly return: 3-5%
  • Max drawdown: 15-18%
  • Sharpe ratio: 2.0-2.2
  • Annual CAGR: 24-28%

Applied Parameters:
  • Default Fraction: 0.25 (Quarter Kelly)
  • Max Fraction: 0.50 (Half Kelly cap)
  • Position Size Cap: 25% of equity

Config Update:
  position_sizing:
    mode: "kelly"
    kelly_fraction: 0.25
    max_fraction: 0.5
    enabled: true
```

## Methodology

### Optimization Target

The optimizer maximizes **Risk-Adjusted Growth (Calmar Ratio)**:

```
Calmar = CAGR / Maximum Drawdown
```

This balances:

- **Growth**: Higher CAGR
- **Risk**: Lower maximum drawdown

**Why Calmar?**

- Measures growth relative to worst-case loss
- Higher values indicate better risk-adjusted performance
- More intuitive than Sharpe for position sizing
- Directly relevant to capital preservation

### Kelly Calculation

The optimizer computes the raw Kelly percentage based on:

1. **Win Rate (W)**: Percentage of profitable trades from OOS data
2. **Win/Loss Ratio (R)**: Average winning trade / Average losing trade

Raw Kelly = W - [(1 - W) / R]

3. **Apply Fraction**: Multiply raw Kelly by default fraction (typically 0.25 for Quarter Kelly)

4. **Safety Check**: Ensure calculated fraction is positive and less than max_fraction

### Recommendation Logic

The optimizer follows this workflow:

1. **Calculate optimal Kelly parameters**
   - Compute raw Kelly from win rate and win/loss ratio
   - Apply default fraction (typically 0.25)
   - Check against max_fraction constraint

2. **Test Kelly sizing on historical data**
   - Run backtest with optimized Kelly sizing
   - Compute all performance metrics
   - Generate complete trade log and equity curve

3. **Compute Calmar ratio**
   - Calmar = CAGR / Max Drawdown
   - Higher Calmar = Better risk-adjusted growth

4. **Calculate sensitivity analysis**
   - Test various fractions (1/8, 1/4, 1/2, full)
   - Compare CAGR, Max DD, Sharpe, Calmar for each
   - Identify optimal balance

5. **Select best fraction based on analysis**
   - Consider Calmar ratio as primary metric
   - Consider secondary metrics (Sharpe, CAGR, recovery)
   - Ensure position size stays within constraints

6. **Apply safety constraints**
   - Max position size: 25% of equity
   - Minimum Kelly fraction: 0.125 (1/8 Kelly) for conservative strategies
   - Minimum trades: 100 for Kelly optimization
   - Ensure Kelly fraction is positive

### Safety Features

- ✅ **Fractional Kelly**: Default Quarter Kelly, max Half Kelly
- ✅ **Position Limits**: Max 25% per trade to prevent overconcentration
- ✅ **Minimum Trades**: Requires 100+ trades for Kelly optimization
- ✅ **Positive Kelly Check**: Ensures strategy has edge before sizing
- ✅ **Fraction Bounds**: Respects default and max fraction constraints
- ✅ **Config Backup**: Always backup before --apply

## Best Practices

### 1. Always Use Fractional Kelly

Never use full Kelly. Always use Quarter (0.25x) or Half (0.5x) Kelly.

**Why**: Full Kelly sits at "edge of ruin". Any estimation error → bankruptcy.

**Example**:

```python
# WRONG: Full Kelly (can be dangerous)
kelly_fraction = 1.0  # Full Kelly

# CORRECT: Fractional Kelly (safest)
kelly_fraction = 0.25  # Quarter Kelly (recommended default)
kelly_fraction = 0.5   # Half Kelly (if you can handle drawdowns)

# MODERATE: Conservative for high-volatility strategies
kelly_fraction = 0.125 # 1/8 Kelly
```

### 2. Require Minimum Trade Count

Kelly Criterion needs 100+ trades for statistical significance.

**Why**: Small sample sizes produce unreliable estimates. A 20-trade sample can easily overestimate edge.

**Guideline**:

- **50+ trades**: Use with caution, results may be unreliable
- **100+ trades**: Kelly Criterion is reasonably reliable
- **200+ trades**: Full confidence in Kelly estimates
- **500+ trades**: Very reliable, can consider higher fractions

**Example**:

```python
# Low trade count (insufficient data)
trades = 30
win_rate = 0.60
wlr = 1.5
kelly = 0.60 - (0.40 / 1.5) = 0.33

# However, with only 30 trades, this is unreliable.
# Use quarter Kelly (8%) instead of raw Kelly (33%).

# High trade count (reliable)
trades = 500
win_rate = 0.58
wlr = 1.6
kelly = 0.58 - (0.42 / 1.6) = 0.35

# With 500 trades, this is much more reliable.
# Quarter Kelly (8.75%) is appropriate.
```

### 3. Update Regularly

Recalculate position sizing monthly or quarterly.

**Why**: Market conditions change, edge decays, and win rate can shift.

**Schedule**:

- **Crypto**: Monthly (more volatile, edge changes faster)
- **Forex**: Monthly to quarterly
- **Stocks**: Quarterly
- **Regime changes**: Immediately if market volatility doubles or strategy performance deteriorates

**Example**:

```bash
# Monthly optimization
python run.py --strategy my_strategy --mode optimize_sizing

# After major market event (e.g., Fed announcement, macro shift)
python run.py --strategy my_strategy --mode optimize_sizing --backtest

# Quarterly review
python run.py --strategy my_strategy --mode optimize_sizing
```

### 4. Backtest Thoroughly

Always test with `--backtest` flag before applying sizing changes.

**Why**: Validate assumptions work on your data. Real markets may differ from historical data.

**Workflow**:

```bash
# Step 1: Optimize sizing
python run.py --strategy my_strategy --mode optimize_sizing

# Step 2: Review report
# - Check Calmar ratios
# - Review sensitivity analysis
# - Verify no catastrophic drawdowns
# - Confirm win rate and edge are stable

# Step 3: Apply with backup
python run.py --strategy my_strategy --mode optimize_sizing --apply

# Step 4: Backtest with new sizing
python run.py --strategy my_strategy --mode backtest --save-reports
```

### 5. Monitor Live Performance

Track if live results match backtest expectations.

**Why**: Real markets may differ from backtest assumptions. Regular monitoring catches drift early.

**Metrics to Track**:

- Actual Sharpe vs expected Sharpe
- Win rate stability
- Drawdown behavior
- Market regime shifts

**Example**:

```python
# Monthly comparison
current_sharpe = calculate_sharpe(live_equity_curve)
expected_sharpe = 2.1  # From backtest

if abs(current_sharpe - expected_sharpe) > 0.5:
    print("Position sizing may need adjustment")
    run_position_sizing_optimization()
```

### 6. Understand Your Risk Tolerance

Choose your Kelly fraction based on your ability to handle drawdowns.

**Conservative (Quarter Kelly)**: Good for most traders

- Captures 75% of Kelly growth
- 25% of Kelly volatility
- Max drawdowns ~15-20%
- Suitable for long-term wealth building

**Moderate (Half Kelly)**: For risk-tolerant traders

- Captures 50% of Kelly growth
- 50% of Kelly volatility
- Max drawdowns ~22-25%
- Good balance of growth and risk

**Aggressive (1/8 Kelly)**: Only for experienced traders

- Captures 12.5% of Kelly growth
- 12.5% of Kelly volatility
- Max drawdowns ~10-15%
- Maximizes capital preservation

### 7. Combine with Stop Losses

Position sizing works best when combined with proper stop loss management.

**Why**: Kelly maximizes growth for a given win rate and reward/risk ratio, but doesn't determine where to enter/exit. Stop losses ensure your edge is actually realized.

**Best Practice**:

- Use Kelly to determine HOW MUCH to risk per trade
- Use stop losses to determine WHEN to exit (ATR-based or fixed %)
- Both are necessary for complete risk management

## Examples

### Example 1: Kelly Optimization

```bash
# Run optimization
python run.py --strategy ema_crossover_rsi --mode optimize_sizing
```

**Output shows**:

```
==================================================
Kelly Criterion Optimization
==================================================

Raw Kelly: 0.35 (35% of equity)
Applied Fraction: 0.09 (Quarter Kelly)

Statistics:
  Win Rate: 55.2%
  Win/Loss Ratio: 1.82
  Trade Count: 342
  Avg Win: $280
  Avg Loss: $154
  Confidence: High

Calculated Position Size: $8,800
Risk Amount: $2,800
Expected Monthly Return: 6.2%
Max Drawdown: 18.5%

Sensitivity Analysis:
  Kelly Fraction | CAGR  | Max DD | Sharpe | Calmar
  ---------------+-------+--------|--------|--------
  0.125 (1/8)    | 18%   | 12%    | 1.8    | 1.5
  0.250 (1/4)    | 24%   | 15%    | 2.1    | 1.6
  0.500 (1/2)    | 30%   | 22%    | 2.4    | 1.4
  1.000 (Full)   | 38%   | 35%    | 2.7    | 1.1

==================================================
RECOMMENDATION: Quarter Kelly (25% fraction)
==================================================

Why:
  • Best Calmar ratio: 1.6
  • Strong statistical edge with 55% win rate
  • 1.82:1 reward/risk ratio provides good compensation
  • 342 trades provides high confidence
  • Quarter Kelly balances growth and risk well

Applied Parameters:
  • Default Fraction: 0.25
  • Max Fraction: 0.50
  • Position Size Cap: 25% of equity
```

**Interpretation**:

- Strategy has strong edge (55% win rate, 1.82:1 reward ratio)
- Quarter Kelly (25% of raw Kelly) suggests 8-9% per trade is optimal
- Sensitivity analysis shows Quarter Kelly provides best balance
- Expected monthly return ~6% with max drawdown ~19%
- This sizing balances growth and risk well

### Example 2: Apply to Config

```bash
# Optimize and apply
python run.py --strategy ema_crossover_rsi --mode optimize_sizing --apply
```

**Output shows**:

```
==================================================
Applying Position Sizing Optimization
==================================================

Backup created: config.yaml.backup_20250210_103000

Updating config.yaml...

  position_sizing:
    mode: "kelly"
    kelly_fraction: 0.25
    max_fraction: 0.5
    enabled: true

Optimization complete. Review changes before next backtest.
```

**Verification**:

```bash
# Check backup was created
ls config/*.backup*

# Review changes
git diff config.yaml

# Run backtest with new sizing
python run.py --strategy ema_crossover_rsi --mode backtest --save-reports
```

### Example 3: Fresh Backtest Optimization

```bash
# Optimization without validation
python run.py --strategy my_strategy --mode optimize_sizing --backtest
```

**Use cases**:

- You want quick sizing optimization without full validation
- Strategy doesn't have sufficient trades for validation yet
- Testing sizing on specific market regime
- Quick feedback on sizing parameters

**Limitations**:

- Doesn't include Monte Carlo simulation
- Only shows single backtest results
- Not suitable for final strategy approval
- Validation still recommended before live trading

### Example 4: Conservative Kelly

```bash
# For high-risk or edge-uncertain strategies
# Adjust config to use 1/8 Kelly (more conservative)
```

**Configuration**:

```yaml
position_sizing_optimization:
  enabled: true
  kelly:
    enabled: true
    default_fraction: 0.125 # 1/8 Kelly (more conservative)
    max_fraction: 0.25 # Quarter Kelly max
  optimization:
    objective: "calmar"
    risk_tolerance: 0.5 # More conservative
```

**Why Conservative Kelly**:

- When win rate is borderline (45-50%)
- When edge is uncertain or recently changed
- When you have limited historical data
- When you want maximum capital preservation

**Example Output**:

```
Raw Kelly: 0.15 (15% of equity)
Applied Fraction: 0.02 (1/8 Kelly)

Statistics:
  Win Rate: 48.5%
  Win/Loss Ratio: 1.3
  Trade Count: 120
  Confidence: Moderate

Calculated Position Size: $2,000
Risk Amount: $500
Expected Monthly Return: 1.5%
Max Drawdown: 10%

Recommendation: 1/8 Kelly for capital preservation
```

## FAQ

**Q: What is the Kelly Criterion and how does it work?**

A: The Kelly Criterion is a mathematical formula that calculates the optimal percentage of your capital to bet based on your expected edge (win rate and win/loss ratio). The formula is: Kelly% = W - [(1 - W) / R]. Fractional Kelly (using 1/4 or 1/2 of the raw Kelly) is recommended for safety.

**Q: How do I calculate my Win Rate and Win/Loss Ratio for Kelly?**

A: Use your out-of-sample trade results. Win rate = (Number of profitable trades / Total trades) × 100. Win/loss ratio = Average winning trade size / Average losing trade size. Both should be calculated on the OOS period from your validation.

**Q: What's the difference between Quarter Kelly and Half Kelly?**

A: Quarter Kelly uses 25% of the raw Kelly percentage, Half Kelly uses 50%. Quarter Kelly captures 75% of maximum growth with 25% of volatility. Half Kelly captures 50% of growth with 50% of volatility. Quarter Kelly is generally recommended for most traders.

**Q: Can Kelly Criterion be negative?**

A: Yes, Kelly can be negative if your strategy's win rate is too low or win/loss ratio is too poor. Negative Kelly means your strategy has no statistical edge and should not be traded. Position sizing cannot fix a strategy with negative expectancy.

**Q: What if my Kelly result is very high (e.g., 50% or more)?**

A: A high Kelly percentage usually indicates either:

1. Very high win rate (> 60%) with good reward/risk ratio
2. Insufficient historical data (sample size too small)
3. Overfitting to historical data

For high Kelly results, consider:

- Using a smaller fraction (Quarter Kelly or Half Kelly)
- Ensuring sufficient OOS data (200+ trades minimum)
- Verifying win rate and edge are stable across regimes

**Q: How often should I re-optimize position sizing?**

A: Monthly to quarterly, or whenever market conditions change significantly. For crypto, monthly is recommended due to higher volatility. For stocks, quarterly may be sufficient. Re-optimize after major market events, regime shifts, or significant changes in strategy performance.

**Q: Can I override the recommended parameters?**

A: Yes. The optimization provides recommendations with reasoning. You can manually edit config.yaml with different values. Consider testing your changes with `--backtest` before applying. If you prefer a conservative approach, use 1/8 or 1/4 Kelly. If you're comfortable with higher risk, consider Half Kelly.

**Q: What's the relationship between position sizing and validation?**

A: Position sizing optimization should be done AFTER validation passes. Use the validated strategy results to optimize sizing. Never optimize sizing without validation to avoid overfitting. The OOS period provides reliable estimates of win rate and win/loss ratio.

**Q: Does position sizing affect walk-forward optimization?**

A: Yes. Each parameter set is optimized with the same sizing. Walk-forward windows test sizing across different market regimes. This ensures sizing works well in both calm and volatile conditions. Always validate with position sizing included.

**Q: How does position sizing impact Monte Carlo simulation?**

A: Monte Carlo simulation resamples trades from out-of-sample period. Position sizing affects trade distribution (larger positions produce larger PnL swings). This makes Monte Carlo analysis more relevant when sizing is optimized. Use the sensitivity analysis to understand trade distribution under different Kelly fractions.

**Q: What if I have multiple strategies?**

A: Run optimization separately for each strategy. Consider portfolio-level effects if trading multiple strategies simultaneously. Each strategy's Kelly sizing should be optimized independently based on its own edge. Aggressive sizing across multiple correlated strategies increases portfolio risk.

**Q: Why is there a max_position_pct of 25%?**

A: This prevents overconcentration in a single position. Even if Kelly suggests 50%, the constraint protects you from a single catastrophic loss. Professional traders typically use 10-25% per position. This cap ensures you never risk too much on a single trade.

**Q: Can I use this for live trading?**

A: This framework is backtest-only. Use the optimization results to inform your live trading position sizing, but always test live with small sizes first. Never use backtest-only sizing decisions without live validation. Monitor live performance and be prepared to adjust sizing if performance differs from expectations.

**Q: What's the relationship between position sizing and stop losses?**

A: Position sizing determines HOW MUCH to risk per trade. Stop losses determine WHERE to exit. Use both together. Kelly maximizes growth for a given edge, but stop losses ensure your edge is realized. Never rely on position sizing alone for risk management.

## See Also

- [Strategy Development Guide](STRATEGY.md)
- [Framework Architecture](FRAMEWORK.md)
- [Validation Framework](FRAMEWORK.md#validation)
- [Kelly Criterion Paper](https://en.wikipedia.org/wiki/Kelly_criterion)
