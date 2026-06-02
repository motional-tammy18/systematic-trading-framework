# Contributing to Systematic Trading Framework

Thank you for your interest in contributing to this framework. This document provides guidelines and instructions for contributing.

---

## How to Contribute

### Reporting Bugs

Before reporting, determine where the bug originates -- this affects what information we need.

**Engine bugs** (in `engine/`, `data/`, `run.py`, `config/`, `visualization/` -- the framework itself):

- Provide a clear description of the issue
- Include reproduction steps using the example strategy or a minimal script
- Attach the full stack trace and error output

**Strategy bugs** (in a custom `strategies/<name>/` implementation):

- Describe the **approach and logic** conceptually (e.g., "an SMA crossover with a volume filter") -- you do not need to share proprietary code or exact parameters
- Specify the timeframe (e.g., 15m, 1h) and symbol if using a standard pair
- Include the full stack trace and error output
- If the error is reproducible with the example strategy, mention that

Bug reports should be filed on GitHub with the label `bug`. Include as much context as you can share -- vague bug reports without reproduction steps cannot be triaged effectively.

### Suggesting Features

We welcome suggestions for new features, including:

- New technical indicators (e.g., custom oscillators, volume-based indicators)
- Additional validation methods or statistical tests
- Performance optimizations
- Documentation improvements

Feature requests should be filed on GitHub with the label `enhancement`. Describe the problem you are trying to solve and why the feature would be useful to the broader community.

### Submitting Pull Requests

- Keep pull requests focused and small -- one feature or fix per PR
- Include tests for new functionality (if applicable)
- Update documentation for any changed behavior
- Ensure all existing backtests still pass before submitting

---

## Development Setup

### Clone the Repository

```bash
git clone https://github.com/dikibagast/systematic-trading-framework.git
cd systematic-trading
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Setup

```bash
python run.py --strategy ema_crossover_rsi --mode backtest
```

This runs a backtest on the included example strategy. If the setup is correct, you will see output showing trade metrics (total trades, win rate, Sharpe ratio, etc.). Any errors at this stage typically indicate a missing dependency or Python version mismatch.

---

## Coding Standards

### Import Organization (MANDATORY)

All Python files must use the following import order:

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

### Type Hints (MANDATORY)

All functions must have type hints for parameters and return types. This is enforced for code consistency and to support static analysis tools.

```python
def run_backtest(
    signals: pl.DataFrame,
    config: dict,
    position_sizing_mode: str = "percent_equity",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Run vectorized backtest on signal data."""
    pass
```

### Docstrings (Google Style)

All public functions must have docstrings in Google format with Args, Returns, and Raises sections:

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
    pass
```

### Naming Conventions

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Private functions: `_leading_underscore`
- Constants: `UPPER_SNAKE_CASE`

### Polars Usage (MANDATORY - NO PANDAS IN STRATEGY/ENGINE CODE)

Use polars expressions for all data operations. Never use pandas in strategy or engine code.

```python
# Correct
df = df.with_columns([
    pl.col("close").ewm_mean(alpha=0.1).alias("ema")
])
signals = pl.when(condition).then(1).otherwise(-1)

# Wrong -- never use pandas in strategy or engine code
import pandas as pd
```

**Exception**: The `visualization/` directory may use pandas for Streamlit/Plotly display purposes.

### Error Handling

Validate inputs at function boundaries. Raise typed exceptions with descriptive messages:

```python
def run_backtest(signals: pl.DataFrame, config: dict) -> Tuple:
    required_cols = {"timestamp", "open", "high", "low", "close", "volume", "signal"}
    if not required_cols.issubset(set(signals.columns)):
        missing = required_cols - set(signals.columns)
        raise ValueError(f"Missing required columns: {missing}")
```

### Configuration Management

All strategy parameters MUST be in `config.yaml`. Access via `self.params.get("key", default)`. Never hardcode values.

---

## Hard Rules

These rules are enforced by the project maintainer and cannot be bypassed:

- **NEVER** use pandas - use polars only
- **NEVER** use `as any`, `@ts-ignore`, `@ts-expect-error`
- **NEVER** install new dependencies without opening an issue first
- **NEVER** hardcode values - use YAML configs
- **NEVER** leave TODO comments - fully implement before committing
- **NEVER** commit unless explicitly requested
- **NO** live trading code
- All strategies must be stateless and deterministic

---

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes (keep scope focused -- do not mix unrelated changes)
4. Run backtest verification: `python run.py --strategy <name> --mode backtest`
5. Open a pull request against the `main` branch
6. Link any related issues in the PR description
7. Respond to review feedback in a timely manner

PRs that violate the coding standards or hard rules will be returned for revision.

---

## Strategy Development Workflow

### Step 1: Copy the Template

```bash
cp -r strategies/_template/ strategies/<your_strategy>/
```

### Step 2: Configure Parameters

Edit `strategies/<your_strategy>/config.yaml` with your parameters and optimization space. Define all tunable values in `param_space` for grid search.

### Step 3: Implement Signals

Implement `generate_signals()` in `strategies/<your_strategy>/signals.py` following the signal contract:

- `1` = Long position
- `-1` = Short position
- `0` = Flat (no position)

The function must return a DataFrame with a `signal` column.

### Step 4: Test

```bash
python run.py --strategy <name> --mode backtest
```

### Step 5: Full Validation (when ready)

```bash
python run.py --strategy <name> --mode full
```

Full validation runs walk-forward optimization and Monte Carlo simulation. This takes significantly longer than backtest mode -- reserve it for final verification.

Reference `docs/STRATEGY.md` for the complete strategy development guide.

---

## Reporting Issues

A good issue report includes:

- **Title**: Clear, descriptive summary of the problem
- **Where it occurred**: Engine (framework) or Strategy (custom implementation)
- **Description**: Detailed explanation of what went wrong
- **Timeframe**: Candle timeframe that was active when the error occurred (if relevant)
- **Symbol**: Trading pair (e.g., BTC/USDT:USDT) (if relevant)
- **Strategy approach**: High-level description of the strategy logic if the issue is strategy-related (proprietary code stays private)
- **Steps to reproduce**: Clear sequence to trigger the issue (use example strategy if possible)
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happened
- **Full error output**: Complete traceback and error message

Do not report issues without reproduction steps. Issues without enough information to diagnose will be closed.

---

## Getting Help

- Read `docs/FRAMEWORK.md` for framework architecture and modification guidelines
- Read `docs/STRATEGY.md` for strategy development and signal contract reference
- Read `docs/POSITION_SIZING.md` for Kelly Criterion optimization details
- Check existing GitHub issues before opening a new one
- Open a GitHub discussion for general questions

---

## Documentation

| Document                  | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| `AGENTS.md`               | AI agent router and development guidelines    |
| `docs/STRATEGY.md`        | Strategy development guide                    |
| `docs/FRAMEWORK.md`       | Framework architecture and modification guide |
| `docs/POSITION_SIZING.md` | Kelly Criterion position sizing methodology   |

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
