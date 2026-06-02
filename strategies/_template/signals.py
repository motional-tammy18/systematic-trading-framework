"""Template strategy with EMA crossover and RSI filter."""

import polars as pl
from engine.base_strategy import BaseStrategy


class TemplateStrategy(BaseStrategy):
    """Template strategy with EMA crossover and RSI filter.

    Entry conditions:
    - Long: Short EMA crosses above Long EMA AND RSI < oversold threshold
    - Short: Short EMA crosses below Long EMA AND RSI > overbought threshold

    Exit conditions:
    - Long exit: Short EMA crosses below Long EMA OR RSI > overbought
    - Short exit: Short EMA crosses above Long EMA OR RSI < oversold

    This template provides a complete, working example that can be
    copied and modified for new strategies.
    """

    def _calculate_ema(self, df: pl.DataFrame, period: int) -> pl.Expr:
        """Calculate Exponential Moving Average using polars expressions.

        EMA = (close * multiplier) + (previous_ema * (1 - multiplier))
        where multiplier = 2 / (period + 1)
        """
        multiplier = 2.0 / (period + 1)
        return (
            pl.col("close")
            .ewm_mean(alpha=multiplier, adjust=False)
            .alias(f"ema_{period}")
        )

    def _calculate_rsi(self, df: pl.DataFrame, period: int) -> pl.Expr:
        """Calculate Relative Strength Index using polars expressions.

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

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals using EMA crossover and RSI filter.

        Entry conditions:
        - Long: Short EMA crosses above Long EMA AND RSI < oversold threshold
        - Short: Short EMA crosses below Long EMA AND RSI > overbought threshold

        Exit conditions:
        - Long exit: Short EMA crosses below Long EMA OR RSI > overbought
        - Short exit: Short EMA crosses above Long EMA OR RSI < oversold

        This template implements position-aware exits using cumulative entry
        tracking. Exit signals only trigger if a corresponding position is open.

        Args:
            data: Polars DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            Polars DataFrame with additional 'signal' column:
            - 1 = long position
            - -1 = short position
            - 0 = flat (no position)
        """
        params = self.params

        ema_short_period = params.get("ema_short", 20)
        ema_long_period = params.get("ema_long", 50)
        rsi_period = params.get("rsi_period", 14)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)

        df = data.with_columns([
            self._calculate_ema(data, ema_short_period),
            self._calculate_ema(data, ema_long_period),
            self._calculate_rsi(data, rsi_period),
        ])

        df = df.rename({
            f"ema_{ema_short_period}": "ema_short",
            f"ema_{ema_long_period}": "ema_long",
            f"rsi_{rsi_period}": "rsi",
        })

        # Initialize signal column
        df = df.with_columns([
            pl.lit(0, dtype=pl.Int8).alias("signal")
        ])

        # Define entry conditions
        long_entry = (
            (pl.col("ema_short") > pl.col("ema_long")) &
            (pl.col("rsi") < rsi_oversold)
        )

        short_entry = (
            (pl.col("ema_short") < pl.col("ema_long")) &
            (pl.col("rsi") > rsi_overbought)
        )

        # Define exit conditions
        long_exit = (pl.col("ema_short") <= pl.col("ema_long")) | (
            pl.col("rsi") > rsi_overbought
        )
        short_exit = (pl.col("ema_short") >= pl.col("ema_long")) | (
            pl.col("rsi") < rsi_oversold
        )

        # Track entries using cumulative sum (stateless position tracking)
        # Use Int32 to prevent overflow from cumulative sum
        df = df.with_columns([
            pl.when(long_entry)
            .then(pl.lit(1, dtype=pl.Int32))
            .when(short_entry)
            .then(pl.lit(-1, dtype=pl.Int32))
            .otherwise(pl.lit(0, dtype=pl.Int32))
            .alias("entry_trigger")
        ])

        # Calculate position state from entries alone
        df = df.with_columns([
            pl.col("entry_trigger")
            .cum_sum()
            .alias("position_state_raw")
        ])

        # Apply position-aware exits: exit only when transitioning from position -> flat
        # Check previous row's position state and current row's exit condition
        df = df.with_columns([
            pl.when((pl.col("position_state_raw").shift(1) > 0) & long_exit)
            .then(pl.lit(0, dtype=pl.Int32))
            .when((pl.col("position_state_raw").shift(1) < 0) & short_exit)
            .then(pl.lit(0, dtype=pl.Int32))
            .otherwise(pl.col("position_state_raw"))
            .alias("position_state")
        ])

        # Generate final signal from position_state
        df = df.with_columns([
            pl.when(pl.col("position_state") > 0)
            .then(pl.lit(1, dtype=pl.Int8))
            .when(pl.col("position_state") < 0)
            .then(pl.lit(-1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("signal")
        ])

        # Clean up temporary columns
        df = df.drop(["entry_trigger", "position_state_raw", "position_state"])

        return df
