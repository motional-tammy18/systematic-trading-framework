import polars as pl
from engine.base_strategy import BaseStrategy


class EMACrossoverRSI(BaseStrategy):
    """EMA crossover with RSI filter and trend quality filters.

    Entry conditions (EMA + RSI + Trend Quality):
    - Long: EMA bullish (short > long) AND RSI < rsi_entry_long AND trending market
    - Short: EMA bearish (short < long) AND RSI > rsi_entry_short AND trending market

    Trend quality filters:
    - EMA separation ratio: ema_short/ema_long must exceed adaptive threshold
    - Long EMA slope: long EMA must be trending in entry direction

    Exit conditions (RSI + Trend):
    - Long exit: RSI > rsi_exit_long OR no longer trending
    - Short exit: RSI < rsi_exit_short OR no longer trending

    RSI levels derived from offsets:
    - rsi_entry_long = 50 - rsi_entry_offset
    - rsi_entry_short = 50 + rsi_entry_offset
    - rsi_exit_long = 100 - rsi_exit_offset
    - rsi_exit_short = 0 + rsi_exit_offset
    """

    def _calculate_ema(self, df: pl.DataFrame, period: int, name: str) -> pl.Expr:
        multiplier = 2.0 / (period + 1)
        return pl.col("close").ewm_mean(alpha=multiplier, adjust=False).alias(name)

    def _calculate_rsi(self, df: pl.DataFrame, period: int, name: str) -> pl.Expr:
        delta = pl.col("close").diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)
        avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False)
        avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.alias(name)

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        params = self.params

        ema_short_period = int(params.get("ema_short", 20))
        ema_long_period = int(params.get("ema_long", 50))
        rsi_period = int(params.get("rsi_period", 14))
        rsi_entry_offset = params.get("rsi_entry_offset", 5)
        rsi_exit_offset = params.get("rsi_exit_offset", 20)
        trend_slope_lookback = int(params.get("trend_slope_lookback", 10))
        trend_separation_mult = params.get("trend_separation_mult", 1.0)

        rsi_entry_long = 50 - rsi_entry_offset
        rsi_entry_short = 50 + rsi_entry_offset
        rsi_exit_long = 100 - rsi_exit_offset
        rsi_exit_short = rsi_exit_offset

        df = data.with_columns(
            [
                self._calculate_ema(data, ema_short_period, "ema_short"),
                self._calculate_ema(data, ema_long_period, "ema_long"),
                self._calculate_rsi(data, rsi_period, "rsi"),
            ]
        )

        df = df.with_columns(
            [(pl.col("ema_short") / pl.col("ema_long")).alias("ema_ratio")]
        )

        df = df.with_columns(
            [
                (
                    (
                        pl.col("ema_long")
                        - pl.col("ema_long").shift(trend_slope_lookback)
                    )
                    / pl.col("ema_long").shift(trend_slope_lookback)
                ).alias("long_ema_slope")
            ]
        )

        df = df.with_columns(
            [
                (
                    pl.col("close")
                    .pct_change()
                    .rolling_std(window_size=trend_slope_lookback)
                    * trend_separation_mult
                ).alias("min_separation")
            ]
        )

        bullish_trending = (
            (pl.col("ema_short") > pl.col("ema_long"))
            & (pl.col("ema_ratio") > (1 + pl.col("min_separation")))
            & (pl.col("long_ema_slope") > 0)
        )

        bearish_trending = (
            (pl.col("ema_short") < pl.col("ema_long"))
            & (pl.col("ema_ratio") < (1 - pl.col("min_separation")))
            & (pl.col("long_ema_slope") < 0)
        )

        long_entry = bullish_trending & (pl.col("rsi") < rsi_entry_long)
        short_entry = bearish_trending & (pl.col("rsi") > rsi_entry_short)

        long_exit = (pl.col("rsi") > rsi_exit_long) | bearish_trending
        short_exit = (pl.col("rsi") < rsi_exit_short) | bullish_trending

        entry_signal = (
            pl.when(long_entry)
            .then(pl.lit(1, dtype=pl.Int8))
            .when(short_entry)
            .then(pl.lit(-1, dtype=pl.Int8))
            .otherwise(None)
        )

        df = df.with_columns(
            [
                entry_signal.alias("entry_signal"),
                long_exit.alias("long_exit"),
                short_exit.alias("short_exit"),
            ]
        )

        df = df.with_columns(
            [
                pl.col("entry_signal")
                .forward_fill()
                .fill_null(0)
                .alias("position_from_entries")
            ]
        )

        df = df.with_columns(
            [
                pl.col("position_from_entries")
                .shift(1)
                .fill_null(0)
                .alias("prev_position")
            ]
        )

        df = df.with_columns(
            [
                ((pl.col("prev_position") == 1) & pl.col("long_exit")).alias(
                    "should_exit_long"
                ),
                ((pl.col("prev_position") == -1) & pl.col("short_exit")).alias(
                    "should_exit_short"
                ),
            ]
        )

        df = df.with_columns(
            [
                pl.when(pl.col("entry_signal").is_not_null())
                .then(pl.col("entry_signal"))
                .when(pl.col("should_exit_long") | pl.col("should_exit_short"))
                .then(pl.lit(0, dtype=pl.Int8))
                .otherwise(None)
                .alias("combined_signal")
            ]
        )

        df = df.with_columns(
            [
                pl.col("combined_signal")
                .shift(1)
                .forward_fill()
                .fill_null(0)
                .alias("signal")
            ]
        )

        df = df.drop(
            [
                "entry_signal",
                "position_from_entries",
                "prev_position",
                "should_exit_long",
                "should_exit_short",
                "combined_signal",
                "ema_ratio",
                "long_ema_slope",
                "min_separation",
            ]
        )

        return df
