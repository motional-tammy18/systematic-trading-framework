"""Order lifecycle management for backtesting.

This module handles order fill simulation, position lifecycle tracking,
and TP/SL/trailing stop management. It sits between signal generation
and PnL calculation, resolving what actually happened on each bar.
"""

from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import polars as pl
import numpy as np


class PositionState(Enum):
    """Position lifecycle states."""

    PENDING_ENTRY = "pending_entry"
    ACTIVE = "active"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class ExitReason(Enum):
    """Reasons for position exit."""

    TP = "TP"
    SL = "SL"
    SIGNAL = "SIGNAL"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"


class OrderManager:
    """Manages order lifecycle and position state for backtesting.

    Tracks positions through their lifecycle (pending_entry → active → closed),
    handles market and limit entries, manages TP/SL levels (fixed % and ATR-based),
    implements trailing stops, and applies bar fill assumptions for realistic
    backtest simulation.

    Args:
        config: Configuration dict containing order_management and backtest settings

    Example:
        >>> config = {
        ...     'order_management': {
        ...         'entry': {'type': 'market'},
        ...         'take_profit': {'type': 'fixed_pct', 'value': 0.02},
        ...         'stop_loss': {'type': 'fixed_pct', 'value': 0.01},
        ...         'trailing_stop': {'enabled': False},
        ...         'atr_period': 14,
        ...     },
        ...     'backtest': {
        ...         'bar_fill_assumption': 'pessimistic',
        ...         'slippage_pct': 0.0005,
        ...     }
        ... }
        >>> manager = OrderManager(config)
        >>> resolved_trades = manager.process_signals(signals_df, data_df)
    """

    def __init__(self, config: dict):
        """Initialize OrderManager with configuration."""
        self.config = config

        # Order management config
        om_config = config.get("order_management", {})
        self.entry_type = om_config.get("entry", {}).get("type", "market")
        self.tp_config = om_config.get("take_profit", {"type": "none"})
        self.sl_config = om_config.get("stop_loss", {"type": "none"})
        self.trailing_config = om_config.get("trailing_stop", {"enabled": False})
        self.atr_period = om_config.get("atr_period", 14)
        self.max_entry_bars = om_config.get("max_entry_bars", 5)
        self.max_holding_bars = om_config.get("max_holding_bars", 0)  # 0 = disabled

        # Backtest config
        bt_config = config.get("backtest", {})
        self.bar_fill = bt_config.get("bar_fill_assumption", "pessimistic")
        self.slippage_pct = bt_config.get("slippage_pct", 0.0005)

        # Validate bar_fill assumption
        if self.bar_fill not in ("pessimistic", "optimistic", "random"):
            raise ValueError(
                f"Invalid bar_fill_assumption: {self.bar_fill}. "
                "Must be 'pessimistic', 'optimistic', or 'random'"
            )

        # Random state for bar_fill="random"
        self.rng = np.random.RandomState(42)

    def _calculate_atr(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate Average True Range (ATR) for the dataset.

        Uses the standard ATR formula:
        TR = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = EMA(TR, period)

        Args:
            data: DataFrame with high, low, close columns

        Returns:
            DataFrame with added 'atr' column
        """
        # Calculate True Range components
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift(1)).abs()
        low_close = (data["low"] - data["close"].shift(1)).abs()

        # True Range is the max of the three
        tr = pl.max_horizontal([high_low, high_close, low_close])

        # ATR is EMA of TR
        atr = tr.ewm_mean(alpha=1.0 / self.atr_period, adjust=False)

        return data.with_columns(atr.alias("atr"))

    def _get_tp_sl_prices(
        self,
        entry_price: float,
        direction: int,
        bar_atr: Optional[float],
        strategy_tp: Optional[float],
        strategy_sl: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate TP and SL prices based on configuration.

        Priority:
        1. Strategy-provided levels (if present)
        2. Config-based calculation (fixed_pct or atr_multiple)
        3. None (no TP/SL)

        Args:
            entry_price: Position entry price
            direction: 1 for long, -1 for short
            bar_atr: ATR value at entry bar (for atr_multiple type)
            strategy_tp: TP price from strategy (optional)
            strategy_sl: SL price from strategy (optional)

        Returns:
            Tuple of (tp_price, sl_price)
        """
        tp_price = None
        sl_price = None

        # Take Profit calculation
        if strategy_tp is not None:
            # Strategy provided explicit TP level
            tp_price = strategy_tp
        elif self.tp_config["type"] == "fixed_pct":
            pct = self.tp_config.get("value", 0.02)
            if direction == 1:  # Long
                tp_price = entry_price * (1 + pct)
            else:  # Short
                tp_price = entry_price * (1 - pct)
        elif self.tp_config["type"] == "atr_multiple":
            if bar_atr is not None and bar_atr > 0:
                multiple = self.tp_config.get("value", 2.0)
                if direction == 1:  # Long
                    tp_price = entry_price + (bar_atr * multiple)
                else:  # Short
                    tp_price = entry_price - (bar_atr * multiple)

        # Stop Loss calculation
        if strategy_sl is not None:
            # Strategy provided explicit SL level
            sl_price = strategy_sl
        elif self.sl_config["type"] == "fixed_pct":
            pct = self.sl_config.get("value", 0.01)
            if direction == 1:  # Long
                sl_price = entry_price * (1 - pct)
            else:  # Short
                sl_price = entry_price * (1 + pct)
        elif self.sl_config["type"] == "atr_multiple":
            if bar_atr is not None and bar_atr > 0:
                multiple = self.sl_config.get("value", 1.0)
                if direction == 1:  # Long
                    sl_price = entry_price - (bar_atr * multiple)
                else:  # Short
                    sl_price = entry_price + (bar_atr * multiple)

        return tp_price, sl_price

    def _check_limit_fill(
        self,
        direction: int,
        limit_price: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
    ) -> Tuple[bool, float]:
        """Check if a limit order would fill within a bar.

        For longs: fills if bar_low <= limit_price <= bar_high
        For shorts: fills if bar_low <= limit_price <= bar_high

        Args:
            direction: 1 for long, -1 for short
            limit_price: Limit order price level
            bar_open: Bar open price
            bar_high: Bar high price
            bar_low: Bar low price

        Returns:
            Tuple of (filled: bool, fill_price: float)
        """
        # Limit fills if price touches the level
        if bar_low <= limit_price <= bar_high:
            # Fill at limit price (or better)
            if direction == 1:  # Long - fill at or below limit
                fill_price = min(limit_price, bar_open)
            else:  # Short - fill at or above limit
                fill_price = max(limit_price, bar_open)
            return True, fill_price
        return False, 0.0

    def _check_tp_sl_hit(
        self,
        direction: int,
        tp_price: Optional[float],
        sl_price: Optional[float],
        bar_high: float,
        bar_low: float,
    ) -> Tuple[Optional[ExitReason], float]:
        """Check if TP or SL was hit within a bar.

        Uses bar_fill_assumption to determine order when both could trigger:
        - pessimistic: SL before TP (worst case)
        - optimistic: TP before SL (best case)
        - random: 50/50 per bar

        Args:
            direction: 1 for long, -1 for short
            tp_price: Take profit price level (None if no TP)
            sl_price: Stop loss price level (None if no SL)
            bar_high: Bar high price
            bar_low: Bar low price

        Returns:
            Tuple of (exit_reason: ExitReason or None, exit_price: float)
        """
        # Check if TP hit
        tp_hit = False
        if tp_price is not None:
            if direction == 1:  # Long: TP hit if high >= tp_price
                tp_hit = bar_high >= tp_price
            else:  # Short: TP hit if low <= tp_price
                tp_hit = bar_low <= tp_price

        # Check if SL hit
        sl_hit = False
        if sl_price is not None:
            if direction == 1:  # Long: SL hit if low <= sl_price
                sl_hit = bar_low <= sl_price
            else:  # Short: SL hit if high >= sl_price
                sl_hit = bar_high >= sl_price

        if not tp_hit and not sl_hit:
            return None, 0.0

        # Determine which hit first based on bar_fill_assumption
        if tp_hit and sl_hit:
            # Both hit - use assumption to decide
            if self.bar_fill == "pessimistic":
                # Worst case: SL before TP
                sl_hit = True
                tp_hit = False
            elif self.bar_fill == "optimistic":
                # Best case: TP before SL
                sl_hit = False
                tp_hit = True
            else:  # random
                # 50/50 chance
                if self.rng.random() < 0.5:
                    sl_hit = True
                    tp_hit = False
                else:
                    sl_hit = False
                    tp_hit = True

        if sl_hit:
            # Exit at SL price (or worse if gapped)
            if direction == 1:  # Long
                exit_price = min(sl_price, bar_low) if sl_price else bar_low
            else:  # Short
                exit_price = max(sl_price, bar_high) if sl_price else bar_high
            return ExitReason.SL, exit_price

        if tp_hit:
            # Exit at TP price (or better if gapped)
            if direction == 1:  # Long
                exit_price = max(tp_price, bar_high) if tp_price else bar_high
            else:  # Short
                exit_price = min(tp_price, bar_low) if tp_price else bar_low
            return ExitReason.TP, exit_price

        return None, 0.0

    def _update_trailing_stop(
        self,
        direction: int,
        current_sl: Optional[float],
        trailing_price: float,
        bar_high: float,
        bar_low: float,
    ) -> Tuple[Optional[float], float, bool]:
        """Update trailing stop level based on price movement.

        Args:
            direction: 1 for long, -1 for short
            current_sl: Current stop loss price (None if not set)
            trailing_price: Current trailing reference price (highest for longs, lowest for shorts)
            bar_high: Bar high price
            bar_low: Bar low price

        Returns:
            Tuple of (new_sl: float or None, new_trailing_price: float, triggered: bool)
        """
        if not self.trailing_config.get("enabled", False):
            return current_sl, trailing_price, False

        distance_pct = self.trailing_config.get("distance_pct", 0.005)

        # Update trailing reference price
        if direction == 1:  # Long - track highest price
            new_trailing = max(trailing_price, bar_high)
            # Calculate new SL level
            new_sl = new_trailing * (1 - distance_pct)
            # Only move SL up (tighten), never down
            if current_sl is not None:
                new_sl = max(new_sl, current_sl)
            # Check if triggered
            triggered = bar_low <= new_sl
        else:  # Short - track lowest price
            new_trailing = min(trailing_price, bar_low)
            # Calculate new SL level
            new_sl = new_trailing * (1 + distance_pct)
            # Only move SL down (tighten), never up
            if current_sl is not None:
                new_sl = min(new_sl, current_sl)
            # Check if triggered
            triggered = bar_high >= new_sl

        return new_sl, new_trailing, triggered

    def process_signals(
        self, signals: pl.DataFrame, data: pl.DataFrame
    ) -> pl.DataFrame:
        """Process signals into resolved trades with entry/exit details.

        Tracks position lifecycle through states:
        - PENDING_ENTRY: Signal changed, waiting for fill
        - ACTIVE: Position open, monitoring TP/SL/trailing
        - CLOSED: Exit triggered, trade complete

        Args:
            signals: DataFrame with signal column (1=long, -1=short, 0=flat)
                     and optional entry_price, tp_price, sl_price columns
            data: DataFrame with OHLCV data aligned with signals

        Returns:
            DataFrame with resolved trades including entry/exit details,
            exit reasons, TP/SL prices, and trade statistics
        """
        # Validate required columns
        required_signal_cols = {"timestamp", "signal"}
        if not required_signal_cols.issubset(set(signals.columns)):
            missing = required_signal_cols - set(signals.columns)
            raise ValueError(f"Signals missing required columns: {missing}")

        required_data_cols = {"timestamp", "open", "high", "low", "close"}
        if not required_data_cols.issubset(set(data.columns)):
            missing = required_data_cols - set(data.columns)
            raise ValueError(f"Data missing required columns: {missing}")

        # Calculate ATR if needed for TP/SL
        if (
            self.tp_config.get("type") == "atr_multiple"
            or self.sl_config.get("type") == "atr_multiple"
        ):
            data = self._calculate_atr(data)

        # Convert to numpy for fast iteration
        timestamps = data["timestamp"].to_numpy()
        opens = data["open"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        signal_values = signals["signal"].to_numpy()

        # Optional strategy-provided levels
        strategy_entry = signals.get_column("entry_price").to_numpy() if "entry_price" in signals.columns else None
        strategy_tp = signals.get_column("tp_price").to_numpy() if "tp_price" in signals.columns else None
        strategy_sl = signals.get_column("sl_price").to_numpy() if "sl_price" in signals.columns else None

        atr_values = data.get_column("atr").to_numpy() if "atr" in data.columns else None

        n_bars = len(data)

        # Trade tracking
        trades: List[Dict[str, Any]] = []

        # Current position state
        state = PositionState.CLOSED
        position_direction = 0
        entry_price = 0.0
        entry_bar = -1
        entry_type_str = "market"
        tp_price: Optional[float] = None
        sl_price: Optional[float] = None
        trailing_price = 0.0
        max_price = 0.0  # Highest for longs, lowest for shorts
        pending_bars = 0

        for i in range(n_bars):
            curr_signal = int(signal_values[i])
            bar_open = float(opens[i])
            bar_high = float(highs[i])
            bar_low = float(lows[i])
            bar_atr = float(atr_values[i]) if atr_values is not None else None

            # Get strategy-provided levels for this bar
            strat_entry = float(strategy_entry[i]) if strategy_entry is not None else None
            strat_tp = float(strategy_tp[i]) if strategy_tp is not None else None
            strat_sl = float(strategy_sl[i]) if strategy_sl is not None else None

            if state == PositionState.CLOSED:
                # Check for new signal
                if curr_signal != 0:
                    if self.entry_type == "market":
                        # Market entry: fill immediately at bar open + slippage
                        slippage_factor = self.rng.uniform(-0.5, 0.5)
                        entry_price = bar_open * (1 + slippage_factor * self.slippage_pct)
                        entry_bar = i
                        entry_type_str = "market"
                        position_direction = curr_signal

                        # Calculate TP/SL
                        tp_price, sl_price = self._get_tp_sl_prices(
                            entry_price, position_direction, bar_atr, strat_tp, strat_sl
                        )

                        # Initialize trailing tracking
                        max_price = entry_price
                        trailing_price = entry_price

                        state = PositionState.ACTIVE
                    else:  # limit entry
                        # Limit entry: go to pending state
                        position_direction = curr_signal
                        entry_price = strat_entry if strat_entry is not None else bar_open
                        entry_type_str = "limit"
                        pending_bars = 0
                        state = PositionState.PENDING_ENTRY

            elif state == PositionState.PENDING_ENTRY:
                # Check for limit fill
                limit_price = strat_entry if strat_entry is not None else entry_price
                filled, fill_price = self._check_limit_fill(
                    position_direction, limit_price, bar_open, bar_high, bar_low
                )

                if filled:
                    # Limit filled - activate position
                    entry_price = fill_price
                    entry_bar = i

                    # Calculate TP/SL
                    tp_price, sl_price = self._get_tp_sl_prices(
                        entry_price, position_direction, bar_atr, strat_tp, strat_sl
                    )

                    # Initialize trailing tracking
                    max_price = entry_price
                    trailing_price = entry_price

                    state = PositionState.ACTIVE
                else:
                    # Not filled - check for cancellation
                    pending_bars += 1
                    if curr_signal == 0 or curr_signal != position_direction:
                        # Signal cancelled or reversed
                        state = PositionState.CANCELLED
                    elif pending_bars >= self.max_entry_bars:
                        # Max wait time exceeded
                        state = PositionState.CANCELLED

                if state == PositionState.CANCELLED:
                    # Reset to closed, no trade recorded
                    state = PositionState.CLOSED
                    position_direction = 0
                    entry_price = 0.0

            elif state == PositionState.ACTIVE:
                # Update max price tracking
                if position_direction == 1:  # Long
                    max_price = max(max_price, bar_high)
                else:  # Short
                    max_price = min(max_price, bar_low)

                # Check for exit conditions
                exit_reason: Optional[ExitReason] = None
                exit_price = 0.0

                # 1. Check TP/SL
                tp_sl_reason, tp_sl_price = self._check_tp_sl_hit(
                    position_direction, tp_price, sl_price, bar_high, bar_low
                )

                if tp_sl_reason is not None:
                    exit_reason = tp_sl_reason
                    exit_price = tp_sl_price

                # 2. Check trailing stop (if no TP/SL hit and trailing enabled)
                if exit_reason is None and self.trailing_config.get("enabled", False):
                    new_sl, new_trailing, triggered = self._update_trailing_stop(
                        position_direction, sl_price, trailing_price, bar_high, bar_low
                    )
                    sl_price = new_sl
                    trailing_price = new_trailing

                    if triggered:
                        exit_reason = ExitReason.TRAILING_STOP
                        exit_price = sl_price if sl_price is not None else (
                            bar_low if position_direction == 1 else bar_high
                        )

                # 3. Check time-based stop loss
                if exit_reason is None and self.max_holding_bars > 0:
                    bars_held = i - entry_bar
                    if bars_held >= self.max_holding_bars:
                        exit_reason = ExitReason.TIME_STOP
                        # Exit at bar open with slippage
                        slippage_factor = self.rng.uniform(-0.5, 0.5)
                        exit_price = bar_open * (1 + slippage_factor * self.slippage_pct)

                # 4. Check signal change
                if exit_reason is None:
                    if curr_signal == 0 or (
                        curr_signal != 0 and curr_signal != position_direction
                    ):
                        exit_reason = ExitReason.SIGNAL
                        # Exit at next bar open (current bar close as proxy)
                        slippage_factor = self.rng.uniform(-0.5, 0.5)
                        exit_price = bar_open * (1 + slippage_factor * self.slippage_pct)

                if exit_reason is not None:
                    # Record trade
                    bars_held = i - entry_bar

                    trade = {
                        "entry_time": timestamps[entry_bar],
                        "exit_time": timestamps[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "direction": position_direction,
                        "entry_type": entry_type_str,
                        "exit_reason": exit_reason.value,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "max_price": max_price,
                        "bars_held": bars_held,
                    }
                    trades.append(trade)

                    # Reset state
                    state = PositionState.CLOSED
                    position_direction = 0
                    entry_price = 0.0
                    tp_price = None
                    sl_price = None

        # Handle any open position at end of data
        if state == PositionState.ACTIVE:
            # Close at last bar close
            last_idx = n_bars - 1
            exit_price = float(closes[last_idx])
            bars_held = last_idx - entry_bar

            trade = {
                "entry_time": timestamps[entry_bar],
                "exit_time": timestamps[last_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": position_direction,
                "entry_type": entry_type_str,
                "exit_reason": ExitReason.SIGNAL.value,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "max_price": max_price,
                "bars_held": bars_held,
            }
            trades.append(trade)

        # Create trade log DataFrame
        if trades:
            trade_log = pl.DataFrame(trades)
        else:
            # Empty trade log with correct schema
            trade_log = pl.DataFrame(
                {
                    "entry_time": [],
                    "exit_time": [],
                    "entry_price": [],
                    "exit_price": [],
                    "direction": [],
                    "entry_type": [],
                    "exit_reason": [],
                    "tp_price": [],
                    "sl_price": [],
                    "max_price": [],
                    "bars_held": [],
                },
                schema={
                    "entry_time": pl.Datetime,
                    "exit_time": pl.Datetime,
                    "entry_price": pl.Float64,
                    "exit_price": pl.Float64,
                    "direction": pl.Int8,
                    "entry_type": pl.Utf8,
                    "exit_reason": pl.Utf8,
                    "tp_price": pl.Float64,
                    "sl_price": pl.Float64,
                    "max_price": pl.Float64,
                    "bars_held": pl.Int64,
                },
            )

        return trade_log

    def get_config_summary(self) -> dict:
        """Return a summary of the order management configuration.

        Returns:
            Dict with configuration summary for logging/debugging
        """
        return {
            "entry_type": self.entry_type,
            "take_profit": self.tp_config,
            "stop_loss": self.sl_config,
            "trailing_stop": self.trailing_config,
            "bar_fill_assumption": self.bar_fill,
            "atr_period": self.atr_period,
            "max_entry_bars": self.max_entry_bars,
            "max_holding_bars": self.max_holding_bars,
        }
