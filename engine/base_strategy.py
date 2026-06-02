"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
import polars as pl


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    All strategies must inherit from this class and implement the
    generate_signals method.
    """

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate signals from OHLCV data.

        Args:
            data: Polars DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            Polars DataFrame with additional 'signal' column (1=long, -1=short, 0=flat)
        """
        pass
