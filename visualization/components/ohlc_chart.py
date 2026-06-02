"""OHLC Chart component with trade markers.

Renders candlestick chart with historical trades overlaid.
"""

from datetime import datetime, timedelta
from typing import Optional

import plotly.graph_objects as go
import polars as pl
import pandas as pd


def render_ohlc_chart(
    ohlc_df: pl.DataFrame,
    trades_df: pl.DataFrame,
    dark_mode: bool = True,
    height: int = 500,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> go.Figure:
    """Render OHLC candlestick chart with trade markers.

    Args:
        ohlc_df: DataFrame with timestamp, open, high, low, close, volume columns
        trades_df: DataFrame with entry_time, exit_time, direction, entry_price, exit_price
        dark_mode: Use dark theme (default True)
        height: Chart height in pixels
        start_date: Start date for filtering (default: last 7 days)
        end_date: End date for filtering (default: None = latest)

    Returns:
        Plotly figure object
    """
    if ohlc_df is None or ohlc_df.height == 0:
        fig = go.Figure()
        fig.add_annotation(text="No OHLC data available", showarrow=False)
        return fig

    ohlc_filtered = ohlc_df
    
    if start_date is not None or end_date is not None:
        if start_date is not None:
            ohlc_filtered = ohlc_filtered.filter(pl.col("timestamp") >= start_date)
        if end_date is not None:
            ohlc_filtered = ohlc_filtered.filter(pl.col("timestamp") <= end_date)
        
        if ohlc_filtered.height == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data in selected date range", showarrow=False)
            return fig

    bg_color = "#0e1117" if dark_mode else "#ffffff"
    text_color = "#fafafa" if dark_mode else "#31333f"
    grid_color = "#262730" if dark_mode else "#e0e0e0"

    timestamps = ohlc_filtered["timestamp"].to_list()
    open_prices = ohlc_filtered["open"].to_list()
    high_prices = ohlc_filtered["high"].to_list()
    low_prices = ohlc_filtered["low"].to_list()
    close_prices = ohlc_filtered["close"].to_list()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name="OHLC",
            increasing_line_color="#00cc96",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#00cc96",
            decreasing_fillcolor="#ef5350",
        )
    )

    if trades_df is not None and trades_df.height > 0:
        trades = trades_df.to_pandas()
        
        if start_date is not None or end_date is not None:
            trades_mask = pd.Series([True] * len(trades))
            if start_date is not None:
                trades_mask = trades_mask & (pd.to_datetime(trades["entry_time"]) >= start_date)
            if end_date is not None:
                trades_mask = trades_mask & (pd.to_datetime(trades["entry_time"]) <= end_date)
            trades = trades[trades_mask]

        if len(trades) > 0:
            long_entries = trades[trades["direction"] == 1]
            if len(long_entries) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=long_entries["entry_time"],
                        y=long_entries["entry_price"],
                        mode="markers",
                        name="Long Entry",
                        marker=dict(
                            symbol="triangle-up",
                            size=12,
                            color="#00E5FF",
                            line=dict(width=2, color="#FFFFFF"),
                        ),
                        hoverinfo="text",
                        hovertext=long_entries.apply(
                            lambda r: f"Long Entry<br>Price: ${r['entry_price']:.2f}<br>Time: {r['entry_time']}",
                            axis=1,
                        ),
                    )
                )

            short_entries = trades[trades["direction"] == -1]
            if len(short_entries) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=short_entries["entry_time"],
                        y=short_entries["entry_price"],
                        mode="markers",
                        name="Short Entry",
                        marker=dict(
                            symbol="triangle-down",
                            size=12,
                            color="#FF00FF",
                            line=dict(width=2, color="#FFFFFF"),
                        ),
                        hoverinfo="text",
                        hovertext=short_entries.apply(
                            lambda r: f"Short Entry<br>Price: ${r['entry_price']:.2f}<br>Time: {r['entry_time']}",
                            axis=1,
                        ),
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=trades["exit_time"],
                    y=trades["exit_price"],
                    mode="markers",
                    name="Exit",
                    marker=dict(
                        symbol="circle",
                        size=10,
                        color="#FFD700",
                        line=dict(width=2, color="#FFFFFF"),
                    ),
                    hoverinfo="text",
                    hovertext=trades.apply(
                        lambda r: f"Exit<br>Price: ${r['exit_price']:.2f}<br>PnL: ${r['pnl']:.2f} ({r['pnl_pct']:.2f}%)<br>Time: {r['exit_time']}",
                        axis=1,
                    ),
                )
            )

    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor=grid_color,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="Price ($)",
            showgrid=True,
            gridcolor=grid_color,
            tickformat=",.0f",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=height,
        hovermode="closest",
    )

    return fig
