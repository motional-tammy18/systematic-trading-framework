import plotly.graph_objects as go
import polars as pl


def render_equity_chart(equity_df: pl.DataFrame, dark_mode: bool = True) -> go.Figure:
    """Render equity curve with drawdown overlay."""
    if equity_df is None or equity_df.height == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    timestamps = equity_df["timestamp"].to_list()
    equity_values = equity_df["equity"].to_list()
    drawdown_values = equity_df["drawdown"].to_list() if "drawdown" in equity_df.columns else [0.0] * len(equity_values)

    bg_color = "#0e1117" if dark_mode else "#ffffff"
    text_color = "#fafafa" if dark_mode else "#31333f"
    grid_color = "#262730" if dark_mode else "#e0e0e0"
    equity_color = "#00cc96"
    drawdown_color = "rgba(239, 85, 59, 0.3)"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=equity_values,
        mode="lines",
        name="Equity",
        line=dict(color=equity_color, width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 204, 150, 0.1)",
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=drawdown_values,
        mode="lines",
        name="Drawdown",
        line=dict(color=drawdown_color, width=1),
        fill="tozeroy",
        fillcolor=drawdown_color,
    ))

    circuit_breaker_idx = None
    if len(equity_values) > 1:
        for i in range(1, len(equity_values)):
            if equity_values[i] == equity_values[i-1] and equity_values[i] < equity_values[0] * 0.5:
                circuit_breaker_idx = i
                break

    if circuit_breaker_idx:
        fig.add_vline(
            x=timestamps[circuit_breaker_idx],
            line_dash="dash",
            line_color="red",
            annotation_text="Circuit Breaker",
            annotation_position="top left"
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
        ),
        yaxis=dict(
            title="Equity ($)",
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
        height=400,
    )

    return fig
