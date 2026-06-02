import streamlit as st
import polars as pl
import pandas as pd


def render_trades_table(trades_df: pl.DataFrame, dark_mode: bool = True) -> None:
    """Render interactive trades table."""
    if trades_df is None or trades_df.height == 0:
        st.info("No trades executed in this backtest.")
        return

    df = trades_df.to_pandas()

    display_cols = [
        "entry_time",
        "exit_time",
        "direction",
        "entry_price",
        "exit_price",
        "pnl",
        "pnl_pct",
    ]
    if "exit_reason" in df.columns:
        display_cols.append("exit_reason")

    available_cols = [c for c in display_cols if c in df.columns]
    df_display = df[available_cols].copy()

    for col in ["entry_time", "exit_time"]:
        if col in df_display.columns:
            df_display[col] = pd.to_datetime(df_display[col]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )

    for col in ["entry_price", "exit_price"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )

    if "pnl" in df_display.columns:
        df_display["pnl"] = df_display["pnl"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
        )

    if "pnl_pct" in df_display.columns:
        df_display["pnl_pct"] = df_display["pnl_pct"].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
        )

    if "direction" in df_display.columns:
        df_display["direction"] = df_display["direction"].map({1: "Long", -1: "Short"})

    def color_direction(val):
        if val == "Long":
            return "color: #00cc96; font-weight: bold"
        elif val == "Short":
            return "color: #ef5350; font-weight: bold"
        return ""

    def color_pnl(val):
        if pd.isna(val):
            return ""
        if isinstance(val, str) and val.startswith("$"):
            try:
                num_val = float(val.replace("$", "").replace(",", ""))
                if num_val > 0:
                    return "color: #00cc96; font-weight: bold"
                elif num_val < 0:
                    return "color: #ef5350; font-weight: bold"
            except:
                pass
        return ""

    def color_pnl_pct(val):
        if pd.isna(val):
            return ""
        if isinstance(val, str) and val.endswith("%"):
            try:
                num_val = float(val.replace("%", ""))
                if num_val > 0:
                    return "color: #00cc96; font-weight: bold"
                elif num_val < 0:
                    return "color: #ef5350; font-weight: bold"
            except:
                pass
        return ""

    styler = df_display.style

    if "direction" in df_display.columns:
        styler = styler.map(color_direction, subset=["direction"])

    if "pnl" in df_display.columns:
        styler = styler.map(color_pnl, subset=["pnl"])

    if "pnl_pct" in df_display.columns:
        styler = styler.map(color_pnl_pct, subset=["pnl_pct"])

    st.dataframe(
        styler,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
    <div style="display: flex; gap: 20px; font-size: 12px; color: #888;">
        <span><span style="color: #00cc96;">▲</span> Long</span>
        <span><span style="color: #ef5350;">▼</span> Short</span>
        <span><span style="color: #00cc96;">+</span> Profit</span>
        <span><span style="color: #ef5350;">−</span> Loss</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.caption(f"Showing {len(df_display)} trades")
