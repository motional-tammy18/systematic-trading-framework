import streamlit as st

from utils.data_loader import (
    discover_reports,
    load_backtest_by_path,
    load_latest_backtest,
    load_ohlc_data,
)
from components.equity_chart import render_equity_chart
from components.ohlc_chart import render_ohlc_chart
from components.metrics_panel import render_metrics_panel
from components.trades_table import render_trades_table
from datetime import datetime, timedelta


st.set_page_config(page_title="Backtest Dashboard", layout="wide", page_icon="")

DARK_MODE = True

st.markdown(
    """
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #262730;
    }
    .section-divider {
        margin: 0.5rem 0;
        border-top: 1px solid #2a2a2a;
    }
    .stPlotlyChart {
        margin-bottom: 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Backtest Results")

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.backtest_data = None
    st.session_state.selected_strategy = None
    st.session_state.selected_date = None
    st.session_state.selected_timestamp = None

available_reports = discover_reports("../reports")

selected_report = None

with st.sidebar:
    st.header("Select Report")

    if not available_reports:
        st.warning("No reports found. Run a backtest with --save-reports first.")
    else:
        strategies = sorted(set(r["strategy"] for r in available_reports))

        default_idx = 0
        if st.session_state.selected_strategy in strategies:
            default_idx = strategies.index(st.session_state.selected_strategy)

        selected_strategy = st.selectbox(
            "Strategy", options=strategies, index=default_idx, key="strategy_selector"
        )

        strategy_reports = [
            r for r in available_reports if r["strategy"] == selected_strategy
        ]
        dates = sorted(set(r["date"] for r in strategy_reports), reverse=True)

        default_date_idx = 0
        if st.session_state.selected_date in dates:
            default_date_idx = dates.index(st.session_state.selected_date)

        selected_date = st.selectbox(
            "Date", options=dates, index=default_date_idx, key="date_selector"
        )

        date_reports = [r for r in strategy_reports if r["date"] == selected_date]
        timestamps = sorted(date_reports, key=lambda x: x["timestamp"], reverse=True)

        timestamp_options = [t["timestamp"].strftime("%H:%M:%S") for t in timestamps]

        default_ts_idx = 0
        if (
            st.session_state.selected_timestamp
            and st.session_state.selected_strategy == selected_strategy
            and st.session_state.selected_date == selected_date
        ):
            for i, t in enumerate(timestamps):
                if t["timestamp_str"] == st.session_state.selected_timestamp:
                    default_ts_idx = i
                    break

        selected_timestamp_display = st.selectbox(
            "Time",
            options=timestamp_options,
            index=default_ts_idx,
            key="timestamp_selector",
        )

        for t in timestamps:
            if t["timestamp"].strftime("%H:%M:%S") == selected_timestamp_display:
                selected_report = t
                break

        st.session_state.selected_strategy = selected_strategy
        st.session_state.selected_date = selected_date
        if selected_report:
            st.session_state.selected_timestamp = selected_report["timestamp_str"]

    if st.button("↻ Refresh Data"):
        st.session_state.data_loaded = False
        st.rerun()

try:
    if not st.session_state.data_loaded:
        if selected_report:
            st.session_state.backtest_data = load_backtest_by_path(
                selected_report["path"], selected_report["timestamp_str"]
            )
        else:
            st.session_state.backtest_data = load_latest_backtest("../reports")
        st.session_state.data_loaded = True

    data = st.session_state.backtest_data
    if data is None:
        raise FileNotFoundError("No data available")

    equity_df = data["equity_df"]
    trades_df = data["trades_df"]
    summary_dict = data["summary_dict"]
    report_info = data["report_info"]

    st.caption(
        f"**{report_info.get('strategy', 'N/A')}** | "
        f"{report_info.get('date', 'N/A')} | "
        f"{', '.join(summary_dict.get('pairs', []))} | "
        f"{summary_dict.get('timeframe', 'N/A')} | "
        f"Data: {summary_dict.get('data_start', 'N/A')[:10]} → {summary_dict.get('data_end', 'N/A')[:10]}"
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.subheader("Performance Metrics")
    render_metrics_panel(summary_dict, dark_mode=DARK_MODE)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.subheader("Equity Curve")
    fig = render_equity_chart(equity_df, dark_mode=DARK_MODE)
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.subheader("Price Chart with Trades")
    pairs = summary_dict.get("pairs", [])
    timeframe = summary_dict.get("timeframe", "15m")

    ohlc_df = None
    if pairs:
        ohlc_df = load_ohlc_data(pairs[0], timeframe, "../data/raw")

    if ohlc_df is not None:
        ohlc_max_ts = ohlc_df["timestamp"].max()
        ohlc_min_ts = ohlc_df["timestamp"].min()

        ohlc_max_date = ohlc_max_ts
        ohlc_min_date = ohlc_min_ts

        default_start = ohlc_max_date - timedelta(days=7)

        with st.expander("📅 Date Range", expanded=False):
            date_range = st.date_input(
                "Select date range",
                value=(default_start.date(), ohlc_max_date.date()),
                min_value=ohlc_min_date.date(),
                max_value=ohlc_max_date.date(),
            )

        if len(date_range) == 2:
            start_date = datetime.combine(date_range[0], datetime.min.time())
            end_date = datetime.combine(date_range[1], datetime.max.time())
        else:
            start_date = default_start
            end_date = None

        fig_ohlc = render_ohlc_chart(
            ohlc_df,
            trades_df,
            dark_mode=DARK_MODE,
            start_date=start_date,
            end_date=end_date,
        )
        st.plotly_chart(fig_ohlc, width="stretch")
    else:
        st.info("OHLC data not available. Place raw OHLC data in data/raw/ directory.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.subheader("Trade Log")
    render_trades_table(trades_df, dark_mode=DARK_MODE)

except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Run a backtest with --save-reports first: `python run.py --strategy ema_crossover --mode backtest --save-reports`"
    )
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
