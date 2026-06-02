import streamlit as st
import polars as pl


def render_metrics_panel(summary_dict: dict, dark_mode: bool = True) -> None:
    """Render metrics panel with all KPI cards organized by category."""
    if not summary_dict or "metrics" not in summary_dict:
        st.warning("No metrics available")
        return

    metrics = summary_dict.get("metrics", {})
    config = summary_dict.get("config", {})
    initial_capital = metrics.get("initial_capital", config.get("initial_capital", 10000))
    total_pnl = metrics.get("total_pnl", 0)
    final_equity = metrics.get("final_equity", initial_capital + total_pnl)
    total_return = metrics.get("total_return_pct", 0)

    # =====================
    # PERFORMANCE METRICS
    # =====================
    st.markdown("### Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sharpe = metrics.get("sharpe_ratio", 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col2:
        sortino = metrics.get("sortino_ratio", 0)
        st.metric("Sortino Ratio", f"{sortino:.2f}")
    with col3:
        cagr = metrics.get("cagr", 0)
        st.metric("CAGR", f"{cagr:.2%}")
    with col4:
        calmar = metrics.get("calmar_ratio", 0)
        st.metric("Calmar Ratio", f"{calmar:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        max_dd = metrics.get("max_drawdown", 0)
        st.metric("Max Drawdown", f"{max_dd:.1%}")
    with col6:
        max_dd_duration = metrics.get("max_drawdown_duration", 0)
        st.metric("Max DD Duration", f"{max_dd_duration:,} bars")
    with col7:
        recovery_factor = metrics.get("recovery_factor", 0)
        st.metric("Recovery Factor", f"{recovery_factor:.2f}")
    with col8:
        profit_factor = metrics.get("profit_factor", 0)
        st.metric("Profit Factor", f"{profit_factor:.2f}")

    col9, col10, col11 = st.columns(3)
    with col9:
        st.metric("Total Return", f"{total_return:+.2f}%")
    with col10:
        st.metric("Total PnL", f"${total_pnl:,.2f}")
    with col11:
        st.metric("Expectancy", f"${metrics.get('expectancy', 0):,.2f}")

    st.markdown("---")

    # =====================
    # TRADE STATISTICS
    # =====================
    st.markdown("### Trade Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_trades = metrics.get("total_trades", 0)
        st.metric("Total Trades", f"{total_trades:,}")
    with col2:
        buy_trades = metrics.get("buy_trades", 0)
        st.metric("Buy Trades", f"{buy_trades:,}")
    with col3:
        sell_trades = metrics.get("sell_trades", 0)
        st.metric("Sell Trades", f"{sell_trades:,}")
    with col4:
        win_rate = metrics.get("win_rate", 0)
        st.metric("Win Rate", f"{win_rate:.1f}%")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        winning_trades = metrics.get("winning_trades", 0)
        st.metric("Winning Trades", f"{winning_trades:,}")
    with col6:
        losing_trades = metrics.get("losing_trades", 0)
        st.metric("Losing Trades", f"{losing_trades:,}")
    with col7:
        avg_trade_pnl = metrics.get("average_trade_pnl", 0)
        st.metric("Avg Trade PnL", f"${avg_trade_pnl:,.2f}")
    with col8:
        expectancy = metrics.get("expectancy", 0)
        st.metric("Expectancy", f"${expectancy:,.2f}")

    col9, col10, col11, col12 = st.columns(4)
    with col9:
        avg_win = metrics.get("average_win", 0)
        st.metric("Avg Win", f"${avg_win:,.2f}")
    with col10:
        avg_loss = metrics.get("average_loss", 0)
        st.metric("Avg Loss", f"${avg_loss:,.2f}")
    with col11:
        best_trade = metrics.get("best_trade", 0)
        st.metric("Best Trade", f"${best_trade:,.2f}")
    with col12:
        worst_trade = metrics.get("worst_trade", 0)
        st.metric("Worst Trade", f"${worst_trade:,.2f}")

    st.markdown("---")

    # =====================
    # PnL SUMMARY
    # =====================
    st.markdown("### PnL Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Initial Capital", f"${initial_capital:,.2f}")
    with col2:
        st.metric("Final Equity", f"${final_equity:,.2f}")
    with col3:
        st.metric("Net Profit", f"${total_pnl:,.2f}", delta=f"{total_return:+.2f}%")
