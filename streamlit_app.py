"""
🔬 A股智能交易信号系统
自动学习归因 · 多周期共振 · T+1买卖建议
"""
import sys
import os
from pathlib import Path

# 确保项目根在 path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from core.data_fetcher import (
    get_main_board_stocks, fetch_stock_data, clear_cache, get_stock_list,
)
from core.indicators import calc_all_indicators, detect_market_regime
from core.patterns import PatternDetector
from core.backtest import BacktestEngine
from core.signals import SignalGenerator

# 页面配置
st.set_page_config(
    page_title="A股智能交易信号",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化引擎
detector = PatternDetector()
backtest = BacktestEngine()
signal_gen = SignalGenerator()

# ================================================================
# 自定义CSS
# ================================================================
st.markdown("""
<style>
.main-title { font-size: 2.2rem; font-weight: bold;
    background: linear-gradient(135deg, #ef5350, #ff7043);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.signal-buy { color: #ef5350; font-weight: bold; }
.signal-sell { color: #26a69a; font-weight: bold; }
.signal-neutral { color: #ff9800; }
.card { border-radius: 12px; padding: 16px; margin: 8px 0;
    border: 1px solid #e0e0e0; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
.stButton > button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# 侧边栏
# ================================================================
with st.sidebar:
    st.markdown("### ⚙️ 控制面板")

    scan_option = st.selectbox(
        "扫描范围",
        ["前100只（快速）", "前300只（推荐）", "前500只", "全部沪深主板+创业板"],
        index=1,
        help="只扫描沪深主板(60/00)和创业板(30)，自动排除科创板和北交所"
    )
    scan_map = {
        "前100只（快速）": 100,
        "前300只（推荐）": 300,
        "前500只": 500,
        "全部沪深主板+创业板": None,
    }
    scan_n = scan_map[scan_option]

    st.divider()
    action = st.button("🔄 刷新全部数据", use_container_width=True)
    if action:
        clear_cache()
        st.rerun()

    st.divider()

    st.markdown("### 📖 策略说明")
    st.markdown("""
    **检测形态**:
    - MACD 顶/底背离
    - 量价背离
    - 均线金叉/死叉
    - RSI 超买/超卖
    - 布林带突破
    - KDJ 金叉/死叉
    - 放量突破
    - 多指标共振

    **回测规则**:
    - T+1 持有 5 日
    - 止损 -5% / 止盈 +10%
    - 最少 20 次信号才纳入统计
    """)

    st.divider()
    st.markdown("### ℹ️ 关于")
    st.markdown("""
    **A股智能交易信号 v2.0**

    基于多周期K线 + 多形态识别
    的T+1交易建议系统

    ⚠️ 仅供参考，不构成投资建议
    股市有风险，投资需谨慎
    """)

    st.caption(f"数据源: akshare | 腾讯财经")

# ================================================================
# 主标题
# ================================================================
st.markdown('<p class="main-title">🔬 A股智能交易信号系统</p>', unsafe_allow_html=True)
st.caption("多形态识别 · 多周期共振 · 自动回测优化 · T+1交易建议")

# ============= Tabs =============
tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 实时买卖信号", "📈 个股深度分析",
    "🔬 策略回测排行", "📊 持仓模拟",
])

# ================================================================
# Tab 1: 实时买卖信号
# ================================================================
with tab1:
    st.subheader("🔥 全市场实时交易信号")

    if st.button("🚀 开始扫描市场", type="primary", key="scan_btn"):
        scope_label = "全部沪深主板+创业板" if scan_n is None else f"前 {scan_n} 只"
        with st.spinner(f"正在扫描 {scope_label}（预计1-5分钟）..."):
            progress_bar = st.progress(0)

            def update_progress(pct):
                progress_bar.progress(pct)

            scan_results = signal_gen.scan_market(n_stocks=scan_n,
                                                   progress_callback=update_progress)
            st.session_state["scan_results"] = scan_results
            st.session_state["scan_time"] = datetime.now().isoformat()

            progress_bar.empty()
            st.success(f"扫描完成！共 {scan_results['scanned']} 只股票（沪深主板+创业板），"
                      f"发现 {len(scan_results['buy'])} 个买入信号，"
                      f"{len(scan_results['sell'])} 个卖出信号")

    # 显示结果
    if "scan_results" in st.session_state:
        scan = st.session_state["scan_results"]
        scan_time = st.session_state.get("scan_time", "")
        st.caption(f"扫描时间: {scan_time[:19]} | 共 {scan['scanned']} 只股票")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟢 买入信号")
            buy_list = scan.get("buy", [])
            if buy_list:
                for s in buy_list[:20]:
                    score = s["signal_score"]
                    with st.container():
                        st.markdown(f"""
                        <div class="card">
                        <b>{s.get('name', s['code'])}</b> ({s['code']})
                        <span class="signal-buy"> ★{score}</span> |
                        现价 ¥{s.get('price', 0):.2f}<br/>
                        <small>{' | '.join(s.get('buy_triggers', [])[:3])}</small><br/>
                        <small>RSI {s.get('indicators', {}).get('rsi', '?')} |
                        量比 {s.get('indicators', {}).get('volume_ratio', '?')}</small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("暂未发现买入信号")

        with col2:
            st.markdown("### 🔴 卖出信号")
            sell_list = scan.get("sell", [])
            if sell_list:
                for s in sell_list[:20]:
                    score = abs(s["signal_score"])
                    with st.container():
                        st.markdown(f"""
                        <div class="card">
                        <b>{s.get('name', s['code'])}</b> ({s['code']})
                        <span class="signal-sell"> ▲{score}</span> |
                        现价 ¥{s.get('price', 0):.2f}<br/>
                        <small>{' | '.join(s.get('sell_triggers', [])[:3])}</small><br/>
                        <small>RSI {s.get('indicators', {}).get('rsi', '?')} |
                        量比 {s.get('indicators', {}).get('volume_ratio', '?')}</small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("暂未发现卖出信号")

        # 多周期确认的强信号
        st.divider()
        st.markdown("### ⭐ 多周期共振确认 (日+周+月)")
        confirmed = [s for s in scan.get("buy", []) if s.get("multi_period_confirm")]
        if confirmed:
            cols = st.columns(min(len(confirmed), 5))
            for i, s in enumerate(confirmed[:5]):
                with cols[i]:
                    st.metric(
                        f"{s.get('name', s['code'])}",
                        f"¥{s.get('price', 0):.2f}",
                        f"★{s['signal_score']}"
                    )
        else:
            st.info("暂无多周期共振信号")

# ================================================================
# Tab 2: 个股深度分析
# ================================================================
with tab2:
    st.subheader("📈 个股深度技术分析")

    col_search, col_period = st.columns([3, 1])
    with col_search:
        stock_input = st.text_input("输入股票代码", placeholder="600519",
                                     key="stock_detail_input")
    with col_period:
        chart_days = st.selectbox("K线周期", [60, 120, 250, 360], index=2)

    if stock_input:
        code = stock_input.strip().zfill(6)
        with st.spinner(f"加载 {code} 数据..."):
            stock_data = fetch_stock_data(code, "")

        if stock_data.get("daily") is not None and not stock_data["daily"].empty:
            df = stock_data["daily"].tail(chart_days)
            df = calc_all_indicators(df)
            df_sig = detector.detect_all(df.copy())
            latest = detector.get_latest_signal(df_sig)
            name = stock_data.get("meta", {}).get("name", code)

            # 股票信息行
            st.markdown(f"## {name} ({code})")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("现价", f"¥{latest['price']:.2f}",
                         f"{latest['indicators']['change_pct']:+.2f}%")
            with col2:
                signal = latest["signal"]
                color = "#ef5350" if "买" in signal else "#26a69a" if "卖" in signal else "#888"
                st.markdown(f"**信号**: <span style='color:{color}'>{signal}</span>",
                           unsafe_allow_html=True)
            with col3:
                st.metric("RSI", f"{latest['indicators']['rsi']:.1f}")
            with col4:
                st.metric("量比", f"{latest['indicators']['volume_ratio']:.2f}")
            with col5:
                regime = detect_market_regime(df)
                regime_cn = {"trending_up": "上升趋势", "trending_down": "下降趋势",
                            "ranging": "震荡", "unknown": "未知"}
                st.metric("市场状态", regime_cn.get(regime, "未知"))

            # 触发信号
            if latest["triggers"]:
                st.markdown("**当前触发信号**:")
                for t in latest["triggers"]:
                    st.markdown(f"- {t}")

            # K线图
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.25, 0.25],
            )

            # 主图：K线+均线+布林带
            colors = ["#ef5350" if c >= o else "#26a69a"
                     for c, o in zip(df["close"], df["open"])]
            fig.add_trace(go.Candlestick(
                x=df["date"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name="K线",
                increasing_line_color="#ef5350",
                decreasing_line_color="#26a69a",
            ), row=1, col=1)

            # 均线
            for ma, color in [("ma_5", "#ff9800"), ("ma_20", "#9c27b0"), ("ma_60", "#607d8b")]:
                if ma in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df["date"], y=df[ma], mode="lines",
                        name=ma.upper(), line=dict(color=color, width=1),
                    ), row=1, col=1)

            # 布林带
            if "bb_upper" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["bb_upper"], mode="lines",
                    name="布林上轨", line=dict(color="#aaa", width=0.5, dash="dash"),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["bb_lower"], mode="lines",
                    name="布林下轨", line=dict(color="#aaa", width=0.5, dash="dash"),
                    fill="tonexty", fillcolor="rgba(128,128,128,0.05)",
                ), row=1, col=1)

            # MACD
            fig.add_trace(go.Bar(
                x=df["date"], y=df["macd_hist"], name="MACD柱",
                marker_color=np.where(df["macd_hist"] > 0, "#ef5350", "#26a69a"),
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["macd_dif"], mode="lines",
                name="DIF", line=dict(color="#2196f3", width=1),
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["macd_dea"], mode="lines",
                name="DEA", line=dict(color="#ff9800", width=1),
            ), row=2, col=1)

            # 成交量
            fig.add_trace(go.Bar(
                x=df["date"], y=df["volume"], name="成交量",
                marker_color=colors, opacity=0.4,
            ), row=3, col=1)

            fig.update_layout(
                title=f"{name} ({code}) 技术分析",
                xaxis_rangeslider_visible=False,
                height=700,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="top", y=-0.05),
            )
            fig.update_yaxes(title_text="价格", row=1, col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            fig.update_yaxes(title_text="成交量", row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # 买卖信号标注表格
            st.subheader("📋 近期交易信号历史")
            signal_rows = df_sig[df_sig["signal_strength"] != "无"].tail(20)
            if not signal_rows.empty:
                st.dataframe(
                    signal_rows[["date", "close", "signal_strength",
                                  "buy_score", "sell_score", "rsi",
                                  "volume_ratio_20", "change_pct"]].tail(15),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.error(f"无法获取 {code} 的数据")

# ================================================================
# Tab 3: 策略回测排行
# ================================================================
with tab3:
    st.subheader("🔬 交易形态胜率排行（自动学习归因）")

    if st.button("📊 运行回测分析", type="primary", key="backtest_btn"):
        with st.spinner("正在回测历史数据，自动学习最优形态..."):
            progress_bar = st.progress(0)
            stocks = get_main_board_stocks(50)  # 回测前50只

            stock_data_map = {}
            for i, (_, row) in enumerate(stocks.iterrows()):
                code = row["code"]
                name = row.get("name", "")
                stock_data_map[code] = fetch_stock_data(code, name)
                progress_bar.progress((i + 1) / len(stocks))

            bt_results = backtest.batch_backtest(stock_data_map)
            st.session_state["bt_results"] = bt_results
            progress_bar.empty()
            st.success("回测完成！")

    if "bt_results" in st.session_state:
        bt = st.session_state["bt_results"]

        # 全局形态排行
        st.markdown("### 🏆 最优交易形态 Top 10")
        ranking = bt.get("global_ranking", [])
        if ranking:
            df_rank = pd.DataFrame(ranking[:10])
            df_rank.columns = ["形态名称", "胜率%", "均收益%", "信号数", "胜次"]
            df_rank["胜率%"] = df_rank["胜率%"].astype(float)

            # 颜色条
            def color_win_rate(val):
                if val >= 70: return "background-color: #ffcdd2; font-weight: bold"
                if val >= 60: return "background-color: #fff3e0"
                return ""
            st.dataframe(
                df_rank.style.map(color_win_rate, subset=["胜率%"]),
                use_container_width=True, hide_index=True,
            )

        # 最强信号个股
        st.markdown("### 🎯 当前信号最强个股 Top 20")
        top_stocks = bt.get("top_stocks", [])
        if top_stocks:
            rows = []
            for s in top_stocks[:20]:
                rows.append({
                    "代码": s["code"],
                    "名称": s["name"],
                    "信号": s["signal"],
                    "评分": s["score"],
                    "现价": s["price"],
                    "触发形态": ", ".join(s.get("triggers", [])[:3]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("当前暂未发现强买入信号")

# ================================================================
# Tab 4: 持仓模拟
# ================================================================
with tab4:
    st.subheader("📊 模拟持仓跟踪")

    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

    col_add, col_view = st.columns([1, 2])

    with col_add:
        st.markdown("#### 添加模拟持仓")
        add_code = st.text_input("股票代码", key="add_code")
        add_price = st.number_input("买入价格", 0.0, 10000.0, 0.0, 0.01, key="add_price")
        add_shares = st.number_input("买入股数", 100, 1000000, 100, 100, key="add_shares")
        add_date = st.date_input("买入日期", datetime.now() - timedelta(days=1))

        if st.button("➕ 添加持仓", use_container_width=True):
            if add_code and add_price > 0:
                st.session_state["portfolio"].append({
                    "code": add_code.strip().zfill(6),
                    "buy_price": add_price,
                    "shares": add_shares,
                    "buy_date": str(add_date),
                })
                st.success(f"已添加 {add_code}")
                st.rerun()

    with col_view:
        st.markdown("#### 当前持仓")
        portfolio = st.session_state["portfolio"]

        if portfolio:
            total_cost = 0
            total_value = 0
            rows = []

            for pos in portfolio:
                code = pos["code"]
                try:
                    data = fetch_stock_data(code, "")
                    daily = data.get("daily")
                    if daily is not None and not daily.empty:
                        current_price = float(daily["close"].iloc[-1])
                    else:
                        current_price = pos["buy_price"]
                except Exception:
                    current_price = pos["buy_price"]

                cost = pos["buy_price"] * pos["shares"]
                value = current_price * pos["shares"]
                pnl = value - cost
                pnl_pct = (current_price / pos["buy_price"] - 1) * 100

                total_cost += cost
                total_value += value

                rows.append({
                    "代码": code,
                    "买入价": f"¥{pos['buy_price']:.2f}",
                    "现价": f"¥{current_price:.2f}",
                    "股数": pos["shares"],
                    "成本": f"¥{cost:,.0f}",
                    "市值": f"¥{value:,.0f}",
                    "盈亏": f"¥{pnl:+,.0f}",
                    "盈亏%": f"{pnl_pct:+.2f}%",
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            total_pnl = total_value - total_cost
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总成本", f"¥{total_cost:,.0f}")
            with col2:
                st.metric("总市值", f"¥{total_value:,.0f}")
            with col3:
                st.metric("总盈亏", f"¥{total_pnl:+,.0f}",
                         f"{total_pnl/total_cost*100:+.2f}%" if total_cost > 0 else "")

            if st.button("🗑 清空持仓", type="secondary"):
                st.session_state["portfolio"] = []
                st.rerun()
        else:
            st.info("暂无模拟持仓，请在左侧添加")

# ================================================================
# 底部免责
# ================================================================
st.divider()
st.caption(
    "⚠️ **免责声明**: 本系统所有信号和策略仅供参考学习，不构成任何投资建议。"
    "所有交易决策由您本人作出。股市有风险，投资需谨慎。"
    "数据来源: akshare, 腾讯财经。"
)
