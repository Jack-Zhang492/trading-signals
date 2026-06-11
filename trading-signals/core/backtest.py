"""
回测引擎 + 策略优化器
自动学习归因，迭代优化，找出胜率最高的买卖战法
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from core.config import BACKTEST_PARAMS
from core.indicators import calc_all_indicators
from core.patterns import PatternDetector


class BacktestEngine:
    """回测引擎"""

    def __init__(self):
        self.params = BACKTEST_PARAMS
        self.detector = PatternDetector()

    def backtest_stock(self, df: pd.DataFrame) -> Dict:
        """
        对单只股票的历史数据进行回测
        计算每个信号形态的胜率和收益
        """
        if df is None or df.empty:
            return {"error": "无数据"}

        # 计算指标和形态
        df = calc_all_indicators(df)
        df = self.detector.detect_all(df)

        results = {
            "summary": {},
            "patterns": {},
            "trades": [],
            "equity_curve": [],
        }

        # 各类信号的统计
        signal_cols = {
            "macd_bullish_div": ("MACD底背离(买)", 1),
            "macd_bearish_div": ("MACD顶背离(卖)", -1),
            "vol_bullish_div": ("量价底背离(买)", 1),
            "vol_bearish_div": ("量价顶背离(卖)", -1),
            "ma_golden_cross": ("MA金叉(买)", 1),
            "ma_death_cross": ("MA死叉(卖)", -1),
            "rsi_oversold": ("RSI超卖(买)", 1),
            "rsi_overbought": ("RSI超买(卖)", -1),
            "bb_lower_touch": ("布林下轨(买)", 1),
            "bb_upper_touch": ("布林上轨(卖)", -1),
            "kdj_golden": ("KDJ金叉(买)", 1),
            "kdj_death": ("KDJ死叉(卖)", -1),
            "vol_breakout_up": ("放量突破(买)", 1),
            "vol_breakout_down": ("放量下跌(卖)", -1),
            "net_signal_3": ("强烈买入(买)", 1),
            "net_signal_neg3": ("强烈卖出(卖)", -1),
        }

        hold_days = self.params["hold_days"]
        stop_loss = self.params["stop_loss"]
        take_profit = self.params["take_profit"]
        n = len(df)

        pattern_stats = defaultdict(lambda: {
            "signals": 0, "wins": 0, "losses": 0,
            "total_return": 0.0, "max_return": 0.0, "min_return": 0.0,
            "avg_hold_return": [],
        })

        all_trades = []
        equity = 1.0
        equity_curve = []

        for i in range(30, n - hold_days - 1):
            row = df.iloc[i]
            entry_price = row["close"]
            entry_date = row.get("date", i)

            for col, (label, direction) in signal_cols.items():
                signal_val = None

                if col == "net_signal_3":
                    signal_val = 1 if row.get("net_signal", 0) >= 3 else 0
                elif col == "net_signal_neg3":
                    signal_val = -1 if row.get("net_signal", 0) <= -3 else 0
                elif col in df.columns:
                    signal_val = row[col]

                if not signal_val or signal_val == 0:
                    continue

                # 模拟持有
                exit_idx = min(i + hold_days, n - 1)
                exit_price = df.iloc[exit_idx]["close"]

                # 检查止损止盈（中间过程）
                for j in range(i + 1, exit_idx + 1):
                    mid_price = df.iloc[j]["close"]
                    ret = (mid_price / entry_price - 1)
                    if ret <= stop_loss:
                        exit_price = mid_price
                        exit_idx = j
                        break
                    if ret >= take_profit:
                        exit_price = mid_price
                        exit_idx = j
                        break

                # 计算收益
                hold_return = direction * (exit_price / entry_price - 1)

                pattern_stats[label]["signals"] += 1
                pattern_stats[label]["total_return"] += hold_return
                pattern_stats[label]["avg_hold_return"].append(hold_return)
                pattern_stats[label]["max_return"] = max(
                    pattern_stats[label]["max_return"], hold_return)
                pattern_stats[label]["min_return"] = min(
                    pattern_stats[label]["min_return"], hold_return)

                if hold_return > 0:
                    pattern_stats[label]["wins"] += 1
                else:
                    pattern_stats[label]["losses"] += 1

                all_trades.append({
                    "date": str(entry_date)[:10],
                    "pattern": label,
                    "direction": "买入" if direction == 1 else "卖出",
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "return": round(hold_return * 100, 2),
                    "hold_days": exit_idx - i,
                    "win": hold_return > 0,
                })

                # 更新权益曲线
                equity *= (1 + hold_return * 0.1)  # 10%仓位
            equity_curve.append({"date": str(entry_date)[:10], "equity": round(equity, 4)})

        # 汇总每个形态的统计
        for label, stats in pattern_stats.items():
            n_sig = stats["signals"]
            if n_sig > 0:
                win_rate = stats["wins"] / n_sig
                avg_ret = stats["total_return"] / n_sig
                returns = stats["avg_hold_return"]
                sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 2 and np.std(returns) > 0 else 0
            else:
                win_rate = 0
                avg_ret = 0
                sharpe = 0

            results["patterns"][label] = {
                "signals": n_sig,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": round(win_rate * 100, 1),
                "avg_return": round(avg_ret * 100, 2),
                "total_return": round(stats["total_return"] * 100, 2),
                "max_return": round(stats["max_return"] * 100, 2),
                "min_return": round(stats["min_return"] * 100, 2),
                "sharpe": round(sharpe, 2),
                "reliable": n_sig >= self.params["min_samples"],
            }

        results["trades"] = all_trades
        results["equity_curve"] = equity_curve

        # 汇总
        total_trades = len(all_trades)
        total_wins = sum(1 for t in all_trades if t["win"])
        results["summary"] = {
            "total_signals": total_trades,
            "total_wins": total_wins,
            "total_losses": total_trades - total_wins,
            "overall_win_rate": round(total_wins / max(total_trades, 1) * 100, 1),
            "avg_return_per_trade": round(
                np.mean([t["return"] for t in all_trades]) if all_trades else 0, 2
            ),
            "total_return": round(
                sum(t["return"] for t in all_trades), 2
            ) if all_trades else 0,
            "final_equity": round(equity, 4),
        }

        return results

    def rank_patterns(self, backtest_results: Dict) -> List[Dict]:
        """
        按胜率和可靠性排序形态
        返回最优交易战法排行
        """
        patterns = backtest_results.get("patterns", {})
        ranked = []

        min_samples = self.params["min_samples"]
        min_win_rate = self.params["min_win_rate"]

        for name, stats in patterns.items():
            if stats["signals"] < min_samples:
                continue
            if stats["win_rate"] / 100 < min_win_rate:
                continue

            # 综合评分：胜率*0.4 + 夏普*0.3 + 平均收益归一化*0.2 + 信号数量归一化*0.1
            score = (
                stats["win_rate"] / 100 * 0.4 +
                max(0, stats["sharpe"]) / 3.0 * 0.3 +
                max(0, stats["avg_return"]) / 10.0 * 0.2 +
                min(stats["signals"] / 50, 1.0) * 0.1
            )
            ranked.append({
                "name": name,
                "win_rate": stats["win_rate"],
                "avg_return": stats["avg_return"],
                "sharpe": stats["sharpe"],
                "signals": stats["signals"],
                "total_return": stats["total_return"],
                "score": round(score * 100, 1),
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def batch_backtest(self, stock_data: Dict[str, Dict]) -> Dict:
        """
        批量回测多只股票，汇总最优策略
        返回: {
            "global_ranking": [...],  # 全局形态排行
            "stock_ranking": {...},   # 每只股票的形态排行
            "top_stocks": [...],      # 信号最强个股
        }
        """
        all_patterns = defaultdict(lambda: {
            "signals": 0, "wins": 0, "losses": 0,
            "total_return": 0.0, "returns": [],
        })
        stock_results = {}
        top_stocks = []

        for code, data in stock_data.items():
            daily = data.get("daily")
            if daily is None or daily.empty:
                continue

            bt = self.backtest_stock(daily)
            stock_results[code] = bt

            name = data.get("meta", {}).get("name", code)

            # 收集形态统计
            for label, stats in bt.get("patterns", {}).items():
                ap = all_patterns[label]
                ap["signals"] += stats["signals"]
                ap["wins"] += stats["wins"]
                ap["losses"] += stats["losses"]
                ap["total_return"] += stats["total_return"]
                ap["returns"].append(stats["avg_return"])

            # 记录最新信号
            signals_df = calc_all_indicators(daily.copy())
            signals_df = self.detector.detect_all(signals_df)
            latest_signal = self.detector.get_latest_signal(signals_df)

            if latest_signal["net_score"] > 0:
                top_stocks.append({
                    "code": code,
                    "name": name,
                    "signal": latest_signal["signal"],
                    "score": latest_signal["net_score"],
                    "price": latest_signal["price"],
                    "triggers": latest_signal["triggers"],
                    "indicators": latest_signal["indicators"],
                })

        # 全局形态排行
        global_ranking = []
        for name, ap in all_patterns.items():
            n = ap["signals"]
            if n < self.params["min_samples"]:
                continue
            wr = ap["wins"] / n
            avg_ret = ap["total_return"] / n
            global_ranking.append({
                "name": name,
                "win_rate": round(wr * 100, 1),
                "avg_return": round(avg_ret * 100, 2),
                "signals": n,
                "wins": ap["wins"],
            })

        global_ranking.sort(key=lambda x: x["win_rate"], reverse=True)

        # 最强信号排序
        top_stocks.sort(key=lambda x: x["score"], reverse=True)

        return {
            "global_ranking": global_ranking,
            "stock_results": stock_results,
            "top_stocks": top_stocks[:50],
        }


# 单例
backtest_engine = BacktestEngine()
