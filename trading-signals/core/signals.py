"""
实时信号生成器 — 基于最新K线数据生成买卖信号
扫描全市场/自选股池，输出可操作的买卖建议
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import TOP_N_STOCKS
from core.data_fetcher import (
    fetch_stock_data, get_main_board_stocks, get_stock_list,
)
from core.indicators import calc_all_indicators, detect_market_regime
from core.patterns import PatternDetector
from core.backtest import BacktestEngine


class SignalGenerator:
    """实时信号生成器"""

    def __init__(self):
        self.detector = PatternDetector()
        self.backtest = BacktestEngine()

    def scan_stock(self, code: str, name: str = "") -> Dict:
        """
        扫描单只股票，生成最新信号
        返回: 完整信号字典
        """
        data = fetch_stock_data(code, name)

        daily = data.get("daily")
        weekly = data.get("weekly")
        monthly = data.get("monthly")

        result = {
            "code": code,
            "name": name,
            "price": 0,
            "signal": "无数据",
            "signal_score": 0,
            "buy_triggers": [],
            "sell_triggers": [],
            "indicators": {},
            "regime": "unknown",
            "multi_period_confirm": False,
        }

        if daily is None or daily.empty:
            return result

        # 计算指标和形态
        daily_ind = calc_all_indicators(daily)
        daily_sig = self.detector.detect_all(daily_ind.copy())

        # 市场状态
        regime = detect_market_regime(daily_ind)

        # 最新信号
        latest = self.detector.get_latest_signal(daily_sig)

        result["price"] = latest["price"]
        result["signal"] = latest["signal"]
        result["signal_score"] = latest["net_score"]
        result["indicators"] = latest["indicators"]
        result["regime"] = regime

        # 分离买卖触发器
        for t in latest["triggers"]:
            t_str = str(t)
            if t_str.startswith("BUY "):
                result["buy_triggers"].append(t_str[4:])
            elif t_str.startswith("SELL "):
                result["sell_triggers"].append(t_str[5:])

        # 多周期确认
        if weekly is not None and not weekly.empty:
            weekly_ind = calc_all_indicators(weekly)
            weekly_sig = self.detector.detect_all(weekly_ind)
            w_latest = self.detector.get_latest_signal(weekly_sig)

            if monthly is not None and not monthly.empty:
                monthly_ind = calc_all_indicators(monthly)
                monthly_sig = self.detector.detect_all(monthly_ind)
                m_latest = self.detector.get_latest_signal(monthly_sig)

                # 日/周/月共振
                daily_bullish = latest["net_score"] > 0
                weekly_bullish = w_latest["net_score"] > 0
                monthly_bullish = m_latest["net_score"] > 0

                if daily_bullish and weekly_bullish and monthly_bullish:
                    result["multi_period_confirm"] = True
                    result["signal_score"] += 2
                elif daily_bullish and weekly_bullish:
                    result["signal_score"] += 1

        return result

    def scan_market(self, n_stocks: int = None,
                    progress_callback=None) -> Dict:
        """
        扫描沪深主板+创业板，返回信号排序
        n_stocks=None 时扫描全部符合条件的股票
        """
        stocks = get_main_board_stocks(n_stocks)
        results = []
        total = len(stocks)

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            for _, row in stocks.iterrows():
                code = row["code"]
                name = row.get("name", "")
                futures[executor.submit(self.scan_stock, code, name)] = code

            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result(timeout=45)
                    results.append(result)
                except Exception:
                    pass

                if progress_callback:
                    progress_callback((i + 1) / total)

        # 按信号分数排序（正值买入在前，负值卖出在前）
        results.sort(key=lambda x: abs(x["signal_score"]), reverse=True)

        # 分类
        buy_signals = [r for r in results if r["signal_score"] > 1]
        sell_signals = [r for r in results if r["signal_score"] < -1]
        neutral = [r for r in results if -1 <= r["signal_score"] <= 1]

        return {
            "buy": buy_signals,
            "sell": sell_signals,
            "neutral": neutral,
            "all": results,
            "scanned": len(results),
            "timestamp": datetime.now().isoformat(),
        }

    def get_buy_recommendations(self, n: int = 20) -> List[Dict]:
        """
        获取买入建议Top N
        """
        scan = self.scan_market()
        buy_list = scan.get("buy", [])
        return buy_list[:n]

    def get_sell_warnings(self, n: int = 20) -> List[Dict]:
        """
        获取卖出警告Top N
        """
        scan = self.scan_market()
        sell_list = scan.get("sell", [])
        return sell_list[:n]


# 单例
signal_gen = SignalGenerator()
