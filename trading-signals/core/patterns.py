"""
形态识别引擎 — 检测经典买卖点形态
MACD顶/底背离 / 量价背离 / 均线金叉死叉 / RSI超买超卖
布林带突破 / KDJ共振 / 放量突破 / 多指标共振
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from core.config import PATTERN_PARAMS, INDICATOR_PARAMS


class PatternDetector:
    """形态检测器"""

    def __init__(self):
        self.p = PATTERN_PARAMS

    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测所有买卖点形态
        返回: DataFrame 包含所有信号列
        """
        if df is None or df.empty:
            return df

        df = df.copy()
        n = len(df)

        # 确保有足够数据
        if n < 60:
            return df

        # ---- 1. MACD 背离 ----
        df = self._detect_macd_divergence(df)

        # ---- 2. 量价背离 ----
        df = self._detect_volume_divergence(df)

        # ---- 3. 均线交叉 ----
        df = self._detect_ma_cross(df)

        # ---- 4. RSI 信号 ----
        df = self._detect_rsi_signals(df)

        # ---- 5. 布林带信号 ----
        df = self._detect_bollinger_signals(df)

        # ---- 6. KDJ 信号 ----
        df = self._detect_kdj_signals(df)

        # ---- 7. 放量突破 ----
        df = self._detect_volume_breakout(df)

        # ---- 8. 综合买卖信号 ----
        df = self._combine_signals(df)

        return df

    # ================================================================
    # MACD 背离
    # ================================================================

    def _detect_macd_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD 背离检测
        顶背离: 价格创新高，MACD DIF 未创新高 → 卖出信号
        底背离: 价格创新低，MACD DIF 未创新低 → 买入信号
        """
        n = len(df)
        lookback = self.p["macd_divergence_lookback"]
        min_gap = self.p["macd_divergence_min_gap"]

        df["macd_bullish_div"] = 0  # 底背离（买入信号）
        df["macd_bearish_div"] = 0  # 顶背离（卖出信号）

        if n < lookback + min_gap:
            return df

        for i in range(lookback + min_gap, n):
            window = df.iloc[i - lookback:i + 1]

            # 底背离检测
            price_low_idx = window["close"].idxmin()
            dif_low_idx = window["macd_dif"].idxmin()

            if (price_low_idx > dif_low_idx + min_gap and
                df.loc[price_low_idx, "close"] < df.loc[dif_low_idx, "close"] and
                df.loc[price_low_idx, "macd_dif"] > df.loc[dif_low_idx, "macd_dif"]):
                df.at[i, "macd_bullish_div"] = 1

            # 顶背离检测
            price_high_idx = window["close"].idxmax()
            dif_high_idx = window["macd_dif"].idxmax()

            if (price_high_idx > dif_high_idx + min_gap and
                df.loc[price_high_idx, "close"] > df.loc[dif_high_idx, "close"] and
                df.loc[price_high_idx, "macd_dif"] < df.loc[dif_high_idx, "macd_dif"]):
                df.at[i, "macd_bearish_div"] = -1

        return df

    # ================================================================
    # 量价背离
    # ================================================================

    def _detect_volume_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        量价背离
        价涨量缩 → 顶背离（卖出）
        价跌量增 → 底背离（买入 — 恐慌盘出清）
        """
        df["vol_bearish_div"] = 0
        df["vol_bullish_div"] = 0

        if len(df) < 10:
            return df

        for i in range(10, len(df)):
            price_delta = df["close"].iloc[i] - df["close"].iloc[i - 10]
            vol_delta = df["volume"].iloc[i] - df["volume"].iloc[i - 10]

            # 价涨量缩
            if price_delta > 0 and vol_delta < -df["volume"].iloc[i - 10] * 0.3:
                df.at[i, "vol_bearish_div"] = -1

            # 价跌量缩（地量见地价）
            if (price_delta < 0 and
                df["volume_ratio_20"].iloc[i] < self.p["volume_dry_ratio"] and
                df["change_20d"].iloc[i] < -0.1):
                df.at[i, "vol_bullish_div"] = 1

        return df

    # ================================================================
    # 均线交叉
    # ================================================================

    def _detect_ma_cross(self, df: pd.DataFrame) -> pd.DataFrame:
        """均线金叉/死叉"""
        df["ma_golden_cross"] = 0
        df["ma_death_cross"] = 0

        if "ma_5" not in df.columns or "ma_20" not in df.columns:
            return df

        for i in range(2, len(df)):
            # MA5 上穿 MA20（金叉）
            if (df["ma_5"].iloc[i - 1] <= df["ma_20"].iloc[i - 1] and
                df["ma_5"].iloc[i] > df["ma_20"].iloc[i]):
                df.at[i, "ma_golden_cross"] = 1

            # MA5 下穿 MA20（死叉）
            if (df["ma_5"].iloc[i - 1] >= df["ma_20"].iloc[i - 1] and
                df["ma_5"].iloc[i] < df["ma_20"].iloc[i]):
                df.at[i, "ma_death_cross"] = -1

        return df

    # ================================================================
    # RSI 信号
    # ================================================================

    def _detect_rsi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI超买超卖信号"""
        df["rsi_oversold"] = 0
        df["rsi_overbought"] = 0

        if "rsi" not in df.columns:
            return df

        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_overbought"] = ((df["rsi"] > 70).astype(int) * -1)

        # RSI 底背离加强
        for i in range(2, len(df)):
            if df["rsi_oversold"].iloc[i] == 1 and df["rsi"].iloc[i] > df["rsi"].iloc[i - 1]:
                df.at[i, "rsi_oversold"] = 2  # 超卖区回升

        return df

    # ================================================================
    # 布林带信号
    # ================================================================

    def _detect_bollinger_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """布林带突破信号"""
        df["bb_lower_touch"] = 0
        df["bb_upper_touch"] = 0
        df["bb_squeeze"] = 0

        if "bb_lower" not in df.columns:
            return df

        # 触及下轨
        df["bb_lower_touch"] = (df["close"] <= df["bb_lower"] * 1.02).astype(int)

        # 突破上轨
        df["bb_upper_touch"] = ((df["close"] >= df["bb_upper"] * 0.98).astype(int) * -1)

        # 布林带收窄（即将变盘）
        if len(df) > 20:
            bb_width_now = df["bb_width"].iloc[-1]
            bb_width_20 = df["bb_width"].iloc[-20:].mean()
            if bb_width_now < bb_width_20 * 0.5:
                df.at[df.index[-1], "bb_squeeze"] = 1

        return df

    # ================================================================
    # KDJ 信号
    # ================================================================

    def _detect_kdj_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """KDJ金叉死叉"""
        df["kdj_golden"] = 0
        df["kdj_death"] = 0

        if "k" not in df.columns or "d" not in df.columns:
            return df

        for i in range(2, len(df)):
            # K上穿D且低位
            if (df["k"].iloc[i - 1] <= df["d"].iloc[i - 1] and
                df["k"].iloc[i] > df["d"].iloc[i] and
                df["k"].iloc[i] < 40):
                df.at[i, "kdj_golden"] = 1

            # K下穿D且高位
            if (df["k"].iloc[i - 1] >= df["d"].iloc[i - 1] and
                df["k"].iloc[i] < df["d"].iloc[i] and
                df["k"].iloc[i] > 60):
                df.at[i, "kdj_death"] = -1

        return df

    # ================================================================
    # 放量突破
    # ================================================================

    def _detect_volume_breakout(self, df: pd.DataFrame) -> pd.DataFrame:
        """放量突破信号"""
        df["vol_breakout_up"] = 0
        df["vol_breakout_down"] = 0

        if "volume_ratio_20" not in df.columns:
            return df

        for i in range(1, len(df)):
            if (df["volume_ratio_20"].iloc[i] > self.p["volume_breakout_ratio"] and
                df["change_pct"].iloc[i] > 0.02):
                df.at[i, "vol_breakout_up"] = 1

            if (df["volume_ratio_20"].iloc[i] > self.p["volume_breakout_ratio"] and
                df["change_pct"].iloc[i] < -0.02):
                df.at[i, "vol_breakout_down"] = -1

        return df

    # ================================================================
    # 综合买卖信号
    # ================================================================

    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        综合所有信号生成最终买卖建议
        每个正向信号+1，负向信号-1
        """
        buy_signals = [
            "macd_bullish_div", "vol_bullish_div", "ma_golden_cross",
            "rsi_oversold", "bb_lower_touch", "kdj_golden", "vol_breakout_up",
        ]
        sell_signals = [
            "macd_bearish_div", "vol_bearish_div", "ma_death_cross",
            "rsi_overbought", "bb_upper_touch", "kdj_death", "vol_breakout_down",
        ]

        df["buy_score"] = 0
        df["sell_score"] = 0

        for sig in buy_signals:
            if sig in df.columns:
                df["buy_score"] += df[sig].fillna(0).clip(lower=0)

        for sig in sell_signals:
            if sig in df.columns:
                df["sell_score"] += df[sig].fillna(0).clip(upper=0).abs()

        # 综合净得分
        df["net_signal"] = df["buy_score"] - df["sell_score"]

        # 信号等级
        df["signal_strength"] = "无"
        df.loc[df["net_signal"] >= 3, "signal_strength"] = "强烈买入"
        df.loc[(df["net_signal"] >= 2) & (df["net_signal"] < 3), "signal_strength"] = "买入"
        df.loc[(df["net_signal"] >= 1) & (df["net_signal"] < 2), "signal_strength"] = "关注买入"
        df.loc[df["net_signal"] <= -3, "signal_strength"] = "强烈卖出"
        df.loc[(df["net_signal"] <= -2) & (df["net_signal"] > -3), "signal_strength"] = "卖出"
        df.loc[(df["net_signal"] <= -1) & (df["net_signal"] > -2), "signal_strength"] = "关注卖出"

        return df

    def get_latest_signal(self, df: pd.DataFrame) -> Dict:
        """获取最新一根K线的信号详情"""
        if df is None or df.empty:
            return {"signal": "无数据", "score": 0}

        latest = df.iloc[-1]
        signal = latest.get("signal_strength", "无")

        def safe_float(val, default=0.0):
            try:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return default
                return float(val)
            except (ValueError, TypeError):
                return default

        def safe_int(val, default=0):
            return int(safe_float(val, default))

        buy_score = safe_int(latest.get("buy_score", 0))
        sell_score = safe_int(latest.get("sell_score", 0))

        # 收集触发的具体信号
        triggers = []
        buy_map = {
            "macd_bullish_div": "MACD底背离",
            "vol_bullish_div": "量价底背离",
            "ma_golden_cross": "MA金叉",
            "rsi_oversold": "RSI超卖",
            "bb_lower_touch": "布林下轨",
            "kdj_golden": "KDJ金叉",
            "vol_breakout_up": "放量突破",
        }
        sell_map = {
            "macd_bearish_div": "MACD顶背离",
            "vol_bearish_div": "量价顶背离",
            "ma_death_cross": "MA死叉",
            "rsi_overbought": "RSI超买",
            "bb_upper_touch": "布林上轨",
            "kdj_death": "KDJ死叉",
            "vol_breakout_down": "放量下跌",
        }

        for col, label in buy_map.items():
            if col in df.columns:
                val = safe_float(latest[col])
                if val > 0:
                    triggers.append(f"BUY {label}")
        for col, label in sell_map.items():
            if col in df.columns:
                val = safe_float(latest[col])
                if val < 0:
                    triggers.append(f"SELL {label}")

        price = safe_float(latest.get("close", 0))
        return {
            "date": str(latest.get("date", ""))[:10],
            "price": price,
            "signal": str(signal),
            "buy_score": buy_score,
            "sell_score": sell_score,
            "net_score": buy_score - sell_score,
            "triggers": triggers,
            "indicators": {
                "rsi": round(safe_float(latest.get("rsi", 50)), 1),
                "macd_dif": round(safe_float(latest.get("macd_dif", 0)), 3),
                "macd_dea": round(safe_float(latest.get("macd_dea", 0)), 3),
                "kdj_k": round(safe_float(latest.get("k", 50)), 1),
                "volume_ratio": round(safe_float(latest.get("volume_ratio_20", 1)), 2),
                "change_pct": round(safe_float(latest.get("change_pct", 0)) * 100, 2),
            },
        }


# 单例
detector = PatternDetector()
