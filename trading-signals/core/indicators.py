"""
技术指标计算引擎
MACD / RSI / 布林带 / 均线 / KDJ / 量价关系
"""
import pandas as pd
import numpy as np
from core.config import INDICATOR_PARAMS


def calc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标
    输入: DataFrame with [open, high, low, close, volume]
    输出: DataFrame with all indicators added
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    p = INDICATOR_PARAMS

    # ---- 均线 ----
    for ma in p["ma_periods"]:
        df[f"ma_{ma}"] = df["close"].rolling(ma).mean()

    # ---- EMA ----
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # ---- MACD ----
    df["macd_dif"] = df["ema_12"] - df["ema_26"]
    df["macd_dea"] = df["macd_dif"].ewm(span=p["macd_signal"], adjust=False).mean()
    df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])  # 柱状图（乘以2）

    # ---- RSI ----
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/p["rsi_period"], adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/p["rsi_period"], adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # ---- 布林带 ----
    df["bb_mid"] = df["close"].rolling(p["bollinger_period"]).mean()
    bb_std = df["close"].rolling(p["bollinger_period"]).std()
    df["bb_upper"] = df["bb_mid"] + p["bollinger_std"] * bb_std
    df["bb_lower"] = df["bb_mid"] - p["bollinger_std"] * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ---- KDJ ----
    low_min = df["low"].rolling(9).min()
    high_max = df["high"].rolling(9).max()
    rsv = (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    df["k"] = rsv.ewm(com=2, adjust=False).mean()
    df["d"] = df["k"].ewm(com=2, adjust=False).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]

    # ---- 成交量相关 ----
    df["volume_ma_5"] = df["volume"].rolling(5).mean()
    df["volume_ma_20"] = df["volume"].rolling(p["volume_ma"]).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_5"].replace(0, np.nan)
    df["volume_ratio_20"] = df["volume"] / df["volume_ma_20"].replace(0, np.nan)

    # ---- ATR (平均真实波幅) ----
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / df["close"]

    # ---- 涨跌幅 ----
    df["change_pct"] = df["close"].pct_change()
    df["change_5d"] = df["close"].pct_change(5)
    df["change_20d"] = df["close"].pct_change(20)

    # ---- 趋势强度 ADX 简化 ----
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    df["plus_di"] = pd.Series(plus_dm).rolling(14).mean() / df["atr"].replace(0, np.nan) * 100
    df["minus_di"] = pd.Series(minus_dm).rolling(14).mean() / df["atr"].replace(0, np.nan) * 100
    dx = (abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"]).replace(0, np.nan)) * 100
    df["adx"] = dx.rolling(14).mean()

    # ---- OBV ----
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    return df


def detect_market_regime(df: pd.DataFrame) -> str:
    """检测市场状态: trending_up / trending_down / ranging"""
    if df is None or df.empty or "adx" not in df.columns:
        return "unknown"

    adx = df["adx"].iloc[-1] if not pd.isna(df["adx"].iloc[-1]) else 20
    close = df["close"].values
    ma_60 = df["ma_60"].iloc[-1] if "ma_60" in df.columns else close[-1]

    if adx > 25:
        if close[-1] > ma_60:
            return "trending_up"
        return "trending_down"
    elif adx < 20:
        return "ranging"
    return "trending_up" if close[-1] > ma_60 else "trending_down"
