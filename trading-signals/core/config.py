"""
全局配置 — 智能交易信号系统
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# 数据范围
DEFAULT_LOOKBACK_DAYS = 360
MINUTE_DATA_DAYS = 60       # 分时数据只取近期（数据量太大）
TOP_N_STOCKS = 300          # 默认分析前N只（市值排序）

# 技术指标参数
INDICATOR_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_period": 14,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "ma_periods": [5, 10, 20, 60, 120, 250],
    "volume_ma": 20,
}

# 策略检测参数
PATTERN_PARAMS = {
    "macd_divergence_lookback": 30,   # 背离检测回溯天数
    "macd_divergence_min_gap": 5,     # 背离最小间隔
    "volume_breakout_ratio": 2.0,     # 放量倍数
    "volume_dry_ratio": 0.5,          # 缩量倍数
}

# 回测参数
BACKTEST_PARAMS = {
    "min_samples": 20,         # 最少样本数才纳入统计
    "hold_days": 5,            # T+1 持有天数
    "stop_loss": -0.05,        # 止损 -5%
    "take_profit": 0.10,       # 止盈 +10%
    "min_win_rate": 0.55,     # 最低胜率阈值
}

# Streamlit Cloud 缓存
CACHE_TTL_DAILY = 3600       # 日线缓存1小时
CACHE_TTL_MINUTE = 900       # 分时缓存15分钟
CACHE_TTL_LIST = 86400       # 股票列表1天
