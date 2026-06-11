"""
数据采集模块 — 自动下载A股全量K线数据
分钟K线 / 日K线 / 周K线 / 月K线 + 成交量
所有数据缓存到本地磁盘，支持增量更新
"""
import time
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import diskcache
import requests

from core.config import (
    DATA_DIR, DEFAULT_LOOKBACK_DAYS, MINUTE_DATA_DAYS, TOP_N_STOCKS,
    CACHE_TTL_DAILY, CACHE_TTL_MINUTE, CACHE_TTL_LIST,
)

# 缓存
cache = diskcache.Cache(str(DATA_DIR / ".cache"))


def _cache_key(prefix: str, *args) -> str:
    raw = f"{prefix}:{':'.join(str(a) for a in args)}"
    return hashlib.md5(raw.encode()).hexdigest()


# ================================================================
# 股票列表
# ================================================================

def get_stock_list(refresh: bool = False) -> pd.DataFrame:
    """获取全A股列表"""
    ck = _cache_key("stock_list")
    if not refresh:
        cached = cache.get(ck)
        if cached is not None:
            return pd.DataFrame(cached)

    try:
        import akshare as ak
        # 使用东财全量实时列表（最稳定）
        df_all = ak.stock_zh_a_spot_em()
        if df_all is not None and not df_all.empty:
            stocks = df_all[["代码", "名称"]].rename(columns={"代码": "code", "名称": "name"}).copy()
            stocks["code"] = stocks["code"].astype(str).str.zfill(6)
            stocks["market"] = stocks["code"].apply(lambda x: "sh" if str(x).startswith(("6", "68")) else "sz")
            stocks = stocks.drop_duplicates(subset=["code"]).reset_index(drop=True)
            cache.set(ck, stocks.to_dict(orient="records"), expire=CACHE_TTL_LIST)
            return stocks
    except Exception as e:
        print(f"股票列表获取失败: {e}")

    # 从缓存兜底
    cached = cache.get(ck)
    if cached:
        return pd.DataFrame(cached)
    return pd.DataFrame(columns=["code", "name", "market"])


# ================================================================
# K线数据下载
# ================================================================

def _parse_code_for_akshare(code: str, market: str) -> str:
    """转换为 akshare 格式的代码"""
    if market == "sh":
        return f"sh{code}"
    return f"sz{code}"


def _get_kline_akshare(code: str, period: str = "daily", days: int = 360) -> Optional[pd.DataFrame]:
    """通过 akshare 获取K线"""
    try:
        import akshare as ak

        if period == "daily":
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "换手率": "turnover",
                })
                df["date"] = pd.to_datetime(df["date"])
                return df[["date", "open", "high", "low", "close", "volume", "amount", "turnover"]]

        elif period == "weekly":
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days * 7 + 30)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(
                symbol=code, period="weekly",
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount",
                })
                df["date"] = pd.to_datetime(df["date"])
                return df[["date", "open", "high", "low", "close", "volume", "amount"]]

        elif period == "monthly":
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days * 31 + 60)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(
                symbol=code, period="monthly",
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount",
                })
                df["date"] = pd.to_datetime(df["date"])
                return df[["date", "open", "high", "low", "close", "volume", "amount"]]

        elif period == "minute":
            # 分时数据（最近60天）— 使用5分钟K线
            try:
                df = ak.stock_zh_a_hist_min_em(
                    symbol=code,
                    period="5",
                    adjust="qfq",
                )
            except Exception:
                # 降级：使用60分钟K线
                df = ak.stock_zh_a_hist(
                    symbol=code, period="60",
                    start_date=(datetime.now() - timedelta(days=MINUTE_DATA_DAYS)).strftime("%Y%m%d"),
                    end_date=datetime.now().strftime("%Y%m%d"),
                    adjust="qfq",
                )
            if df is not None and not df.empty:
                # 统一列名
                rename_map = {}
                for col in df.columns:
                    if "时间" in str(col) or "日期" in str(col):
                        rename_map[col] = "date"
                    elif "开" in str(col):
                        rename_map[col] = "open"
                    elif "收" in str(col):
                        rename_map[col] = "close"
                    elif "高" in str(col):
                        rename_map[col] = "high"
                    elif "低" in str(col):
                        rename_map[col] = "low"
                    elif "量" in str(col):
                        rename_map[col] = "volume"
                    elif "额" in str(col):
                        rename_map[col] = "amount"
                df = df.rename(columns=rename_map)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    cutoff = datetime.now() - timedelta(days=MINUTE_DATA_DAYS)
                    df = df[df["date"] >= cutoff]
                    keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
                    return df[keep_cols]
            return None

    except Exception as e:
        pass

    return None


def _get_kline_tencent(code: str, period: str = "daily", days: int = 360) -> Optional[pd.DataFrame]:
    """通过腾讯财经获取K线（备用）"""
    try:
        market_prefix = "sh" if code.startswith(("6", "68")) else "sz"
        symbol = f"{market_prefix}{code}"

        period_map = {"daily": "day", "weekly": "week", "monthly": "month"}
        p = period_map.get(period, "day")

        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},{p},,,{days},qfq"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        klines = data.get("data", {}).get(symbol, {}).get(p, [])
        if not klines:
            klines = data.get("data", {}).get(symbol, {}).get(f"qfq{p}", [])

        if not klines:
            return None

        records = []
        for row in klines:
            records.append({
                "date": pd.to_datetime(row[0]),
                "open": float(row[1]),
                "close": float(row[2]),
                "high": float(row[3]),
                "low": float(row[4]),
                "volume": float(row[5]),
            })

        return pd.DataFrame(records)

    except Exception:
        return None


def fetch_stock_data(code: str, name: str = "", force_refresh: bool = False) -> Dict:
    """
    获取单只股票的所有周期K线数据
    返回: {daily, weekly, monthly, minute, meta}
    """
    result = {"daily": None, "weekly": None, "monthly": None, "minute": None}

    for period, ttl, days in [
        ("daily", CACHE_TTL_DAILY, DEFAULT_LOOKBACK_DAYS),
        ("weekly", CACHE_TTL_DAILY * 4, DEFAULT_LOOKBACK_DAYS),
        ("monthly", CACHE_TTL_DAILY * 8, DEFAULT_LOOKBACK_DAYS * 2),
        ("minute", CACHE_TTL_MINUTE, MINUTE_DATA_DAYS),
    ]:
        ck = _cache_key(f"kline_{period}", code)
        if not force_refresh:
            cached = cache.get(ck)
            if cached is not None:
                result[period] = pd.DataFrame(cached)
                continue

        # 先尝试 akshare
        df = _get_kline_akshare(code, period, days)
        # 失败则尝试腾讯
        if df is None or df.empty:
            df = _get_kline_tencent(code, period, days)

        if df is not None and not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
            cache.set(ck, df.to_dict(orient="records"), expire=ttl)
            result[period] = df

    result["meta"] = {"code": code, "name": name, "fetched_at": datetime.now().isoformat()}

    return result


def fetch_batch_stocks(codes: List[str], names: List[str] = None,
                       max_workers: int = 8, progress_callback=None) -> Dict[str, Dict]:
    """
    批量获取多只股票数据
    返回: {code: {daily, weekly, monthly, minute, meta}}
    """
    if names is None:
        names = [""] * len(codes)

    results = {}
    total = len(codes)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_stock_data, code, name): code
            for code, name in zip(codes, names)
        }

        for i, future in enumerate(as_completed(futures)):
            code = futures[future]
            try:
                results[code] = future.result(timeout=60)
            except Exception as e:
                results[code] = {"daily": None, "weekly": None, "monthly": None,
                                 "minute": None, "meta": {"code": code, "error": str(e)}}

            if progress_callback:
                progress_callback((i + 1) / total)

    return results


def get_top_stocks(n: int = TOP_N_STOCKS) -> pd.DataFrame:
    """获取成交活跃的前N只股票（仅沪深主板+创业板，排除科创板/北交所）"""
    return get_main_board_stocks(n)


def get_main_board_stocks(n: int = None) -> pd.DataFrame:
    """
    获取沪深主板 + 创业板股票列表
    排除: 科创板(688/689)、北交所(8/4)、ST、退市
    按成交额排序，n 为 None 时返回全部
    """
    ck = _cache_key("main_board", n or 0)
    cached = cache.get(ck)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            df = df.rename(columns={"代码": "code", "名称": "name"})
            df["code"] = df["code"].astype(str).str.zfill(6)

            # 只保留沪深主板 + 创业板
            # 主板: 60XXXX (沪), 00XXXX (深)
            # 创业板: 30XXXX
            # 排除: 688/689 (科创板), 8/4 (北交所/新三板), ST, 退市, N新股
            df = df[
                df["code"].str.match(r"^(60|00|30)\d{4}$") &
                ~df["name"].str.contains("ST|退|N", na=False)
            ]

            if "成交额" in df.columns:
                df["成交额"] = pd.to_numeric(df["成交额"], errors="coerce")
                df = df.sort_values("成交额", ascending=False)

            if n is not None:
                df = df.head(n)

            result = df[["code", "name"]].copy()
            cache.set(ck, result.to_dict(orient="records"), expire=CACHE_TTL_LIST)
            return result
    except Exception as e:
        print(f"股票列表获取失败: {e}")

    # 降级：用通用列表过滤
    stocks = get_stock_list()
    stocks = stocks[stocks["code"].str.match(r"^(60|00|30)\d{4}$")]
    if n is not None:
        stocks = stocks.head(n)
    return stocks[["code", "name"]]


def clear_cache(period: str = None):
    """清除缓存"""
    if period:
        for key in list(cache.iterkeys()):
            if period in key:
                cache.delete(key)
    else:
        cache.clear()
