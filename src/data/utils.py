"""通用工具：带重试的请求、交易日历。"""
import time
from functools import lru_cache

import akshare as ak
import pandas as pd

from .config import REQUEST_SLEEP


def fetch_with_retry(func, *args, retries=3, **kwargs):
    """调用 AkShare 接口，失败时等待后重试（网络抖动、限流很常见）。"""
    for attempt in range(1, retries + 1):
        try:
            result = func(*args, **kwargs)
            time.sleep(REQUEST_SLEEP)
            return result
        except Exception as e:  # noqa: BLE001 - AkShare 抛出的异常类型不统一
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  请求失败（{e}），{wait} 秒后第 {attempt + 1} 次重试…")
            time.sleep(wait)


@lru_cache(maxsize=1)
def trading_calendar() -> pd.DatetimeIndex:
    """A 股交易日历（来自新浪，含未来已公布的交易日）。"""
    df = fetch_with_retry(ak.tool_trade_date_hist_sina)
    return pd.DatetimeIndex(pd.to_datetime(df["trade_date"])).sort_values()
