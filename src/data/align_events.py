"""步骤 1.5：给每条公告/新闻分配“可交易日期”（trade_date），防止未来函数。

运行：python -m src.data.align_events

规则（以 15:00 收盘为界）：
- 交易日 15:00 之前发布 → 当天可用（trade_date = 当天）
- 15:00 及之后发布，或在非交易日发布 → 下一个交易日可用
- 只有日期、没有具体时间（时间为 00:00:00）→ 保守起见视为盘后发布，下一个交易日可用
  （巨潮的公告大多只有日期，宁可晚用一天，也不能“提前知道消息”）

输出：data/processed/events.parquet
字段：code, trade_date, publish_time, title, content, source, url
"""
import warnings

import numpy as np
import pandas as pd

from .config import MARKET_CLOSE, PROCESSED_DIR
from .utils import trading_calendar

# pandas 内部与新版 NumPy 的兼容性警告，不影响结果
warnings.filterwarnings("ignore", message=".*generic.*unit.*", category=DeprecationWarning)

SOURCES = ["announcements_raw.parquet", "news_raw.parquet"]


def assign_trade_date(publish_time: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    t = pd.to_datetime(publish_time)
    day = t.dt.normalize()
    time_of_day = t - day
    close = pd.Timedelta(MARKET_CLOSE)

    no_time = time_of_day == pd.Timedelta(0)
    after_close = (time_of_day >= close) | no_time
    is_trading_day = day.isin(calendar)

    # 当天可用：交易日且盘中/盘前发布；否则顺延到“严格晚于发布日”的第一个交易日
    same_day = is_trading_day & ~after_close
    next_idx = np.searchsorted(calendar.values, day.values, side="right")
    next_idx = np.clip(next_idx, 0, len(calendar) - 1)
    next_day = pd.Series(calendar.values[next_idx], index=t.index)
    return day.where(same_day, next_day)


def main():
    frames = [pd.read_parquet(PROCESSED_DIR / f) for f in SOURCES if (PROCESSED_DIR / f).exists()]
    if not frames:
        print("没有找到公告或新闻数据，请先运行 fetch_announcements / fetch_news")
        return
    df = pd.concat(frames, ignore_index=True)

    calendar = trading_calendar()
    df["trade_date"] = assign_trade_date(df["publish_time"], calendar)
    df = df.dropna(subset=["title"]).drop_duplicates(["code", "title", "trade_date"])
    df = df[["code", "trade_date", "publish_time", "title", "content", "source", "url"]]
    df = df.sort_values(["code", "trade_date", "publish_time"]).reset_index(drop=True)

    out = PROCESSED_DIR / "events.parquet"
    df.to_parquet(out, index=False)
    shifted = (df["trade_date"] != df["publish_time"].dt.normalize()).mean()
    print(f"对齐完成：{len(df)} 条，其中 {shifted:.0%} 被顺延到之后的交易日 → {out}")


if __name__ == "__main__":
    main()
