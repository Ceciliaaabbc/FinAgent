"""步骤 1.3：下载沪深 300 指数日线，作为计算“超额收益”的基准。

运行：python -m src.data.fetch_index
"""
import akshare as ak
import pandas as pd

from .config import END_DATE, INDEX_SYMBOL, PROCESSED_DIR, START_DATE
from .utils import fetch_with_retry


def main():
    df = fetch_with_retry(ak.stock_zh_index_daily, symbol=INDEX_SYMBOL)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.sort_values("date").reset_index(drop=True)

    out = PROCESSED_DIR / "benchmark.parquet"
    df.to_parquet(out, index=False)
    print(f"基准指数：{len(df)} 个交易日（{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}）→ {out}")


if __name__ == "__main__":
    main()
