"""步骤 1.4（方案 C）：每日增量采集个股新闻（东方财富），从今天开始积累新闻库。

运行：python -m src.data.fetch_news             # 全部股票，约 5 分钟
      python -m src.data.fetch_news --limit 5   # 测试

为什么要每天运行：这个接口只返回每只股票最近约 10 条新闻，拿不到历史。
每天运行一次，新数据会追加到新闻库中（按链接去重），时间越长积累越多。
设置定时任务（macOS / Linux，每个交易日 18:00 运行）：
    crontab -e
    0 18 * * 1-5 cd /Users/cyt/Documents/FinAgent && .venv/bin/python -m src.data.fetch_news >> data/news_cron.log 2>&1
"""
import argparse

import akshare as ak
import pandas as pd
from tqdm import tqdm

from .config import PROCESSED_DIR
from .utils import fetch_with_retry

NEWS_STORE = PROCESSED_DIR / "news_raw.parquet"


def download_one(code: str) -> pd.DataFrame:
    df = fetch_with_retry(ak.stock_news_em, symbol=code)
    df = df.rename(columns={"新闻标题": "title", "新闻内容": "content", "发布时间": "publish_time",
                            "文章来源": "media", "新闻链接": "url"})
    df["code"] = code
    df["source"] = "eastmoney_news"
    return df[["code", "publish_time", "title", "content", "source", "url"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只采集前 N 只（测试用）")
    args = parser.parse_args()

    codes = pd.read_csv(PROCESSED_DIR / "universe.csv", dtype={"code": str})["code"].tolist()
    codes = codes[: args.limit] if args.limit else codes

    frames = []
    for code in tqdm(codes, desc="采集新闻"):
        try:
            frames.append(download_one(code))
        except Exception as e:  # noqa: BLE001
            print(f"  {code} 采集失败：{e}")
    if not frames:
        print("本次没有采集到新闻")
        return

    new = pd.concat(frames, ignore_index=True)
    new["publish_time"] = pd.to_datetime(new["publish_time"])
    old = pd.read_parquet(NEWS_STORE) if NEWS_STORE.exists() else new.iloc[0:0]
    merged = (pd.concat([old, new], ignore_index=True)
              .drop_duplicates(["code", "url"])
              .sort_values(["code", "publish_time"]))
    merged.to_parquet(NEWS_STORE, index=False)
    print(f"本次新增 {len(merged) - len(old)} 条，新闻库共 {len(merged)} 条 → {NEWS_STORE}")


if __name__ == "__main__":
    main()
