"""步骤 1.2：下载股票池中每只股票的日线行情（前复权）。

运行：python -m src.data.fetch_prices            # 全部股票
      python -m src.data.fetch_prices --limit 5  # 只下载前 5 只，用于测试

- 每只股票单独存一个文件，已下载的会跳过 → 中断后重新运行即可断点续传。
- 前复权（qfq）：消除分红送股造成的价格跳空，否则程序会误以为股价暴跌。
- 数据源：依次尝试新浪 → 腾讯 → 东方财富，前一个失败就换下一个（网络不稳定时很常见）。统一成相同的列：
  date, code, open, high, low, close, volume, amount, turnover（小数，0.01 = 1%）, pct_chg（小数）
"""
import argparse

import akshare as ak
import pandas as pd
from tqdm import tqdm

from .config import END_DATE, PROCESSED_DIR, RAW_DIR, START_DATE
from .utils import fetch_with_retry

PRICE_DIR = RAW_DIR / "prices"
OUT_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover"]


def with_exchange_prefix(code: str) -> str:
    """600519 → sh600519；000001 → sz000001（新浪接口需要交易所前缀）。"""
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    return f"sz{code}"


def from_sina(code: str) -> pd.DataFrame:
    df = fetch_with_retry(ak.stock_zh_a_daily, symbol=with_exchange_prefix(code),
                          start_date=START_DATE, end_date=END_DATE, adjust="qfq")
    df["code"] = code
    return df  # 新浪的 turnover 本身就是小数


def from_tencent(code: str) -> pd.DataFrame:
    df = fetch_with_retry(ak.stock_zh_a_hist_tx, symbol=with_exchange_prefix(code),
                          start_date=START_DATE, end_date=END_DATE, adjust="qfq")
    df["code"] = code
    return df  # 腾讯的 volume 单位是股，turnover 是小数，与新浪一致


def from_eastmoney(code: str) -> pd.DataFrame:
    df = fetch_with_retry(ak.stock_zh_a_hist, symbol=code, period="daily",
                          start_date=START_DATE, end_date=END_DATE, adjust="qfq")
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                            "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover"})
    df["code"] = code
    df["turnover"] = df["turnover"] / 100  # 东方财富是百分数，统一成小数
    return df


SOURCES = [("新浪", from_sina), ("腾讯", from_tencent), ("东方财富", from_eastmoney)]


def download_one(code: str) -> pd.DataFrame:
    for i, (name, fetch) in enumerate(SOURCES):
        try:
            df = fetch(code)
            break
        except Exception as e:  # noqa: BLE001
            if i == len(SOURCES) - 1:
                raise
            print(f"  {code} {name}失败（{str(e)[:80]}），改用{SOURCES[i + 1][0]}")
    df = df[OUT_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_chg"] = df["close"].pct_change()  # 用复权价计算日收益率
    return df


def merge_all() -> pd.DataFrame | None:
    files = sorted(PRICE_DIR.glob("*.parquet"))
    if not files:
        print("没有已下载的行情文件，跳过合并")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    out = PROCESSED_DIR / "prices.parquet"
    df.to_parquet(out, index=False)
    print(f"合并完成：{df['code'].nunique()} 只股票，{len(df)} 行 → {out}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只下载前 N 只（测试用）")
    args = parser.parse_args()

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    codes = pd.read_csv(PROCESSED_DIR / "universe.csv", dtype={"code": str})["code"].tolist()
    codes = codes[: args.limit] if args.limit else codes

    failed = []
    for code in tqdm(codes, desc="下载行情"):
        path = PRICE_DIR / f"{code}.parquet"
        if path.exists():
            continue
        try:
            df = download_one(code)
            if df.empty:
                failed.append(code)
                continue
            df.to_parquet(path, index=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {code} 下载失败：{e}")
            failed.append(code)

    if failed:
        print(f"失败 {len(failed)} 只：{failed}（重新运行脚本会自动重试）")
    merge_all()


if __name__ == "__main__":
    main()
