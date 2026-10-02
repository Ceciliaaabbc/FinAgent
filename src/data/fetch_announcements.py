"""步骤 1.4（方案 B）：下载每只股票的历史公告（巨潮资讯网），作为历史回测用的文本数据。

运行：python -m src.data.fetch_announcements            # 全部股票（约 2 小时）
      python -m src.data.fetch_announcements --limit 5  # 测试

为什么用公告：免费新闻接口只返回最近几条，拿不到多年历史；
公告是上市公司的官方披露（业绩、增减持、中标、诉讼等），历史完整、来源权威。
每只股票单独存文件，支持断点续传。
"""
import argparse
import warnings

import akshare as ak
import pandas as pd
from tqdm import tqdm

from .config import END_DATE, PROCESSED_DIR, RAW_DIR, START_DATE
from .utils import fetch_with_retry

ANN_DIR = RAW_DIR / "announcements"
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def download_one(code: str) -> pd.DataFrame:
    df = fetch_with_retry(ak.stock_zh_a_disclosure_report_cninfo, symbol=code, market="沪深京",
                          start_date=START_DATE, end_date=END_DATE)
    df = df.rename(columns={"公告标题": "title", "公告时间": "publish_time", "公告链接": "url"})
    df["publish_time"] = pd.to_datetime(df["publish_time"].astype(str), format="mixed")
    df["code"] = code
    df["content"] = ""      # 公告接口只有标题；标题已包含主要信息（如“业绩预增”“股东减持”）
    df["source"] = "cninfo"
    return df[["code", "publish_time", "title", "content", "source", "url"]]


def merge_all() -> pd.DataFrame | None:
    files = sorted(ANN_DIR.glob("*.parquet"))
    if not files:
        print("没有已下载的公告文件，跳过合并")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["code", "title", "publish_time"]).sort_values(["code", "publish_time"])
    out = PROCESSED_DIR / "announcements_raw.parquet"
    df.to_parquet(out, index=False)
    print(f"合并完成：{df['code'].nunique()} 只股票，{len(df)} 条公告 → {out}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只下载前 N 只（测试用）")
    args = parser.parse_args()

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    codes = pd.read_csv(PROCESSED_DIR / "universe.csv", dtype={"code": str})["code"].tolist()
    codes = codes[: args.limit] if args.limit else codes

    failed = []
    for code in tqdm(codes, desc="下载公告"):
        path = ANN_DIR / f"{code}.parquet"
        if path.exists():
            continue
        try:
            download_one(code).to_parquet(path, index=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {code} 下载失败：{e}")
            failed.append(code)

    if failed:
        print(f"失败 {len(failed)} 只：{failed}（重新运行脚本会自动重试）")
    merge_all()


if __name__ == "__main__":
    main()
