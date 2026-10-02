"""步骤 1.1：获取沪深 300 成分股列表。

运行：python -m src.data.fetch_universe

注意：这里拿到的是“当前”成分股。用当前成分股回测过去会有“幸存者偏差”
（过去被剔除的差公司不在名单里，回测结果偏乐观）。起步阶段可以接受，
但要在报告中写明这一局限。
"""
import akshare as ak

from .config import INDEX_CODE, PROCESSED_DIR
from .utils import fetch_with_retry


def main():
    df = fetch_with_retry(ak.index_stock_cons_csindex, symbol=INDEX_CODE)
    df = df.rename(columns={"成分券代码": "code", "成分券名称": "name", "交易所": "exchange"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[["code", "name", "exchange"]].drop_duplicates("code")

    out = PROCESSED_DIR / "universe.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"股票池：{len(df)} 只 → {out}")


if __name__ == "__main__":
    main()
