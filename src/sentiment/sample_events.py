"""步骤 2.2（上）：从全部文本中抽取一批样本，准备交给老师模型标注。

运行：python -m src.sentiment.sample_events --n 5000

抽样策略：
1. 标题去重：很多公告标题一模一样（如“关联交易公告”），重复标注没有意义。
2. 按年份均匀抽样：保证每年都有数据，后面才能按时间划分训练集和测试集。
3. 控制例行公告的比例（默认 30%）：例行公告数量极多且几乎都是“中性”，
   全部按自然比例抽样会把标注预算浪费在它们身上，模型也学不到利好/利空。
"""
import argparse

import pandas as pd

from .config import LLM_DIR, ROUTINE_PATTERNS
from .events import load_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="抽样条数")
    parser.add_argument("--routine-ratio", type=float, default=0.3, help="例行公告占比")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ev = load_events().drop_duplicates("title")
    ev["routine"] = ev["title"].str.contains("|".join(ROUTINE_PATTERNS))
    ev["year"] = ev["trade_date"].dt.year

    years = sorted(ev["year"].unique())
    quota = args.n // len(years)
    parts = []
    for year in years:
        g = ev[ev["year"] == year]
        normal, routine = g[~g["routine"]], g[g["routine"]]
        n_normal = min(quota - int(quota * args.routine_ratio), len(normal))
        n_routine = min(quota - n_normal, len(routine))
        n_normal = min(quota - n_routine, len(normal))   # 一类不够时由另一类补足
        parts += [normal.sample(n_normal, random_state=args.seed),
                  routine.sample(n_routine, random_state=args.seed)]

    sample = pd.concat(parts).sort_values("trade_date").reset_index(drop=True)
    cols = ["id", "code", "name", "trade_date", "source", "title", "content", "routine"]
    out = LLM_DIR / "to_label.parquet"
    sample[cols].to_parquet(out, index=False)

    print(f"抽样 {len(sample)} 条（去重后可选 {len(ev)} 条）→ {out}")
    print(f"每年条数：{sample['trade_date'].dt.year.value_counts().sort_index().to_dict()}")
    print(f"例行公告占比：{sample['routine'].mean():.0%}")


if __name__ == "__main__":
    main()
