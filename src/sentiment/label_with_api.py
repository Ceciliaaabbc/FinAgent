"""步骤 2.2（下）：用老师模型（强模型 API）批量标注抽样数据。

运行：python -m src.sentiment.label_with_api                      # 调用 .env 中配置的老师模型
      python -m src.sentiment.label_with_api --limit 50           # 先标 50 条，检查效果和成本
      python -m src.sentiment.label_with_api --provider keyword   # 不调 API，用关键词基线跑通流程

为什么这样做：人工标注 5000 条太慢。先让强模型按规则标注，再用这些标签训练小模型，
这种做法叫“知识蒸馏”。中断后重新运行会跳过已标注的条目（断点续传）。
"""
import argparse

import pandas as pd

from .config import LLM_DIR, TEACHER
from .llm_client import make_model, run_batch
from .prompts import teacher_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["api", "keyword"], default="api")
    parser.add_argument("--workers", type=int, default=8, help="并发请求数（被限流时调小）")
    parser.add_argument("--limit", type=int, default=None, help="只标注前 N 条（测试用）")
    args = parser.parse_args()

    samples = pd.read_parquet(LLM_DIR / "to_label.parquet")
    rows = samples.to_dict("records")[: args.limit]
    model = make_model(args.provider, TEACHER)
    labeler = TEACHER["model"] if args.provider == "api" else "keyword"

    cache = LLM_DIR / f"labels_raw_{args.provider}.jsonl"
    result = run_batch(rows, teacher_messages, model, cache, workers=args.workers, desc="标注")
    if result.empty:
        print("没有得到任何标注结果")
        return

    valid = result[result["label"].notna()]
    labels = samples.merge(valid[["id", "label", "score", "reason"]], on="id")
    labels["labeler"] = labeler
    out = LLM_DIR / "labels.parquet"
    labels.to_parquet(out, index=False)

    fail = 1 - len(valid) / len(result)
    tokens_in, tokens_out = result["prompt_tokens"].sum(), result["completion_tokens"].sum()
    cost = (tokens_in * TEACHER["price_in"] + tokens_out * TEACHER["price_out"]) / 1e6
    print(f"\n标注完成：{len(labels)} 条有效，解析失败率 {fail:.1%} → {out}")
    print(f"标签分布：{labels['label'].value_counts().to_dict()}")
    print(f"例行公告中“中性”占比：{(labels[labels['routine']]['label'] == '中性').mean():.0%}")
    if args.provider == "api":
        print(f"tokens：输入 {tokens_in:,}，输出 {tokens_out:,}；估算成本 ¥{cost:.2f}"
              "（需在 .env 中填写价格）")


if __name__ == "__main__":
    main()
