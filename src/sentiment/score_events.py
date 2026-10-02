"""步骤 2.9：用部署好的模型给全部文本打分，输出给阶段 3 计算情绪因子。

运行：
  python -m src.sentiment.score_events                      # 用 .env 中的学生模型（vLLM 部署的微调 Qwen）
  python -m src.sentiment.score_events --provider keyword   # 还没训练好模型时，先用关键词基线顶上
  python -m src.sentiment.score_events --since 2026-01-01   # 只给新数据打分

支持断点续传；每个模型的结果分开缓存，换模型不会混在一起。
输出：data/processed/sentiment_scores.parquet
字段：id, code, trade_date, title, label, score, signed_score（利好为正、利空为负、中性为 0）, reason, scorer
"""
import argparse
import re

from .config import LLM_DIR, SCORES_PATH, SIGN, STUDENT
from .events import load_events
from .llm_client import make_model, run_batch
from .prompts import PROMPTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["api", "keyword"], default="api")
    parser.add_argument("--model", help="覆盖 .env 中的学生模型名")
    parser.add_argument("--prompt", choices=["teacher", "student"], default="student")
    parser.add_argument("--workers", type=int, default=32, help="vLLM 能同时处理很多请求，可以开大")
    parser.add_argument("--since", help="只处理该日期及之后的文本，如 2026-01-01")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    events = load_events()
    if args.since:
        events = events[events["trade_date"] >= args.since]
    rows = events.to_dict("records")[: args.limit]

    endpoint = dict(STUDENT, **({"model": args.model} if args.model else {}))
    model = make_model(args.provider, endpoint)
    scorer = endpoint["model"] if args.provider == "api" else "keyword"
    cache = LLM_DIR / f"scores_raw_{re.sub(r'[^0-9A-Za-z._-]', '_', scorer)}.jsonl"

    result = run_batch(rows, PROMPTS[args.prompt], model, cache, workers=args.workers, desc="打分")
    valid = result[result["label"].notna()]
    scores = events.merge(valid[["id", "label", "score", "reason"]], on="id")
    scores["signed_score"] = scores["label"].map(SIGN) * scores["score"]
    scores["scorer"] = scorer
    scores = scores[["id", "code", "trade_date", "title", "label", "score", "signed_score", "reason", "scorer"]]
    scores = scores.sort_values(["code", "trade_date"]).reset_index(drop=True)
    scores.to_parquet(SCORES_PATH, index=False)

    print(f"打分完成：{len(scores)} / {len(rows)} 条（解析失败 {len(result) - len(valid)} 条）→ {SCORES_PATH}")
    print(f"标签分布：{scores['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
