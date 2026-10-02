"""步骤 2.7：在测试集上评估模型，并生成对比表。

运行示例：
  # 关键词基线
  python -m src.sentiment.evaluate --name keyword --provider keyword
  # 老师模型（大模型 API），用完整规则提示词
  python -m src.sentiment.evaluate --name teacher_api --target teacher
  # 原始 Qwen（零样本），用完整规则提示词，vLLM 中的模型名为基座模型路径
  python -m src.sentiment.evaluate --name qwen3b_base --target student --model Qwen/Qwen2.5-3B-Instruct --prompt teacher
  # 微调后的 Qwen，用简短提示词（与训练时一致），vLLM 中的 LoRA 模块名为 sentiment
  python -m src.sentiment.evaluate --name qwen3b_sft --target student --model sentiment --prompt student
  # 汇总所有结果
  python -m src.sentiment.evaluate --summary

注意：测试集的标准答案是老师模型的标签。所以这里衡量的是“学生学到了老师几成”；
老师与人工的一致率见 review_report.md。
"""
import argparse
import json
import time

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from .config import EVAL_DIR, LABELS, LLM_DIR, SIGN, STUDENT, TEACHER
from .llm_client import make_model, run_batch
from .prompts import PROMPTS

INVALID = "解析失败"


def evaluate(args) -> None:
    gold = pd.read_parquet(LLM_DIR / f"{args.split}.parquet")
    rows = gold.to_dict("records")[: args.limit]
    endpoint = dict(TEACHER if args.target == "teacher" else STUDENT)
    if args.model:
        endpoint["model"] = args.model
    if args.base_url:
        endpoint["base_url"] = args.base_url
    prompt = args.prompt or ("student" if args.target == "student" else "teacher")
    model = make_model(args.provider, endpoint)

    preds_path = EVAL_DIR / f"preds_{args.name}_{args.split}.jsonl"
    start = time.perf_counter()
    preds = run_batch(rows, PROMPTS[prompt], model, preds_path,
                      workers=args.workers, resume=False, desc=f"评估 {args.name}")
    wall = time.perf_counter() - start

    df = pd.DataFrame(rows)[["id", "label", "score"]].merge(
        preds[["id", "label", "score", "latency", "prompt_tokens", "completion_tokens"]],
        on="id", how="left", suffixes=("", "_pred"))
    df["label_pred"] = df["label_pred"].fillna(INVALID)   # 请求失败或格式错误都算错
    df["score_pred"] = df["score_pred"].fillna(0)

    y_true, y_pred = df["label"], df["label_pred"]
    present = [lab for lab in LABELS if lab in set(y_true)]
    f1_each = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    signed_true = df["label"].map(SIGN) * df["score"]
    signed_pred = df["label_pred"].map(SIGN).fillna(0) * df["score_pred"]
    tokens_in, tokens_out = df["prompt_tokens"].sum(), df["completion_tokens"].sum()
    cost = (tokens_in * endpoint["price_in"] + tokens_out * endpoint["price_out"]) / 1e6

    result = {
        "name": args.name, "split": args.split, "n": len(df),
        "model": endpoint["model"] if args.provider == "api" else "keyword", "prompt": prompt,
        "accuracy": float((y_true == y_pred).mean()),
        # 宏 F1 只对测试集中实际出现的类别求平均，否则缺失的类别会被算作 0 分
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        **{f"f1_{lab}": float(v) for lab, v in zip(LABELS, f1_each)},
        "parse_fail": float((y_pred == INVALID).mean()),
        "signed_score_spearman": float(signed_true.corr(signed_pred, method="spearman")),
        "avg_latency_s": float(df["latency"].mean()),
        "throughput_per_s": len(df) / wall if wall > 0 else None,
        "cost_per_1k_yuan": cost / len(df) * 1000,
    }
    (EVAL_DIR / f"{args.name}_{args.split}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0, digits=3))
    print("混淆矩阵（行 = 标准答案，列 = 预测）：")
    print(pd.crosstab(y_true, y_pred))
    print(f"\n准确率 {result['accuracy']:.3f}｜宏 F1 {result['macro_f1']:.3f}｜"
          f"解析失败 {result['parse_fail']:.1%}｜带符号强度 Spearman {result['signed_score_spearman']:.3f}")
    print(f"结果 → {EVAL_DIR / f'{args.name}_{args.split}.json'}，逐条预测 → {preds_path}")


def summary() -> None:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(EVAL_DIR.glob("*.json"))]
    if not results:
        raise SystemExit("还没有评估结果")
    header = "| 模型 | 数据集 | 条数 | 准确率 | 宏 F1 | 利好 F1 | 利空 F1 | 解析失败 | 平均延迟(秒) | 吞吐(条/秒) | 每千条成本(元) |"
    lines = [header, "|" + "---|" * 11]
    for r in sorted(results, key=lambda r: (r["split"], -r["macro_f1"])):
        tp = f"{r['throughput_per_s']:.1f}" if r.get("throughput_per_s") else "-"
        lines.append(
            f"| {r['name']} | {r['split']} | {r['n']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
            f"{r['f1_利好']:.3f} | {r['f1_利空']:.3f} | {r['parse_fail']:.1%} | "
            f"{r['avg_latency_s']:.2f} | {tp} | {r['cost_per_1k_yuan']:.2f} |")
    table = "\n".join(lines)
    (EVAL_DIR / "summary.md").write_text("# 情绪模型评估对比\n\n" + table + "\n", encoding="utf-8")
    print(table)
    print(f"\n→ {EVAL_DIR / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="这次评估的名字，如 qwen3b_sft")
    parser.add_argument("--provider", choices=["api", "keyword"], default="api")
    parser.add_argument("--target", choices=["teacher", "student"], default="student",
                        help="使用 .env 中哪个接口配置")
    parser.add_argument("--model", help="覆盖 .env 中的模型名")
    parser.add_argument("--base-url", help="覆盖 .env 中的接口地址")
    parser.add_argument("--prompt", choices=["teacher", "student"],
                        help="提示词：teacher = 完整规则，student = 简短指令（微调模型用）")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--summary", action="store_true", help="汇总所有评估结果")
    args = parser.parse_args()

    if args.summary:
        summary()
    elif args.name:
        evaluate(args)
    else:
        parser.error("需要 --name 或 --summary")


if __name__ == "__main__":
    main()
