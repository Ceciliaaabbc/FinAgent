"""步骤 2.8（加分项）：构造 DPO 偏好数据。

思路：让微调后的模型在训练集上预测一遍，挑出它答错的样本：
  chosen   = 老师模型的正确答案
  rejected = 微调模型自己的错误输出
DPO 训练会让模型更倾向 chosen、远离 rejected，专门纠正它“自己常犯的错”。

运行：
  # 1. 先用微调模型预测训练集（会生成 outputs/eval/preds_qwen3b_sft_train.jsonl）
  python -m src.sentiment.evaluate --name qwen3b_sft --target student --model sentiment --split train
  # 2. 构造偏好数据
  python -m src.sentiment.build_dpo --preds outputs/eval/preds_qwen3b_sft_train.jsonl
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .build_dataset import update_dataset_info
from .config import LLM_DIR
from .prompts import STUDENT_INSTRUCTION, format_event, target_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="微调模型在训练集上的预测文件（jsonl）")
    parser.add_argument("--max-pairs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gold = pd.read_parquet(LLM_DIR / "train.parquet")
    preds = pd.read_json(Path(args.preds), lines=True)[["id", "raw", "label"]]
    df = gold.merge(preds, on="id", suffixes=("", "_pred"))

    wrong = df[(df["label_pred"] != df["label"]) & df["raw"].str.strip().astype(bool)]
    wrong = wrong.sample(min(len(wrong), args.max_pairs), random_state=args.seed)
    pairs = [{"instruction": STUDENT_INSTRUCTION,
              "input": format_event(row),
              "chosen": target_output(row["label"], row["score"], row["reason"]),
              "rejected": row["raw"].strip()}
             for row in wrong.to_dict("records")]

    out = LLM_DIR / "dpo_train.json"
    out.write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    update_dataset_info({"finagent_sentiment_dpo": {
        "file_name": "dpo_train.json", "ranking": True,
        "columns": {"prompt": "instruction", "query": "input", "chosen": "chosen", "rejected": "rejected"},
    }})
    print(f"训练集 {len(df)} 条中模型答错 {len(df[df['label_pred'] != df['label']])} 条，"
          f"生成偏好对 {len(pairs)} 条 → {out}")
    if len(pairs) < 200:
        print("偏好对太少（< 200），DPO 意义不大，可以跳过这一步")


if __name__ == "__main__":
    main()
