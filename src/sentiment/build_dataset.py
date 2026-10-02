"""步骤 2.4 + 2.5：按时间划分数据集，并转换成 LLaMA-Factory 的微调格式。

运行：python -m src.sentiment.build_dataset

- 按时间划分（默认 80% / 10% / 10%）：早期数据训练，后期数据测试。
  随机划分会让模型提前见到“未来”的同类公告，测试分数会虚高。
- 只对训练集下采样“中性”（默认最多 50%）：避免模型偷懒、什么都答“中性”。
  验证集和测试集保持原始分布，这样评估结果才真实。

输出（data/llm/sentiment/）：
- train.json / val.json / test.json：alpaca 格式 {instruction, input, output}
- dataset_info.json：告诉 LLaMA-Factory 数据集在哪里
- train.parquet / val.parquet / test.parquet：带标签的原始数据，评估时使用
"""
import argparse
import json

import pandas as pd

from .config import LLM_DIR
from .prompts import STUDENT_INSTRUCTION, format_event, target_output


def to_alpaca(df: pd.DataFrame) -> list[dict]:
    return [{"instruction": STUDENT_INSTRUCTION,
             "input": format_event(row),
             "output": target_output(row["label"], row["score"], row["reason"])}
            for row in df.to_dict("records")]


def downsample_neutral(df: pd.DataFrame, max_ratio: float, seed: int) -> pd.DataFrame:
    neutral, other = df[df["label"] == "中性"], df[df["label"] != "中性"]
    keep = int(len(other) * max_ratio / (1 - max_ratio))
    if len(neutral) <= keep:
        return df
    print(f"训练集“中性”过多：保留 {keep} / {len(neutral)} 条，训练集从 {len(df)} 条降为 {len(other) + keep} 条")
    return pd.concat([other, neutral.sample(keep, random_state=seed)]).sort_values("trade_date")


def update_dataset_info(entries: dict) -> None:
    """合并写入 dataset_info.json（保留 DPO 等其他已有条目）。"""
    path = LLM_DIR / "dataset_info.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    info.update(entries)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--max-neutral", type=float, default=0.5, help="训练集中“中性”的最大占比")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = pd.read_parquet(LLM_DIR / "labels.parquet").sort_values("trade_date").reset_index(drop=True)
    n = len(labels)
    n_test, n_val = int(n * args.test), int(n * args.val)
    splits = {
        "train": labels.iloc[: n - n_val - n_test],
        "val": labels.iloc[n - n_val - n_test: n - n_test],
        "test": labels.iloc[n - n_test:],
    }
    splits["train"] = downsample_neutral(splits["train"], args.max_neutral, args.seed)

    for name, df in splits.items():
        df.to_parquet(LLM_DIR / f"{name}.parquet", index=False)
        (LLM_DIR / f"{name}.json").write_text(
            json.dumps(to_alpaca(df), ensure_ascii=False, indent=1), encoding="utf-8")
        dist = df["label"].value_counts(normalize=True).round(2).to_dict()
        print(f"{name:5s}：{len(df):5d} 条，"
              f"{df['trade_date'].min():%Y-%m-%d} ~ {df['trade_date'].max():%Y-%m-%d}，标签占比 {dist}")

    update_dataset_info({
        "finagent_sentiment_train": {"file_name": "train.json"},
        "finagent_sentiment_val": {"file_name": "val.json"},
    })
    print(f"\n微调数据已写入 {LLM_DIR}")
    print("样例：")
    print(json.dumps(to_alpaca(splits["train"].head(1))[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
