"""步骤 2.3：人工抽检老师模型的标注质量。

第一步，导出抽检表：
    python -m src.sentiment.review --export --n 200
用 Excel 或 WPS 打开 data/llm/sentiment/review.csv，在 human_label 列填写你的判断（利好/中性/利空），
有分歧的可以在 human_note 列写原因。保存时保持 CSV 格式。

第二步，计算一致率：
    python -m src.sentiment.review --score

为什么要做：用数字说明训练数据的质量（如“老师模型标注与人工一致率 91%”）。
这个一致率也是学生模型能达到的大致上限，因为学生是向老师学的。
"""
import argparse

import pandas as pd

from .config import LABELS, LLM_DIR

REVIEW_PATH = LLM_DIR / "review.csv"


def export(n: int, seed: int, force: bool) -> None:
    if REVIEW_PATH.exists() and not force:
        raise SystemExit(f"{REVIEW_PATH} 已存在，可能已有你的人工标注。确认要覆盖请加 --force")
    labels = pd.read_parquet(LLM_DIR / "labels.parquet")
    sample = labels.sample(min(n, len(labels)), random_state=seed)
    sheet = pd.DataFrame({
        "id": sample["id"], "code": sample["code"], "name": sample["name"],
        "title": sample["title"], "content": sample["content"].str[:200],
        "llm_label": sample["label"], "llm_score": sample["score"], "llm_reason": sample["reason"],
        "human_label": "", "human_note": "",
    })
    sheet.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")  # utf-8-sig：Excel 打开不乱码
    print(f"已导出 {len(sheet)} 条 → {REVIEW_PATH}\n请在 human_label 列填写：{' / '.join(LABELS)}")


def score() -> None:
    sheet = pd.read_csv(REVIEW_PATH, dtype=str).fillna("")
    sheet["human_label"] = sheet["human_label"].str.strip()
    done = sheet[sheet["human_label"].isin(LABELS)]
    if done.empty:
        raise SystemExit("还没有填写 human_label")

    agree = (done["llm_label"] == done["human_label"]).mean()
    matrix = pd.crosstab(done["human_label"], done["llm_label"],
                         rownames=["人工"], colnames=["老师模型"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
    diff = done[done["llm_label"] != done["human_label"]]

    lines = [
        "# 老师模型标注质量抽检", "",
        f"- 已抽检：{len(done)} / {len(sheet)} 条",
        f"- 一致率：**{agree:.1%}**", "",
        "## 混淆矩阵（行 = 人工，列 = 老师模型）", "", "```", matrix.to_string(), "```", "",
        "## 不一致的样本", "",
    ]
    lines += [f"- {r.title}｜模型：{r.llm_label}｜人工：{r.human_label}｜{r.human_note}"
              for r in diff.itertuples()]
    report = LLM_DIR / "review_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print(matrix)
    print(f"\n完整报告 → {report}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="导出抽检表")
    parser.add_argument("--score", action="store_true", help="计算一致率")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="覆盖已有的抽检表")
    args = parser.parse_args()
    if args.export:
        export(args.n, args.seed, args.force)
    elif args.score:
        score()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
