"""步骤 1.6：数据质量检查，生成报告和图表。

运行：python -m src.data.check_data
输出：data/processed/data_report.md 和 data/processed/price_samples.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import PROCESSED_DIR  # noqa: E402

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def check_prices(lines: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    universe = pd.read_csv(PROCESSED_DIR / "universe.csv", dtype={"code": str})
    lines += [
        "## 行情数据",
        f"- 股票数：{df['code'].nunique()} / 股票池 {len(universe)}",
        f"- 时间范围：{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}，"
        f"共 {df['date'].nunique()} 个交易日，{len(df)} 行",
        f"- 缺失值：{int(df[['open', 'high', 'low', 'close', 'volume']].isna().sum().sum())} 个",
        f"- 重复行（同股票同日期）：{int(df.duplicated(['code', 'date']).sum())}",
        f"- 成交量为 0 的行（多为停牌）：{int((df['volume'] == 0).sum())}",
        f"- 价格异常（最高价 < 最低价 或 收盘价 ≤ 0）："
        f"{int(((df['high'] < df['low']) | (df['close'] <= 0)).sum())}",
        f"- 单日涨跌幅超过 ±21%（可能是数据错误或新股）：{int((df['pct_chg'].abs() > 0.21).sum())}",
    ]
    missing = sorted(set(universe["code"]) - set(df["code"]))
    if missing:
        lines.append(f"- 未下载成功的股票：{missing}")

    # 上市较晚的股票交易日会偏少，这是正常的
    days = df.groupby("code")["date"].count()
    lines.append(f"- 每只股票交易日数：最少 {days.min()}，中位数 {int(days.median())}，最多 {days.max()}")
    return df


def check_events(lines: list[str]) -> None:
    path = PROCESSED_DIR / "events.parquet"
    if not path.exists():
        lines += ["## 公告/新闻数据", "- 尚未生成 events.parquet"]
        return
    ev = pd.read_parquet(path)
    per_stock = ev.groupby("code").size()
    yearly = ev.groupby(ev["trade_date"].dt.year).size()
    lines += [
        "## 公告/新闻数据",
        f"- 总条数：{len(ev)}，覆盖股票 {ev['code'].nunique()} 只",
        f"- 来源分布：{ev['source'].value_counts().to_dict()}",
        f"- 每只股票条数：最少 {per_stock.min()}，中位数 {int(per_stock.median())}，最多 {per_stock.max()}",
        f"- 每年条数：{yearly.to_dict()}",
        f"- 标题为空：{int(ev['title'].isna().sum())}",
        f"- 可交易日早于发布日（应为 0，否则有未来函数）："
        f"{int((ev['trade_date'] < ev['publish_time'].dt.normalize()).sum())}",
    ]


def plot_samples(df: pd.DataFrame, n: int = 4) -> None:
    codes = df["code"].drop_duplicates().head(n)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    for ax, code in zip(axes, codes):
        s = df[df["code"] == code]
        ax.plot(s["date"], s["close"], linewidth=1)
        ax.set_title(code, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(PROCESSED_DIR / "price_samples.png", dpi=120)


def main():
    lines = ["# 阶段 1 数据质量报告", ""]
    prices = check_prices(lines)
    lines.append("")
    check_events(lines)
    lines += ["", "## 已知局限",
              "- 股票池使用当前沪深 300 成分股，回测存在幸存者偏差。",
              "- 历史文本以公告标题为主；新闻需每日采集逐步积累。",
              "- 只有日期没有时间的公告，统一顺延到下一个交易日（保守处理）。"]
    plot_samples(prices)

    report = PROCESSED_DIR / "data_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告 → {report}\n价格图 → {PROCESSED_DIR / 'price_samples.png'}")


if __name__ == "__main__":
    main()
