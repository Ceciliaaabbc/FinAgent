"""一键运行阶段 1 全部步骤。

运行：python -m src.data.run_stage1             # 全量（首次约 2–3 小时，主要是公告）
      python -m src.data.run_stage1 --limit 5   # 只用前 5 只股票快速测试
"""
import subprocess
import sys

STEPS = [
    ("1.1 股票池", "src.data.fetch_universe", False),
    ("1.3 基准指数", "src.data.fetch_index", False),
    ("1.2 日线行情", "src.data.fetch_prices", True),
    ("1.4 历史公告", "src.data.fetch_announcements", True),
    ("1.4 最新新闻", "src.data.fetch_news", True),
    ("1.5 时间对齐", "src.data.align_events", False),
    ("1.6 质量检查", "src.data.check_data", False),
]


def main():
    extra = sys.argv[1:]
    for name, module, accepts_limit in STEPS:
        print(f"\n===== 步骤 {name} =====")
        cmd = [sys.executable, "-m", module] + (extra if accepts_limit else [])
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
