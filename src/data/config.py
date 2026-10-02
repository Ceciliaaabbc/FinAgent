"""阶段 1 的全局配置：路径、时间范围、股票池。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # 原始下载（按股票/按日期分文件，支持断点续传）
PROCESSED_DIR = DATA_DIR / "processed"  # 合并清洗后的数据

START_DATE = "20200101"
END_DATE = "20260930"

INDEX_CODE = "000300"        # 沪深 300（中证指数代码，用于获取成分股）
INDEX_SYMBOL = "sh000300"    # 沪深 300（新浪代码，用于获取指数行情）

# A 股收盘时间：此时间之后发布的信息只能在下一个交易日使用
MARKET_CLOSE = "15:00:00"

REQUEST_SLEEP = 0.5  # 每次请求间隔（秒），避免被限流

for d in (RAW_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)
