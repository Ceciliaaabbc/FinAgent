"""阶段 2 的配置：路径、标签定义、模型接口（从项目根目录的 .env 读取）。"""
import hashlib
import os

from dotenv import load_dotenv

from ..data.config import PROCESSED_DIR, ROOT

load_dotenv(ROOT / ".env")

LLM_DIR = ROOT / "data" / "llm" / "sentiment"   # 标注数据、微调数据集
OUTPUT_DIR = ROOT / "outputs"                    # 模型权重、评估结果
EVAL_DIR = OUTPUT_DIR / "eval"
EVENTS_PATH = PROCESSED_DIR / "events.parquet"
UNIVERSE_PATH = PROCESSED_DIR / "universe.csv"
SCORES_PATH = PROCESSED_DIR / "sentiment_scores.parquet"

LABELS = ["利好", "中性", "利空"]
SIGN = {"利好": 1, "中性": 0, "利空": -1}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# “老师模型”：强模型 API，用来标注训练数据（任意 OpenAI 兼容接口）
TEACHER = {
    "base_url": _env("TEACHER_BASE_URL", "https://api.deepseek.com"),
    "api_key": _env("TEACHER_API_KEY"),
    "model": _env("TEACHER_MODEL", "deepseek-chat"),
    "price_in": float(_env("TEACHER_PRICE_IN", "0")),    # 元 / 百万输入 tokens
    "price_out": float(_env("TEACHER_PRICE_OUT", "0")),  # 元 / 百万输出 tokens
    # 推理模型（如 Qwen3.5）默认先“思考”再回答，又慢又会耗尽输出长度；填 none 可关闭思考
    "reasoning_effort": _env("TEACHER_REASONING_EFFORT"),
}

# “学生模型”：用 vLLM 部署的 Qwen（原始或微调后），vLLM 提供 OpenAI 兼容接口
STUDENT = {
    "base_url": _env("STUDENT_BASE_URL", "http://localhost:8000/v1"),
    "api_key": _env("STUDENT_API_KEY", "EMPTY"),
    "model": _env("STUDENT_MODEL", "sentiment"),
    "reasoning_effort": _env("STUDENT_REASONING_EFFORT"),
    "price_in": 0.0,
    "price_out": 0.0,
}

# 例行程序性公告的标题关键词：采样时控制它们的比例，避免标注预算浪费在大量“中性”上
ROUTINE_PATTERNS = [
    "独立董事", "法律意见", "股东大会", "董事会决议", "监事会决议", "述职报告", "付息", "兑付",
    "募集说明书", "信用评级报告", "月报表", "章程", "议事规则", "会议资料", "会议文件", "制度",
    "候选人", "声明与承诺", "核查意见", "保荐", "持续督导", "更正", "摘要",
]


def event_id(code: str, trade_date, title: str) -> str:
    """每条文本的唯一 ID，用于断点续传和跨文件关联。"""
    key = f"{code}|{str(trade_date)[:10]}|{title}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


for d in (LLM_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)
