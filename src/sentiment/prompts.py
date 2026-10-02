"""步骤 2.1：标注规则与提示词。

- 老师提示词（teacher）：完整规则 + 示例，给强模型 API 标注数据用，也用来测试原始 Qwen 的零样本能力。
- 学生提示词（student）：一句简短指令，用于微调和微调后推理。规则由模型从训练数据中学会，
  所以推理时不需要长提示词，速度更快、成本更低。
"""
import json
import re

from .config import LABELS

TEACHER_SYSTEM = """你是一名资深 A 股研究员。任务：判断一条上市公司公告或新闻，对该公司股价短期（未来 1–2 周）的影响。

判断规则：
1. 利好：可能推动股价上涨的信息。例如业绩预增或扭亏、大额订单或中标、股东或高管增持、股份回购、重要产品获批、超预期分红。
2. 利空：可能导致股价下跌的信息。例如业绩预减或预亏、股东减持、立案调查或行政处罚、重大诉讼、债务违约、评级下调、重要项目终止。
3. 中性：例行程序性信息，例如董事会或股东大会的通知与决议、独立董事述职、法律意见书、债券付息、章程修订、月报表；以及影响方向不明确的信息。
4. 只根据给出的文字判断，不要猜测没有提到的内容。拿不准时选“中性”。

强度 score（1–5）：
1 = 几乎没有影响（“中性”一律为 1）；2 = 轻微；3 = 中等；4 = 较大；5 = 重大（如业绩大幅变脸、立案调查、重大资产重组）。

示例：
标题：关于签订 20 亿元重大销售合同的公告 → {"label": "利好", "score": 4, "reason": "大额合同显著增加未来收入"}
标题：关于召开 2024 年第一次临时股东大会的通知 → {"label": "中性", "score": 1, "reason": "例行会议通知"}
标题：关于公司实际控制人被立案调查的公告 → {"label": "利空", "score": 5, "reason": "实控人被立案，治理风险大"}
标题：关于持股 5% 以上股东减持股份计划的预披露公告 → {"label": "利空", "score": 3, "reason": "大股东减持带来抛压"}

只输出一行 JSON，不要输出任何其他内容：
{"label": "利好/中性/利空", "score": 1-5, "reason": "不超过 30 字的理由"}"""

STUDENT_INSTRUCTION = (
    '判断下面这条信息对该公司股价的短期影响。只输出 JSON：'
    '{"label": "利好/中性/利空", "score": 1-5, "reason": "简短理由"}'
)

SOURCE_NAMES = {"cninfo": "公司公告", "eastmoney_news": "财经新闻"}


def format_event(row: dict) -> str:
    """把一条文本整理成模型输入。"""
    lines = [
        f"公司：{row.get('name') or ''}（{row['code']}）",
        f"来源：{SOURCE_NAMES.get(row.get('source'), row.get('source', ''))}",
        f"标题：{row['title']}",
    ]
    content = (row.get("content") or "").strip()
    if content:
        lines.append(f"内容：{content[:500]}")
    return "\n".join(lines)


def teacher_messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": format_event(row)}]


def student_messages(row: dict) -> list[dict]:
    # 与 LLaMA-Factory 的 alpaca 格式一致：instruction 和 input 用换行拼接成一条用户消息，
    # 保证推理时的输入和训练时完全相同
    return [{"role": "user", "content": STUDENT_INSTRUCTION + "\n" + format_event(row)}]


PROMPTS = {"teacher": teacher_messages, "student": student_messages}


def target_output(label: str, score: int, reason: str) -> str:
    """训练目标（模型应该输出的文本）。"""
    return json.dumps({"label": label, "score": int(score), "reason": reason}, ensure_ascii=False)


def parse_output(text: str | None) -> dict | None:
    """从模型输出中解析 JSON；格式不对就返回 None（计为“解析失败”）。"""
    if not text:
        return None
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        label = obj["label"]
        score = int(obj["score"])
    except (ValueError, KeyError, TypeError):
        return None
    if label not in LABELS:
        return None
    score = 1 if label == "中性" else min(max(score, 1), 5)
    return {"label": label, "score": score, "reason": str(obj.get("reason", ""))[:60]}
