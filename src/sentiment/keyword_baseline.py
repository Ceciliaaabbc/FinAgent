"""关键词词典基线：不用任何模型，只靠“利好词/利空词”计分。

两个用途：
1. 评估时的传统方法基线，用来证明大模型确实更好；
2. 没有 API Key 或 GPU 时，用它跑通整个流程（--provider keyword）。
"""
import json
import re

POSITIVE = {
    "预增": 3, "扭亏": 4, "大幅增长": 3, "增长": 1, "中标": 3, "重大合同": 3, "签订": 1,
    "增持": 3, "回购": 2, "获批": 2, "注册证": 2, "批准": 1, "分红": 1, "利润分配": 1,
    "上调": 2, "突破": 1,
}
NEGATIVE = {
    "预减": 3, "预亏": 4, "亏损": 3, "下降": 1, "减持": 3, "立案": 5, "处罚": 4, "警示函": 3,
    "诉讼": 2, "仲裁": 2, "冻结": 3, "质押": 1, "违约": 4, "终止": 2, "退市": 5,
    "风险提示": 2, "下调": 2, "问询函": 1, "失信": 4, "被执行": 3,
}


def keyword_sentiment(text: str) -> dict:
    pos = sum(w for k, w in POSITIVE.items() if k in text)
    neg = sum(w for k, w in NEGATIVE.items() if k in text)
    net = pos - neg
    if abs(net) < 2:
        return {"label": "中性", "score": 1, "reason": "无明显利好/利空关键词"}
    label = "利好" if net > 0 else "利空"
    return {"label": label, "score": min(abs(net), 5), "reason": f"关键词净得分 {net}"}


class KeywordModel:
    """与 ChatModel 接口相同：输入 messages，输出包含 raw 文本的字典。"""

    def __call__(self, messages: list[dict]) -> dict:
        text = messages[-1]["content"]
        match = re.search(r"标题：.*", text, re.S)  # 只看标题和内容，忽略公司名
        result = keyword_sentiment(match.group(0) if match else text)
        return {"raw": json.dumps(result, ensure_ascii=False), "latency": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0}
