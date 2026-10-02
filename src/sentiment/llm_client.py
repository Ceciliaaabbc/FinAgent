"""调用大模型的通用工具：OpenAI 兼容客户端 + 并发批处理（带断点续传）。

DeepSeek、通义千问等 API，以及 vLLM 部署的本地模型，都提供 OpenAI 兼容接口，
所以同一套代码既能调用老师模型，也能调用学生模型，只需换 base_url 和 model。
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from .keyword_baseline import KeywordModel
from .prompts import parse_output


class ChatModel:
    def __init__(self, base_url: str, api_key: str, model: str,
                 max_tokens: int = 120, temperature: float = 0.0, **_):
        if not api_key:
            raise SystemExit("缺少 API Key：请复制 .env.example 为 .env 并填写")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=3)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature  # 0 = 输出尽量确定，便于复现

    def __call__(self, messages: list[dict]) -> dict:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, max_tokens=self.max_tokens)
        usage = resp.usage
        return {
            "raw": resp.choices[0].message.content or "",
            "latency": time.perf_counter() - start,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }


def make_model(provider: str, endpoint: dict | None = None):
    """provider = "api"（调用 endpoint 指定的接口）或 "keyword"（关键词基线）。"""
    if provider == "keyword":
        return KeywordModel()
    return ChatModel(**endpoint)


def run_batch(rows: list[dict], build_messages: Callable[[dict], list[dict]], model,
              cache_path: Path, workers: int = 8, resume: bool = True, desc: str = "") -> pd.DataFrame:
    """并发处理 rows（每行必须有 id），结果逐条追加写入 cache_path（jsonl）。

    resume=True 时跳过已处理的 id，中断后重新运行即可续传；请求出错的行不会写入，下次自动重试。
    返回 DataFrame：id, raw, label, score, reason, latency, prompt_tokens, completion_tokens
    """
    if not resume and cache_path.exists():
        cache_path.unlink()
    done: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["id"]] = rec

    def work(row: dict) -> dict:
        try:
            out = model(build_messages(row))
        except Exception as e:  # noqa: BLE001 - 网络、限流、超时等都在这里统一处理
            return {"id": row["id"], "error": str(e)}
        out["id"] = row["id"]
        out.update(parse_output(out["raw"]) or {"label": None, "score": None, "reason": None})
        return out

    todo = [r for r in rows if r["id"] not in done]
    errors = []
    with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(work, r) for r in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc or "调用模型"):
            rec = fut.result()
            if "error" in rec:
                errors.append(rec["error"])
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[rec["id"]] = rec

    if errors:
        print(f"  {len(errors)} 条请求出错（重新运行会自动重试），示例：{errors[0][:200]}")
    return pd.DataFrame([done[r["id"]] for r in rows if r["id"] in done])
