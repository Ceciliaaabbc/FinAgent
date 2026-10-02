"""读取阶段 1 产出的文本事件，补上公司名称和唯一 ID。"""
import pandas as pd

from .config import EVENTS_PATH, UNIVERSE_PATH, event_id


def load_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        raise SystemExit("找不到 events.parquet，请先运行阶段 1：python -m src.data.run_stage1")
    ev = pd.read_parquet(EVENTS_PATH)
    names = pd.read_csv(UNIVERSE_PATH, dtype={"code": str})[["code", "name"]]
    ev = ev.merge(names, on="code", how="left")
    ev["content"] = ev["content"].fillna("")
    ev["id"] = [event_id(c, d, t) for c, d, t in zip(ev["code"], ev["trade_date"], ev["title"])]
    return ev.drop_duplicates("id").reset_index(drop=True)
