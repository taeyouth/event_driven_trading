# src/scoring/basic_scoring.py
# 목적:
# - events.parquet 에 polarity(−1~+1), impact_strength(0~1), confidence(0~1), base_signal(−1~+1)을 추가
# - 간단한 룰/가중치 기반 (params.yaml 활용)
#
# 실행:
#   python -m src.scoring.basic_scoring

from __future__ import annotations
from typing import Dict, List
import math
import pandas as pd
from datetime import datetime, timezone

from src.common.logging import setup_logger
from src.common.config import load_config
from src.common.constants import EVENTS_FILE

def _kw_score(text: str, kws: List[str]) -> int:
    if not isinstance(text, str) or not text:
        return 0
    t = text.lower()
    s = 0
    for kw in kws or []:
        if not kw:
            continue
        if kw.lower() in t:
            s += 1
    return s

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _clip11(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))

def _polarity(title: str, summary: str, wpol: Dict) -> float:
    pos = _kw_score(title, wpol.get("positive_kws", [])) + _kw_score(summary, wpol.get("positive_kws", []))
    neg = _kw_score(title, wpol.get("negative_kws", [])) + _kw_score(summary, wpol.get("negative_kws", []))
    # 간단 정규화: (pos - neg) / (pos + neg + ε)
    denom = (pos + neg) if (pos + neg) > 0 else 1.0
    score = (pos - neg) / denom
    return _clip11(score)  # −1 ~ +1

def _impact(row: pd.Series, wimp: Dict) -> float:
    etype = (row.get("event_type") or "other")
    type_w = float(wimp.get(etype, 1.0))
    sal_w = float(wimp.get("salience_weight", 0.5))
    nov_w  = float(wimp.get("novelty_weight", 0.5))
    sal = float(row.get("salience", 0.0) or 0.0)   # 0~1
    nov = float(row.get("novelty", 0.0) or 0.0)    # 0/1
    raw = type_w * (sal_w * sal + nov_w * nov)
    # 0~1로 간단 스케일 (type_w가 1.2 등일 수 있으므로 1.2 기준으로 클립)
    return _clip01(raw / 1.2)

def _minutes_since(ts: pd.Timestamp) -> float:
    if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
        return 1e9
    now = datetime.now(timezone.utc)
    return max(0.0, (now - ts.to_pydatetime()).total_seconds() / 60.0)

def _confidence(row: pd.Series, wconf: Dict) -> float:
    source_trust_tbl = (wconf.get("source_trust") or {})
    default_trust = float(source_trust_tbl.get("default", 0.5))
    # feed_name / source 중 있으면 우선
    feed = (row.get("feed_name") or "").lower()
    src  = (row.get("source") or "").lower()
    trust = float(source_trust_tbl.get(feed, source_trust_tbl.get(src, default_trust)))

    # 시간 감쇠 (신규 이벤트일수록 신뢰↑, 오래될수록 감쇠)
    tdecay = float(wconf.get("time_decay_minutes", 180))
    mins = _minutes_since(row.get("published_ts"))
    time_factor = 1.0 if mins <= 0 else 1.0 / (1.0 + (mins / max(1.0, tdecay)))  # ~ (0,1]

    conf = trust * time_factor
    conf = max(conf, float(wconf.get("min_confidence", 0.2)))
    return _clip01(conf)

def run() -> None:
    logger = setup_logger("scoring")
    cfg = load_config()

    if not EVENTS_FILE.exists():
        logger.warning("events.parquet 이 없습니다. 먼저 event_normalizer를 실행하세요.")
        return

    df = pd.read_parquet(EVENTS_FILE)
    if df.empty:
        logger.info("events.parquet 이 비어있습니다.")
        return

    wpol = ((cfg.params or {}).get("weights") or {}).get("polarity", {}) or {}
    wimp = ((cfg.params or {}).get("weights") or {}).get("impact", {}) or {}
    wconf = ((cfg.params or {}).get("weights") or {}).get("confidence", {}) or {}

    # 결측 컬럼 대비
    for c in ["title", "summary", "event_type", "salience", "novelty", "published_ts", "feed_name", "source"]:
        if c not in df.columns:
            df[c] = "" if c in ("title", "summary", "event_type", "feed_name", "source") else 0.0

    # 계산
    df["polarity"] = df.apply(lambda r: _polarity(r["title"], r["summary"], wpol), axis=1)              # −1 ~ +1
    df["impact_strength"] = df.apply(lambda r: _impact(r, wimp), axis=1)                                # 0 ~ 1
    df["confidence"] = df.apply(lambda r: _confidence(r, wconf), axis=1)                                # 0 ~ 1

    # base_signal: polarity * impact_strength * confidence (−1 ~ +1)
    df["base_signal"] = df["polarity"] * df["impact_strength"] * df["confidence"]
    # 안전상 NaN 처리
    df["base_signal"] = df["base_signal"].fillna(0.0).clip(-1.0, 1.0)

    df.to_parquet(EVENTS_FILE, index=False)
    logger.info(f"events(updated) -> {EVENTS_FILE} | rows={len(df)} | cols={list(df.columns)}")

if __name__ == "__main__":
    run()
