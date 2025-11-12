# src/eventing/event_normalizer.py
# 목적:
# - data/raw/rss_*.csv (가장 최근 파일 1개) 정규화
# - 중복 제거(id/link 기준)
# - 간단한 이벤트 타입 분류(정책/유명인/콜라보/산업/기타)
# - salience/novelty의 1차 지표(가벼운 룰 기반)
# - 결과를 data/processed/events.parquet 로 저장

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import re
import hashlib
from datetime import datetime
import pandas as pd

from src.common.logging import setup_logger
from src.common.config import load_config
from src.common.constants import RAW_DIR, PROCESSED_DIR, EVENTS_FILE

# ──────────────────────────────────────────────────────────────────────────────
# 1) 유틸
# ──────────────────────────────────────────────────────────────────────────────
def _latest_raw_csv() -> Path | None:
    files = sorted(RAW_DIR.glob("rss_*.csv"))
    return files[-1] if files else None

def _gen_event_id(row: Dict[str, str]) -> str:
    # id가 있으면 우선 사용하고, 없으면 title+link로 해시 생성
    base = row.get("id") or f"{row.get('title','')}|{row.get('link','')}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]

def _parse_ts(ts: str) -> pd.Timestamp | None:
    # RSS published 문자열을 pandas에서 최대한 파싱
    ts = ts.replace(" KST", "+09:00")
    if not ts:
        return None
    try:
        return pd.to_datetime(ts, utc=True, errors="coerce")
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# 2) 간단한 이벤트 타입 분류 룰
#    - 가벼운 사전 + 키워드 매칭(POC 1차)
#    - params.yaml에 override 키워드가 있으면 합칩니다(없으면 기본만)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_RULES = {
    "policy": [
        "정책", "규제", "완화", "정부", "보도자료", "금융위원회", "공정거래위원회", "관세", "관보", "지원 확대"
    ],
    "celebrity_comment": [
        "발언", "코멘트", "견해", "트윗", "인터뷰", "유명", "인플루언서", "거물", "빌리어네어", "대주주"
    ],
    "collaboration": [
        "콜라보", "협업", "제휴", "파트너십", "MOU", "공동", "협력", "공동개발"
    ],
    "industry": [
        "산업", "업계", "공급망", "수출", "수입", "생산", "라인", "증설", "감산", "출시", "신제품"
    ]
}

def _merge_rules(cfg_rules: Dict) -> Dict:
    rules = {k: set(v) for k, v in DEFAULT_RULES.items()}
    if cfg_rules:
        for k, arr in cfg_rules.items():
            rules.setdefault(k, set()).update(arr or [])
    return {k: sorted(v) for k, v in rules.items()}

def _classify_event_type(title: str, summary: str, rules: Dict[str, List[str]]) -> str:
    text = f"{title} {summary}"
    text = text.lower()
    for etype, kws in rules.items():
        for kw in kws:
            if kw.lower() in text:
                return etype
    return "other"

# ──────────────────────────────────────────────────────────────────────────────
# 3) 간단 특징치: salience / novelty
#    - salience: 제목·요약에서 '강조 키워드' 출현 수를 간단 가중치로 환산 (0~1 스케일)
#    - novelty: 동일 제목·링크가 이미 존재하면 0, 아니면 1
#      (향후: 유사도 기반으로 확장 가능)
# ──────────────────────────────────────────────────────────────────────────────
EMPHASIS_KWS = [
    "긴급", "속보", "전격", "대규모", "확대", "중단", "폐지", "인하", "인상",
    "완화", "강화", "수혜", "악재", "호재", "수주", "계약", "출시"
]

def _salience_score(title: str, summary: str) -> float:
    text = f"{title} {summary}"
    count = 0
    for kw in EMPHASIS_KWS:
        count += text.count(kw)
    # 단순 스케일(상한 1.0)
    return min(count / 3.0, 1.0)

def _novelty_flag(df: pd.DataFrame) -> pd.Series:
    # 동일 제목 또는 동일 링크가 있으면 0, 아니면 1
    dup_title = df["title"].duplicated(keep="first")
    dup_link = df["link"].duplicated(keep="first")
    nov = (~(dup_title | dup_link)).astype(int)
    return nov

# ──────────────────────────────────────────────────────────────────────────────
# 4) 메인
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    logger = setup_logger("eventing")
    cfg = load_config()

    latest = _latest_raw_csv()
    if latest is None:
        logger.warning("data/raw/rss_*.csv 가 없습니다. 먼저 RSS 수집을 실행하세요.")
        return

    logger.info(f"load raw: {latest.name}")
    df = pd.read_csv(latest)

    # 필드 보정(비어 있으면 빈 문자열)
    for col in ["id", "title", "link", "published", "summary", "source", "feed_name"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # published 파싱
    df["published_ts"] = df["published"].apply(_parse_ts)

    # 중복 제거 (id나 link 기준)
    df["_key"] = df["id"].where(df["id"].str.len() > 0, df["link"])
    before = len(df)
    df = df.drop_duplicates(subset=["_key"], keep="first").copy()
    after = len(df)
    if before != after:
        logger.info(f"dedup by id/link: {before} -> {after}")

    # 이벤트 id 생성
    df["event_id"] = df.apply(lambda r: _gen_event_id(r.to_dict()), axis=1)

    # 이벤트 유형 룰 머지(params.yaml > rules.event_types.*)
    cfg_rules = (((cfg.params or {}).get("rules") or {}).get("event_types")) if cfg and cfg.params else None
    rules = _merge_rules(cfg_rules)

    # 타입 분류
    df["event_type"] = df.apply(
        lambda r: _classify_event_type(r["title"], r["summary"], rules), axis=1
    )

    # 특징치 1차
    df["salience"] = df.apply(lambda r: _salience_score(r["title"], r["summary"]), axis=1)
    df["novelty"] = _novelty_flag(df)

    # 컬럼 정리(표준 스키마)
    cols = [
        "event_id", "published_ts", "event_type",
        "title", "summary", "link", "source", "feed_name",
        "salience", "novelty"
    ]
    out = df[cols].sort_values("published_ts", na_position="last").reset_index(drop=True)

    # 저장
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(EVENTS_FILE, index=False)
    logger.info(f"events -> {EVENTS_FILE} (rows={len(out)})")

if __name__ == "__main__":
    run()
