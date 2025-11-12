# src/mapping/entity_linker.py
# 목적:
# - events.parquet 과 configs/sources.yaml의 aliases 사전을 사용해
#   제목/요약 내 키워드 매칭 기반의 이벤트↔티커 매핑 생성
# - 룰기반 relevance_score 산출
# - 결과를 data/processed/mapping.parquet 로 저장
#
# 설계 포인트:
# - 한국어/영문을 모두 소문자 비교 (간단 정규화)
# - names, brands 모두 매칭 대상
# - title/summary 내 등장 여부, salience를 가중합
# - features를 JSON 문자열로 함께 저장(사후 분석/튜닝용)

from __future__ import annotations
import json
from typing import Dict, List, Any
import re
import pandas as pd

from src.common.logging import setup_logger
from src.common.config import load_config
from src.common.constants import EVENTS_FILE, PROCESSED_DIR, MAPPING_FILE

# ──────────────────────────────────────────────────────────────────────────────
# 텍스트 정규화(가벼운 버전)
# ──────────────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    # 필요 시 기초 치환(특수문자 단순화)
    s = re.sub(r"\s+", " ", s)
    return s

def _contains_any(text: str, keys: List[str]) -> int:
    """keys 중 하나라도 부분문자열로 존재하면 1, 없으면 0"""
    text = _norm(text)
    for k in keys:
        kk = _norm(k)
        if kk and kk in text:
            return 1
    return 0

def _count_in(text: str, keys: List[str]) -> int:
    """keys의 총 출현 횟수(대략치)"""
    text = _norm(text)
    cnt = 0
    for k in keys:
        kk = _norm(k)
        if not kk:
            continue
        cnt += text.count(kk)
    return cnt

# ──────────────────────────────────────────────────────────────────────────────
# 스코어 계산
# ──────────────────────────────────────────────────────────────────────────────
def _get_rel_weights(cfg) -> Dict[str, float]:
    # params.yaml → weights.relevance.*
    w = (((cfg.params or {}).get("weights") or {}).get("relevance") or {})
    return {
        "name_hit": float(w.get("name_hit", 1.0)),
        "brand_hit": float(w.get("brand_hit", 0.8)),
        "title_keyword": float(w.get("title_keyword", 0.5)),
        "summary_keyword": float(w.get("summary_keyword", 0.2)),
        "salience_boost": float(w.get("salience_boost", 0.2)),
    }

def _score_row(title: str, summary: str, salience: float,
               names: List[str], brands: List[str], w: Dict[str, float]) -> Dict[str, Any]:
    # 기본 히트
    name_hit_title = _contains_any(title, names)
    name_hit_summary = _contains_any(summary, names)
    brand_hit_title = _contains_any(title, brands)
    brand_hit_summary = _contains_any(summary, brands)

    # 키워드 카운트(다수 등장 시 가중)
    name_count_title = _count_in(title, names)
    brand_count_title = _count_in(title, brands)
    name_count_summary = _count_in(summary, names)
    brand_count_summary = _count_in(summary, brands)

    # 스코어: 히트 여부와 카운트를 모두 반영
    # (간단 가중합; 필요 시 min/max로 클리핑)
    score = 0.0
    score += w["name_hit"] * (name_hit_title or name_hit_summary)
    score += w["brand_hit"] * (brand_hit_title or brand_hit_summary)
    score += w["title_keyword"] * (name_count_title + brand_count_title)
    score += w["summary_keyword"] * (name_count_summary + brand_count_summary)

    # salience 보너스(이벤트 자체 중요도)
    score += w["salience_boost"] * float(salience or 0.0)

    # 정규화/상한(선형 상한 5.0 → 0~1 스케일로 예시)
    max_raw = 5.0
    norm_score = min(score / max_raw, 1.0)

    features = {
        "name_hit": int((name_hit_title or name_hit_summary) > 0),
        "brand_hit": int((brand_hit_title or brand_hit_summary) > 0),
        "name_count_title": int(name_count_title),
        "brand_count_title": int(brand_count_title),
        "name_count_summary": int(name_count_summary),
        "brand_count_summary": int(brand_count_summary),
        "salience": float(salience or 0.0),
        "raw_score": float(score),
        "norm_score": float(norm_score),
    }
    return {"score": norm_score, "features": features}

# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    logger = setup_logger("mapping")
    cfg = load_config()

    if not EVENTS_FILE.exists():
        logger.warning("events.parquet 이 없습니다. 먼저 event_normalizer를 실행하세요.")
        return

    df = pd.read_parquet(EVENTS_FILE)
    if df.empty:
        logger.info("events.parquet 이 비어있습니다.")
        return

    aliases = (cfg.sources or {}).get("aliases") or {}
    if not aliases:
        logger.warning("sources.yaml 의 aliases 가 비어 있습니다. 매핑이 생성되지 않습니다.")
        return

    w = _get_rel_weights(cfg)

    # 표준 스키마 컬럼 보장
    for c in ["event_id", "title", "summary", "salience"]:
        if c not in df.columns:
            df[c] = "" if c in ("title", "summary") else 0.0

    rows_out = []
    for _, r in df.iterrows():
        title = r["title"] or ""
        summary = r["summary"] or ""
        salience = float(r.get("salience", 0.0) or 0.0)
        event_id = r["event_id"]

        for tkr, ent in aliases.items():
            names = (ent or {}).get("names") or []
            brands = (ent or {}).get("brands") or []
            if not names and not brands:
                continue
            result = _score_row(title, summary, salience, names, brands, w)
            score = result["score"]
            if score <= 0.0:
                continue  # 완전 0점은 스킵(튜닝 가능)

            rows_out.append({
                "event_id": event_id,
                "ticker": str(tkr),
                "relevance_score": float(score),
                "features": json.dumps(result["features"], ensure_ascii=False),
            })

    if not rows_out:
        logger.info("생성된 매핑 결과가 없습니다. aliases 확장 또는 weights 조정이 필요합니다.")
        return

    out = pd.DataFrame(rows_out).sort_values(
        ["event_id", "relevance_score"], ascending=[True, False]
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MAPPING_FILE, index=False)
    logger.info(f"mapping -> {MAPPING_FILE} (rows={len(out)}, events={out['event_id'].nunique()})")

if __name__ == "__main__":
    run()
