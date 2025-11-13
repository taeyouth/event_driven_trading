# src/signal/signal_builder.py
# 목적:
# - events.parquet( base_signal, polarity, impact_strength, confidence )
# - rankings.parquet( relevance Top N_R )
# - rankings_volume.parquet( volume Top N_V )
# 세 개를 결합해 최종 buy_score(0~1), decision(매수/관망/제외)을 산출
# 결과는 data/processed/signals.parquet 로 저장
#
# 실행:
#   python -m src.signal.signal_builder

from __future__ import annotations
import json
from typing import Dict
import pandas as pd
import numpy as np

from src.common.logging import setup_logger
from src.common.config import load_config
from src.common.constants import (
    EVENTS_FILE, PROCESSED_DIR, RANKINGS_FILE
)

RANKINGS_VOLUME_FILE = PROCESSED_DIR / "rankings_volume.parquet"
OUT_FILE = PROCESSED_DIR / "signals.parquet"

# ──────────────────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────────────────
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _safe(df: pd.DataFrame, col: str, default):
    if col not in df.columns:
        df[col] = default
    return df

def _norm_rank_score(rank_col: pd.Series, max_n: int) -> pd.Series:
    # 순위(1=최상) → 0~1 점수로 선형 변환: (max_n - rank + 1)/max_n
    s = (max_n - rank_col + 1) / max_n
    return s.clip(lower=0.0, upper=1.0)

def _norm_uplift(u: pd.Series) -> pd.Series:
    # 거래량 uplift는 음수~양수. 양수일수록 +.
    # 간단 정규화: 0~1 스케일로 매핑 (u<=0 -> 0, u>=1 -> 1)
    return np.clip(u, 0.0, 1.0)

def _build_reason(row: pd.Series) -> str:
    pieces = []
    # 이벤트 요약
    if isinstance(row.get("event_type"), str) and row.get("event_type"):
        pieces.append(f"이벤트유형={row['event_type']}")
    if pd.notna(row.get("published_ts")):
        pieces.append(f"발생시각(UTC)={str(row['published_ts'])}")
    # 근거 점수
    pieces.append(f"base={row.get('base_signal_norm'):.2f}")
    pieces.append(f"rel={row.get('rel_score'):.2f}(rank={int(row.get('rank_relevance', 0))})")
    pieces.append(f"vol={row.get('vol_score'):.2f}(uplift={row.get('volume_uplift'):.2f}, z={row.get('abn_volume_z'):.2f})")
    return " | ".join(pieces)

# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    logger = setup_logger("signal")
    cfg = load_config()

    # 가중치/임계값
    wsig = (((cfg.params or {}).get("weights") or {}).get("signal") or {})
    w_base = float(wsig.get("w_base", 0.50))
    w_rel  = float(wsig.get("w_rel",  0.30))
    w_vol  = float(wsig.get("w_vol",  0.20))
    w_mix  = float(wsig.get("w_mix",  0.00))

    th = ((cfg.params or {}).get("thresholds") or {})
    thr_buy   = float(th.get("buy_score_buy",   0.70))
    thr_watch = float(th.get("buy_score_watch", 0.50))
    require_positive_base = bool(th.get("require_positive_base", True))

    N_R = int((((cfg.params or {}).get("topn") or {}).get("relevance", 10)))
    N_V = int((((cfg.params or {}).get("topn") or {}).get("volume", 10)))

    # 파일 로드
    if not EVENTS_FILE.exists():
        logger.warning("events.parquet 이 없습니다.")
        return
    if not RANKINGS_FILE.exists():
        logger.warning("rankings.parquet(관련도 TopN_R) 이 없습니다.")
        return
    if not RANKINGS_VOLUME_FILE.exists():
        logger.warning("rankings_volume.parquet(거래량 TopN_V) 이 없습니다.")
        return

    ev  = pd.read_parquet(EVENTS_FILE)
    rel = pd.read_parquet(RANKINGS_FILE)
    vol = pd.read_parquet(RANKINGS_VOLUME_FILE)

    if ev.empty or (rel.empty and vol.empty):
        logger.warning("입력 데이터가 비어 있습니다.")
        return

    # 필수 컬럼 보정
    for c, d in [("event_id",""), ("ticker",""), ("base_signal",0.0),
                 ("published_ts", pd.NaT), ("event_type","")]:
        ev = _safe(ev, c, d)

    rel = _safe(rel, "rank_relevance", np.nan)
    rel = _safe(rel, "relevance_score", 0.0)

    # volume 파일 필수 컬럼
    for c, d in [("rank_volume", np.nan), ("volume_uplift", 0.0), ("abn_volume_z", 0.0)]:
        vol = _safe(vol, c, d)

    # 결합 키: event_id + ticker (두 축 어디에 있든 outer 방식으로 결합)
    # 1) 관련도 축
    rel_k = rel[["event_id","ticker","rank_relevance","relevance_score"]].drop_duplicates()
    # 2) 거래량 축
    vol_k = vol[["event_id","ticker","rank_volume","volume_uplift","abn_volume_z"]].drop_duplicates()
    # 3) 이벤트 메타
    ev_k  = ev[["event_id","published_ts","event_type","base_signal","title","summary"]].drop_duplicates("event_id")

    base = pd.merge(rel_k, vol_k, on=["event_id","ticker"], how="outer")
    base = pd.merge(base, ev_k, on="event_id", how="left")

    if base.empty:
        logger.info("결합 결과가 비어 있습니다. 입력(aliases, 랭킹, 거래량)을 점검하세요.")
        return

    # 스코어 구성요소 계산
    # A) base_signal 정규화: -1~+1 → 0~1 (음/양 방향 유지 위해 별도 보관)
    base["base_signal"] = pd.to_numeric(base["base_signal"], errors="coerce").fillna(0.0)
    base["base_signal_norm"] = (base["base_signal"] + 1.0) / 2.0  # 0~1

    # B) 관련도 점수(순위 기반)
    if "rank_relevance" in base.columns and base["rank_relevance"].notna().any():
        base["rel_score"] = _norm_rank_score(base["rank_relevance"].fillna(N_R + 1), N_R)
    else:
        base["rel_score"] = 0.0

    # C) 거래량 점수(uplift 기반)
    base["vol_score"] = _norm_uplift(pd.to_numeric(base["volume_uplift"], errors="coerce").fillna(0.0))

    # D) 상호작용항(옵션): rel_score * vol_score
    base["mix_term"] = base["rel_score"] * base["vol_score"]

    # 최종 buy_score
    base["buy_score_raw"] = (
        w_base * base["base_signal_norm"] +
        w_rel  * base["rel_score"] +
        w_vol  * base["vol_score"] +
        w_mix  * base["mix_term"]
    )
    base["buy_score"] = base["buy_score_raw"].astype(float).apply(_clip01)

    # 의사결정 로직
    # - base_signal이 음수인데 require_positive_base=True이면 강등
    decision = []
    for _, r in base.iterrows():
        bs = float(r["buy_score"])
        base_sig = float(r["base_signal"])

        if require_positive_base and base_sig < 0:
            # 기본 신호가 음수 → 관망 또는 제외
            if bs >= thr_watch:
                decision.append("관망")
            else:
                decision.append("제외")
            continue

        if bs >= thr_buy:
            decision.append("매수")
        elif bs >= thr_watch:
            decision.append("관망")
        else:
            decision.append("제외")
    base["decision"] = decision

    # 설명 문자열(근거) 및 메타
    base["reason"] = base.apply(_build_reason, axis=1)

    # 정렬(우선순위: event_id, buy_score desc)
    out_cols = [
        "event_id", "ticker", "buy_score", "decision", "reason",
        "base_signal", "base_signal_norm",
        "rank_relevance", "relevance_score",
        "rank_volume", "volume_uplift", "abn_volume_z",
        "event_type", "published_ts", "title", "summary"
    ]
    out = base[out_cols].sort_values(["event_id", "buy_score"], ascending=[True, False]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_FILE, index=False)
    logger.info(f"signals -> {OUT_FILE} (rows={len(out)}, events={out['event_id'].nunique()})")

if __name__ == "__main__":
    run()
