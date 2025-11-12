# src/ranking/relevance_topn.py
# 목적:
# - mapping.parquet(이벤트↔티커, relevance_score) + events.parquet(base_signal)
# - 이벤트별 "관련도 Top N_R" 랭킹을 산출하여 data/processed/rankings.parquet 에 저장
#
# 실행:
#   python -m src.ranking.relevance_topn

from __future__ import annotations
import pandas as pd

from src.common.logging import setup_logger
from src.common.config import load_config
from src.common.constants import (
    MAPPING_FILE, EVENTS_FILE, PROCESSED_DIR, RANKINGS_FILE
)

def run() -> None:
    logger = setup_logger("ranking")
    cfg = load_config()

    if not MAPPING_FILE.exists():
        logger.warning("mapping.parquet 이 없습니다. 먼저 entity_linker를 실행하세요.")
        return
    if not EVENTS_FILE.exists():
        logger.warning("events.parquet 이 없습니다. 먼저 event_normalizer/ scoring을 실행하세요.")
        return

    mp = pd.read_parquet(MAPPING_FILE)
    ev = pd.read_parquet(EVENTS_FILE)

    if mp.empty:
        logger.info("mapping.parquet 이 비어있습니다.")
        return
    if "base_signal" not in ev.columns:
        ev["base_signal"] = 0.0

    # 이벤트 메타(시간/타입/신호) 붙이기
    cols = ["event_id", "published_ts", "event_type", "base_signal", "salience", "novelty"]
    evm = ev[cols].drop_duplicates("event_id")
    df = mp.merge(evm, on="event_id", how="left")

    # 이벤트별 Top N_R
    N_R = int((((cfg.params or {}).get("topn") or {}).get("relevance", 10)))
    df["rank_relevance"] = df.groupby("event_id")["relevance_score"].rank("dense", ascending=False)

    top_rel = df[df["rank_relevance"] <= N_R].sort_values(
        ["event_id", "rank_relevance", "relevance_score"], ascending=[True, True, False]
    ).reset_index(drop=True)

    # 저장
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    top_rel.to_parquet(RANKINGS_FILE, index=False)
    logger.info(f"rankings(relevance) -> {RANKINGS_FILE} (rows={len(top_rel)}, events={top_rel['event_id'].nunique()}, N={N_R})")

if __name__ == "__main__":
    run()
