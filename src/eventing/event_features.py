# src/eventing/event_features.py
# 목적:
# - 정규화된 이벤트 파일(events.parquet)을 불러와
#   특징치(salience/novelty 등)를 재계산하거나 추가 지표를 부여
#   (이번 단계에선 event_normalizer와 동일 로직을 함수화만 해둠)

from __future__ import annotations
import pandas as pd
from src.common.logging import setup_logger
from src.common.constants import EVENTS_FILE

def recompute_features() -> None:
    logger = setup_logger("eventing")
    if not EVENTS_FILE.exists():
        logger.warning("events.parquet 가 없습니다. 먼저 event_normalizer 를 실행하세요.")
        return

    df = pd.read_parquet(EVENTS_FILE)
    # 필요 시 향후 추가 특징치 계산 위치
    # ex) df["some_feature"] = ...
    df.to_parquet(EVENTS_FILE, index=False)
    logger.info(f"events features recomputed -> {EVENTS_FILE} (rows={len(df)})")

if __name__ == "__main__":
    recompute_features()
