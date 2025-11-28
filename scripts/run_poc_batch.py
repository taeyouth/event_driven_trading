# scripts/run_poc_batch.py
# 전체 파이프라인을 순차적으로 실행하는 오케스트레이터

import sys
import logging
from pathlib import Path

# 프로젝트 루트를 path에 추가하여 모듈 import 문제 해결
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.constants import (
    DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR, LOGS_DIR,
    PARAMS_YAML, SOURCES_YAML, ENV_FILE
)

# 각 모듈의 메인 함수 import
from src.ingestion import rss_ingestor, marketdata_ingestor
from src.eventing import event_normalizer
from src.mapping import entity_linker
from src.scoring import basic_scoring
from src.ranking import relevance_topn, volume_topn
from src.signal import signal_builder

def ensure_dirs():
    """필수 디렉토리 생성"""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        # raw 하위 디렉토리도 보장
        (RAW_DIR / "market").mkdir(parents=True, exist_ok=True)

def run_step(step_name, module_func):
    """각 단계를 실행하고 예외 처리"""
    logger = logging.getLogger("poc")
    logger.info(f"--- [STEP START] {step_name} ---")
    try:
        module_func()
        logger.info(f"--- [STEP DONE] {step_name} ---\n")
    except Exception as e:
        logger.error(f"!!! [STEP FAILED] {step_name} !!!")
        logger.exception(e)
        raise e  # 파이프라인 중단

def main():
    logger = setup_logger("poc")
    cfg = load_config()

    logger.info("=== POC Batch Pipeline Started ===")
    
    # 0. 환경 점검
    ensure_dirs()
    
    # 1. Ingestion (데이터 수집)
    # RSS 수집
    run_step("RSS Ingestion", rss_ingestor.run)
    # 시장 데이터 수집 (주의: data/raw/market/*.csv 파일이 있어야 작동함)
    run_step("Market Data Ingestion", marketdata_ingestor.run)

    # 2. Event Processing (이벤트 정규화 & 특징 추출)
    run_step("Event Normalizer", event_normalizer.run)

    # 3. Text Analysis & Mapping (뉴스-종목 연결)
    run_step("Entity Linker", entity_linker.run)

    # 4. Scoring (감성/임팩트 점수 산출)
    run_step("Basic Scoring", basic_scoring.run)

    # 5. Ranking (관련도 및 거래량 분석)
    run_step("Ranking (Relevance)", relevance_topn.run)
    run_step("Ranking (Volume)", volume_topn.run)

    # 6. Signal Generation (최종 매매 신호 생성)
    run_step("Signal Builder", signal_builder.run)

    logger.info("=== POC Batch Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()