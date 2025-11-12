# scripts/run_poc_batch.py
# 최소 엔트리포인트: 설정 로드, 로거 초기화, 디렉터리 준비 확인

from pathlib import Path
from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.constants import (
    REPO_ROOT, DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR, LOGS_DIR,
    PARAMS_YAML, SOURCES_YAML, ENV_FILE
)

def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def main():
    logger = setup_logger("poc")
    cfg = load_config()

    logger.info("=== POC batch start ===")
    logger.info(f"ENV={cfg.env}")
    logger.info(f"PARAMS_YAML exists: {PARAMS_YAML.exists()}")
    logger.info(f"SOURCES_YAML exists: {SOURCES_YAML.exists()}")
    logger.info(f".env exists: {ENV_FILE.exists()}")

    ensure_dirs()
    for p in [DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR, LOGS_DIR]:
        logger.info(f"dir ready: {p}")

    logger.info("=== POC batch end (no-op) ===")

if __name__ == "__main__":
    main()
