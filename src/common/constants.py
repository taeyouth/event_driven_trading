"""
공통 상수 정의
- 경로/파일명 등 하드코딩 상수 모아두는 곳
"""

from pathlib import Path

# 레포 루트 기준
REPO_ROOT = Path(__file__).resolve().parents[2]

# 데이터/로그 디렉터리 (gitignore 대상)
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

LOGS_DIR = REPO_ROOT / "logs"

# 설정 파일 경로
CONFIGS_DIR = REPO_ROOT / "configs"
PARAMS_YAML = CONFIGS_DIR / "params.yaml"
SOURCES_YAML = CONFIGS_DIR / "sources.yaml"
ENV_FILE = REPO_ROOT / ".env"

# 기본 파일명(POC 산출물)
EVENTS_FILE = PROCESSED_DIR / "events.parquet"
MAPPING_FILE = PROCESSED_DIR / "mapping.parquet"
RANKINGS_FILE = PROCESSED_DIR / "rankings.parquet"
SIGNALS_FILE = PROCESSED_DIR / "signals.parquet"
