"""
설정 로더
- .env (선택) + YAML(params/sources) 로딩
- 외부 라이브러리 최소화: PyYAML 권장
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv  # 선택 설치: python-dotenv
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False

try:
    import yaml  # 설치 필요: PyYAML
except Exception as e:
    raise ImportError("PyYAML 미설치: `pip install PyYAML` 후 재시도하세요") from e

from .constants import PARAMS_YAML, SOURCES_YAML, ENV_FILE

@dataclass
class AppConfig:
    env: str
    params: Dict[str, Any]
    sources: Dict[str, Any]

def _load_env() -> None:
    # .env 는 있으면 읽고, 없으면 건너뜀
    if _HAS_DOTENV and ENV_FILE.exists():
        load_dotenv(ENV_FILE)

def _load_yaml(path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config() -> AppConfig:
    _load_env()
    app_env = os.getenv("APP_ENV", "LOCAL")

    params = _load_yaml(PARAMS_YAML)
    sources = _load_yaml(SOURCES_YAML)

    return AppConfig(
        env=app_env,
        params=params,
        sources=sources
    )
