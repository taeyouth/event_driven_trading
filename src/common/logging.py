"""
로깅 설정 유틸
- 콘솔 + 파일 핸들러(회전) 세팅
"""

import logging
from logging.handlers import RotatingFileHandler
from .constants import LOGS_DIR

def setup_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 중복 핸들러 방지
    if logger.handlers:
        return logger

    # 콘솔
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # 파일(회전)
    fh = RotatingFileHandler(
        LOGS_DIR / f"{name}.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
