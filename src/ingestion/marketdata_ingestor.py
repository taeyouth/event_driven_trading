# src/ingestion/marketdata_ingestor.py
# 목적:
# - data/raw/market/*.csv 파일들을 읽어 표준 스키마로 정규화 후
#   분봉으로 집계하여 data/processed/market.parquet 로 저장
#
# 입력 CSV 유연 스키마(자동 인식):
#   [티커 후보]   : ticker, symbol, 종목코드, 코드
#   [시각 후보]   : ts, timestamp, datetime, time, 거래시각, 체결시각, 일시, 날짜시간
#                   (date+time 컬럼이 분리돼 있어도 자동 결합)
#   [거래량 후보] : volume, 거래량, 체결량
#   [가격 후보]   : price, close, 체결가, 종가, 현재가 (있으면 보존)
#
# 실행:
#   python -m src.ingestion.marketdata_ingestor
#
# 요구사항:
#   - pandas
#   - pyarrow (parquet 저장용)
#   - tzdata (Windows에서 권장: Asia/Seoul 타임존 처리)
#
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from src.common.logging import setup_logger
from src.common.constants import RAW_DIR, PROCESSED_DIR

OUT_PATH = PROCESSED_DIR / "market.parquet"

# 후보 컬럼명 사전  ─────────────────────────────────────────────────────────────
TICKER_CANDIDATES = ["ticker", "symbol", "종목코드", "코드"]
TS_CANDIDATES      = ["ts", "timestamp", "datetime", "time", "거래시각", "체결시각", "일시", "날짜시간"]
DATE_CANDIDATES    = ["date", "날짜", "일자"]
TIME_CANDIDATES    = ["time", "시간", "시각"]
VOLUME_CANDIDATES  = ["volume", "거래량", "체결량"]
PRICE_CANDIDATES   = ["price", "close", "체결가", "종가", "현재가"]

def _list_raw_csvs() -> List[Path]:
    p = RAW_DIR / "market"
    return sorted(p.glob("*.csv")) if p.exists() else []

def _find_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None

def _resolve_columns(df: pd.DataFrame) -> Tuple[str, str, str, Optional[str]]:
    """티커/시각/거래량/가격 컬럼명 자동 탐지. 가격은 Optional."""
    ticker_col = _find_first(df, TICKER_CANDIDATES)
    ts_col     = _find_first(df, TS_CANDIDATES)
    vol_col    = _find_first(df, VOLUME_CANDIDATES)
    price_col  = _find_first(df, PRICE_CANDIDATES)

    # ### 변경: date+time 분리 컬럼을 ts로 결합 지원
    if ts_col is None:
        dcol = _find_first(df, DATE_CANDIDATES)
        tcol = _find_first(df, TIME_CANDIDATES)
        if dcol and tcol:
            ts_col = "__tmp_ts__"
            df[ts_col] = df[dcol].astype(str).str.strip() + " " + df[tcol].astype(str).str.strip()

    missing = []
    if ticker_col is None: missing.append("ticker(예: ticker/symbol/종목코드/코드)")
    if ts_col is None:     missing.append("ts(예: ts/timestamp/datetime/거래시각/체결시각/일시)")
    if vol_col is None:    missing.append("volume(예: volume/거래량/체결량)")
    if missing:
        raise ValueError("필수 컬럼이 없습니다: " + ", ".join(missing))

    return ticker_col, ts_col, vol_col, price_col

def _parse_ts_series(s: pd.Series) -> pd.Series:
    """문자열 시각 → UTC 분 단위 Timestamp.
       - 'KST' 같은 약어가 들어와도 최대한 파싱 후 Asia/Seoul 로컬라이즈 → UTC 변환
       - tz 정보가 없으면 Asia/Seoul 가정(운영 환경이 한국 시장이므로)
    """
    # ### 변경: 광범위한 문자열 파싱 후 타임존 처리
    ts = pd.to_datetime(s, errors="coerce", utc=False)
    # tz가 없는 naive는 KST로 가정 → UTC 변환
    try:
        # pandas >= 1.4: zoneinfo 사용 가능, Windows는 tzdata 설치 권장
        ts = ts.dt.tz_localize("Asia/Seoul", nonexistent="shift_forward", ambiguous="NaT", errors="coerce")
    except Exception:
        # 이미 tz-aware인 값은 통과를 시도
        try:
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("Asia/Seoul", errors="coerce")
        except Exception:
            pass
    # UTC로 통일
    try:
        ts = ts.dt.tz_convert("UTC")
    except Exception:
        # tz 없는 일부는 그대로 UTC 가정
        ts = pd.to_datetime(s, errors="coerce", utc=True)
    return ts.dt.floor("min")

def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ticker_col, ts_col, vol_col, price_col = _resolve_columns(df)

    # 타입/정규화
    df[ticker_col] = df[ticker_col].astype(str).str.strip()
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0)

    ts = _parse_ts_series(df[ts_col].astype(str))
    df = df.loc[ts.notna()].copy()
    df["ts"] = ts[ts.notna()]

    # 표준 스키마로 리턴
    out_cols = ["ticker", "ts", "volume"]
    df_out = pd.DataFrame({
        "ticker": df[ticker_col].values,
        "ts": df["ts"].values,
        "volume": df[vol_col].values.astype(float),
    })

    # 가격 보존(있을 때만)
    if price_col:
        df_out["price"] = pd.to_numeric(df[price_col], errors="coerce")

    return df_out

def run() -> None:
    logger = setup_logger("ingestion")
    csvs = _list_raw_csvs()
    if not csvs:
        logger.warning("data/raw/market/*.csv 가 없습니다. 최소 한 개의 CSV를 추가하세요.")
        return

    parts = []
    for p in csvs:
        try:
            dfp = _read_one(p)
            parts.append(dfp)
            logger.info(f"[ok] {p.name} loaded | rows={len(dfp)}")
        except Exception as e:
            logger.exception(f"[skip] {p.name}: {e}")

    if not parts:
        logger.warning("유효한 시장 데이터가 없습니다.")
        return

    df = pd.concat(parts, ignore_index=True)
    # 분봉 집계 (ticker, ts)
    agg = df.groupby(["ticker", "ts"], as_index=False).agg(volume=("volume", "sum"))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT_PATH, index=False)
    logger.info(f"market -> {OUT_PATH} (rows={len(agg)}, tickers={agg['ticker'].nunique()})")

if __name__ == "__main__":
    run()
