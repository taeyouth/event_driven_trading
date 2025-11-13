# ./src/ranking/topn_signals.py

from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

from src.ranking.krx_master import fetch_krx_master


def load_signals(parquet_path: str) -> pd.DataFrame:
    """
    signals.parquet를 읽어서 pandas DataFrame으로 반환.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"signals.parquet 파일을 찾을 수 없습니다: {path}")

    table = pq.read_table(path)
    df = table.to_pandas()
    return df


def attach_krx_master(
    signals_df: pd.DataFrame,
    krx_path: str = "data/krx/krx_master.csv",
    krx_encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    시그널 데이터에 KRX 종목 마스터를 조인하여 종목명을 붙인다.

    - 실행 시마다 fetch_krx_master를 호출해서 최신 마스터를 받아온다.
    """
    # 1) 최신 KRX 마스터 다운로드 + 저장
    krx_df = fetch_krx_master(save_path=krx_path, encoding="cp949")

    # 2) ticker 6자리 보정
    df = signals_df.copy()
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)

    # 3) 조인
    merged = df.merge(krx_df, on="ticker", how="left")

    missing_name_cnt = merged["name"].isna().sum()
    if missing_name_cnt > 0:
        print(f"[WARN] 종목명이 매핑되지 않은 티커 개수: {missing_name_cnt}")

    return merged


def select_topn(
    df_with_name: pd.DataFrame,
    top_n: int = 5,
    relevance_col: str = "relevance_score",
    volume_col: str = "volume_uplift",
    event_col: str = "event_id",
) -> pd.DataFrame:
    """
    이벤트별로
      - 관련도 기준 top N
      - 거래량 상승 기준 top N
    을 추출하여 하나의 DataFrame으로 반환한다.
    """
    df = df_with_name.copy()

    if relevance_col not in df.columns:
        raise KeyError(f"관련도 컬럼이 없습니다: {relevance_col}")
    if volume_col not in df.columns:
        raise KeyError(f"거래량 상승 컬럼이 없습니다: {volume_col}")
    if event_col not in df.columns:
        raise KeyError(f"이벤트 컬럼이 없습니다: {event_col}")

    # 관련도 랭킹
    df["rank_relevance"] = (
        df.groupby(event_col)[relevance_col]
        .rank(method="dense", ascending=False)
    )
    top_rel = df[df["rank_relevance"] <= top_n].copy()
    top_rel["signal_type"] = "relevance"

    # 거래량 랭킹
    df["rank_volume"] = (
        df.groupby(event_col)[volume_col]
        .rank(method="dense", ascending=False)
    )
    top_vol = df[df["rank_volume"] <= top_n].copy()
    top_vol["signal_type"] = "volume"

    # 두 결과 병합
    final = pd.concat([top_rel, top_vol], ignore_index=True)

    final = final.sort_values(
        by=[event_col, "signal_type", "rank_relevance", "rank_volume"]
    ).reset_index(drop=True)

    return final


def export_results(
    df_final: pd.DataFrame,
    csv_path: str = "data/processed/final_signals.csv",
    parquet_path: str = "data/processed/final_signals.parquet",
):
    """
    최종 결과를 CSV 및 Parquet으로 저장한다.
    """
    csv_out = Path(csv_path)
    parquet_out = Path(parquet_path)

    csv_out.parent.mkdir(parents=True, exist_ok=True)

    df_final.to_csv(csv_out, index=False, encoding="utf-8")
    df_final.to_parquet(parquet_out, index=False)

    print(f"[INFO] CSV 저장: {csv_out}")
    print(f"[INFO] Parquet 저장: {parquet_out}")
