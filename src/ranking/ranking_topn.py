# ranking_topn.py

import pyarrow.parquet as pq
import pandas as pd

from krx_master import fetch_krx_master

def load_signals(parquet_path: str) -> pd.DataFrame:
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    return df

def preprocess_signals(df: pd.DataFrame, krx_master_path: str = "krx_master.csv") -> pd.DataFrame:
    # 6자리 ticker 보정
    df['ticker'] = df['ticker'].astype(str).str.zfill(6)

    # 종목명 매핑
    master = pd.read_csv(krx_master_path, dtype={'ticker': str})
    df = df.merge(master, on='ticker', how='left')
    return df

def select_topn(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    # 관련도 랭킹
    df['rank_relevance'] = df.groupby('event_id')['relevance_score'] \
                             .rank(method='dense', ascending=False)
    top_rel = df[df['rank_relevance'] <= top_n].copy()
    top_rel['type'] = 'relevance'

    # 거래량 상승 랭킹
    df['rank_volume'] = df.groupby('event_id')['volume_uplift'] \
                          .rank(method='dense', ascending=False)
    top_vol = df[df['rank_volume'] <= top_n].copy()
    top_vol['type'] = 'volume'

    # 병합 및 정리
    final = pd.concat([top_rel, top_vol], ignore_index=True)
    final = final.sort_values(['event_id', 'type', 'ticker']).reset_index(drop=True)
    return final

def export_results(df_final: pd.DataFrame, csv_path: str = "final_signals.csv", parquet_path: str = "final_signals.parquet"):
    df_final.to_csv(csv_path, index=False, encoding='utf-8')
    df_final.to_parquet(parquet_path, index=False)
    print(f"Exported results to {csv_path} and {parquet_path}")

if __name__ == "__main__":
    # 예시 실행
    signals_df = load_signals("data/processed/signals.parquet")
    _ = fetch_krx_master("data/krx_master.csv")
    prep = preprocess_signals(signals_df, "data/krx_master.csv")
    final = select_topn(prep, top_n=5)
    export_results(final, "data/processed/final_signals.csv", "data/processed/final_signals.parquet")
