# ./run_ranking_topn.py

from src.ranking.topn_signals import (
    load_signals,
    attach_krx_master,
    select_topn,
    export_results,
)


def main():
    signals_path = "data/processed/signals.parquet"
    krx_master_path = "data/krx/krx_master.csv"

    final_csv_path = "data/processed/final_signals.csv"
    final_parquet_path = "data/processed/final_signals.parquet"

    top_n = 5  # 이벤트별 상위 N개

    print(f"[INFO] Loading signals from: {signals_path}")
    signals_df = load_signals(signals_path)

    print(f"[INFO] Fetching & attaching KRX master to signals")
    signals_with_name = attach_krx_master(
        signals_df,
        krx_path=krx_master_path,
        krx_encoding="utf-8",
    )

    print(f"[INFO] Selecting top-{top_n} per event (relevance & volume)")
    final_df = select_topn(
        signals_with_name,
        top_n=top_n,
        relevance_col="relevance_score",
        volume_col="volume_uplift",
        event_col="event_id",
    )

    export_results(
        final_df,
        csv_path=final_csv_path,
        parquet_path=final_parquet_path,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
