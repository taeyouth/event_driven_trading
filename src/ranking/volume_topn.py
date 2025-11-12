# src/ranking/volume_topn.py  (수정판)
from __future__ import annotations
import pandas as pd
from datetime import timedelta
from src.common.logging import setup_logger
from src.common.config import load_config
from src.common.constants import EVENTS_FILE, RANKINGS_FILE, PROCESSED_DIR

MARKET_FILE = PROCESSED_DIR / "market.parquet"
OUT_FILE = PROCESSED_DIR / "rankings_volume.parquet"

def _ensure_ts_utc(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce", utc=False)
    try:
        # tz-naive는 KST로 간주 후 UTC로 변환
        ts = ts.dt.tz_localize("Asia/Seoul", nonexistent="shift_forward", ambiguous="NaT", errors="coerce")
    except Exception:
        # 이미 tz 있는 경우 그대로 시도
        pass
    try:
        ts = ts.dt.tz_convert("UTC")
    except Exception:
        # 마지막 방어
        ts = pd.to_datetime(s, errors="coerce", utc=True)
    return ts

def _calc_metrics(mkt: pd.DataFrame, ticker: str, t0: pd.Timestamp,
                  baseline_days: int, post_minutes: int) -> dict:
    if pd.isna(t0):
        return {"baseline_mean": 0.0, "baseline_std": 0.0, "post_mean": 0.0,
                "volume_uplift": 0.0, "abn_volume_z": 0.0}
    t0u = pd.Timestamp(t0)
    if t0u.tzinfo is None:
        t0u = t0u.tz_localize("UTC")
    t_start = t0u - timedelta(days=baseline_days)
    t_end_post = t0u + timedelta(minutes=post_minutes)

    sub = mkt[(mkt["ticker"] == str(ticker)) & (mkt["ts"] >= t_start) & (mkt["ts"] < t_end_post)]
    if sub.empty:
        return {"baseline_mean": 0.0, "baseline_std": 0.0, "post_mean": 0.0,
                "volume_uplift": 0.0, "abn_volume_z": 0.0}

    base = sub[sub["ts"] < t0u]
    post = sub[sub["ts"] >= t0u]
    base_mean = float(base["volume"].mean()) if not base.empty else 0.0
    base_std  = float(base["volume"].std(ddof=1)) if len(base) > 1 else 0.0
    post_mean = float(post["volume"].mean()) if not post.empty else 0.0

    uplift = (post_mean - base_mean) / (base_mean if base_mean > 0 else 1e-9)
    z = (post_mean - base_mean) / (base_std if base_std > 0 else 1e-9)
    return {"baseline_mean": base_mean, "baseline_std": base_std,
            "post_mean": post_mean, "volume_uplift": float(uplift), "abn_volume_z": float(z)}

def run() -> None:
    logger = setup_logger("ranking")
    cfg = load_config()

    if not (PROCESSED_DIR / "market.parquet").exists():
        logger.warning("market.parquet 이 없습니다. 먼저 marketdata_ingestor 를 실행하세요.")
        return
    if not EVENTS_FILE.exists():
        logger.warning("events.parquet 이 없습니다.")
        return
    if not RANKINGS_FILE.exists():
        logger.warning("rankings.parquet(관련도 TopN_R) 이 없습니다.")
        return

    mkt = pd.read_parquet(PROCESSED_DIR / "market.parquet")
    ev  = pd.read_parquet(EVENTS_FILE)
    rel = pd.read_parquet(RANKINGS_FILE)

    if mkt.empty or rel.empty:
        logger.warning("입력 데이터가 비어 있습니다.")
        return

    # 파라미터
    baseline_days = int((((cfg.params or {}).get("windows") or {}).get("baseline_days", 5)))
    post_minutes  = int((((cfg.params or {}).get("windows") or {}).get("post_minutes", 60)))
    N_V           = int((((cfg.params or {}).get("topn") or {}).get("volume", 10)))

    # 1) rel에 published_ts가 이미 있으면 그대로 사용, 없으면 ev에서 머지
    if "published_ts" in rel.columns:
        df = rel.copy()
    else:
        evm = ev[["event_id", "published_ts"]].drop_duplicates("event_id")
        df = rel.merge(evm, on="event_id", how="left")

    # 2) 혹시 중복 머지로 suffix가 생겼다면 정리  ← ★★ 수정 포인트
    if "published_ts" not in df.columns:
        for cand in ["published_ts_x", "published_ts_y"]:
            if cand in df.columns:
                df["published_ts"] = df[cand]
                break

    # 3) 타입 보정
    df["published_ts"] = _ensure_ts_utc(df["published_ts"])
    mkt["ts"] = _ensure_ts_utc(mkt["ts"])

    # 4) 메트릭 계산
    rows = []
    for _, r in df.iterrows():
        metrics = _calc_metrics(mkt, r["ticker"], r["published_ts"], baseline_days, post_minutes)
        rows.append({**r.to_dict(), **metrics})

    out = pd.DataFrame(rows)

    # 5) 이벤트별 Top N_V 산출
    out["rank_volume"] = out.groupby("event_id")["volume_uplift"].rank("dense", ascending=False)
    topv = out.sort_values(
        ["event_id", "rank_volume", "volume_uplift", "abn_volume_z"],
        ascending=[True, True, False, False]
    )
    topv_final = topv[topv["rank_volume"] <= N_V].reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    topv_final.to_parquet(PROCESSED_DIR / "rankings_volume.parquet", index=False)
    logger.info(f"rankings(volume) -> {PROCESSED_DIR / 'rankings_volume.parquet'} "
                f"(rows={len(topv_final)}, events={topv_final['event_id'].nunique()}, N={N_V})")

if __name__ == "__main__":
    run()
