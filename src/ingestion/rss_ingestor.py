# src/ingestion/rss_ingestor.py
# 목적: configs/sources.yaml의 rss 소스들을 읽어 제목/링크/발행시각 등을 수집해 data/raw/에 CSV로 저장

from typing import List, Dict, Any
from datetime import datetime
import csv
import feedparser

from src.common.config import load_config
from src.common.logging import setup_logger
from src.common.constants import RAW_DIR

# ──────────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────────────────────────────────────
def _parse_entry(e: Any, feed_name: str) -> Dict[str, str]:
    """RSS 엔트리에서 안전하게 필드를 추출"""
    # RSS마다 지원 필드가 달라서 getattr로 방어적으로 접근
    entry_id = getattr(e, "id", "") or getattr(e, "guid", "") or getattr(e, "link", "")
    published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
    source_title = ""
    try:
        source_title = getattr(getattr(e, "source", None), "title", "") or ""
    except Exception:
        source_title = ""

    return {
        "id": entry_id,
        "title": getattr(e, "title", "") or "",
        "link": getattr(e, "link", "") or "",
        "published": published,
        "summary": getattr(e, "summary", "") or "",
        "source": source_title,
        "feed_name": feed_name,
    }

def _dedup_by_id(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """id 기준 중복 제거 (id가 비어있으면 link로 대체)"""
    seen = set()
    out = []
    for r in rows:
        key = r["id"] or r["link"]
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out

def _dump_csv(rows: List[Dict[str, str]]) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"rss_{ts}.csv"

    # 필드 고정(일부 엔트리가 비어 있어도 헤더 안정화)
    fieldnames = ["id", "title", "link", "published", "summary", "source", "feed_name"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return str(path)

# ──────────────────────────────────────────────────────────────────────────────
# 메인 함수
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    """
    configs/sources.yaml 의 rss 리스트를 읽어 들여 배치 수집을 수행.
    결과는 data/raw/rss_YYYYMMDD_HHMMSS.csv 로 저장.
    """
    logger = setup_logger("ingestion")
    cfg = load_config()

    rss_list = cfg.sources.get("rss", [])
    if not rss_list:
        logger.warning("sources.yaml -> rss 리스트가 비어 있습니다.")
        return

    logger.info(f"rss sources: {len(rss_list)}")

    collected: List[Dict[str, str]] = []
    for src in rss_list:
        url = (src or {}).get("url")
        name = (src or {}).get("name", "unknown_rss")
        if not url:
            logger.warning(f"[skip] name={name} url 없음")
            continue

        try:
            feed = feedparser.parse(url)
            n = 0
            for e in getattr(feed, "entries", []):
                row = _parse_entry(e, name)
                # 제목/링크 둘 다 비어 있으면 의미가 없으니 스킵
                if not row["title"] and not row["link"]:
                    continue
                collected.append(row)
                n += 1
            logger.info(f"[ok] {name}: {n} entries")
        except Exception as ex:
            logger.exception(f"[fail] {name}: {url} / {ex}")

    # 중복 제거
    before = len(collected)
    collected = _dedup_by_id(collected)
    after = len(collected)
    if before != after:
        logger.info(f"dedup: {before} -> {after}")

    # 저장
    out_path = _dump_csv(collected)
    logger.info(f"rss dump -> {out_path} (rows={len(collected)})")

if __name__ == "__main__":
    run()
