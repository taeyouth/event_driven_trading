# ./src/ranking/krx_master.py

import requests
import pandas as pd
from io import BytesIO
from pathlib import Path


def _find_column(df: pd.DataFrame, candidates: list[str], logical_name: str) -> str:
    """
    DataFrame에서 후보 컬럼명 리스트 중 실제로 존재하는 첫 번째 컬럼명을 찾는다.
    없으면 에러 발생.
    """
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"[KRX 마스터] {logical_name}에 해당하는 컬럼을 찾지 못했습니다.\n"
        f"  logical_name = {logical_name}\n"
        f"  candidates   = {candidates}\n"
        f"  actual cols  = {list(df.columns)}"
    )


def fetch_krx_master(
    save_path: str = "data/krx/krx_master.csv",
    encoding: str = "cp949",
):
    """
    KRX 상장종목 전체 마스터 파일을 다운로드하여
    ticker(6자리) + name 컬럼으로 저장한다.

    반환:
      - DataFrame (컬럼: ticker, name)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # OTP 생성 URL
    gen_url = "http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd"
    gen_params = {
        "mktId": "ALL",
        "share": "1",
        "csvxls_isNo": "false",
        "name": "fileDown",
        "url": "dbms/MDC/STAT/standard/MDCSTAT01901",
    }
    headers = {
        "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101",
        "User-Agent": "Mozilla/5.0",
    }

    # 1) OTP 코드 요청
    r1 = requests.get(gen_url, params=gen_params, headers=headers)
    r1.raise_for_status()
    otp = r1.content.decode()

    # 2) 실제 다운로드 요청
    down_url = "http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd"
    r2 = requests.post(down_url, data={"code": otp}, headers=headers)
    r2.raise_for_status()
    r2.encoding = encoding

    # 3) DataFrame으로 변환
    df_raw = pd.read_csv(BytesIO(r2.content), encoding=encoding)
    print("[KRX 마스터] 원본 컬럼 목록:", list(df_raw.columns))

    # 4) 종목명/티커 컬럼 자동 매핑
    name_col_candidates = ["종목명", "한글 종목명", "한글명", "표준종목명"]
    ticker_col_candidates = ["단축코드", "종목코드", "단축 종목코드"]

    name_col = _find_column(df_raw, name_col_candidates, logical_name="name")
    ticker_col = _find_column(df_raw, ticker_col_candidates, logical_name="ticker")

    df = df_raw[[name_col, ticker_col]].rename(
        columns={name_col: "name", ticker_col: "ticker"}
    )

    # 5) ticker 앞자리 0 채우기 → 6자리
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)

    # 6) 중복 제거 / 정렬
    df = df.drop_duplicates(subset=["ticker"])
    df = df.sort_values("ticker").reset_index(drop=True)

    # 7) 저장
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"[KRX 마스터] Saved to {save_path}, rows = {len(df)}")

    return df
