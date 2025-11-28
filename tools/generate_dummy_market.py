# tools/generate_dummy_market.py (수정 버전)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트 찾기
PROJECT_ROOT = Path(__file__).resolve().parents[1] # tools 폴더 상위
MARKET_DIR = PROJECT_ROOT / "data" / "raw" / "market"

def generate_dummy_data():
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    
    # [수정됨] 1. 설정: 기간을 '과거 30일'로 늘려서 11월 초 데이터 커버
    # 테스트용 종목: 삼성전자(005930) 외에 결과에 뜬 015120, 000432 등도 포함하면 좋지만
    # 우선 매핑이 확실한 삼성전자/하이닉스/네이버 위주로 생성
    tickers = ["005930", "000660", "035420"] 
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=35) # 넉넉하게 35일 전부터 생성
    
    print(f"Generating dummy market data from {start_date.date()} to {end_date.date()}...")

    dfs = []
    freq = "1min"
    
    current = start_date
    while current <= end_date:
        if current.weekday() < 5: # 주말 제외
            day_str = current.strftime("%Y%m%d")
            
            # 당일 09:00 ~ 15:30
            day_start = current.replace(hour=9, minute=0, second=0, microsecond=0)
            day_end = current.replace(hour=15, minute=30, second=0, microsecond=0)
            
            times = pd.date_range(start=day_start, end=day_end, freq=freq)
            n = len(times)
            
            for tkr in tickers:
                # [수정됨] 2. 거래량 급등 이벤트 시뮬레이션
                # 11월 11일, 12일 즈음에 삼성전자(005930) 거래량이 폭발하도록 조작
                is_event_day = (day_str in ["20251111", "20251112"]) and (tkr == "005930")
                
                if is_event_day:
                    # 평소보다 거래량 5배, 가격 상승
                    vol = np.random.lognormal(mean=12, sigma=1.0, size=n).astype(int) 
                    price = np.random.normal(loc=72000, scale=1500, size=n).astype(int)
                else:
                    # 평상시
                    vol = np.random.lognormal(mean=10, sigma=1.0, size=n).astype(int)
                    price = np.random.normal(loc=70000, scale=1000, size=n).astype(int)
                
                df_temp = pd.DataFrame({
                    "ticker": tkr,
                    "ts": times,
                    "volume": vol,
                    "price": price
                })
                dfs.append(df_temp)
        
        current += timedelta(days=1)

    if dfs:
        full_df = pd.concat(dfs)
        # 날짜 범위가 넓으므로 파일명에 start_end 표시
        filename = f"kospi_dummy_{start_date.strftime('%m%d')}_{end_date.strftime('%m%d')}.csv"
        save_path = MARKET_DIR / filename
        
        # 기존 파일 덮어쓰거나 삭제하고 저장하는 것이 좋음
        full_df.to_csv(save_path, index=False)
        print(f"✅ Created: {save_path} ({len(full_df)} rows)")
    else:
        print("❌ No data generated.")

if __name__ == "__main__":
    generate_dummy_data()