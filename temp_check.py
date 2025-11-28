import pandas as pd; df = pd.read_csv('data/processed/final_signals.csv', encoding='cp949')
print(df[['ticker', 'decision', 'buy_score', 'reason']].head().to_string())