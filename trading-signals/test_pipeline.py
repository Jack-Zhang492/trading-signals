"""Quick pipeline test"""
from core.data_fetcher import fetch_stock_data
from core.indicators import calc_all_indicators, detect_market_regime
from core.patterns import PatternDetector
from core.backtest import BacktestEngine

# Fetch data
data = fetch_stock_data("600519", "Moutai")
daily = data.get("daily")
print(f"Fetched {len(daily)} daily bars")

# Indicators
df = calc_all_indicators(daily)
ind_cols = [c for c in df.columns if c not in ["open","high","low","close","volume","amount","turnover","date"]]
print(f"Indicators computed: {len(ind_cols)} ({', '.join(ind_cols[:10])}...)")

# Patterns
detector = PatternDetector()
df_sig = detector.detect_all(df)
latest = detector.get_latest_signal(df_sig)
print(f"\nLatest signal: {latest['signal']} (score={latest['net_score']})")
print(f"Triggers: {latest['triggers']}")
print(f"Buy score: {latest['buy_score']}, Sell: {latest['sell_score']}")
print(f"Indicators: RSI={latest['indicators']['rsi']}, MACD={latest['indicators']['macd_dif']}")
print(f"RSI: {latest['indicators']['rsi']}")
print(f"Regime: {detect_market_regime(df)}")

# Backtest
engine = BacktestEngine()
bt = engine.backtest_stock(daily)
ranking = engine.rank_patterns(bt)
print(f"\nTop 5 patterns:")
for r in ranking[:5]:
    print(f"  {r['name']}: {r['win_rate']}% win, {r['avg_return']}% avg ret, score={r['score']}")

print(f"\nSummary: {bt['summary']}")
print("\nAll OK!")
