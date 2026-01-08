import numpy as np
import pandas as pd
from csv_loader import load_a_share_csv_folder
from dynamic_scorer import dynamic_select

# 测试数据加载
print("Testing data loading...")
market_data, dates = load_a_share_csv_folder(
    data_dir="data",
    start_date="2020-01-01",
    end_date="2024-12-31"
)

print(f"Loaded {len(market_data)} stocks")
print(f"Date range: {dates.min()} to {dates.max()}")
print(f"Number of dates: {len(dates)}")

# 测试股票数据
print("\nTesting stock data...")
for stock, df in list(market_data.items())[:3]:
    print(f"\nStock: {stock}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Close prices (first 5): {df['close'].head().values}")
    print(f"Returns (first 5): {df['ret'].head().values}")
    print(f"NaN in close: {df['close'].isna().sum()}")
    print(f"NaN in ret: {df['ret'].isna().sum()}")

# 测试 dynamic_select
print("\nTesting dynamic_select...")
test_date = dates[min(100, len(dates)-1)]  # 使用一个中间日期
print(f"Testing date: {test_date}")
stocks = dynamic_select(market_data, test_date)
print(f"Selected stocks: {stocks}")
print(f"Number of selected stocks: {len(stocks)}")

# 测试 rolling_train 中的年份检查
print("\nTesting year ranges...")
for year in range(2020, 2025):
    train_start = pd.Timestamp(f"{year}-01-01")
    train_end = pd.Timestamp(f"{year+1}-01-01")
    
    start_in = train_start in dates
    end_in = train_end in dates
    print(f"Year {year}: start={train_start} in dates={start_in}, end={train_end} in dates={end_in}")

# 测试 hs300 和 market_breadth 生成
hs300 = pd.Series(
    np.cumprod(1 + np.random.normal(0, 0.005, len(dates))),
    index=dates
).pct_change().fillna(0)

market_breadth = pd.Series(
    np.random.uniform(0.2, 0.8, len(dates)),
    index=dates
)

print("\nTesting hs300 and market_breadth...")
print(f"hs300 shape: {hs300.shape}")
print(f"hs300 first 5: {hs300.head().values}")
print(f"market_breadth shape: {market_breadth.shape}")
print(f"market_breadth first 5: {market_breadth.head().values}")
