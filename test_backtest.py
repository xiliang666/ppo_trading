import numpy as np
import pandas as pd
from csv_loader import load_a_share_csv_folder
from dynamic_scorer import dynamic_select

print("Testing backtest components...")

# 加载数据
market_data, dates = load_a_share_csv_folder(
    data_dir="data",
    start_date="2020-01-01",
    end_date="2024-12-31"
)

print(f"Loaded {len(market_data)} stocks")
print(f"Date range: {dates.min()} to {dates.max()}")

# 生成市场指标
hs300 = pd.Series(
    np.cumprod(1 + np.random.normal(0, 0.005, len(dates))),
    index=dates
).pct_change().fillna(0)

market_breadth = pd.Series(
    np.random.uniform(0.2, 0.8, len(dates)),
    index=dates
)

print(f"Generated hs300: {len(hs300)} values")
print(f"Generated market_breadth: {len(market_breadth)} values")

# 测试日期索引
print("\nTesting date indices...")
test_date = dates[100]  # 使用一个中间日期
print(f"Test date: {test_date}")
print(f"Index of test date: {dates.get_loc(test_date)}")

# 测试股票选择
print("\nTesting stock selection...")
stocks = dynamic_select(market_data, test_date)
print(f"Selected {len(stocks)} stocks")
print(f"First 5 stocks: {stocks[:5]}")

# 测试环境初始化
print("\nTesting environment initialization...")
try:
    from trading_env import AShareTradingEnv
    
    # 使用一个较小的日期范围进行测试
    test_start_idx = 100
    test_end_idx = min(150, len(dates)-1)
    
    env = AShareTradingEnv(
        market_data, dates,
        test_start_idx, test_end_idx,
        hs300, market_breadth
    )
    
    obs, _ = env.reset()
    print(f"Environment reset successfully")
    print(f"Observation shape: {obs.shape}")
    print(f"Observation has NaN: {np.isnan(obs).any()}")
    
    # 测试一步
    action = np.zeros(31)  # 随机动作
    obs, reward, done, _, _ = env.step(action)
    print(f"Step executed successfully")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"New observation has NaN: {np.isnan(obs).any()}")
    
    print("\nEnvironment test passed!")
    
except Exception as e:
    print(f"Error in environment test: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")
