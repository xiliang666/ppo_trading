import numpy as np
import pandas as pd
from csv_loader import load_a_share_csv_folder
from trading_env import AShareTradingEnv

# 加载数据
market_data, dates = load_a_share_csv_folder(
    data_dir="data",
    start_date="2020-01-01",
    end_date="2024-12-31"
)

# 生成模拟的市场数据
hs300 = pd.Series(
    np.cumprod(1 + np.random.normal(0, 0.005, len(dates))),
    index=dates
).pct_change().fillna(0)

market_breadth = pd.Series(
    np.random.uniform(0.2, 0.8, len(dates)),
    index=dates
)

print(f"Total dates: {len(dates)}")
print(f"Date range: {dates.min()} to {dates.max()}")

# 测试单个环境实例
test_start_idx = 0
test_end_idx = min(20, len(dates) - 1)

print(f"\nTesting environment with start_idx={test_start_idx}, end_idx={test_end_idx}")
test_env = AShareTradingEnv(
    market_data, dates,
    test_start_idx, test_end_idx,
    hs300, market_breadth
)

print(f"Environment dates length: {len(test_env.dates)}")
print(f"Environment dates: {test_env.dates[:5]}...{test_env.dates[-5:]}")

# 测试环境重置和步骤
obs, _ = test_env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial equity_curve: {test_env.equity_curve}")

done = False
step_count = 0
while not done and step_count < 10:
    action = np.random.rand(31)
    obs, reward, done, _, _ = test_env.step(action)
    step_count += 1
    print(f"Step {step_count}: equity_curve={test_env.equity_curve}, reward={reward}")

print(f"Final equity_curve: {test_env.equity_curve}")
print(f"Length of equity_curve: {len(test_env.equity_curve)}")
