import numpy as np
import pandas as pd
from csv_loader import load_a_share_csv_folder
from rolling_train import rolling_train

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

# 测试rolling_train函数
try:
    print("\nCalling rolling_train...")
    equity = rolling_train(market_data, dates, hs300, market_breadth)
    print(f"Equity returned: {equity}")
    print(f"Equity length: {len(equity)}")
    print(f"Equity values: {equity.values}")
    
    # 计算回测指标
    returns = equity.pct_change().dropna()
    print(f"Returns: {returns}")
    
    if len(returns) > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        max_dd = ((equity.cummax() - equity) / equity.cummax()).max()
        calmar = returns.mean() * 252 / max_dd
        
        print("\nBacktest Summary")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Calmar Ratio: {calmar:.2f}")
    else:
        print("\nNo returns to calculate metrics")
except Exception as e:
    print(f"Error in rolling_train: {e}")
    import traceback
    traceback.print_exc()
