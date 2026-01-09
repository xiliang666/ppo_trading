import numpy as np
import pandas as pd

from csv_loader import load_a_share_csv_folder
from rolling_train import rolling_train


# 模拟数据
market_data, dates = load_a_share_csv_folder(
    data_dir="data",
    start_date="2020-01-01",
    end_date="2024-12-31"
)

hs300 = pd.Series(
    np.cumprod(1 + np.random.normal(0, 0.005, len(dates))),
    index=dates
).pct_change().fillna(0)

market_breadth = pd.Series(
    np.random.uniform(0.2, 0.8, len(dates)),
    index=dates
)

# 回测
equity = rolling_train(
    market_data, dates, hs300, market_breadth
)

returns = equity.pct_change().dropna()
sharpe = returns.mean() / returns.std() * np.sqrt(252)
max_dd = ((equity.cummax() - equity) / equity.cummax()).max()
calmar = returns.mean() * 252 / max_dd

print("Backtest Summary")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Calmar Ratio: {calmar:.2f}")
