import numpy as np
import pandas as pd

print("Testing basic functionality...")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# 测试CSV加载
try:
    from csv_loader import load_a_share_csv_folder
    print("csv_loader imported successfully")
    
    # 加载数据
    market_data, dates = load_a_share_csv_folder(
        data_dir="data",
        start_date="2020-01-01",
        end_date="2024-12-31"
    )
    print(f"Data loaded successfully: {len(market_data)} stocks, {len(dates)} dates")
except Exception as e:
    print(f"Error in csv_loader: {e}")
    import traceback
    traceback.print_exc()

# 测试trading_env
try:
    from trading_env import AShareTradingEnv
    print("trading_env imported successfully")
except Exception as e:
    print(f"Error in trading_env: {e}")
    import traceback
    traceback.print_exc()

# 测试dynamic_scorer
try:
    from dynamic_scorer import dynamic_select
    print("dynamic_scorer imported successfully")
except Exception as e:
    print(f"Error in dynamic_scorer: {e}")
    import traceback
    traceback.print_exc()

# 测试math_utils
try:
    from math_utils import zscore
    print("math_utils imported successfully")
except Exception as e:
    print(f"Error in math_utils: {e}")
    import traceback
    traceback.print_exc()

print("Basic test completed")
