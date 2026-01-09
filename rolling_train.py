import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_env import AShareTradingEnv

device = "cuda" if torch.cuda.is_available() else "cpu"


def rolling_train(market_data, dates, hs300, market_breadth):
    print("Starting real rolling backtest...")
    print(f"Total dates: {len(dates)}")
    print(f"Date range: {dates.min()} to {dates.max()}")

    def make_env():
        return AShareTradingEnv(
            universe=market_data,
            dates=dates,
            start_idx=0,
            end_idx=len(dates),
            hs300=hs300,
            market_breadth=market_breadth
        )

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        verbose=1
    )

    # === 关键：训练 PPO ===
    model.learn(total_timesteps=len(dates) * 2)

    obs = env.reset()
    equity_list = []

    done = False
    while not done:
        # === PPO 决策 ===
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        equity_list.append(info[0]["equity"])

    equity = pd.Series(equity_list, index=dates[1:len(equity_list)+1])

    print(f"Generated equity series with {len(equity)} values")
    print(f"Equity range: {equity.min():.4f} to {equity.max():.4f}")

    return equity
