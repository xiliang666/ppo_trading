import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from dynamic_scorer import dynamic_select
from math_utils import zscore


class AShareTradingEnv(gym.Env):
    def __init__(self, universe, dates, start_idx, end_idx,
                 hs300, market_breadth):
        super().__init__()
        self.universe = universe
        self.dates = dates[start_idx:end_idx]
        self.hs300 = hs300
        self.market_breadth = market_breadth

        self.idx = 0
        self.equity_curve = []

        self.action_space = spaces.Box(
            low=0, high=1, shape=(31,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(30 * 4 + 2,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.equity_curve = [1.0]  # 明确初始净值
        return self._get_obs(), {}


    def _get_obs(self):
        date = self.dates[self.idx]
        stocks = dynamic_select(self.universe, date)

        obs = []
        for s in stocks:
            df = self.universe[s].loc[:date]
            # 处理 NaN 值
            macd = df["dif"].iloc[-1] - df["dea"].iloc[-1]
            macd = 0.0 if pd.isna(macd) else macd
            
            dif_signal = 1.0 if (not pd.isna(df["dif"].iloc[-1]) and df["dif"].iloc[-1] < 0) else 0.0
            
            ret_sum = df["ret"].tail(5).sum()
            ret_sum = 0.0 if pd.isna(ret_sum) else ret_sum
            
            obs.extend([
                macd,
                dif_signal,
                ret_sum,
                1.0
            ])

        while len(obs) < 30 * 4:
            obs.append(0.0)

        # 处理市场指标的 NaN 值
        hs300_val = zscore(self.hs300).loc[date]
        hs300_val = 0.0 if pd.isna(hs300_val) else hs300_val
        
        breadth_val = zscore(self.market_breadth).loc[date]
        breadth_val = 0.0 if pd.isna(breadth_val) else breadth_val
        
        obs.append(hs300_val)
        obs.append(breadth_val)

        # 确保没有 NaN 值
        obs = [0.0 if pd.isna(x) else x for x in obs]

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        weights = np.exp(action) / np.sum(np.exp(action))
        stock_weights = weights[:-1]

        date = self.dates[self.idx]
        stocks = dynamic_select(self.universe, date)

        daily_ret = 0.0
        for i, s in enumerate(stocks):
            daily_ret += stock_weights[i] * self.universe[s].loc[date]["ret"]

        turnover = np.sum(np.abs(stock_weights))
        daily_ret -= 0.001 * turnover

        equity = (self.equity_curve[-1] if self.equity_curve else 1.0)
        equity *= (1 + daily_ret)
        self.equity_curve.append(equity)

        peak = max(self.equity_curve)
        drawdown = (peak - equity) / peak
        reward = daily_ret - 2.0 * min(drawdown, 0.15) - 0.05 * turnover

        self.idx += 1
        done = self.idx >= len(self.dates) - 1

        return self._get_obs(), reward, done, False, {
        "equity": equity,
        "daily_return": daily_ret
        }

