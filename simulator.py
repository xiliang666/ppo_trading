import numpy as np
import pandas as pd
from .indicators import compute_macd

def simulate_market_data(
    n_stocks=200,
    start="2020-01-01",
    end="2025-12-31"
):
    dates = pd.bdate_range(start, end)
    data = {}

    for i in range(n_stocks):
        prices = 100 * np.cumprod(
            1 + np.random.normal(0, 0.01, len(dates))
        )
        df = pd.DataFrame({"close": prices}, index=dates)
        dif, dea = compute_macd(df["close"])
        df["dif"] = dif
        df["dea"] = dea
        df["ret"] = df["close"].pct_change().fillna(0)
        data[f"Stock_{i:03d}"] = df

    return data, dates
