import os
import pandas as pd
from indicators import compute_macd


def load_a_share_csv_folder(
    data_dir: str,
    start_date=None,
    end_date=None
):
    """
    读取一个文件夹下的 A 股 CSV
    返回格式与 simulate_market_data 完全一致
    """
    universe = {}
    all_dates = set()

    for fname in os.listdir(data_dir):
        if not fname.endswith(".csv"):
            continue

        stock_code = fname.replace(".csv", "")
        path = os.path.join(data_dir, fname)

        df = pd.read_csv(path)

        # 统一列名
        df.columns = [c.lower() for c in df.columns]
        # 处理日期列，支持 trade_date 或 date
        if "trade_date" in df.columns:
            df["date"] = pd.to_datetime(df["trade_date"])
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            raise ValueError(f"CSV 文件 {fname} 中缺少日期列 (trade_date 或 date)")
        df = df.sort_values("date").set_index("date")

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        if len(df) < 100:
            continue  # 数据太少直接扔掉

        # 技术指标
        dif, dea = compute_macd(df["close"])
        df["dif"] = dif
        df["dea"] = dea

        # 日收益
        df["ret"] = df["close"].pct_change(fill_method=None).fillna(0)

        universe[stock_code] = df
        all_dates.update(df.index)

    if not universe:
        raise RuntimeError("未成功加载任何股票 CSV")

    dates = pd.DatetimeIndex(sorted(all_dates))
    return universe, dates
