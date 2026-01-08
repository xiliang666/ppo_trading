import pandas as pd

def dynamic_select(
    universe: dict,
    date,
    lookback=60,
    topk=30
):
    scores = {}

    for name, df in universe.items():
        if date not in df.index:
            continue

        sub = df.loc[:date].tail(lookback)
        if len(sub) < lookback:
            continue

        golden = (
            (sub["dif"].shift(1) < sub["dea"].shift(1)) &
            (sub["dif"] > sub["dea"])
        )

        if golden.sum() < 3:
            continue

        future_ret = sub["ret"].rolling(5).sum()
        avg_ret = future_ret[golden].mean()
        prob = (future_ret[golden] > 0).mean()
        weight = 1.5 if sub["dif"].iloc[-1] < 0 else 1.0

        scores[name] = avg_ret * prob * weight

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [x[0] for x in ranked[:topk * 2]]

    returns = pd.DataFrame({
        s: universe[s].loc[:date]["ret"].tail(60)
        for s in candidates
    })

    corr = returns.corr().abs()
    selected = []

    for s in candidates:
        if len(selected) >= topk:
            break
        if all(corr.loc[s, t] < 0.8 for t in selected):
            selected.append(s)

    return selected[:topk]
