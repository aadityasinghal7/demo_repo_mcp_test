import numpy as np
import pandas as pd


def weigthed_average(df: pd.DataFrame, weights: dict) -> pd.Series:
    total_weight = sum(weights.values())
    weighted_sum = sum(df[col] * weight for col, weight in weights.items())
    return weighted_sum / total_weight


def groupweightedaverage(df: pd.DataFrame, groupby: str, value: str, weights: str) -> pd.Series:
    weight_contrib = df[weights] * df[value]
    sum_weight_contrib = weight_contrib.groupby(df[groupby]).sum()
    sum_weights = df[weights].groupby(df[groupby]).sum()
    return sum_weight_contrib / sum_weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="weighted_avg", choices=["weighted_avg", "group"])
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    N_DATA = 100
    data = pd.DataFrame(
        {
            "brand": rng.choice(["A", "B", "C"], size=N_DATA),
            "A": rng.standard_normal(N_DATA),
            "B": rng.standard_normal(N_DATA) * 50 + 20,
            "C": rng.standard_normal(N_DATA) * 100 + 1000,
        }
    )
    if args.style == "weighted_avg":
        weighted_avg = weigthed_average(data, weights={"A": 0.2, "B": 0.3, "C": 0.5})
    elif args.style == "group":
        weighted_avg = groupweightedaverage(data, groupby="brand", value="A", weights="B")
    print(weighted_avg)
