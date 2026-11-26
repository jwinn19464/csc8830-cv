"""Module 2 - validation: compare measured vs. true dimensions (20 measurements).

CSV columns: label, true, measured   (same unit, e.g. cm)
    python -m module2.validate --csv module2/measurements.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def add_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error"] = df["measured"] - df["true"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = 100 * df["abs_error"] / df["true"]
    return df


def stats(df: pd.DataFrame) -> dict:
    e = df["error"].to_numpy(float)
    n = len(e)
    sd = float(e.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    r = float(np.corrcoef(df["true"], df["measured"])[0, 1]) if n > 2 else float("nan")
    return {
        "n": n,
        "bias (mean error)": float(e.mean()),
        "std of error": sd,
        "95% CI of bias": (float(e.mean() - half), float(e.mean() + half)),
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "max abs error": float(np.abs(e).max()),
        "mean % error": float(df["pct_error"].mean()),
        "median % error": float(df["pct_error"].median()),
        "max % error": float(df["pct_error"].max()),
        "Pearson r": r,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    df = add_errors(pd.read_csv(a.csv))
    for k, v in stats(df).items():
        print(f"{k:>18}: {v}")


if __name__ == "__main__":
    main()
