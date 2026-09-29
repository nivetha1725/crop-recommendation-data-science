"""
Feature engineering used by BOTH train_model.py and app.py.

It lives in its own small file so the saved pipeline (models/*.pkl) can be
loaded by app.py. This guarantees training and prediction use the SAME code.
"""
import pandas as pd


def add_features(df):
    """Add simple soil-nutrient features (only if N, P and K columns exist)."""
    df = pd.DataFrame(df).copy()
    if all(col in df.columns for col in ["N", "P", "K"]):
        df["NPK_Total"] = df["N"] + df["P"] + df["K"]      # overall nutrient level
        df["NPK_Average"] = df["NPK_Total"] / 3            # average nutrient level
    return df
