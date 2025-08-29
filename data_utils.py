import os
import pandas as pd
import numpy as np
from typing import Tuple

DEFAULT_FEATURES = [
    "age", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]
TARGET = "cardio"


def load_and_engineer(csv_path: str) -> pd.DataFrame:
    # The CSV from the cardio dataset typically uses ';' as separator
    sep = ";" if csv_path.endswith(".csv") else ","
    df = pd.read_csv(csv_path, sep=sep)

    # Basic cleaning/engineering aligned with your prior script
    # - Convert age (days) to years (int)
    if "age" in df.columns and df["age"].max() > 200:  # likely days
        df["age"] = (df["age"] / 365).astype(int)

    # Add simple derived features that help tabular models
    if "ap_hi" in df.columns and "ap_lo" in df.columns:
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    if "height" in df.columns and "weight" in df.columns:
        h_m = df["height"] / 100.0
        df["bmi"] = df["weight"] / (h_m ** 2)

    # Keep only columns present
    cols = [c for c in (
        DEFAULT_FEATURES + ["pulse_pressure", "bmi", TARGET]
    ) if c in df.columns]
    df = df[cols].dropna().reset_index(drop=True)
    return df


def split_by_client(df: pd.DataFrame, num_clients: int, cid: int) -> Tuple[pd.DataFrame, pd.Series]:
    # Deterministic row-wise split: shard i gets rows where (index % num_clients) == i
    idx = np.arange(len(df))
    mask = (idx % num_clients) == cid
    shard = df.loc[mask].reset_index(drop=True)
    X = shard.drop(columns=[TARGET])
    y = shard[TARGET].astype(int)
    return X, y
