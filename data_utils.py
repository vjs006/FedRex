import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler

# Features to use
DEFAULT_FEATURES = [
    "age", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]
TARGET = "cardio"


def load_and_engineer(csv_path: str) -> pd.DataFrame:
    sep = ";" if csv_path.endswith(".csv") else ","
    df = pd.read_csv(csv_path, sep=sep)

    # Age in years
    if "age" in df.columns and df["age"].max() > 200:
        df["age"] = (df["age"] / 365).astype(int)

    # Feature engineering
    if "ap_hi" in df.columns and "ap_lo" in df.columns:
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    if "height" in df.columns and "weight" in df.columns:
        h_m = df["height"] / 100.0
        df["bmi"] = df["weight"] / (h_m ** 2)

    # Outlier removal
    df = remove_outliers(df)

    # Keep only relevant columns
    cols = [c for c in DEFAULT_FEATURES + ["pulse_pressure", "bmi", TARGET] if c in df.columns]
    df = df[cols].dropna().reset_index(drop=True)

    # Scale continuous features
    continuous_features = ["age", "height", "weight", "ap_hi", "ap_lo", "pulse_pressure", "bmi"]
    continuous_features = [c for c in continuous_features if c in df.columns]
    df = scale_features(df, continuous_features)

    return df


def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    # Absolute thresholds (your original rules)
    df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 250)]
    df = df[(df["ap_lo"] >= 50) & (df["ap_lo"] <= 150)]
    df = df[(df["height"] >= 140) & (df["height"] <= 210)]
    df = df[(df["weight"] >= 40) & (df["weight"] <= 200)]
    if "bmi" in df.columns:
        df = df[(df["bmi"] >= 15) & (df["bmi"] <= 50)]

    # Z-score based removal for continuous features
    continuous_features = ["age", "height", "weight", "ap_hi", "ap_lo"]
    if "bmi" in df.columns:
        continuous_features.append("bmi")
    
    for col in continuous_features:
        mean = df[col].mean()
        std = df[col].std()
        df = df[np.abs(df[col] - mean) <= z_thresh * std]

    return df.reset_index(drop=True)



def scale_features(df: pd.DataFrame, continuous_features: list) -> pd.DataFrame:
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])
    return df


def split_by_client(df: pd.DataFrame, num_clients: int, cid: int) -> Tuple[pd.DataFrame, pd.Series]:
    idx = np.arange(len(df))
    mask = (idx % num_clients) == cid
    shard = df.loc[mask].reset_index(drop=True)
    X = shard.drop(columns=[TARGET])
    y = shard[TARGET].astype(int)
    return X, y
