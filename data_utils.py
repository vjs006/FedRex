import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

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
    # Add interaction features
    if "age" in df.columns and "bmi" in df.columns:
        df["age_bmi"] = df["age"] * df["bmi"]
    if "pulse_pressure" in df.columns and "cholesterol" in df.columns:
        df["pp_chol"] = df["pulse_pressure"] * df["cholesterol"]
    # Add polynomial feature
    if "bmi" in df.columns:
        df["bmi2"] = df["bmi"] ** 2
    
    df = df.replace([np.inf, -np.inf], np.nan)

    # Impute missing values
    df = impute_missing(df)

    # Outlier removal
    df = remove_outliers(df)

    # Keep only relevant columns
    cols = [c for c in DEFAULT_FEATURES + ["pulse_pressure", "bmi", "age_bmi", "pp_chol", "bmi2", TARGET] if c in df.columns]
    df = df[cols].reset_index(drop=True)

    # One-hot encode categorical features
    cat_features = ["cholesterol", "gluc", "smoke", "alco", "active"]
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype(str)
    df = pd.get_dummies(df, columns=[c for c in cat_features if c in df.columns], drop_first=True)

    # Remove highly correlated features
    df = drop_high_corr(df, threshold=0.95)

    # Robust scaling for continuous features
    continuous_features = ["age", "height", "weight", "ap_hi", "ap_lo", "pulse_pressure", "bmi", "age_bmi", "pp_chol", "bmi2"]
    continuous_features = [c for c in continuous_features if c in df.columns]
    df = scale_features(df, continuous_features)

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            imp = SimpleImputer(strategy="mean")
        else:
            imp = SimpleImputer(strategy="most_frequent")
        df[col] = imp.fit_transform(df[[col]])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    # Absolute thresholds
    if "ap_hi" in df.columns:
        df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 250)]
    if "ap_lo" in df.columns:
        df = df[(df["ap_lo"] >= 50) & (df["ap_lo"] <= 150)]
    if "height" in df.columns:
        df = df[(df["height"] >= 140) & (df["height"] <= 210)]
    if "weight" in df.columns:
        df = df[(df["weight"] >= 40) & (df["weight"] <= 200)]
    if "bmi" in df.columns:
        df = df[(df["bmi"] >= 15) & (df["bmi"] <= 50)]

    # Z-score based removal for continuous features
    continuous_features = ["age", "height", "weight", "ap_hi", "ap_lo", "bmi"]
    for col in continuous_features:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[np.abs(df[col] - mean) <= z_thresh * std]
    return df.reset_index(drop=True)


def scale_features(df: pd.DataFrame, continuous_features: list) -> pd.DataFrame:
    scaler = RobustScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])
    df[continuous_features] = np.nan_to_num(df[continuous_features], nan=0.0, posinf=0.0, neginf=0.0)
    return df



def drop_high_corr(df: pd.DataFrame, threshold=0.95) -> pd.DataFrame:
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)


def split_by_client(df: pd.DataFrame, num_clients: int, cid: int) -> Tuple[pd.DataFrame, pd.Series]:
    n = len(df)
    shard_size = n // num_clients
    start = cid * shard_size
    end = (cid + 1) * shard_size if cid < num_clients - 1 else n
    shard = df.iloc[start:end].reset_index(drop=True)

    if len(shard) == 0:
        print(f"[Client {cid}] Empty dataset shard! Adjusting range manually.")
        start = max(0, start - shard_size)
        end = min(n, start + shard_size)
        shard = df.iloc[start:end].reset_index(drop=True)

    print(f"[Client {cid}] Data split: {len(shard)} samples")
    X = shard.drop(columns=[TARGET])
    y = shard[TARGET].astype(int)
    return X, y

