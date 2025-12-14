import os
import json
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_utils import load_and_engineer, split_by_client


def train_lgbm_on_shard(data_path: str, cid: int, num_clients: int, outdir: str = "lgbm_runs"):
    df = load_and_engineer(data_path)
    X, y = split_by_client(df, num_clients, cid)

    # 80/20 split within the shard (same as torch client)
    n = len(X)
    n_train = int(0.8 * n)
    X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_val = y.iloc[:n_train], y.iloc[n_train:]

    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=600, learning_rate=0.05,
        num_leaves=63, subsample=0.9, colsample_bytree=0.8, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "acc": float(accuracy_score(y_val, pred)),
        "prec": float(precision_score(y_val, pred, zero_division=0)),
        "rec": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
    }

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(outdir / f"client{cid}_lgbm.txt"))
    (outdir / f"client{cid}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Client {cid} LGBM metrics: {metrics}")


if __name__ == "__main__":
    # Example: CLIENT_ID=0 NUM_CLIENTS=3 DATA_PATH=cardio_train.csv python lgbm_local_baseline.py
    cid = int(os.environ.get("CLIENT_ID", "0"))
    num_clients = int(os.environ.get("NUM_CLIENTS", "3"))
    data_path = os.environ.get("DATA_PATH", "cardio_train.csv")
    train_lgbm_on_shard(data_path, cid, num_clients)

