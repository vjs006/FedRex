import os
import time
import json
from typing import Dict, List, Tuple
import io

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from data_utils import DEFAULT_FEATURES
from flwr.common import ndarrays_to_parameters
import shap
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, mutual_info_score

from data_utils import load_and_engineer, split_by_client


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden1=64, hidden2=32, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_features),
            nn.Sigmoid(),  # or remove sigmoid if using BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)


def get_tensors(X, y):
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    return X, y


def make_loaders(X, y, batch_size=64):
    # simple 80/20 split per client
    n = len(X)
    n_train = int(0.8 * n)
    X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_val = y.iloc[:n_train], y.iloc[n_train:]
    Xtr_t, ytr_t = get_tensors(X_train, y_train)
    Xva_t, yva_t = get_tensors(X_val, y_val)
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * xb.size(0)
    return loss_sum / len(loader.dataset)

def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy().ravel()
            preds.append(out)
            trues.append(yb.numpy().ravel())
    p = (np.concatenate(preds) >= 0.5).astype(int)
    t = (np.concatenate(trues)).astype(int)
    acc = accuracy_score(t, p)
    loss = float(nn.BCELoss()(torch.tensor(p, dtype=torch.float32).view(-1,1), torch.tensor(t, dtype=torch.float32).view(-1,1)))
    return loss, acc


class TorchClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_clients: int, data_path: str):
        df = load_and_engineer(data_path)
        X, y = split_by_client(df, num_clients, cid)
        # Only keep DEFAULT_FEATURES for training and stats
        X = X[[c for c in DEFAULT_FEATURES if c in X.columns]].copy()
        self.feature_names = list(X.columns)  # <-- Set after column selection!
        self.cid = cid
        self.num_clients = num_clients
        self.device = torch.device("cpu")

        if len(X) < 50:
            raise RuntimeError(f"Client {cid}: too few rows in shard ({len(X)}).")
        self.train_loader, self.val_loader = make_loaders(X, y)

        # Store full validation features as a tensor for SHAP
        X_val_full = X.iloc[int(0.8*len(X)):]  # matches the val split
        y_val_full = y.iloc[int(0.8*len(y)):]
        self.X_val_tensor, self.y_val_tensor = get_tensors(X_val_full, y_val_full)

        # Model
        self.model = MLP(in_features=X.shape[1]).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    # Flower API
    def get_parameters(self, config):
        return [val.cpu().numpy().astype(np.float32) for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.model.state_dict()
        for (k, _), arr in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(arr).clone().detach()
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        try:
            self.set_parameters(parameters)
            # Train for n epochs
            for _ in range(1):  # or config["epochs"] if passed
                train_loss = train_one_epoch(
                    self.model, self.train_loader, 
                    self.criterion, self.optimizer, self.device
                )
        
            # Compute various metrics
            val_acc, val_loss = evaluate(self.model, self.val_loader, self.device)
            
            # Get all stats
            metrics = self.compute_data_quality_stats()  # Now returns flattened metrics
            
            # Add other metrics
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)
            metrics["val_acc"] = float(val_acc)
            
            # Add privacy metrics
            metrics["privacy_epsilon"] = 0.1  # example
            metrics["privacy_delta"] = 1e-5   # example
            
            # Add robustness metrics
            # FIX: convert to float64 before squaring
            params_np = [np.frombuffer(p, dtype=np.float32).astype(np.float64) for p in parameters]
            metrics["robustness_norm"] = float(np.sum([np.sum(p**2) for p in params_np]))
            metrics["robustness_diversity"] = 0.8  # example
            
            return self.get_parameters({}), len(self.train_loader.dataset), metrics
        except Exception as e:
            print(f"[Client {self.cid}] Exception in fit: {e}")
            raise

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.val_loader, self.device)

        shap_vec = self.compute_shap_summary()  # full vector
        shap_flat = np.array(shap_vec).flatten().tolist()
        metrics = {
            "val_acc": float(acc),
            "shap_vec": shap_vec.tolist(),  # send full vector
        }

        # Flatten shap_vec to a 1D list of floats
        shap_flat = np.array(shap_vec).flatten().tolist()
        metrics = {
            "val_acc": float(acc),
        }
        for i, v in enumerate(shap_flat):
            metrics[f"shap_{i}"] = float(v)

        return float(loss), len(self.val_loader.dataset), metrics



    def compute_shap_summary(self, max_background=50, max_eval=200):
        # Use the stored full validation tensor
        bg = self.X_val_tensor[:max_background].to(self.device)
        eval_x = self.X_val_tensor[:max_eval].to(self.device)

        self.model.eval()
        # Do NOT use torch.no_grad() here; SHAP needs gradients
        explainer = shap.DeepExplainer(self.model, bg)
        shap_vals = explainer.shap_values(eval_x)
        
        # shap returns a list for multi-output; for binary just pick first
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = shap_vals.cpu().numpy() if torch.is_tensor(shap_vals) else np.array(shap_vals)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)  # shape (n_features,)
        return mean_abs

    def compute_data_quality_stats(self):
        df = self.train_loader.dataset.tensors[0].numpy()
        df = pd.DataFrame(df, columns=self.feature_names)
        y = self.train_loader.dataset.tensors[1].numpy().ravel()

        metrics = {}
        for f in df.columns:
            x = df[f]

            # Completeness
            S_comp = float(np.mean(~pd.isnull(x)))

            # Validity (dummy check, since no per-feature GT available)
            # If categorical, check frequency of most common value
            if len(np.unique(x)) > 1:
                S_valid = float(np.mean(x == np.median(x)))
            else:
                S_valid = 1.0

            # Uniqueness
            S_uniq = float(len(np.unique(x)) / len(x))

            # Outliers (IQR rule)
            Q1, Q3 = np.percentile(x, 25), np.percentile(x, 75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            S_out = float(np.mean((x < lower) | (x > upper)))

            # Mutual Information with labels
            try:
                if np.issubdtype(x.dtype, np.floating):
                    x_disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform') \
                                .fit_transform(x.reshape(-1, 1)).ravel()
                else:
                    x_disc = x
                I = mutual_info_score(x_disc, y)
                I_max = 5.0
                xi = 0.7
                S_mi = float(np.log(1 + I) / (np.log(1 + I_max) ** xi))
            except Exception:
                S_mi = 0.0

            # Flattened naming: data_quality_<feature>_<stat>
            metrics[f"data_quality_{f}_comp"] = S_comp
            metrics[f"data_quality_{f}_valid"] = S_valid
            metrics[f"data_quality_{f}_uniq"] = S_uniq
            metrics[f"data_quality_{f}_out"] = S_out
            metrics[f"data_quality_{f}_mi"] = S_mi

        return metrics


    def get_privacy_info(self):
        # Example: DP parameters (epsilon, delta)
        epsilon = 0.5
        delta = 1e-5
        return {"epsilon": epsilon, "delta": delta}

    def get_robustness_stats(self, parameters):
        # Convert bytes to numpy arrays
        params_np = [np.frombuffer(p, dtype=np.float32) for p in parameters]
        update = np.concatenate([p.ravel() for p in params_np])
        norm = float(np.linalg.norm(update))
        diversity = 0.8
        return {"norm": norm, "diversity": diversity}

def main():
    try:
        cid = int(os.environ.get("CLIENT_ID", "0"))
        num_clients = int(os.environ.get("NUM_CLIENTS", "3"))
        data_path = os.environ.get("DATA_PATH", "cardio_train.csv")

        client = TorchClient(cid, num_clients, data_path)

        server_addr = os.environ.get("SERVER_ADDRESS", "0.0.0.0:8080")
        fl.client.start_client(
            server_address=server_addr,
            client=client.to_client()
        )
    except Exception as e:
        print(f"[Client {os.environ.get('CLIENT_ID', 'X')}] Fatal error: {e}")

if __name__ == "__main__":
    main()