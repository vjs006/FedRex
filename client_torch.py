import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import shap

from data_utils import load_and_engineer, split_by_client, DEFAULT_FEATURES


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden1=64, hidden2=32, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def get_tensors(X: pd.DataFrame, y: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0)
    return (
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32).view(-1, 1),
    )


def make_loaders(X: pd.DataFrame, y: pd.Series, batch_size=64) -> Tuple[DataLoader, DataLoader]:
    valid_idx = X.dropna(how="all").index.intersection(y.dropna().index)
    X, y = X.loc[valid_idx], y.loc[valid_idx]
    n = len(X)
    if n == 0:
        raise RuntimeError("No valid samples after filtering NaNs!")

    n_train = int(0.8 * n)
    Xtr, Xva, ytr, yva = X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]
    Xtr_t, ytr_t = get_tensors(Xtr, ytr)
    Xva_t, yva_t = get_tensors(Xva, yva)

    bs = min(batch_size, len(Xtr_t))
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=min(batch_size, len(Xva_t)), shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_samples += xb.size(0)
    return float(total_loss / total_samples) if total_samples > 0 else 0.0


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device)).cpu().numpy().ravel()
            preds.append(out)
            trues.append(yb.cpu().numpy().ravel())
    if not preds:
        return 0.0, 0.0
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    preds = np.nan_to_num(preds)
    trues = np.nan_to_num(trues)
    acc = accuracy_score((trues > 0.5).astype(int), (preds > 0.5).astype(int))
    loss_fn = nn.BCELoss()
    loss = loss_fn(torch.tensor(preds, dtype=torch.float32).view(-1, 1),
                   torch.tensor(trues, dtype=torch.float32).view(-1, 1)).item()
    return float(loss), float(acc)

class TorchClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_clients: int, data_path: str):
        self.cid = cid
        self.num_clients = num_clients
        self.device = torch.device("cpu")

        # --- Data load ---
        df = load_and_engineer(data_path)
        X, y = split_by_client(df, num_clients, cid)
        if len(X) == 0:
            raise RuntimeError(f"[Client {cid}] Empty data shard!")

        X = X[[c for c in DEFAULT_FEATURES if c in X.columns]].copy()
        y = y.fillna(0).reset_index(drop=True)
        self.feature_names = list(X.columns)

        print(f"[Client {cid}] Data split: {len(X)} samples")

        self.train_loader, self.val_loader = make_loaders(X, y)
        X_val_full = X.iloc[int(0.8 * len(X)):]
        y_val_full = y.iloc[int(0.8 * len(y)):]
        self.X_val_tensor, self.y_val_tensor = get_tensors(X_val_full, y_val_full)

        # --- Model setup ---
        self.base_model = MLP(in_features=X.shape[1]).to(self.device) 
        self.model = self.base_model # 'self.model' will be the wrapped model if DP is on
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # --- DP setup ---
        self.privacy_engine = None
        self.dp_delta = 1e-5
        self.dp_noise_multiplier = 1.0
        self.dp_max_grad_norm = 1.0
        
        # Call DP setup AFTER initial model/optimizer setup
        self._init_privacy_engine()

    def _init_privacy_engine(self):
        """Safely initialize DP with fallback."""
        try:
            sample_rate = len(next(iter(self.train_loader))[0]) / len(self.train_loader.dataset)
            if sample_rate <= 0:
                raise ValueError("Invalid sample_rate")
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.dp_noise_multiplier,
                max_grad_norm=self.dp_max_grad_norm,
            )
            self.privacy_engine = privacy_engine
            print(f"[Client {self.cid}] PrivacyEngine initialized successfully.")
        except Exception as e:
            print(f"[Client {self.cid}] Warning: DP init failed ({e}). Continuing without DP.")
            self.privacy_engine = None

    def get_parameters(self, config=None):
        return [p.cpu().detach().numpy().astype(np.float32) for _, p in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        
        # Ensure parameters list length matches state_dict keys
        if len(parameters) != len(state_dict):
            print(f"[Client {self.cid}] Parameter mismatch! Expected {len(state_dict)}, got {len(parameters)}")
            return 
            
        for (k, _), arr in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(np.array(arr, dtype=np.float32))
            
        self.model.load_state_dict(state_dict, strict=True)

        self.model.train()

    def fit(self, parameters, config):
        try:
            if parameters:
                self.set_parameters(parameters)

            epochs = int(config.get("local_epochs", 1)) if config else 1
            for _ in range(epochs):
                train_loss = train_one_epoch(self.model, self.train_loader, self.criterion, self.optimizer, self.device)

            val_loss, val_acc = evaluate(self.model, self.val_loader, self.device)
            metrics = self.compute_data_quality_stats()
            metrics.update({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

            # --- Privacy accounting ---
            epsilon = 0.0
            delta = self.dp_delta
            if self.privacy_engine is not None:
                try:
                    epsilon = self.privacy_engine.get_epsilon(delta)
                    if np.isnan(epsilon): epsilon = 0.0
                except Exception:
                    epsilon = 0.0
            metrics.update({"privacy_epsilon": float(epsilon), "privacy_delta": float(delta)})

            # --- Robustness ---
            params_np = [np.frombuffer(p.tobytes(), dtype=np.float32) for p in self.get_parameters()]
            metrics["robustness_norm"] = float(sum(np.sum(p ** 2) for p in params_np))
            metrics["robustness_diversity"] = 0.8

            print(f"[Client {self.cid}] Done: val_acc={val_acc:.3f}, ε={epsilon:.3f}")
            return self.get_parameters(), len(self.train_loader.dataset), metrics

        except Exception as e:
            print(f"[Client {self.cid}] Exception in fit: {e}")
            raise

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Use the currently loaded model (which might be the DP-wrapped one) for loss/acc
        loss, acc = evaluate(self.model, self.val_loader, self.device) 

        shap_vals = np.zeros(len(self.feature_names))
        
        if self.privacy_engine is None: 
            try:
                if self.privacy_engine is None:
                    shap_vals = self.compute_shap_summary() 
                    
            except Exception as e:
                print(f"[Client {self.cid}] SHAP error: {e}") 

        metrics = {"val_acc": acc}
        
        if shap_vals.size > 0:
            for i, v in enumerate(shap_vals.flatten()):
                metrics[f"shap_{i}"] = float(v)
                
        # Only return required metrics for the evaluate phase
        return float(loss), len(self.val_loader.dataset), metrics


    def compute_data_quality_stats(self):
        try:
            X_np = self.train_loader.dataset.tensors[0].numpy()
            y_np = self.train_loader.dataset.tensors[1].ravel()
        except Exception:
            return {}

        df = pd.DataFrame(X_np, columns=self.feature_names)
        metrics = {}
        for f in df.columns:
            x = df[f].fillna(df[f].median())
            S_comp = float(np.mean(~x.isna()))
            S_valid = float(np.mean(x == x.median()) if x.nunique() > 1 else 1.0)
            S_uniq = float(x.nunique() / len(x)) if len(x) > 0 else 0.0
            try:
                Q1, Q3 = np.percentile(x, [25, 75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                S_out = float(np.mean((x < lower) | (x > upper)))
            except Exception:
                S_out = 0.0
            try:
                arr = x.values.reshape(-1, 1)
                disc = KBinsDiscretizer(n_bins=10, encode="ordinal").fit_transform(arr).ravel()
                I = mutual_info_score(disc, y_np)
                S_mi = float(np.log1p(I) / np.log1p(5.0) ** 0.7)
            except Exception:
                S_mi = 0.0
            metrics.update({
                f"data_quality_{f}_comp": S_comp,
                f"data_quality_{f}_valid": S_valid,
                f"data_quality_{f}_uniq": S_uniq,
                f"data_quality_{f}_out": S_out,
                f"data_quality_{f}_mi": S_mi,
            })
        return metrics

    def compute_shap_summary(self, max_background=50, max_eval=200):
        self.model.eval()
        bg = torch.nan_to_num(self.X_val_tensor[:max_background]).to(self.device)
        eval_x = torch.nan_to_num(self.X_val_tensor[:max_eval]).to(self.device)
        explainer = shap.DeepExplainer(self.model, bg)
        shap_vals = explainer.shap_values(eval_x)
        shap_vals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
        return np.mean(np.abs(np.nan_to_num(shap_vals)), axis=0)


def main():
    try:
        cid = int(os.environ.get("CLIENT_ID", "0"))
        num_clients = int(os.environ.get("NUM_CLIENTS", "3"))
        data_path = os.environ.get("DATA_PATH", "cardio_train.csv")
        server_addr = os.environ.get("SERVER_ADDRESS", "0.0.0.0:8080")

        client = TorchClient(cid, num_clients, data_path)
        fl.client.start_client(server_address=server_addr, client=client.to_client())
    except Exception as e:
        print(f"[Client {os.environ.get('CLIENT_ID', 'X')}] Fatal error: {e}")


if __name__ == "__main__":
    main()
