import os
import time
import json
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from data_utils import DEFAULT_FEATURES
import shap

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
        self.cid = cid
        self.num_clients = num_clients
        self.device = torch.device("cpu")
        # Load data shard
        df = load_and_engineer(data_path)
        X, y = split_by_client(df, num_clients, cid)
        if len(X) < 50:
            raise RuntimeError(f"Client {cid}: too few rows in shard ({len(X)}).")
        X = X[[c for c in DEFAULT_FEATURES if c in X.columns]].copy()
        self.train_loader, self.val_loader = make_loaders(X, y)
        # Model
        self.model = MLP(in_features=X.shape[1]).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    # Flower API
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.model.state_dict()
        for (k, _), np_val in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(np_val)
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        if parameters:  # set global params
            self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 1))
        for _ in range(epochs):
            train_one_epoch(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        loss, acc = evaluate(self.model, self.val_loader, self.device)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"val_loss": loss, "val_acc": acc}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.val_loader, self.device)

        shap_mean = 0.0
        try:
            shap_summary = self.compute_shap_summary()

            # Save locally to disk per round
            save_dir = f"shap_outputs/client_{self.cid}"
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"round_{config.get('server_round', 0)}.npy"),
                    shap_summary)

            # Optional: also keep JSON if you prefer
            with open(os.path.join(save_dir, f"round_{config.get('server_round', 0)}.json"), "w") as f:
                json.dump(shap_summary.tolist(), f)

            # Send only a scalar summary back to server
            shap_mean = float(np.mean(np.abs(shap_summary)))

        except Exception as e:
            print(f"[Client {self.cid}] SHAP failed: {e}")

        metrics = {
            "val_acc": float(acc),
            "shap_mean": shap_mean,   # ✅ scalar, safe for Flower
        }

        return float(loss), len(self.val_loader.dataset), metrics



    def compute_shap_summary(self, max_background=50, max_eval=200):
        # Use a small background + subset for speed
        X_val = next(iter(self.val_loader))[0]  # just grab the features
        bg = X_val[:max_background].to(self.device)
        eval_x = X_val[:max_eval].to(self.device)

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



def main():
    cid = int(os.environ.get("CLIENT_ID", "0"))
    num_clients = int(os.environ.get("NUM_CLIENTS", "3"))
    data_path = os.environ.get("DATA_PATH", "cardio_train.csv")

    client = TorchClient(cid, num_clients, data_path)

    # Start Flower client (connect to server at 0.0.0.0:8080 by default)
    server_addr = os.environ.get("SERVER_ADDRESS", "0.0.0.0:8080")
    fl.client.start_client(
        server_address=server_addr,
        client=client.to_client()
    )



if __name__ == "__main__":
    main()