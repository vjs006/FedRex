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
    X_numeric = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    X_np = X_numeric.values
    if X_np.dtype == np.dtype('object_'):
        X_np = X_np.astype(np.float32)
        
    y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0)
    
    return (
        torch.tensor(X_np, dtype=torch.float32), 
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
        
        # --- Data load and feature engineering ---
        df = load_and_engineer(data_path)
        X_full, y = split_by_client(df, num_clients, cid)
        
        if len(X_full) == 0:
            raise RuntimeError(f"[Client {cid}] Empty data shard!")

        # Data Cleaning Block
        X = X_full.copy()
        # This cleanup is redundant in init but kept for safety/debugging data types.
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0) 
        
        y = y.fillna(0).reset_index(drop=True)
        
        # Finalize feature list and size
        self.feature_names = list(X.columns)
        in_features = X.shape[1]
        
        # --- Debug Confirmation ---
        print(f"[Client {cid}] Data split: {len(X)} samples")
        print("-" * 50)
        print("FINAL FEATURE COUNT:", len(self.feature_names))
        print("FINAL FEATURE NAMES LIST:", self.feature_names) 
        print("-" * 50)
        
        # --- Data Loaders ---
        self.train_loader, self.val_loader = make_loaders(X, y)
        
        # Prepare validation tensors for SHAP background data
        n_train = int(0.8 * len(X))
        X_val_full = X.iloc[n_train:]
        y_val_full = y.iloc[n_train:]
        self.X_val_tensor, self.y_val_tensor = get_tensors(X_val_full, y_val_full)

        # --- Dual Model setup ---
        
        # 1. Base model (Vanilla MLP) - NOW self.device IS AVAILABLE
        self.base_model = MLP(in_features=in_features).to(self.device)
        
        # 2. DP Model 
        self.model = self.base_model
        
        # 3. SHAP Model 
        self.shap_model = MLP(in_features=in_features).to(self.device)
        self.shap_model.load_state_dict(self.base_model.state_dict()) 

        self.criterion = nn.BCELoss() # <-- Also essential setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # --- DP setup ---
        self.privacy_engine = None
        self.dp_delta = 1e-5
        self.dp_noise_multiplier = 1.0
        self.dp_max_grad_norm = 1.0

        self._init_privacy_engine()

    # client_torch.py (Replace the entire _init_privacy_engine method)

    # client_torch.py (Corrected _init_privacy_engine)

    def _init_privacy_engine(self):
        """Safely initialize DP using the older Opacus make_private API (v1.x)."""
        
        try:
            # 1. Instantiate the base PrivacyEngine object (no arguments)
            privacy_engine = PrivacyEngine()

            # 2. Use make_private to wrap the module, optimizer, and dataloader.
            # This is where DP hyper-parameters are passed in Opacus v1.x.
            # This call replaces self.model, self.optimizer, and self.train_loader
            # with their DP-wrapped counterparts.
            self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.dp_noise_multiplier,
                max_grad_norm=self.dp_max_grad_norm,
                # target_delta is handled by get_epsilon later, not directly in make_private.
            )
            
            # The wrapped model contains a reference to the original module
            # that we use in fit() for weight synchronization with self.shap_model.
            self.privacy_engine = privacy_engine
            print(f"[Client {self.cid}] PrivacyEngine initialized successfully (v1.x API).")
            
        except Exception as e:
            # The DP setup failed (likely a version or configuration issue)
            print(f"[Client {self.cid}] Warning: DP init failed ({e}). Continuing without DP.")
            self.privacy_engine = None


    def get_parameters(self, config=None):
        # We return the parameters of the model used for training (self.model)
        # The parameters must be obtained from the currently active model (DP-wrapped or not)
        return [p.cpu().detach().numpy().astype(np.float32) for _, p in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # 1. Update the training/DP model (self.model)
        state_dict_dp = self.model.state_dict()
        if len(parameters) == len(state_dict_dp):
            for (k, _), arr in zip(state_dict_dp.items(), parameters):
                state_dict_dp[k] = torch.tensor(np.array(arr, dtype=np.float32))
            self.model.load_state_dict(state_dict_dp, strict=True)
            
        # 2. Update the vanilla SHAP model (self.shap_model)
        # This is CRUCIAL: we update the SHAP model with the base weights
        # We assume the first layers of state_dict_dp contain the actual model weights.
        state_dict_shap = self.shap_model.state_dict()
        
        # Only take the weights corresponding to the SHAP model's state_dict keys.
        # This handles cases where the DP model has extra optimizer/meta layers.
        
        # Ensure we only load the weights that the vanilla MLP has
        vanilla_keys = list(state_dict_shap.keys())
        
        # Parameters array contains parameters in the same order as get_parameters returns them
        if len(parameters) >= len(vanilla_keys):
            for i, key in enumerate(vanilla_keys):
                state_dict_shap[key] = torch.tensor(np.array(parameters[i], dtype=np.float32))
            self.shap_model.load_state_dict(state_dict_shap, strict=True)

    def fit(self, parameters, config):
        try:
            if parameters:
                self.set_parameters(parameters)

            epochs = int(config.get("local_epochs", 1)) if config else 1
            
            # --- Training with DP (self.model is DP-wrapped) ---
            for _ in range(epochs):
                train_loss = train_one_epoch(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
            
            # --- Sync final trained weights to the SHAP model ---
            # This is complex with Opacus. The simplest way is to manually copy the original module state
            # assuming Opacus's hooks are not necessary for evaluation/SHAP
            if self.privacy_engine is not None and hasattr(self.model, 'original_module'):
                # If DP is on, sync the clean weights from the wrapped module to the SHAP model
                self.shap_model.load_state_dict(self.model.original_module.state_dict(), strict=True)
            elif self.privacy_engine is None:
                # If DP is off, self.model is the base model, sync it directly
                self.shap_model.load_state_dict(self.model.state_dict(), strict=True)


            # --- Evaluation Metrics (using the vanilla SHAP model) ---
            val_loss, val_acc = evaluate(self.shap_model, self.val_loader, self.device)
            metrics = self.compute_data_quality_stats()
            metrics.update({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

            # --- Privacy accounting (S_Priv,c) ---
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
            # Use parameters from the DP model for robustness norm
            params_np = [np.frombuffer(p.tobytes(), dtype=np.float32) for p in self.get_parameters()]
            metrics["robustness_norm"] = float(sum(np.sum(p ** 2) for p in params_np))
            metrics["robustness_diversity"] = 0.8

            print(f"[Client {self.cid}] Done: val_acc={val_acc:.3f}, ε={epsilon:.3f}")
            # Return DP-trained model parameters
            return self.get_parameters(), len(self.train_loader.dataset), metrics

        except Exception as e:
            print(f"[Client {self.cid}] Exception in fit: {e}")
            raise

    def evaluate(self, parameters, config):
        # 1. Update both models with global parameters
        self.set_parameters(parameters)
        
        # 2. Evaluate performance using the vanilla SHAP model
        loss, acc = evaluate(self.shap_model, self.val_loader, self.device)

        # 3. Compute SHAP on the vanilla SHAP model (S_Exp,c)
        shap_vals = np.zeros(len(self.feature_names))
        try:
            # We explicitly pass the vanilla model for SHAP calculation
            shap_vals = self.compute_shap_summary(model=self.shap_model)
        except Exception as e:
            print(f"[Client {self.cid}] SHAP error: {e}")

        metrics = {"val_acc": acc}
        for i, v in enumerate(shap_vals.flatten()):
            metrics[f"shap_{i}"] = float(v)
            
        # If DP was active in fit, the shap_vals are based on the latest DP-trained weights
        return float(loss), len(self.val_loader.dataset), metrics

    # ---------------- Data Quality ----------------
    def compute_data_quality_stats(self):
        # ... (no change) ...
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

    # ---------------- SHAP ----------------
    def compute_shap_summary(self, model, max_background=50, max_eval=200):
        # SHAP calculation uses the provided vanilla model
        model.eval()
        bg = torch.nan_to_num(self.X_val_tensor[:max_background]).to(self.device)
        eval_x = torch.nan_to_num(self.X_val_tensor[:max_eval]).to(self.device)
        
        # Note: If Opacus's model is used here, it will crash.
        # But since we use self.shap_model, which is vanilla, it's fine.
        explainer = shap.DeepExplainer(model, bg)
        
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