import json
import os
import numpy as np
import flwr as fl
import pandas as pd
from typing import List, Tuple, Dict, Any
import torch
from flwr.common import Parameters
import io
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


def _decode_bytes_to_ndarray(t: bytes) -> np.ndarray:
    if isinstance(t, (bytes, np.bytes_)):
        buf = io.BytesIO(t)
        return np.load(buf, allow_pickle=False)
    elif isinstance(t, np.ndarray):
        return t
    else:
        raise TypeError(f"Unsupported parameter type: {type(t)}")


def weighted_average(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    results: Dict[str, float] = {}

    for num_examples, m in metrics:
        for k, v in m.items():
            results[k] = results.get(k, 0.0) + v * num_examples

    for k in results:
        results[k] /= total_examples
    return results

class FedReX(fl.server.strategy.FedAvg):
    def __init__(self, alpha_dict=None, **kwargs):
        super().__init__(**kwargs)
        # Trust score weights (sum to 1)
        self.alpha_dict = alpha_dict or {
            "DQ": 0.2, "Perf": 0.2, "Exp": 0.2, "His": 0.15, "Priv": 0.1, "Rob": 0.15
        }
        self.historical_scores = {}  # {cid: score}
        self.global_shap = None      # reference SHAP vector (update each round)
        self.global_shap_history = {}  # Store all rounds' SHAP

    # --- Trust score sub-components ---
    def score_data_quality(self, metrics, data_stats):
        # Use reported stats from client
        # For each feature, compute S_{c,f} as per formula
        alpha1, alpha2, alpha3, alpha4, alpha5 = 0.2, 0.2, 0.2, 0.2, 0.2
        kappa = 0.1
        Z = alpha1 + alpha2 + alpha3 + alpha4 + alpha5 + kappa
        S_feat = []
        for f, v in data_stats.items():
            S_comp = v["comp"]
            S_valid = v["valid"]
            S_uniq = v["uniq"]
            S_out = v["out"]
            S_mi = v["mi"]
            S_f = (alpha1 * S_comp + alpha2 * S_valid + alpha3 * S_uniq + alpha4 * S_out + alpha5 * S_mi + kappa * S_comp * S_valid) / Z
            S_feat.append(S_f)
        # Aggregate to client-level DQ
        w_feat = np.ones(len(S_feat)) / len(S_feat)
        beta = 1.2
        zeta = 0.5
        S_prod = np.prod([s ** w for s, w in zip(S_feat, w_feat)])
        S_var = np.var(S_feat)
        S_DQ_c = (S_prod ** beta) / (1 + zeta * S_var)
        return float(S_DQ_c)

    def aggregate_evaluate(self, rnd, results, failures):
        # Call the default FedAvg aggregation for loss/acc etc.
        agg_results = super().aggregate_evaluate(rnd, results, failures)
        
        # --- SHAP aggregation to update self.global_shap ---
        shap_vecs, weights = [], []
        for _, eval_res in results:
            metrics = eval_res.metrics
            # Only consider clients that successfully computed SHAP (i.e., when DP was off)
            shap_metrics = {k: v for k, v in metrics.items() if k.startswith("shap_")}
            if shap_metrics:
                # Reconstruct the vector based on index
                max_idx = max(int(k.split("_")[1]) for k in shap_metrics)
                shap_vec = np.zeros(max_idx + 1)
                for k, v in shap_metrics.items():
                    idx = int(k.split("_")[1])
                    if idx <= max_idx:
                        shap_vec[idx] = v
                
                shap_vecs.append(shap_vec)
                weights.append(eval_res.num_examples)

        if shap_vecs:
            # Pad vectors if they have different lengths (due to varying features per client)
            max_len = max(len(v) for v in shap_vecs)
            padded_shap_vecs = []
            for v in shap_vecs:
                padded_v = np.pad(v, (0, max_len - len(v)), 'constant', constant_values=0)
                padded_shap_vecs.append(padded_v)
                
            shap_vecs = np.vstack(padded_shap_vecs)
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            self.global_shap = np.average(shap_vecs, axis=0, weights=weights)
            self.global_shap_history[rnd] = self.global_shap
            
            # Print top-5 features (as in FedAvgWithSHAP)
            if self.global_shap.size > 0:
                top_idx = np.argsort(self.global_shap)[::-1][:5]
                print(f"[Round {rnd}] Global SHAP top-5 features (by index):")
                for i in top_idx:
                    print(f"  Feature {i}: {self.global_shap[i]:.6f}")

        return agg_results

    def score_performance(self, metrics, global_metrics):
        # Balanced accuracy improvement
        # Assume metrics["val_acc"] is balanced accuracy
        Perf_c = metrics.get("val_acc", 0.5)
        Perf_global = global_metrics.get("val_acc", 0.5) if global_metrics else 0.5
        Delta_c = max(0, Perf_c - Perf_global)
        lambda_ = 0.05
        S_Perf_c = Delta_c / (Delta_c + lambda_)
        return float(S_Perf_c)

    def score_explanation(self, shap_vec, global_shap):
        # L1 and cosine similarity
        if shap_vec is None or global_shap is None:
            return 1.0
        shap_vec = np.array(shap_vec)
        global_shap = np.array(global_shap)
        eps = 1e-8
        S_L1 = 1 - np.sum(np.abs(shap_vec - global_shap)) / (np.sum(np.abs(global_shap)) + eps)
        S_cos = np.dot(shap_vec, global_shap) / ((np.linalg.norm(shap_vec) + eps) * (np.linalg.norm(global_shap) + eps))
        S_Exp_c = 0.5 * (S_L1 + S_cos)
        return float(S_Exp_c)

    def score_history(self, cid):
        prev = self.historical_scores.get(cid, 1.0)
        beta = 0.8
        # For demo, just return previous score
        return prev

    def score_privacy(self, privacy_info):
        epsilon = privacy_info.get("epsilon", 0.5)
        delta = privacy_info.get("delta", 1e-5)
        alpha, eta = 0.2, 0.5
        S_Priv_c = np.exp(-alpha * epsilon) * ((1 - delta) ** eta)
        return float(S_Priv_c)

    def score_robustness(self, update, ref_update, stats):
        def params_to_vec(params):
            if params is None:
                return np.zeros(1, dtype=np.float64)

            if isinstance(params, Parameters):
                # Decode bytes -> np arrays
                arrays = parameters_to_ndarrays(params)
                return np.concatenate([arr.flatten() for arr in arrays])

            # fallback if already array-like
            try:
                return np.array(params).flatten()
            except Exception:
                return np.zeros(1, dtype=np.float64)

        update_vec = params_to_vec(update)
        ref_vec = params_to_vec(ref_update)

        update_vec = np.nan_to_num(update_vec, nan=0.0, posinf=0.0, neginf=0.0)
        ref_vec = np.nan_to_num(ref_vec, nan=0.0, posinf=0.0, neginf=0.0)

        norm_update = np.linalg.norm(update_vec)
        norm_ref = np.linalg.norm(ref_vec)
        if norm_update == 0 or norm_ref == 0:
            return 0.0

        return 0.5 * (1 + np.dot(update_vec, ref_vec) / (norm_update * norm_ref))


    def compute_trust_score(self, cid, metrics, data_stats, shap_vec, privacy_info, update, ref_update, stats, global_metrics, global_shap):
        if not hasattr(self, "global_shap") or self.global_shap is None:
            self.global_shap = [np.zeros_like(param) for param in parameters_to_ndarrays(update)]
        scores = {
            "DQ": self.score_data_quality(metrics, data_stats),
            "Perf": self.score_performance(metrics, global_metrics),
            "Exp": self.score_explanation(shap_vec, global_shap),
            "His": self.score_history(cid),
            "Priv": self.score_privacy(privacy_info),
            "Rob": self.score_robustness(update, ref_update, stats),
        }
        ts = sum(self.alpha_dict[k] * scores[k] for k in scores)
        self.historical_scores[cid] = ts
        return ts

    def _decode_bytes_to_ndarray(tensor_bytes: bytes) -> np.ndarray:

        # First load attempt
        try:
            buf = io.BytesIO(tensor_bytes)
            arr = np.load(buf, allow_pickle=False)
        except Exception as e:
            raise RuntimeError(f"np.load failed on provided bytes: {e}")

        # If arr is a numpy scalar of bytes (0-d array with string dtype),
        # then it's likely we saw a double-serialized .npy inside a .npy.
        # Convert to Python bytes and try loading again.
        if np.isscalar(arr) and isinstance(arr, (bytes, np.bytes_)):
            inner_bytes = bytes(arr)  # arr is np.bytes_
            try:
                inner_buf = io.BytesIO(inner_bytes)
                arr2 = np.load(inner_buf, allow_pickle=False)
                arr = arr2
            except Exception as e:
                # fallback: return the bytes as raw array (not ideal)
                raise RuntimeError(f"Nested np.load failed while unwrapping inner bytes: {e}")

        # If arr is an ndarray with string dtype (e.g., dtype='|S1408' and shape=()),
        # convert to bytes and attempt to load again.
        if isinstance(arr, np.ndarray) and arr.dtype.kind in ("S", "U") and arr.size == 1:
            try:
                inner_bytes = arr.tobytes()
                inner_buf = io.BytesIO(inner_bytes)
                arr2 = np.load(inner_buf, allow_pickle=False)
                arr = arr2
            except Exception:
                # If we cannot unwrap, raise to make failure explicit
                raise RuntimeError("Decoded array has string dtype and could not be unwrapped to a numeric array.")

        # Final sanity: ensure numeric dtype
        if not isinstance(arr, np.ndarray):
            raise RuntimeError("Decoded object is not a numpy ndarray.")

        if arr.dtype.kind in ("S", "U", "O"):
            raise RuntimeError(f"Decoded ndarray has non-numeric dtype: {arr.dtype}")

        return arr

    
    # server.py (Modified FedReX.aggregate_fit)

    def aggregate_fit(self, rnd, results, failures):
        trust_scores = []
        updates = []
        
        # --- Trust Score Calculation (Your original logic) ---
        global_metrics = None # You'd need to compute/fetch this if needed
        # We need a reference for robustness score if we want to use it properly
        ref_update = self.current_parameters if hasattr(self, 'current_parameters') else None

        for cid, fit_res in results:
            # cid is actually the client ID (int) provided by Flower, not the fit_res object
            
            # Use the index in the results list as a temporary ID for historical_scores if Flower's cid is complex
            client_id = str(cid) 
            data_stats, privacy_info, robustness_stats, shap_vec = extract_client_stats(fit_res.metrics)
            
            # If round 1, global_shap is None, so Exp score defaults to 1.0 (as coded)
            ts = self.compute_trust_score(
                cid=client_id,
                metrics=fit_res.metrics,
                data_stats=data_stats,
                shap_vec=shap_vec,
                privacy_info=privacy_info,
                update=fit_res.parameters,
                ref_update=ref_update, 
                stats=robustness_stats,
                global_metrics=global_metrics, # Pass aggregated metrics from previous round if available
                global_shap=self.global_shap,
            )
            trust_scores.append(float(ts))
            updates.append(fit_res.parameters)
            
        # ... (Normalize trust scores safely - your original logic is fine) ...
        weights = np.array(trust_scores, dtype=np.float64)
        weights = np.clip(weights, a_min=0.0, a_max=None)
        denom = weights.sum()
        if denom <= 0.0:
            weights = np.ones_like(weights) / float(len(weights))
        else:
            weights = weights / denom
            
        # Aggregate parameters (returns list of np.ndarrays)
        agg_ndarrays = self.aggregate_parameters_weighted(updates, weights)

        # CRUCIAL: Convert list of ndarrays back to Flower Parameters object
        agg_params_obj = ndarrays_to_parameters(agg_ndarrays)
        
        # Store the aggregated parameters for the next round's robustness calculation
        self.current_parameters = agg_params_obj 
        
        # Return aggregated parameters and metrics (FedReX doesn't aggregate fit metrics, but should return a dictionary)
        return agg_params_obj, {}



    def aggregate_parameters_weighted(self, updates: List[Parameters], weights: np.ndarray):
        
        updates_np = []
        for u in updates:
            # Flower's standard way to get ndarrays from Parameters
            arrs = parameters_to_ndarrays(u)
            # Convert to float64 for stable aggregation
            updates_np.append([np.array(a, dtype=np.float64) for a in arrs])

        # ... (weights normalization remains the same) ...

        # Layer-wise aggregation
        agg_params = []
        # Use zip(*updates_np) to iterate over layers
        for layer_arrays in zip(*updates_np): 
            stacked = np.stack(layer_arrays, axis=0)
            agg_layer = np.average(stacked, axis=0, weights=weights)
            agg_layer = np.nan_to_num(agg_layer, nan=0.0, posinf=0.0, neginf=0.0)
            # Convert back to float32 for model consistency
            agg_params.append(agg_layer.astype(np.float32))

        # IMPORTANT: Return a list of np.ndarrays
        return agg_params

class FedAvgWithSHAP(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        # Call the default FedAvg aggregation (accuracy, loss, etc.)
        agg_metrics = super().aggregate_evaluate(rnd, results, failures)

        # Collect SHAP vectors from clients
        shap_vecs, weights = [], []
        for _, eval_res in results:
            metrics = eval_res.metrics
            if "shap" in metrics:
                shap_vecs.append(np.array(metrics["shap"]))
                weights.append(eval_res.num_examples)

        # Weighted average of SHAP importances across clients
        if shap_vecs:
            shap_vecs = np.vstack(shap_vecs)
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            global_shap = np.average(shap_vecs, axis=0, weights=weights)

            # Print top-5 features each round (indices only, since server doesn’t know names)
            top_idx = np.argsort(global_shap)[::-1][:5]
            print(f"[Round {rnd}] Global SHAP top-5 features (by index):")
            for i in top_idx:
                print(f"  Feature {i}: {global_shap[i]:.6f}")

        return agg_metrics

    def get_parameters(self, config):
        # Use np.save to serialize each parameter to bytes
        param_bytes = []
        for _, val in self.model.state_dict().items():
            buf = io.BytesIO()
            np.save(buf, val.cpu().numpy().astype(np.float32), allow_pickle=False)
            param_bytes.append(buf.getvalue())
        return param_bytes

    def set_parameters(self, parameters: List[bytes]):
        state_dict = self.model.state_dict()
        for (k, _), param_bytes in zip(state_dict.items(), parameters):
            buf = io.BytesIO(param_bytes)
            np_val = np.load(buf, allow_pickle=False)
            state_dict[k] = torch.tensor(np_val)
        self.model.load_state_dict(state_dict, strict=True)


def extract_client_stats(metrics):
    data_stats = {}
    privacy_info = {}
    robustness_stats = {}
    shap_vec = None

    # Data quality
    for k, v in metrics.items():
        if k.startswith("data_quality_"):
            # Split from the right: feature may have underscores
            rest = k[len("data_quality_"):]  # e.g. "pulse_pressure_comp"
            feat, stat = rest.rsplit("_", 1) # e.g. "pulse_pressure", "comp"
            if feat not in data_stats:
                data_stats[feat] = {}
            data_stats[feat][stat] = v
        elif k.startswith("privacy_"):
            stat = k.replace("privacy_", "")
            privacy_info[stat] = v
        elif k.startswith("robustness_"):
            stat = k.replace("robustness_", "")
            robustness_stats[stat] = v
        elif k.startswith("shap_"):
            try:
                idx = int(k.split("_")[1])
                if shap_vec is None:
                    shap_vec = []
                shap_vec.append(v)
            except ValueError:
                continue

    if shap_vec is not None:
        shap_vec = np.array(shap_vec)

    return data_stats, privacy_info, robustness_stats, shap_vec


def get_strategy():
    def fit_config_fn(server_round: int):
        return {"local_epochs": 2}

    # return FedAvgWithSHAP(
    return FedReX(
        fraction_fit=1.0,        # sample all clients each round (since we have 3)
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_config_fn,
        accept_failures=False,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

def main():
    address = os.environ.get("BIND_ADDRESS", "0.0.0.0:8080")
    strategy = get_strategy()
    print(f"Starting Flower server on {address} …")
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=10),  # try 5 or 10 rounds
        strategy=strategy,
    )


def compute_data_quality_stats(df, y):
    # Example implementation for computing data quality stats
    stats = {}
    for f in df.columns:
        x = df[f]
        # Completeness
        S_comp = x.notnull().mean()
        # Validity (assuming y is the ground truth label)
        S_valid = (x == y).mean()
        # Uniqueness
        S_uniq = x.nunique() / len(x)
        # Outliers (assuming outliers are values outside 1.5*IQR)
        Q1 = x.quantile(0.25)
        Q3 = x.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        S_out = ((x < lower_bound) | (x > upper_bound)).mean()
        # Mutual Information
        try:
            # Discretize x if it's continuous
            if np.issubdtype(x.dtype, np.floating):
                x_disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit_transform(x.reshape(-1, 1)).ravel()
            else:
                x_disc = x
            I = mutual_info_score(x_disc, y)
            I_max = 5.0  # can be adjusted
            xi = 0.7
            S_mi = np.log(1 + I) / (np.log(1 + I_max) ** xi)
        except Exception:
            S_mi = 0.0

        stats[f] = {"comp": S_comp, "valid": S_valid, "uniq": S_uniq, "out": S_out, "mi": S_mi}

    return stats

if __name__ == "__main__":
    main()