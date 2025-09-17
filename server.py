import json
import os
import numpy as np
import flwr as fl
from typing import List, Tuple, Dict, Any

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
        # TODO: Implement S_DQ,c using your LaTeX formulas
        return 1.0

    def score_performance(self, metrics, global_metrics):
        # TODO: Implement S_Perf,c (e.g., balanced accuracy improvement)
        return 1.0

    def score_explanation(self, shap_vec, global_shap):
        # TODO: Implement S_Exp,c (L1/cosine similarity)
        return 1.0

    def score_history(self, cid):
        # EWMA smoothing
        prev = self.historical_scores.get(cid, 1.0)
        beta = 0.8
        # For demo, just return previous score
        return prev

    def score_privacy(self, privacy_info):
        # TODO: Implement S_Priv,c (DP parameters)
        return 1.0

    def score_robustness(self, update, ref_update, stats):
        # TODO: Implement S_Rob,c (cosine, norm, diversity)
        return 1.0

    def compute_trust_score(self, cid, metrics, data_stats, shap_vec, privacy_info, update, ref_update, stats, global_metrics, global_shap):
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

    def aggregate_fit(self, rnd, results, failures):
        # Trust-weighted aggregation
        trust_scores, updates = [], []
        for cid, fit_res in results:
            # Extract needed info from fit_res (extend client reporting as needed)
            ts = self.compute_trust_score(
                cid=cid,
                metrics=fit_res.metrics,
                data_stats=None,      # TODO: pass client data stats
                shap_vec=None,        # TODO: pass client SHAP vector
                privacy_info=None,    # TODO: pass privacy info
                update=fit_res.parameters,
                ref_update=None,      # TODO: pass reference update
                stats=None,           # TODO: pass norm/diversity stats
                global_metrics=None,  # TODO: pass global metrics
                global_shap=self.global_shap
            )
            trust_scores.append(ts)
            updates.append(fit_res.parameters)

        # Normalize trust scores
        weights = np.array(trust_scores)
        weights /= weights.sum() if weights.sum() > 0 else 1.0

        # Weighted aggregation (implement as needed)
        agg_params = self.aggregate_parameters_weighted(updates, weights)
        return agg_params, {}

    def aggregate_parameters_weighted(self, updates, weights):
        # Weighted aggregation of model parameters
        # TODO: implement actual weighted averaging
        return updates[0]  # placeholder

    def aggregate_evaluate(self, rnd, results, failures):
        agg_metrics = super().aggregate_evaluate(rnd, results, failures)

        shap_vecs, weights = [], []
        for _, eval_res in results:
            metrics = eval_res.metrics
            # Collect SHAP values as a list of floats (shap_0, shap_1, ...)
            shap = [metrics[k] for k in sorted(metrics) if k.startswith("shap_")]
            if shap:
                shap_vecs.append(np.array(shap))
                weights.append(eval_res.num_examples)

        if shap_vecs:
            shap_vecs = np.vstack(shap_vecs)
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            global_shap = np.average(shap_vecs, axis=0, weights=weights)

            # Print top-5 features each round
            top_idx = np.argsort(global_shap)[::-1][:5]
            print(f"[Round {rnd}] Global SHAP top-5 features (by index):")
            for i in top_idx:
                print(f"  Feature {i}: {global_shap[i]:.6f}")

            # Store in history and write to single file
            self.global_shap_history[rnd] = global_shap.tolist()
            os.makedirs("shap_outputs", exist_ok=True)
            out_path = "shap_outputs/global_shap.json"
            with open(out_path, "w") as f:
                json.dump(self.global_shap_history, f, indent=2)

        return agg_metrics


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


if __name__ == "__main__":
    main()