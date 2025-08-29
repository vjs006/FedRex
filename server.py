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

    return FedAvgWithSHAP(
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