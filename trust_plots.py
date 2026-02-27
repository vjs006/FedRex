import matplotlib.pyplot as plt

fedavg_scores = [0.61, 0.64, 0.66, 0.67, 0.69, 0.70, 0.71, 0.71, 0.72, 0.72]
fedrex_scores = [0.68, 0.71, 0.72, 0.72, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73]

rounds = range(1, len(fedavg_scores) + 1)

plt.figure(figsize=(7, 5))
plt.plot(rounds, fedavg_scores, marker="o", label="FedAvg")
plt.plot(rounds, fedrex_scores, marker="s", label="FedReX")

plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("FedAvg vs FedReX Convergence")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("fedavg_vs_fedrex_convergence.png", dpi=300)
plt.show()
