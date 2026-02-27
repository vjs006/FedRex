
import matplotlib.pyplot as plt
import numpy as np

rounds = np.arange(1, 11)

C1 = np.array([0.169, 0.18, 0.23, 0.235, 0.245, 0.22, 0.255, 0.24, 0.265, 0.26])
#C2 = np.array([0.189, 0.195, 0.205, 0.21, 0.215, 0.215, 0.22, 0.225, 0.225, 0.23])
#C3 = np.array([0.199, 0.175, 0.175, 0.17, 0.17, 0.165, 0.165, 0.18, 0.16, 0.155])
#C4 = np.array([0.159, 0.16, 0.155, 0.155, 0.15, 0.15, 0.145, 0.145, 0.14, 0.15])
#C5 = np.array([0.149, 0.18, 0.13, 0.125, 0.115, 0.14, 0.11, 0.105, 0.105, 0.105])
#C6 = np.array([0.135, 0.11, 0.105, 0.105, 0.105, 0.11, 0.105, 0.105, 0.105, 0.10])

# Sanity check
#assert np.allclose(C1 + C2 + C3 + C4 + C5 + C6, 1.0)

plt.figure(figsize=(7, 4.8))

plt.plot(rounds, C1, marker='o', label="Client C1")
#plt.plot(rounds, C2, marker='o', label="Client C2 (Reliable)")
#plt.plot(rounds, C3, marker='o', label="Client C3 (Moderate)")
#plt.plot(rounds, C4, marker='o', label="Client C4 (Moderate)")
#plt.plot(rounds, C5, marker='o', label="Client C5 (Inconsistent)")
#plt.plot(rounds, C6, marker='o', label="Client C6 (Noisy)")

plt.xlabel("Round")
plt.ylabel("Trust Weight")
plt.title("Trust score across rounds for client 1")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()



"""
import matplotlib.pyplot as plt

rounds = list(range(1, 11))

fedavg = [0.61, 0.64, 0.66, 0.67, 0.69, 0.70, 0.71, 0.71, 0.72, 0.72]
fedrex = [0.66, 0.70, 0.72, 0.73, 0.735, 0.738, 0.740, 0.742, 0.743, 0.745]

plt.figure(figsize=(6.8, 4.6))

plt.plot(rounds, fedavg, marker='o', label="FedAvg")
plt.plot(rounds, fedrex, marker='o', label="FedReX")

plt.xlabel("Round")
plt.ylabel("Global Accuracy")
plt.title("FedAvg vs FedReX Accuracy Convergence")
plt.grid(True)
plt.legend()

plt.ylim(0.60, 0.76)
plt.tight_layout()
plt.show()
"""