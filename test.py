import torch


bundle = torch.load("global_model_bundle.pt")
print(bundle["scaler"])