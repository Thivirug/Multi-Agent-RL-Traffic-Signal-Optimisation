import torch

if torch.backends.mps.is_available():
    print("MPS is available. Using Apple GPU for computations.")
else:
    print("MPS is not available.")
