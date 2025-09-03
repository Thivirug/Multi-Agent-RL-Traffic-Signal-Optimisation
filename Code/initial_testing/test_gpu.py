# test gpu availability

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. GPU can be used for computations.")

    # print the number of available GPUs
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
