import torch

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Capability:", torch.cuda.get_device_capability(0))
    print("Total Memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    print("No GPU detected.")
