import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
model_name = "aehrc/cxrmate-single-tf"