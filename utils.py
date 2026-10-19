import random
import numpy as np
import torch

from model import ResNet18

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(path, device):
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()