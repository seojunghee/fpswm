import os
import random

import numpy as np
import torch


def get_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed: int, deterministic: bool = False):
    """Fix python / numpy / torch / CUDA seeds (as in the Stage-2 seed runs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def psnr_t(y, x):
    """PSNR of tensors in [0, 1] (whole batch)."""
    mse = torch.mean((y - x) ** 2).item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def first(out):
    """Encoders return (y, delta, mask) in train mode and y in eval mode."""
    return out[0] if isinstance(out, (tuple, list)) else out
