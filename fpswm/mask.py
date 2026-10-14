import torch.nn.functional as F

from .config import HP


def compute_flatness_mask(x, kernel_size=HP.mask_kernel, var_scale=HP.mask_scale, floor=HP.mask_floor):
    """Content-adaptive flatness mask  M = f + (1 - f) * v / (v + s),  v = local variance (k x k).

    x: (B, 3, H, W) in [0, 1]. Returns (B, 1, H, W) in [f, 1]; low values = smooth regions.
    """
    gray = x.mean(dim=1, keepdim=True)
    pad = kernel_size // 2
    mean = F.avg_pool2d(gray, kernel_size, stride=1, padding=pad, count_include_pad=False)
    mean_sq = F.avg_pool2d(gray * gray, kernel_size, stride=1, padding=pad, count_include_pad=False)
    var = (mean_sq - mean * mean).clamp(min=0)
    return floor + (1.0 - floor) * var / (var + var_scale)
