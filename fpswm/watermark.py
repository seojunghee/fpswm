import cv2
import numpy as np
import torch

from .config import HP


def load_watermark_bits(path, hw=HP.wm_hw, threshold=127):
    """Binary QR image -> (1, 1, hw, hw) float map of 0/1 (nearest-neighbour resize, > threshold = 1)."""
    wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"cannot read watermark image: {path}")
    wm = cv2.resize(wm, (hw, hw), interpolation=cv2.INTER_NEAREST)
    bits = (wm > threshold).astype(np.uint8)
    return torch.from_numpy(bits).float()[None, None]
