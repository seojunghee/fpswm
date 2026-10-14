import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    """Differentiable single-level Haar DWT / IDWT (tensor indexing and arithmetic only)."""

    def forward_dwt(self, x):
        _, _, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        a, b = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
        c, d = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
        ll = (a + b + c + d) * 0.25
        lh = (a - b + c - d) * 0.25
        hl = (a + b - c - d) * 0.25
        hh = (a - b - c + d) * 0.25
        return ll, lh, hl, hh

    def inverse_dwt(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = ll + lh + hl + hh
        out[:, :, 0::2, 1::2] = ll - lh + hl - hh
        out[:, :, 1::2, 0::2] = ll + lh - hl - hh
        out[:, :, 1::2, 1::2] = ll - lh - hl + hh
        return out
