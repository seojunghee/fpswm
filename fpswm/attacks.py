"""Attacks.

attack_val_v2            deterministic evaluation attacks (severity in [0, 1]); JPEG is the real PIL
                         codec at subsampling 4:4:4, QF = round(100 * (1 - 0.6 * severity)) -> 58 at 0.7
differentiable_attack_v2 stochastic training attack layer; JPEG is the differentiable approximation
                         built from the standard luminance quantization table (diffjpeg_attack)
"""
import io
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


# --------------------------------------------------------------------------- real JPEG
@torch.no_grad()
def real_jpeg_attack_eval(y, quality_float, subsampling=0):
    """Real PIL JPEG. y: (B, 3, H, W) in [0, 1]; quality_float in (0, 1] (x100 -> PIL quality)."""
    q = int(np.clip(int(round(float(np.clip(quality_float, 0.01, 1.0)) * 100.0)), 1, 100))
    out = []
    for img in y.detach().cpu().clamp(0, 1):
        u8 = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(u8).save(buf, format="JPEG", quality=q, subsampling=subsampling, optimize=False)
        buf.seek(0)
        with Image.open(buf) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        out.append(torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous())
    return torch.stack(out, 0).to(device=y.device, dtype=y.dtype)


# --------------------------------------------------------------------------- noise
def noise_lowfreq_preserve(y, std=0.02, kernel_size=5, sigma=1.5):
    """Gaussian noise with its own low-frequency part removed (no global tone shift)."""
    noise = torch.randn_like(y) * std
    noise_high = noise - TF.gaussian_blur(noise, kernel_size=kernel_size, sigma=sigma)
    return (y + noise_high).clamp(0, 1)


# --------------------------------------------------------------------------- evaluation attacks
@torch.no_grad()
def attack_val_v2(y, atk, severity):
    """Deterministic evaluation attack (paper, Section IV).

    severity 0.7: JPEG QF 58, crop to 72 % (centre) and resize back, rotation 10.5 deg clockwise,
    resize to 72 % and back, blur sigma 1.14, noise std 0.021.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    atk = str(atk).lower().strip()
    _, _, H, W = y.shape
    if atk == "none":
        return y.clamp(0, 1)
    if atk == "noise":
        return noise_lowfreq_preserve(y, std=0.03 * severity).clamp(0, 1)
    if atk == "blur":
        sigma = 0.3 + 1.2 * severity
        return TF.gaussian_blur(y, kernel_size=5 if sigma < 1.0 else 7, sigma=sigma).clamp(0, 1)
    if atk == "rotate":
        return TF.rotate(y, angle=-15.0 * severity, interpolation=InterpolationMode.BILINEAR,
                         expand=False, fill=0).clamp(0, 1)
    if atk == "resize":
        s = 1.0 - 0.4 * severity
        nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
        y2 = F.interpolate(y, size=(nh, nw), mode="bilinear", align_corners=False)
        return F.interpolate(y2, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
    if atk == "crop":
        s = 1.0 - 0.30 * severity
        ch, cw = max(1, int(round(H * s))), max(1, int(round(W * s)))
        top, left = max(0, (H - ch) // 2), max(0, (W - cw) // 2)
        return TF.resized_crop(y, top=top, left=left, height=ch, width=cw, size=(H, W),
                               interpolation=InterpolationMode.BILINEAR, antialias=True).clamp(0, 1)
    if atk == "jpeg":
        return real_jpeg_attack_eval(y, quality_float=1.0 - 0.6 * severity).clamp(0, 1)
    raise ValueError(f"unknown attack: {atk}")


# --------------------------------------------------------------------------- differentiable JPEG
_JPEG_QTABLE = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)


def dct_2d_basis(n, device):
    k = torch.arange(n, dtype=torch.float32, device=device)
    i = torch.arange(n, dtype=torch.float32, device=device)
    C = torch.cos(math.pi * k.unsqueeze(1) * (2 * i.unsqueeze(0) + 1) / (2 * n))
    C[0] *= math.sqrt(1.0 / n)
    C[1:] *= math.sqrt(2.0 / n)
    return C


def _blockwise(x, M_left, M_right):
    B, Ch, H, W = x.shape
    n = M_left.shape[0]
    blocks = x.unfold(2, n, n).unfold(3, n, n)
    Hn, Wn = blocks.shape[2], blocks.shape[3]
    blocks = blocks.contiguous().view(B * Ch * Hn * Wn, n, n)
    out = (M_left @ blocks @ M_right).view(B, Ch, Hn, Wn, n, n)
    return out.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, Ch, H, W)


def block_dct2(x, basis):
    return _blockwise(x, basis, basis.t())


def block_idct2(X, basis):
    return _blockwise(X, basis.t(), basis)


def diffjpeg_attack(x, quality_float):
    """Differentiable JPEG approximation (standard luminance table; rounding passed straight-through)."""
    B, C, H, W = x.shape
    n = 8
    pad_h, pad_w = (n - H % n) % n, (n - W % n) % n
    x_p = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h or pad_w) else x
    basis = dct_2d_basis(n, x.device)
    coef = block_dct2(x_p * 255.0 - 128.0, basis)
    q = max(1, min(100, int(round(quality_float * 100))))
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    qtable = torch.clamp(_JPEG_QTABLE.to(x.device) * scale / 100.0, min=1.0)
    Hp, Wp = x_p.shape[-2:]
    qmap = qtable.repeat(Hp // n, Wp // n)
    cq = coef / qmap
    cq = cq + (torch.round(cq) - cq).detach()
    y = (block_idct2(cq * qmap, basis) + 128.0) / 255.0
    if pad_h or pad_w:
        y = y[:, :, :H, :W]
    return y.clamp(0, 1)


# --------------------------------------------------------------------------- training attacks
def differentiable_attack_v2(y, cfg_atk, epoch=None, total_epochs=None, return_attack_name=False):
    """Stochastic training attack; strength ramps up with epoch / total_epochs."""
    p = float(cfg_atk.get("attack_prob", 0.85))
    if random.random() > p:
        return (y, "none") if return_attack_name else y
    _, _, H, W = y.shape
    t = 0.0
    if epoch is not None and total_epochs and total_epochs > 1:
        t = float(epoch) / float(total_epochs - 1)

    crop_lo, crop_hi = cfg_atk.get("crop_range", (0.70, 0.95))
    noise_std = float(cfg_atk.get("noise_std", 0.025))
    rot_lo, rot_hi = cfg_atk.get("rot_range", (-15, 15))
    res_lo, res_hi = cfg_atk.get("resize_range", (0.65, 1.0))
    blur_lo, blur_hi = cfg_atk.get("blur_sigma", (0.3, 1.5))
    jpeg_lo, jpeg_hi = cfg_atk.get("jpeg_down", (0.55, 1.0))

    crop_scale = random.uniform(crop_hi - (crop_hi - crop_lo) * t, crop_hi)
    rot_max = max(abs(rot_lo), abs(rot_hi)) * (0.2 + 0.8 * t)
    rs_scale = random.uniform(res_hi - (res_hi - res_lo) * t, res_hi)
    sigma = random.uniform(blur_lo, blur_hi) * (0.3 + 0.7 * t)
    jpeg_qual = random.uniform(jpeg_hi - (jpeg_hi - jpeg_lo) * t, jpeg_hi)
    ns = noise_std * (0.5 + 0.5 * t)

    weights = cfg_atk.get("weights", {"jpeg": 0.20, "noise": 0.20, "rotate": 0.18,
                                      "crop": 0.18, "blur": 0.12, "resize": 0.12})
    atk = random.choices(list(weights.keys()), list(weights.values()), k=1)[0]

    if atk == "noise":
        hi_map = (y - TF.gaussian_blur(y, kernel_size=7, sigma=2.0)).abs()
        hi_map = hi_map / (hi_map.max() + 1e-8)
        y2 = (y + torch.randn_like(y) * ns * (0.5 + 0.5 * hi_map)).clamp(0, 1)
    elif atk == "blur":
        y2 = TF.gaussian_blur(y, kernel_size=5 if sigma < 1.0 else 7, sigma=float(sigma))
    elif atk == "rotate":
        angle = abs(random.uniform(-rot_max, rot_max))
        y2 = TF.rotate(y, angle=-angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
    elif atk == "crop":
        nh, nw = max(1, int(H * crop_scale)), max(1, int(W * crop_scale))
        if cfg_atk.get("center_crop_only", False):
            top, left = max(0, (H - nh) // 2), max(0, (W - nw) // 2)
        else:
            top, left = random.randint(0, max(0, H - nh)), random.randint(0, max(0, W - nw))
        y2 = TF.resized_crop(y, top=top, left=left, height=nh, width=nw, size=(H, W),
                             interpolation=InterpolationMode.BILINEAR, antialias=True)
    elif atk == "resize":
        nh, nw = max(1, int(H * rs_scale)), max(1, int(W * rs_scale))
        y2 = F.interpolate(y, (nh, nw), mode="bilinear", align_corners=False)
        y2 = F.interpolate(y2, (H, W), mode="bilinear", align_corners=False)
    elif atk == "jpeg":
        y2 = diffjpeg_attack(y, quality_float=jpeg_qual)
    else:
        y2 = y
    y2 = y2.clamp(0, 1)
    return (y2, atk) if return_attack_name else y2
