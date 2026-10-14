"""Evaluation helpers shared by the experiment scripts."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as calc_ssim

from .attacks import attack_val_v2
from .config import HP
from .data import batch_x
from .search import ber_from_pred, extract_watermark
from .utils import first

ATTACKS7 = ("none", "jpeg", "crop", "resize", "rotate", "blur", "noise")


@torch.no_grad()
def eval_multiattack(enc, dec, loader, wm_bits_map, device, attacks=ATTACKS7, severity=HP.severity,
                     max_batches=None, verbose=False):
    """Plain decoding (no search) under each attack. Returns ({attack: {BER, BitAcc, NCC, IoU}}, mean)."""
    was = enc.training, dec.training
    enc.eval(); dec.eval()
    wm = (wm_bits_map > 0.5).float().to(device)
    per, eps = {}, 1e-8
    for atk in attacks:
        ber, ncc, iou, nb = [], [], [], 0
        for batch in loader:
            x = batch_x(batch, device)
            wm_b = wm.repeat(x.shape[0], 1, 1, 1)
            y = first(enc(x, wm_b))
            logits = dec(attack_val_v2(y, atk, severity))
            if logits.shape[-2:] != wm_b.shape[-2:]:
                logits = F.interpolate(logits, size=wm_b.shape[-2:], mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits) > 0.5).float()
            ber.append((pred != wm_b).float().mean().item())
            p, t = pred.flatten(1), wm_b.flatten(1)
            pz, tz = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
            ncc.append((((pz * tz).sum(1)) / ((pz.pow(2).sum(1) + eps).sqrt() * (tz.pow(2).sum(1) + eps).sqrt() + eps)).mean().item())
            inter = (pred * wm_b).sum(dim=(1, 2, 3))
            union = ((pred + wm_b) > 0).float().sum(dim=(1, 2, 3)) + eps
            iou.append((inter / union).mean().item())
            nb += 1
            if max_batches is not None and nb >= max_batches:
                break
        per[atk] = {"BER": float(np.mean(ber)), "BitAcc": 1 - float(np.mean(ber)),
                    "NCC": float(np.mean(ncc)), "IoU": float(np.mean(iou)), "n_batches": nb}
    bers = [per[a]["BER"] for a in attacks]
    mean = {"BER": float(np.mean(bers)), "BitAcc": 1 - float(np.mean(bers)),
            "NCC": float(np.mean([per[a]["NCC"] for a in attacks])),
            "IoU": float(np.mean([per[a]["IoU"] for a in attacks]))}
    if verbose:
        print(f"[VAL] severity={severity} mean_BER={mean['BER']:.6f} | " +
              " ".join(f"{a}:{per[a]['BER']:.4f}" for a in attacks))
    enc.train(was[0]); dec.train(was[1])
    return per, mean


@torch.no_grad()
def eval_wm_visual_quality(enc, dec, loader, wm_bits_map, device,
                           attacks=("none", "noise", "jpeg", "crop", "rotate"), severity=HP.severity,
                           max_batches=10):
    """BER plus confidence and SSIM / PSNR of the soft watermark map (used for Stage-2 model selection).

    As in the original notebook, both networks are left in train mode on return.
    """
    enc.eval(); dec.eval()
    wm = (wm_bits_map > 0.5).float().to(device)
    res = {}
    for atk in attacks:
        ber, ss, ps, conf = [], [], [], []
        for nb, batch in enumerate(loader):
            x = batch_x(batch, device)
            wm_b = wm.repeat(x.shape[0], 1, 1, 1)
            y = first(enc(x, wm_b))
            logits = dec(attack_val_v2(y, atk, severity))
            if logits.shape[-2:] != wm_b.shape[-2:]:
                logits = F.interpolate(logits, size=wm_b.shape[-2:], mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits)
            ber.append(((prob > 0.5).float() != wm_b).float().mean().item())
            conf.append((prob - 0.5).abs().mul(2).mean().item())
            ss.append(calc_ssim(prob, wm_b, data_range=1.0, size_average=True).item())
            ps.append(10 * math.log10(1.0 / (F.mse_loss(prob, wm_b).item() + 1e-8)))
            if max_batches and nb + 1 >= max_batches:
                break
        res[atk] = {"BER": round(float(np.mean(ber)), 4), "Confidence": round(float(np.mean(conf)), 4),
                    "WM_SSIM": round(float(np.mean(ss)), 4), "WM_PSNR": round(float(np.mean(ps)), 2)}
    enc.train(); dec.train()
    return res


def summarize_crop_rotate(per_attack):
    c, r = per_attack["crop"]["BER"], per_attack["rotate"]["BER"]
    return c, r, 0.5 * (c + r), max(c, r)


@torch.no_grad()
def region_residual(enc, loader, wm_bits_map, device, theta=HP.theta, max_batches=None):
    """Inference-mode R_flat / R_tex: channel-mean |y - x| averaged over {M < theta} and its complement."""
    enc.eval()
    wm = (wm_bits_map > 0.5).float().to(device)
    flat, tex = [], []
    for nb, batch in enumerate(loader):
        x = batch_x(batch, device)
        y = first(enc(x, wm.repeat(x.shape[0], 1, 1, 1)))
        mag = (y - x).abs().mean(dim=1, keepdim=True)
        fl = enc.flatness_mask(x) < theta
        if fl.any():
            flat.append(mag[fl].mean().item())
        if (~fl).any():
            tex.append(mag[~fl].mean().item())
        if max_batches is not None and nb + 1 >= max_batches:
            break
    return float(np.mean(flat)), float(np.mean(tex))


@torch.no_grad()
def ber_with_search(enc, dec, loader, wm_bits_map, device, attacks=("crop", "rotate"), severity=HP.severity,
                    max_batches=None):
    """BER of geometric attacks with the 41-candidate inference-time search."""
    enc.eval(); dec.eval()
    wm = (wm_bits_map > 0.5).float().to(device)
    out = {}
    for atk in attacks:
        bers = []
        for nb, batch in enumerate(loader):
            x = batch_x(batch, device)
            gt = wm.repeat(x.shape[0], 1, 1, 1)
            y = first(enc(x, gt))
            pred, _ = extract_watermark(dec, attack_val_v2(y, atk, severity))
            bers.append(ber_from_pred(pred, gt))
            if max_batches is not None and nb + 1 >= max_batches:
                break
        out[f"{atk}_trick"] = float(np.mean(bers))
    return out


@torch.no_grad()
def mean_psnr(enc, loader, wm_bits_map, device, max_batches=None):
    enc.eval()
    wm = (wm_bits_map > 0.5).float().to(device)
    ps = []
    for nb, batch in enumerate(loader):
        x = batch_x(batch, device)
        y = first(enc(x, wm.repeat(x.shape[0], 1, 1, 1)))
        for b in range(x.shape[0]):
            ps.append(10 * np.log10(1.0 / max(torch.mean((y[b] - x[b]) ** 2).item(), 1e-12)))
        if max_batches is not None and nb + 1 >= max_batches:
            break
    return float(np.mean(ps))
