"""Table 7 (block-DCT column): classical blind block-DCT watermark at matched PSNR.

One bit per 8x8 luminance block, coefficient pair (3,4)/(4,3), margin alpha calibrated on 32
validation images to the PSNR of the proposed model (first 50 images); same attacks as the proposed method.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.fftpack import dct, idct

from fpswm.attacks import attack_val_v2
from fpswm.cli import base_parser, setup
from fpswm.evaluate import ATTACKS7, mean_psnr
from fpswm.models import load_models

PAIR = ((3, 4), (4, 3))
p = base_parser(__doc__)
p.add_argument("--alphas", type=float, nargs="+", default=[22, 24, 26, 28, 30, 32, 36])
args = p.parse_args()
device, wm, loader = setup(args)


def _dct2(b):
    return dct(dct(b.T, norm='ortho').T, norm='ortho')


def _idct2(b):
    return idct(idct(b.T, norm='ortho').T, norm='ortho')


def dct_embed(img, bits, alpha):
    ycc = np.array(Image.fromarray(img).convert("YCbCr")).astype(np.float32)
    Y = ycc[:, :, 0]
    nby, nbx = Y.shape[0] // 8, Y.shape[1] // 8
    (r1, c1), (r2, c2) = PAIR
    k = 0
    for by in range(nby):
        for bx in range(nbx):
            if k >= len(bits):
                break
            C = _dct2(Y[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8])
            a, b = C[r1, c1], C[r2, c2]
            m = (a + b) / 2
            if bits[k] == 1 and a - b < alpha:
                C[r1, c1], C[r2, c2] = m + alpha / 2, m - alpha / 2
            elif bits[k] == 0 and b - a < alpha:
                C[r1, c1], C[r2, c2] = m - alpha / 2, m + alpha / 2
            Y[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8] = _idct2(C)
            k += 1
    ycc[:, :, 0] = np.clip(Y, 0, 255)
    return np.array(Image.fromarray(ycc.astype(np.uint8), mode="YCbCr").convert("RGB"))


def dct_extract(img, n_bits):
    Y = np.array(Image.fromarray(img).convert("YCbCr")).astype(np.float32)[:, :, 0]
    out, k = np.zeros(n_bits, dtype=np.uint8), 0
    (r1, c1), (r2, c2) = PAIR
    for by in range(Y.shape[0] // 8):
        for bx in range(Y.shape[1] // 8):
            if k >= n_bits:
                break
            C = _dct2(Y[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8])
            out[k] = 1 if C[r1, c1] > C[r2, c2] else 0
            k += 1
    return out


def psnr_u8(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(max(mse, 1e-9)))


to_u8 = lambda t: (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
bits = (wm > 0.5).float().flatten().cpu().numpy().astype(np.uint8)
imgs = [to_u8(b[0]) for _, b in zip(range(32), loader)]

enc, _ = load_models(args.ckpt, "hybrid", device)
target = mean_psnr(enc, loader, wm, device, max_batches=50)
print(f"proposed PSNR (first 50 images): {target:.2f} dB")
best = None
for a in args.alphas:
    m = float(np.mean([psnr_u8(im, dct_embed(im, bits, a)) for im in imgs]))
    print(f"  alpha={a:>4} -> PSNR {m:.2f} dB")
    if best is None or abs(m - target) < abs(best[1] - target):
        best = (a, m)
alpha = best[0]
print(f"alpha = {alpha}")

rows = []
for atk in ATTACKS7:
    bers, ps, cov = [], [], []
    for batch in loader:
        img = to_u8(batch[0])
        wimg = dct_embed(img, bits, alpha)
        if atk == "none":
            att = wimg.copy()
        else:
            t = torch.from_numpy(wimg.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            att = to_u8(attack_val_v2(t, atk, 0.7)[0].cpu())
        k = min((img.shape[0] // 8) * (img.shape[1] // 8), len(bits))
        pred = dct_extract(att, len(bits))
        bers.append(float(np.mean(pred[:k] != bits[:k])))
        cov.append(k / len(bits))
        ps.append(psnr_u8(img, wimg))
    rows.append({"attack": atk, "BER": np.mean(bers), "PSNR": np.mean(ps), "coverage": np.mean(cov)})
    print(f"{atk:<8} BER={np.mean(bers):.4f} PSNR={np.mean(ps):.2f}")
pd.DataFrame(rows).to_csv(os.path.join(args.out, "p1_table7_baseline_dct.csv"), index=False)
