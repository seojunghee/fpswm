"""Table 9: BER, adopted extraction path and attacked-image PSNR vs severity.

Geometric attacks (crop, rotate) use the inference-time search; the others use plain decoding.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

from fpswm.attacks import attack_val_v2
from fpswm.cli import base_parser, setup
from fpswm.data import batch_x
from fpswm.models import load_models
from fpswm.search import extract_watermark
from fpswm.utils import first

SEVERITIES = (0.1, 0.3, 0.5, 0.7)
ATTACKS = ("none", "jpeg", "crop", "resize", "rotate", "blur", "noise")
GEOMETRIC = ("crop", "rotate")
args = base_parser(__doc__).parse_args()
device, wm, loader = setup(args)
enc, dec = load_models(args.ckpt, "hybrid", device)
to_u8 = lambda t: (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def psnr_u8(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


rows = []
with torch.no_grad():
    for sev in SEVERITIES:
        for atk in ATTACKS:
            search = atk in GEOMETRIC
            pe, pa, se, sa, bers, methods, n = [], [], [], [], [], {}, 0
            for batch in loader:
                x = batch_x(batch, device)
                y = first(enc(x, wm))
                img_x, img_y = to_u8(x[0]), to_u8(y[0])
                if atk == "none":
                    ya, img_a = y, img_y.copy()
                else:
                    ya = attack_val_v2(y, atk, sev)
                    img_a = (ya[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                if search:
                    pred, info = extract_watermark(dec, ya)
                    m = info["method"]
                else:
                    pred, m = (torch.sigmoid(dec(ya)) > 0.5).float(), "plain"
                if pred.shape[-2:] != wm.shape[-2:]:
                    pred = F.interpolate(pred, size=wm.shape[-2:], mode="nearest")
                pe.append(psnr_u8(img_x, img_y)); se.append(structural_similarity(img_x, img_y, channel_axis=2, data_range=255))
                pa.append(psnr_u8(img_x, img_a)); sa.append(structural_similarity(img_x, img_a, channel_axis=2, data_range=255))
                bers.append((pred != wm).float().mean().item())
                methods[m] = methods.get(m, 0) + 1
                n += 1
            top = max(methods.items(), key=lambda kv: kv[1])[0]
            rows.append(dict(severity=sev, attack=atk, BER=np.mean(bers), PSNR_embed=np.mean(pe),
                             SSIM_embed=np.mean(se), PSNR_attacked=np.mean(pa), SSIM_attacked=np.mean(sa),
                             search=search, top_method=top, plain_ratio=methods.get("plain", 0) / n, N=n))
            print(f"sev={sev} {atk:<7} BER={np.mean(bers):.4f} PSNR_att={np.mean(pa):.2f} top={top}")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.out, "p1_table9_severity.csv"), index=False)
print(df.pivot(index="attack", columns="severity", values="BER").round(4))
print(df.pivot(index="attack", columns="severity", values="PSNR_attacked").round(2))
