"""Tables 4 and 5: per-seed imperceptibility and BER.

For each checkpoint: PSNR / SSIM (cover vs watermarked), R_flat / R_tex (inference mode),
BER under the seven attacks at severity 0.7 with plain decoding, and crop / rotation with the
41-candidate search. 200 full-resolution validation images.

    python experiments/eval_seeds.py --val-dir COCO/images/val --ckpt checkpoints/stage2_seed17 \
        checkpoints/stage2_seed42 checkpoints/stage2_seed137 --labels 17 42 137
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from pytorch_msssim import ssim

from fpswm.cli import base_parser, setup
from fpswm.data import batch_x
from fpswm.evaluate import ATTACKS7, ber_with_search, eval_multiattack
from fpswm.config import HP
from fpswm.models import load_models
from fpswm.utils import first

p = base_parser(__doc__, ckpt=False)
p.add_argument("--ckpt", nargs="+", required=True)
p.add_argument("--labels", nargs="+", default=None)
args = p.parse_args()
device, wm, loader = setup(args)
labels = args.labels or [os.path.basename(c.rstrip("/")) for c in args.ckpt]


@torch.no_grad()
def invisibility(enc):
    enc.eval()
    ps, ss, fl_, tx_ = [], [], [], []
    for batch in loader:
        x = batch_x(batch, device)
        y = first(enc(x, wm))
        ps.append(10 * np.log10(1.0 / max(torch.mean((y - x) ** 2).item(), 1e-12)))
        ss.append(ssim(y, x, data_range=1.0, size_average=True).item())
        mag = (y - x).abs().mean(dim=1, keepdim=True)
        fl = enc.flatness_mask(x) < HP.theta
        if fl.any():
            fl_.append(mag[fl].mean().item())
        if (~fl).any():
            tx_.append(mag[~fl].mean().item())
    return dict(PSNR=np.mean(ps), SSIM=np.mean(ss), R_flat=np.mean(fl_), R_tex=np.mean(tx_))


rows = []
for lab, ck in zip(labels, args.ckpt):
    enc, dec = load_models(ck, "hybrid", device)
    r = {"seed": lab, **invisibility(enc)}
    per, mean = eval_multiattack(enc, dec, loader, wm, device, attacks=ATTACKS7)
    r.update({a: per[a]["BER"] for a in ATTACKS7})
    r.update(ber_with_search(enc, dec, loader, wm, device))
    r["mean_BER"] = mean["BER"]
    rows.append(r)
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    del enc, dec

df = pd.DataFrame(rows).set_index("seed")
df.to_csv(os.path.join(args.out, "p1_table4_5_seeds.csv"))
print("\nmean +- population SD over seeds")
for c in df.columns:
    v = df[c].values.astype(float)
    print(f"{c:14s} {v.mean():.6f} +- {v.std(ddof=0):.6f}  [{v.min():.6f}, {v.max():.6f}]")
