"""Table 6: BER as a function of the JPEG quality factor, 4:4:4 and 4:2:0 (real PIL codec, plain decoding)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from fpswm.cli import base_parser, setup
from fpswm.data import batch_x
from fpswm.models import load_models
from fpswm.utils import first

QUALITIES = (100, 95, 90, 85, 80, 75, 70, 65, 60, 58, 55, 50)
args = base_parser(__doc__).parse_args()
device, wm, loader = setup(args)
enc, dec = load_models(args.ckpt, "hybrid", device)


def jpeg_trunc(y, q, subsampling):
    """As in the notebook for this table: float -> uint8 by truncation, then PIL JPEG."""
    out = []
    for b in range(y.shape[0]):
        arr = (y[b].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(q), subsampling=subsampling, optimize=False)
        buf.seek(0)
        out.append(torch.from_numpy(np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0).permute(2, 0, 1))
    return torch.stack(out).to(y.device)


@torch.no_grad()
def sweep(subsampling):
    rows = []
    for q in QUALITIES:
        bers = []
        for batch in loader:
            x = batch_x(batch, device)
            y = first(enc(x, wm))
            ya = jpeg_trunc(y, q, subsampling)
            lg = dec(ya)
            if lg.shape[-2:] != wm.shape[-2:]:
                lg = F.interpolate(lg, size=wm.shape[-2:], mode="bilinear", align_corners=False)
            bers.append(((torch.sigmoid(lg) > 0.5).float() != wm).float().mean().item())
        rows.append({"quality": q, "BER": float(np.mean(bers))})
        print(f"subsampling={subsampling} QF={q:3d} BER={rows[-1]['BER']:.4f}")
    return pd.DataFrame(rows)


d444, d420 = sweep(0), sweep(2)
df = d444.rename(columns={"BER": "BER_444"}).merge(d420.rename(columns={"BER": "BER_420"}), on="quality")
df.to_csv(os.path.join(args.out, "p1_table6_jpeg_sweep.csv"), index=False)
print(df.round(4).to_string(index=False))
