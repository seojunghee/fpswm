"""Table 8: (1) Stage-1 high-frequency-only model, (2) frozen Stage-2 model (proposed),
(3) Stage-2 model trained without freezing fb.

Note: in the measurement reported in the paper, model (1) was evaluated with the Stage-2 mask floor
(f = 0.20) and bound (delta_max = 0.022), which are the defaults here. Use --stage1-native to evaluate it
with its training values (f = 0.35, delta_max = 0.015).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fpswm.cli import base_parser, setup
from fpswm.config import HP
from fpswm.evaluate import ATTACKS7, ber_with_search, eval_multiattack, mean_psnr
from fpswm.models import load_models

p = base_parser(__doc__, ckpt=False)
p.add_argument("--stage1", required=True)
p.add_argument("--frozen", nargs="+", required=True, help="Stage-2 seed folders (proposed)")
p.add_argument("--unfrozen", nargs="+", required=True, help="Stage-2 seed folders trained with --unfreeze-detail")
p.add_argument("--stage1-native", action="store_true")
args = p.parse_args()
device, wm, loader = setup(args)


def evaluate(enc, dec, tag, path):
    per, _ = eval_multiattack(enc, dec, loader, wm, device, attacks=ATTACKS7)
    r = {"model": tag, "ckpt": path, **{a: per[a]["BER"] for a in ATTACKS7}}
    r.update(ber_with_search(enc, dec, loader, wm, device))
    r["PSNR"] = mean_psnr(enc, loader, wm, device)
    print(r)
    return r


rows = []
enc, dec = load_models(args.stage1, "stage1", device)
if not args.stage1_native:
    enc.mask_floor, enc.dmax = HP.mask_floor, HP.dmax
rows.append(evaluate(enc, dec, "(1) high-frequency only", args.stage1))
for tag, dirs in (("(2) frozen", args.frozen), ("(3) not frozen", args.unfrozen)):
    for d in dirs:
        enc, dec = load_models(d, "hybrid", device)
        rows.append(evaluate(enc, dec, tag, d))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.out, "p1_table8_ablation.csv"), index=False)
num = df.drop(columns=["ckpt"]).groupby("model")
print(num.mean().round(4).to_string())
print(num.std(ddof=0).round(4).to_string())
