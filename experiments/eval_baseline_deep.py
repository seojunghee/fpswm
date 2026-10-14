"""Table 7 (HiDDeN and MBRS columns), evaluated with the same attacks as the proposed method.

Requires the authors' code and weights (not redistributed here):
  HiDDeN  https://github.com/ando-khachatryan/HiDDeN   run folder "combined-noise" (400 epochs)
  MBRS    https://github.com/jzyustc/MBRS                weights results/MBRS_256_m256/models/EC_42.pth
The message is the leading L bits of the QR map. HiDDeN is reported at its training resolution (128x128)
and additionally at the original resolution; MBRS at 256x256.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fpswm.attacks import attack_val_v2
from fpswm.cli import base_parser, setup
from fpswm.data import batch_x

p = base_parser(__doc__, ckpt=False)
p.add_argument("--hidden-repo", required=True)
p.add_argument("--hidden-run", required=True, help="e.g. <HiDDeN>/experiments/combined-noise")
p.add_argument("--mbrs-repo", required=True)
p.add_argument("--mbrs-ckpt", required=True)
args = p.parse_args()
device, wm, loader = setup(args)
ATTACKS = ("none", "jpeg", "resize", "blur", "noise", "crop", "rotate")
_root_paths = [args.hidden_repo, args.mbrs_repo]


def _isolate(repo):
    for o in _root_paths:
        while o in sys.path:
            sys.path.remove(o)
    sys.path.insert(0, repo)
    for name in list(sys.modules):
        if name == "utils" or name.startswith(("utils.", "network", "model", "noise_layers", "options")):
            del sys.modules[name]


to_signed = lambda x: x * 2.0 - 1.0
to_unit = lambda x: ((x + 1.0) / 2.0).clamp(0, 1)


@torch.no_grad()
def run_baseline(name, embed_fn, extract_fn, in_hw, L, full_res=False):
    msg = (wm > 0.5).float().flatten()[:L].view(1, L).to(device)
    row = {"model": name, "input": "original" if full_res else f"{in_hw[0]}x{in_hw[1]}"}
    psnrs, px = [], []
    for atk in ATTACKS:
        bers = []
        for batch in loader:
            x = batch_x(batch, device)
            if not full_res:
                x = F.interpolate(x, size=in_hw, mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)
            y = to_unit(embed_fn(to_signed(x), msg))
            if atk == "none":
                psnrs.append(10 * np.log10(1.0 / max(torch.mean((y - x) ** 2).item(), 1e-12)))
                px.append(x.shape[-2] * x.shape[-1])
            ya = attack_val_v2(y, atk, 0.7) if atk != "none" else y
            lg = extract_fn(to_signed(ya))
            pred = (lg > 0).float() if lg.min() < 0 else (lg > 0.5).float()
            bers.append((pred != msg).float().mean().item())
        row[atk] = float(np.mean(bers))
        print(f"[{name}] {atk:<7} BER={row[atk]:.4f}")
    row.update(PSNR=float(np.mean(psnrs)), bits=L, bpp=L / float(np.mean(px)))
    row["bits_recovered_jpeg"] = L * (1 - row["jpeg"])
    return row


def load_hidden():
    _isolate(args.hidden_repo)
    import utils as hutils
    from model.hidden import Hidden
    from noise_layers.noiser import Noiser
    _, hcfg, _ = hutils.load_options(os.path.join(args.hidden_run, "options-and-config.pickle"))
    model = Hidden(hcfg, device, Noiser([], device), None)
    ckpt, _ = hutils.load_last_checkpoint(os.path.join(args.hidden_run, "checkpoints"))
    hutils.model_from_checkpoint(model, ckpt)
    ed = model.encoder_decoder
    ed = ed.module if hasattr(ed, "module") else ed
    ed.eval()

    def embed(xs, m):
        ed.encoder.H, ed.encoder.W = xs.shape[-2], xs.shape[-1]
        return ed.encoder(xs, m)
    return embed, ed.decoder, (hcfg.H, hcfg.W), hcfg.message_length


def load_mbrs(H=256, W=256, L=256):
    _isolate(args.mbrs_repo)
    from network.Network import Network
    net = Network(H, W, L, ["Identity()"], device, 1, 1e-3, False)
    ed = net.encoder_decoder.module if hasattr(net.encoder_decoder, "module") else net.encoder_decoder
    ed.load_state_dict(torch.load(args.mbrs_ckpt, map_location=device))
    ed.eval()
    return (lambda xs, m: ed.encoder(xs, m)), (lambda ys: ed.decoder(ys)), (H, W), L


rows = []
e, d, hw, L = load_hidden()
rows.append(run_baseline("HiDDeN", e, d, hw, L, full_res=False))
rows.append(run_baseline("HiDDeN", e, d, hw, L, full_res=True))
e, d, hw, L = load_mbrs()
rows.append(run_baseline("MBRS", e, d, hw, L, full_res=False))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.out, "p1_table7_baseline_deep.csv"), index=False)
print(df.round(4).to_string(index=False))
