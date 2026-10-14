"""Stage 1 (Section III-C): train the level-3 high-frequency generators fb and the decoder.

Embedding only in LH3 / HL3 / HH3, differentiable JPEG approximation, mask floor 0.35, clamp 0.015.

    python train/train_stage1.py --train-dir COCO/images/train --val-dir COCO/images/val \
        --wm data/watermark_qr_64.png --out checkpoints/stage1
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fpswm.attacks import differentiable_attack_v2
from fpswm.config import HP
from fpswm.data import ResizedCoverDS, batch_x, list_images
from fpswm.evaluate import eval_multiattack, eval_wm_visual_quality, summarize_crop_rotate
from fpswm.losses import compute_loss_v3, flat_visibility_loss
from fpswm.models import build_models
from fpswm.utils import get_device
from fpswm.watermark import load_watermark_bits

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--train-dir", required=True)
p.add_argument("--val-dir", required=True)
p.add_argument("--wm", default="data/watermark_qr_64.png")
p.add_argument("--out", default="checkpoints/stage1")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--val-batches", type=int, default=20)
p.add_argument("--lambda-flat", type=float, default=25.0)
p.add_argument("--device", default=None)
p.add_argument("--num-workers", type=int, default=2)
args = p.parse_args()

device = get_device(args.device)
os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
wm_bits_map = load_watermark_bits(args.wm).to(device)

dl = DataLoader(ResizedCoverDS(list_images(args.train_dir), HP.train_hw, HP.train_hw),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                pin_memory=True, drop_last=True)
dl_val = DataLoader(ResizedCoverDS(list_images(args.val_dir), HP.train_hw, HP.train_hw),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

enc, dec = build_models("stage1", device)          # detail-only forward, f = 0.35, dmax = 0.015
opt = torch.optim.Adam([{"params": enc.parameters(), "lr": args.lr},
                        {"params": dec.parameters(), "lr": args.lr}])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

L_IMG, L_MSG, L_FREQ, L_CONF = 8.0, 14.0, 3.0, 0.8
DETAIL_ROBUST_BOOST = {"jpeg": 1.4, "blur": 1.3, "resize": 1.3}     # beta_det
VAL_ATTACKS = ["none", "jpeg", "crop", "resize", "rotate", "blur", "noise"]
base_attack = {"attack_prob": 0.88, "crop_range": (0.60, 0.92), "noise_std": 0.028, "rot_range": (-20, 20),
               "resize_range": (0.65, 1.0), "blur_sigma": (0.3, 1.5), "jpeg_down": (0.40, 0.92),
               "center_crop_only": False}


def attack_weights(epoch):
    if epoch < 20:
        return {"crop": 0.22, "rotate": 0.22, "jpeg": 0.22, "noise": 0.18, "resize": 0.08, "blur": 0.08}
    return {"crop": 0.20, "rotate": 0.20, "jpeg": 0.20, "noise": 0.20, "resize": 0.10, "blur": 0.10}


def geo_boost(epoch):                                  # beta_geo
    return 2.0 if epoch < 30 else 1.5


best_ber, best_epoch, metrics, wm_cache = 1e9, -1, defaultdict(list), None
theta1 = HP.mask_floor_stage1 + 0.5 * (1 - HP.mask_floor_stage1)
for epoch in range(args.epochs):
    acfg = dict(base_attack, weights=attack_weights(epoch), attack_prob=0.82 if epoch < 5 else 0.88)
    lam_msg_eff = L_MSG * (0.3 + 0.7 * epoch / max(args.epochs - 1, 1))
    enc.train(); dec.train()
    sums, n, atk_count = defaultdict(float), 0, Counter()
    for batch in dl:
        x = batch_x(batch, device)
        if wm_cache is None or wm_cache.shape[0] != x.shape[0]:
            wm_cache = wm_bits_map.repeat(x.shape[0], 1, 1, 1)
        y, delta, mask = enc(x, wm_cache)
        y_att, atk = differentiable_attack_v2(y, acfg, epoch=epoch, total_epochs=args.epochs, return_attack_name=True)
        atk_count[atk] += 1
        logits = dec(y_att)
        if logits.shape[-2:] != wm_cache.shape[-2:]:
            logits = F.interpolate(logits, size=wm_cache.shape[-2:], mode="bilinear", align_corners=False)
        lam_msg = lam_msg_eff * (geo_boost(epoch) if atk in ("crop", "rotate") else 1.0)
        lam_msg *= DETAIL_ROBUST_BOOST.get(atk, 1.0)
        lam_stab = 0.0 if atk in ("crop", "rotate") else 0.02 * L_IMG
        total, l_img, l_msg, _, l_freq, l_conf = compute_loss_v3(
            y, y_att, x, logits, wm_cache, atk, lambda_img=L_IMG, lambda_msg=lam_msg,
            lambda_freq=L_FREQ, lambda_stab=lam_stab, lambda_conf=L_CONF)
        total = total + args.lambda_flat * flat_visibility_loss(delta, mask)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), max_norm=1.0)
        opt.step()
        sums["img"] += l_img.item(); sums["msg"] += l_msg.item(); sums["tot"] += total.item(); n += 1
        with torch.no_grad():
            mag = delta.abs().mean(dim=1, keepdim=True)
            fl = mask < theta1
            if fl.any():
                sums["flat"] += mag[fl].mean().item()
            if (~fl).any():
                sums["tex"] += mag[~fl].mean().item()
    sched.step()
    torch.save({"epoch": epoch + 1, "enc": enc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict()},
               os.path.join(args.out, "checkpoints", f"ckpt_epoch_{epoch + 1:03d}.pt"))

    per, mean = eval_multiattack(enc, dec, dl_val, wm_bits_map, device, attacks=VAL_ATTACKS,
                                 severity=HP.severity, max_batches=args.val_batches)
    vis = None
    if epoch % 5 == 0 or epoch == args.epochs - 1:
        vis = eval_wm_visual_quality(enc, dec, dl_val, wm_bits_map, device, max_batches=10)
    crop_ber, rot_ber, _, _ = summarize_crop_rotate(per)
    print(f"Epoch {epoch + 1}/{args.epochs} | L_img={sums['img'] / n:.5f} L_msg={sums['msg'] / n:.4f} | "
          f"VAL BER={mean['BER']:.4f} | crop={crop_ber:.4f} rot={rot_ber:.4f} | "
          f"flat={sums['flat'] / n:.5f} tex={sums['tex'] / n:.5f}")
    with open(os.path.join(args.out, f"val_epoch_{epoch + 1:03d}.json"), "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch + 1, "per_attack": per, "mean": mean, "wm_visual": vis}, f, indent=2)
    is_best = mean["BER"] < best_ber
    if is_best:
        best_ber, best_epoch = mean["BER"], epoch + 1
        torch.save(enc.state_dict(), os.path.join(args.out, "enc_best.pt"))
        torch.save(dec.state_dict(), os.path.join(args.out, "dec_best.pt"))
        with open(os.path.join(args.out, "best_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"best_epoch": best_epoch, "best_val_mean_ber": best_ber, "mean": mean,
                       "per_attack": per, "wm_visual": vis}, f, indent=2)
        print(f"  [BEST] epoch={best_epoch} BER={best_ber:.4f}")
    metrics["Epoch"].append(epoch + 1)
    metrics["ValMean_BER"].append(mean["BER"])
    metrics["IsBest"].append(int(is_best))
    for a in VAL_ATTACKS:
        metrics[f"Val_BER_{a}"].append(per[a]["BER"])
    pd.DataFrame(metrics).to_csv(os.path.join(args.out, "stage1_metrics.csv"), index=False)

print(f"done: best_epoch={best_epoch} best_val_BER={best_ber:.4f} -> {args.out}")
