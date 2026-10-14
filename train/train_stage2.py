"""Stage 2 (Section III-C): add fu (LL2) and fmid (HL2/LH2) on top of the Stage-1 model.

The level-3 generators fb are frozen by default, the clamp is 0.022 * M, lambda_flat = 40.

  final model (Tables 4, 5, 7, 8 (2)):
      --seeds 17 42 137
  ablation (3) "not frozen" (Table 8):
      --seeds 17 42 137 --unfreeze-detail --detail-lr-mult 0.5 --patience 100

    python train/train_stage2.py --stage1 checkpoints/stage1 --train-dir ... --val-dir ... --out checkpoints
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fpswm.attacks import differentiable_attack_v2
from fpswm.config import HP
from fpswm.data import ResizedCoverDS, batch_x, list_images, make_seeded_dataloader
from fpswm.evaluate import eval_multiattack, eval_wm_visual_quality
from fpswm.losses import compute_loss_v3, flat_visibility_loss
from fpswm.models import DWTWatermarkEncoder, FreqWatermarkDecoder, make_mid_encoder
from fpswm.utils import get_device, set_all_seeds
from fpswm.watermark import load_watermark_bits

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--stage1", required=True, help="Stage-1 folder with enc_best.pt / dec_best.pt")
p.add_argument("--train-dir", required=True)
p.add_argument("--val-dir", required=True)
p.add_argument("--wm", default="data/watermark_qr_64.png")
p.add_argument("--out", default="checkpoints", help="parent folder; one sub-folder per run")
p.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 137])
p.add_argument("--lambda-flat", type=float, nargs="+", default=[40.0])
p.add_argument("--no-clamp", action="store_true", help="train without the clamp")
p.add_argument("--unfreeze-detail", action="store_true", help="also train fb (LH3/HL3/HH3)")
p.add_argument("--detail-lr-mult", type=float, default=None,
               help="separate LR multiplier for fb when unfrozen (ablation (3): 0.5); "
                    "default: same parameter group as fu/fmid")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--val-batches", type=int, default=20)
p.add_argument("--device", default=None)
p.add_argument("--num-workers", type=int, default=2)
args = p.parse_args()

device = get_device(args.device)
wm_bits_map = load_watermark_bits(args.wm).to(device)
ds_train = ResizedCoverDS(list_images(args.train_dir), HP.train_hw, HP.train_hw)
dl_val = DataLoader(ResizedCoverDS(list_images(args.val_dir), HP.train_hw, HP.train_hw),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

L_IMG, L_MSG, L_FREQ, L_CONF = 8.0, 14.0, 3.0, 0.8
ROBUST_BOOST = {"jpeg": 2.0}
GEO_BOOST = 1.2
VAL_ATTACKS = ["none", "jpeg", "crop", "resize", "rotate", "blur", "noise"]
SSIM_WEIGHT, MIN_SSIM_FLOOR = 0.5, 0.10
base_attack = {"attack_prob": 0.88, "crop_range": (0.60, 0.92), "noise_std": 0.028, "rot_range": (-20, 20),
               "resize_range": (0.65, 1.0), "blur_sigma": (0.3, 1.5), "jpeg_down": (0.40, 0.92),
               "center_crop_only": False,
               "weights": {"crop": 0.08, "rotate": 0.14, "jpeg": 0.28, "noise": 0.18, "resize": 0.16, "blur": 0.16}}


def run_name(seed, lam):
    if args.no_clamp:
        return f"noclamp_{'unfrz_' if args.unfreeze_detail else ''}lflat{int(lam)}_seed{seed}"
    if args.unfreeze_detail:
        return f"nofreeze_seed{seed}"
    return f"stage2_seed{seed}" if lam == 40.0 else f"stage2_lflat{int(lam)}_seed{seed}"


def make_models():
    """Same construction order as the notebook (affects random initialisation of fmid)."""
    enc = DWTWatermarkEncoder(base=32)
    dec = FreqWatermarkDecoder(out_hw=HP.wm_hw, base=64)
    enc.add_detail_branches()
    enc.mode = "detail_only"                                     # Stage-1 layout for loading
    enc.load_state_dict(torch.load(os.path.join(args.stage1, "enc_best.pt"), map_location="cpu", weights_only=True))
    dec.load_state_dict(torch.load(os.path.join(args.stage1, "dec_best.pt"), map_location="cpu", weights_only=True))
    enc.wm_encoder_mid = make_mid_encoder()                      # new fmid (seed-dependent init)
    enc.mode = "hybrid"
    enc.mask_floor, enc.dmax = HP.mask_floor, HP.dmax
    enc.use_clamp = not args.no_clamp
    enc.freeze_detail = not args.unfreeze_detail
    return enc.to(device), dec.to(device)


for seed in args.seeds:
    for lam_flat in args.lambda_flat:
        set_all_seeds(seed)
        out = os.path.join(args.out, run_name(seed, lam_flat))
        os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
        enc, dec = make_models()
        for prm in enc.detail_parameters():
            prm.requires_grad_(args.unfreeze_detail)

        trainable = list(enc.wm_encoder_ll.parameters()) + list(enc.wm_encoder_mid.parameters())
        groups = [{"params": trainable, "lr": args.lr}, {"params": dec.parameters(), "lr": args.lr * 0.5}]
        clip_params = trainable + list(dec.parameters())
        if args.unfreeze_detail:
            if args.detail_lr_mult is None:
                trainable += enc.detail_parameters()
                groups[0]["params"] = trainable
            else:
                groups.append({"params": enc.detail_parameters(), "lr": args.lr * args.detail_lr_mult})
            clip_params = trainable + list(dec.parameters()) + (
                enc.detail_parameters() if args.detail_lr_mult is not None else [])
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
        dl = make_seeded_dataloader(ds_train, batch_size=args.batch_size, seed=seed, num_workers=args.num_workers)

        best_comb, best_ber, best_epoch, no_improve = 1e9, 1e9, -1, 0
        metrics, wm_cache = defaultdict(list), None
        print(f"=== {os.path.basename(out)} | clamp={not args.no_clamp} freeze_detail={not args.unfreeze_detail} "
              f"lambda_flat={lam_flat} ===")
        for epoch in range(args.epochs):
            enc.train(); dec.train()
            s_img = s_msg = s_flat = s_tex = 0.0
            n = 0
            for batch in dl:
                x = batch_x(batch, device)
                if wm_cache is None or wm_cache.shape[0] != x.shape[0]:
                    wm_cache = wm_bits_map.repeat(x.shape[0], 1, 1, 1)
                y, delta, mask = enc(x, wm_cache)
                y_att, atk = differentiable_attack_v2(y, base_attack, epoch=epoch, total_epochs=args.epochs,
                                                      return_attack_name=True)
                logits = dec(y_att)
                if logits.shape[-2:] != wm_cache.shape[-2:]:
                    logits = F.interpolate(logits, size=wm_cache.shape[-2:], mode="bilinear", align_corners=False)
                lam_msg = L_MSG * (GEO_BOOST if atk in ("crop", "rotate") else 1.0) * ROBUST_BOOST.get(atk, 1.0)
                lam_stab = 0.0 if atk in ("crop", "rotate") else 0.02 * L_IMG
                total, l_img, l_msg, _, _, _ = compute_loss_v3(
                    y, y_att, x, logits, wm_cache, atk, lambda_img=L_IMG, lambda_msg=lam_msg,
                    lambda_freq=L_FREQ, lambda_stab=lam_stab, lambda_conf=L_CONF)
                total = total + lam_flat * flat_visibility_loss(delta, mask)
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                opt.step()
                s_img += l_img.item(); s_msg += l_msg.item(); n += 1
                with torch.no_grad():
                    mag = delta.abs().mean(dim=1, keepdim=True)
                    fl = mask < HP.theta
                    if fl.any():
                        s_flat += mag[fl].mean().item()
                    if (~fl).any():
                        s_tex += mag[~fl].mean().item()
            sched.step()
            torch.save(enc.state_dict(), os.path.join(out, "checkpoints", "enc_last.pt"))
            torch.save(dec.state_dict(), os.path.join(out, "checkpoints", "dec_last.pt"))

            per, mean = eval_multiattack(enc, dec, dl_val, wm_bits_map, device, attacks=VAL_ATTACKS,
                                         severity=HP.severity, max_batches=args.val_batches)
            vis = eval_wm_visual_quality(enc, dec, dl_val, wm_bits_map, device, max_batches=10)
            wm_ssim = float(np.mean([vis[a]["WM_SSIM"] for a in vis]))
            comb = mean["BER"] - SSIM_WEIGHT * wm_ssim
            print(f"Epoch {epoch + 1}/{args.epochs} | L_img={s_img / n:.5f} L_msg={s_msg / n:.4f} | "
                  f"VAL BER={mean['BER']:.4f} | SSIM_avg={wm_ssim:.4f} | flat={s_flat / n:.5f} tex={s_tex / n:.5f}")
            is_best = comb < best_comb and wm_ssim > MIN_SSIM_FLOOR
            if is_best:
                best_comb, best_ber, best_epoch, no_improve = comb, mean["BER"], epoch + 1, 0
                torch.save(enc.state_dict(), os.path.join(out, "enc_best.pt"))
                torch.save(dec.state_dict(), os.path.join(out, "dec_best.pt"))
                with open(os.path.join(out, "best_summary.json"), "w", encoding="utf-8") as f:
                    json.dump({"best_epoch": best_epoch, "best_val_mean_ber": best_ber, "wm_ssim_avg": wm_ssim,
                               "delta_flat_mean": round(s_flat / n, 6), "per_attack": per,
                               "use_clamp": not args.no_clamp, "lambda_flat": lam_flat,
                               "freeze_detail": not args.unfreeze_detail, "seed": seed}, f, indent=2)
                print(f"  [BEST] ep={best_epoch} BER={best_ber:.4f} SSIM={wm_ssim:.4f}")
            else:
                no_improve += 1
            metrics["Epoch"].append(epoch + 1)
            metrics["ValMean_BER"].append(mean["BER"])
            metrics["WM_SSIM_avg"].append(wm_ssim)
            metrics["Delta_flat_mean"].append(round(s_flat / n, 6))
            metrics["IsBest"].append(int(is_best))
            for a in VAL_ATTACKS:
                metrics[f"Val_BER_{a}"].append(per[a]["BER"])
            pd.DataFrame(metrics).to_csv(os.path.join(out, "metrics.csv"), index=False)
            if no_improve >= args.patience:
                print(f"early stopping, best_epoch={best_epoch}")
                break
        print(f"done {os.path.basename(out)}: best_epoch={best_epoch} best_BER={best_ber:.4f}")
        del opt, sched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
