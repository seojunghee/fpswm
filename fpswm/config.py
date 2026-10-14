"""Hyperparameters of the final model (paper, Table 3).

Values are the ones used in the experiments. They are stored as default attributes on the
encoder (see ``DWTWatermarkEncoder``), so an experiment changes them on the model object
(e.g. ``enc.dmax = 0.0243``) instead of through notebook globals.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Hyper:
    # image / watermark
    train_hw: int = 256            # training images are resized to 256x256
    wm_hw: int = 64                # 64x64 QR map = 4,096 bits

    # flatness mask  M = f + (1 - f) * v / (v + s)
    mask_kernel: int = 9           # k
    mask_scale: float = 1.5e-3     # s
    mask_floor: float = 0.20       # f   (Stage 2 / final model)
    mask_floor_stage1: float = 0.35

    # residual generators
    ll2_strength: float = 0.035    # a2 (fu, and 0.3*a2 for fmid)
    level3_strength: float = 0.10  # a3 (fb)

    # content-adaptive hard clamp  |Delta| <= dmax * M
    dmax: float = 0.022
    dmax_stage1: float = 0.015

    # flat-region threshold used for R_flat (theta = f + 0.5 (1 - f) = 0.60)
    theta: float = 0.60

    # evaluation
    severity: float = 0.7          # JPEG QF 58, crop 72 %, rotation 10.5 deg
    n_eval: int = 200              # first 200 validation images (sorted file names)


HP = Hyper()
