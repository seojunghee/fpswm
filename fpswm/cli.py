"""Common command-line options for the experiment scripts."""
import argparse
import os

from .config import HP
from .data import fullres_loader
from .utils import get_device
from .watermark import load_watermark_bits


def base_parser(description, ckpt=True):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--val-dir", required=True, help="COCO validation image folder")
    p.add_argument("--val-list", default=None,
                   help="optional list of relative image paths (default: sorted folder listing)")
    p.add_argument("--wm", default="data/watermark_qr_64.png", help="64x64 binary QR watermark image")
    if ckpt:
        p.add_argument("--ckpt", required=True, help="folder containing enc_best.pt and dec_best.pt")
    p.add_argument("--out", default="results", help="output folder for CSV / figures")
    p.add_argument("--n-eval", type=int, default=HP.n_eval, help="number of validation images")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=2)
    return p


def setup(args, crop8=False):
    """Returns device, wm_bits_map (1,1,64,64 on device), full-resolution loader."""
    os.makedirs(args.out, exist_ok=True)
    device = get_device(args.device)
    wm = load_watermark_bits(args.wm).to(device)
    loader = fullres_loader(args.val_dir, args.n_eval, args.val_list, crop8=crop8, num_workers=args.num_workers)
    return device, wm, loader
