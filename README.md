# FPS-WM

Code and trained models for

**FPS-WM: Frequency-path separation and inference-time search for high-capacity blind image watermarking**
Jung Hee Seo, submitted to IEEE Access, 2026.

FPS-WM embeds a 64×64 binary QR watermark (4,096 bits) in the subbands of a three-level Haar DWT.
- **Stage 1** trains the level-3 high-frequency residual generators fb (LH3, HL3, HH3) under a differentiable JPEG approximation.
- **Stage 2** adds the level-2 low-frequency generator fu (LL2) and the mid-band generator fmid (HL2, LH2), with fb held fixed.
- A four-path decoder recovers the watermark.
- At inference, 41 rotation and scale candidates are searched and adopted under a central-confidence margin rule.

> Status: submitted to IEEE Access (under review).

## Repository layout

```
fpswm/                 package
  models.py            encoder (fb, fu, fmid, residual clamp) and four-path decoder; checkpoint loading
  dwt.py               differentiable Haar DWT
  mask.py              flatness mask  M = f + (1 - f) v / (v + s)
  attacks.py           evaluation attacks (real PIL JPEG, ...) and the training attack layer (differentiable JPEG)
  search.py            41-candidate rotation / scale search, central-confidence margin rule
  losses.py            composite training loss (Eq. (12), Table 2)
  evaluate.py          BER / residual / PSNR helpers
  data.py, watermark.py, cli.py, config.py, utils.py
train/                 Stage-1 and Stage-2 training
experiments/           scripts for Tables 4-9
tools/make_val_list.py writes the list of evaluation images
tests/test_smoke.py    quick check without data or checkpoints
data/                  watermark image and image list (no COCO images)
```

## Installation

Tested with Python 3.13, PyTorch 2.11.0 and torchvision 0.26.0 (CUDA 13.0, NVIDIA A100, Google Colab).

```bash
git clone https://github.com/seojunghee/fpswm.git
cd fpswm
pip install -r requirements.txt
python tests/test_smoke.py        # prints "ok"
```

## Data

- **Images.** MS-COCO is not redistributed here.
  - The training folder holds 3,083 COCO images and the validation folder 1,000 COCO images, randomly sampled from COCO train2017 and val2017, respectively.
  - Training resizes every image to 256×256 (bicubic), with batch size 4.
  - Model selection during training uses the first 20 validation batches (80 images, 256×256).
  - Evaluation uses the first 200 validation files (sorted by name) at their original resolution. The exact list is `data/val_list.txt`, written by `python tools/make_val_list.py --val-dir <val folder> --n 200`.
- **Watermark.** `data/watermark_qr_64.png` is the 64×64 binary QR image (pixels > 127 are 1).

## Pretrained models

Download `fpswm_checkpoints_v1.0.zip` from the [Release v1.0](https://github.com/seojunghee/fpswm/releases) page and unpack it into `checkpoints/`.

| Folder | Model | Used for |
|---|---|---|
| `stage1/` | Stage 1 (fb only, trained with δmax = 0.015, f = 0.35) | Stage-2 initialisation; Table 8 (1) |
| `final_r3/` | Stage-2 reference run | Tables 6, 7 (PSNR calibration of block DCT), 9 |
| `stage2_seed17/`, `stage2_seed42/`, `stage2_seed137/` | Stage 2, three seeds (proposed) | Tables 4, 5, 7, 8 (2) |
| `nofreeze_seed17/`, `nofreeze_seed42/`, `nofreeze_seed137/` | Stage 2 without freezing fb | Table 8 (3) |

Each folder contains `enc_best.pt` and `dec_best.pt`:

```python
from fpswm import load_models, load_watermark_bits, extract_watermark
enc, dec = load_models("checkpoints/stage2_seed17", kind="hybrid", device="cuda")
wm = load_watermark_bits("data/watermark_qr_64.png").cuda()
y = enc(x, wm)                           # x: (1, 3, H, W) in [0, 1]
bits, info = extract_watermark(dec, y)   # (1, 1, 64, 64), with rotation / scale search
```

## Training

```bash
# Stage 1: fb only, differentiable JPEG, 100 epochs, lr 1e-3
python train/train_stage1.py --train-dir $TRAIN --val-dir $VAL --out checkpoints/stage1

# Stage 2: fu + fmid, fb frozen, lr 3e-4, three seeds
python train/train_stage2.py --stage1 checkpoints/stage1 --train-dir $TRAIN --val-dir $VAL \
    --out checkpoints --seeds 17 42 137

# Table 8 (3): Stage 2 without freezing fb
python train/train_stage2.py --stage1 checkpoints/stage1 --train-dir $TRAIN --val-dir $VAL \
    --out checkpoints --seeds 17 42 137 --unfreeze-detail --detail-lr-mult 0.5 --patience 100
```

## Reproducing the tables

`$C` stands for `--val-dir $VAL --val-list data/val_list.txt`. Results are written as CSV files to `results/`.

| Result | Command |
|---|---|
| Tables 4, 5 | `python experiments/eval_seeds.py $C --ckpt checkpoints/stage2_seed17 checkpoints/stage2_seed42 checkpoints/stage2_seed137 --labels 17 42 137` |
| Table 6 | `python experiments/eval_jpeg_sweep.py $C --ckpt checkpoints/final_r3` |
| Table 7, block DCT | `python experiments/eval_baseline_dct.py $C --ckpt checkpoints/final_r3` |
| Table 7, HiDDeN / MBRS | `python experiments/eval_baseline_deep.py $C --hidden-repo <HiDDeN> --hidden-run <HiDDeN>/experiments/combined-noise --mbrs-repo <MBRS> --mbrs-ckpt <MBRS>/results/MBRS_256_m256/models/EC_42.pth` |
| Table 8 | `python experiments/eval_ablation.py $C --stage1 checkpoints/stage1 --frozen checkpoints/stage2_seed17 checkpoints/stage2_seed42 checkpoints/stage2_seed137 --unfrozen checkpoints/nofreeze_seed17 checkpoints/nofreeze_seed42 checkpoints/nofreeze_seed137` |
| Table 9 | `python experiments/eval_severity.py $C --ckpt checkpoints/final_r3` |

In Table 8, all three models are evaluated with the same bound, δmax = 0.022 and f = 0.20. This is the default of `eval_ablation.py`. Add `--stage1-native` to evaluate model (1) with its training values instead (δmax = 0.015, f = 0.35).

HiDDeN (https://github.com/ando-khachatryan/HiDDeN) and MBRS (https://github.com/jzyustc/MBRS) are used through their authors' code and weights, which are not redistributed here.

## Evaluation protocol

- 200 validation images at original resolution, batch size 1, inference mode.
- Severity 0.7 for every attack:
  - JPEG: real PIL codec, QF 58, 4:4:4 subsampling.
  - Crop: centre crop to 72 %, resized back.
  - Rotation: 10.5° clockwise.
  - Resize: to 72 % and back.
  - Blur: Gaussian, σ = 1.14.
  - Noise: σ = 0.021, with its low-frequency part removed.
- Crop and rotation are decoded with the inference-time search, and all other conditions with plain decoding.

## Reproducibility

- Stage 2 fixes the Python, NumPy and torch seeds and the data loader shuffle order. cuDNN kernels are not forced to be deterministic, so retraining reproduces the reported values within run-to-run variation. The released checkpoints reproduce the tables exactly.
- The forward pass, attacks, differentiable JPEG and search were checked against the original notebook code. On identical weights the outputs agree to within 2.4 × 10⁻⁷, and the attacks and search agree exactly.

## Citation

```bibtex
@article{fpswm2026,
  title   = {FPS-WM: Frequency-path separation and inference-time search for high-capacity blind image watermarking},
  author  = {Jung Hee Seo},
  journal = {submitted to IEEE Access},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). MS-COCO images are subject to their own terms of use.
