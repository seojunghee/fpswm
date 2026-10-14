"""Data loading.

Training : images resized to 256x256 (bicubic), batch 4.
Evaluation: original resolution, batch 1, the first ``n`` images of the sorted validation folder.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root_dir):
    """All images under root_dir (recursive), sorted by relative path."""
    items = []
    root_dir = os.path.abspath(root_dir)
    for cur, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                p = os.path.join(cur, f)
                items.append((p, os.path.relpath(p, root_dir)))
    items.sort(key=lambda t: t[1])
    return [p for p, _ in items]


def read_list(list_file, root_dir):
    """Image list file (one relative path per line) -> absolute paths."""
    with open(list_file, encoding="utf-8") as f:
        rel = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return [os.path.join(root_dir, r) for r in rel]


def load_image(path, crop8=False):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if crop8:
        a = a[: a.shape[0] - a.shape[0] % 8, : a.shape[1] - a.shape[1] % 8]
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)


class FullResCoverDS(Dataset):
    """Original resolution, no resizing (batch size must be 1)."""

    def __init__(self, paths, crop8=False):
        self.paths, self.crop8 = paths, crop8

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_image(self.paths[i], self.crop8)


class ResizedCoverDS(Dataset):
    def __init__(self, paths, H=256, W=256):
        self.paths, self.H, self.W = paths, H, W

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((self.W, self.H), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def make_seeded_dataloader(dataset, batch_size, seed, shuffle=True, num_workers=2):
    """Shuffle order and worker RNG tied to the seed."""
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id):
        ws = seed + worker_id
        np.random.seed(ws % (2 ** 32))
        random.seed(ws)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True, generator=g, worker_init_fn=_worker_init)


def val_paths(val_dir, n=None, list_file=None):
    paths = read_list(list_file, val_dir) if list_file else list_images(val_dir)
    return paths[:n] if n else paths


def fullres_loader(val_dir, n=None, list_file=None, crop8=False, num_workers=2):
    """Evaluation loader: original resolution, batch 1, first n images."""
    ds = FullResCoverDS(val_paths(val_dir, n, list_file), crop8=crop8)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)


def batch_x(batch, device):
    return (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
