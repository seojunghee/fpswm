"""Write the sorted list of validation images used in the papers (first N files of the folder).

    python tools/make_val_list.py --val-dir /path/to/coco/images/val --n 200 --out data/val_list.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpswm.data import list_images  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--val-dir", required=True)
p.add_argument("--n", type=int, default=200)
p.add_argument("--out", default="data/val_list.txt")
a = p.parse_args()
paths = list_images(a.val_dir)[: a.n]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for q in paths:
        f.write(os.path.relpath(q, os.path.abspath(a.val_dir)) + "\n")
print(f"wrote {len(paths)} paths to {a.out}")
