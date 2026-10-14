"""Quick checks that need no data or checkpoints:  python -m pytest tests  (or python tests/test_smoke.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fpswm import attack_val_v2, build_models, compute_flatness_mask, extract_watermark


def test_bound_and_shapes():
    torch.manual_seed(0)
    enc, dec = build_models("hybrid")
    enc.eval(); dec.eval()
    wm = torch.randint(0, 2, (1, 1, 64, 64)).float()
    for shape in [(1, 3, 256, 256), (1, 3, 203, 157)]:
        x = torch.rand(shape)
        with torch.no_grad():
            y = enc(x, wm)
            assert y.shape == x.shape
            ratio = ((y - x).abs() / (enc.dmax * compute_flatness_mask(x))).max().item()
            assert ratio <= 1 + 1e-5, ratio
            assert dec(attack_val_v2(y, "jpeg", 0.7)).shape == (1, 1, 64, 64)
            pred, info = extract_watermark(dec, attack_val_v2(y, "rotate", 0.7))
            assert pred.shape == (1, 1, 64, 64)


def test_state_dict_roundtrip():
    enc, dec = build_models("hybrid")
    enc2, dec2 = build_models("hybrid")
    enc2.load_state_dict(enc.state_dict(), strict=True)
    dec2.load_state_dict(dec.state_dict(), strict=True)
    enc1, _ = build_models("stage1")
    enc1.load_state_dict(enc1.state_dict(), strict=True)


if __name__ == "__main__":
    test_bound_and_shapes(); test_state_dict_roundtrip(); print("ok")
