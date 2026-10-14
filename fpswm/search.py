"""Inference-time search (paper, Section III): 41 rotation / scale candidates, central-confidence margin rule."""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

MARGIN = 0.03


def center_confidence(logits, frac=0.6):
    """Mean |sigmoid - 0.5| over the central frac x frac window (ignores padded borders)."""
    prob = torch.sigmoid(logits)
    _, _, H, W = prob.shape
    h0, h1 = int(H * (1 - frac) / 2), int(H * (1 + frac) / 2)
    w0, w1 = int(W * (1 - frac) / 2), int(W * (1 + frac) / 2)
    return (prob[:, :, h0:h1, w0:w1] - 0.5).abs().mean().item()


@torch.no_grad()
def extract_watermark(dec, image, rotation_angles=range(-20, 21, 2), scale_candidates=None, verbose=False):
    """Decode with search. Pass rotation_angles=[] and scale_candidates=[] for plain decoding.

    Default candidates: plain + 20 rotations (-20..20 deg, step 2, excluding 0) + 20 scales (0.98..0.60).
    A candidate is adopted only if its central confidence exceeds the current best by MARGIN.
    Returns (pred_bits (B,1,64,64), {"method", "confidence"}).
    """
    if scale_candidates is None:
        scale_candidates = [round(1.0 - 0.02 * i, 2) for i in range(21)]
    _, _, H, W = image.shape
    logits = dec(image)
    cands = [("plain", logits, center_confidence(logits))]
    for a in rotation_angles:
        if a == 0:
            continue
        y = TF.rotate(image, angle=a, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0).clamp(0, 1)
        lg = dec(y)
        cands.append((f"rotate{a:+d}", lg, center_confidence(lg)))
    for s in scale_candidates:
        if s >= 0.999:
            continue
        nh, nw = max(1, int(H * s)), max(1, int(W * s))
        small = F.interpolate(image, size=(nh, nw), mode='bilinear', align_corners=False)
        pt, pl = (H - nh) // 2, (W - nw) // 2
        y = torch.zeros_like(image)
        y[:, :, pt:pt + nh, pl:pl + nw] = small
        lg = dec(y)
        cands.append((f"scale{s:.2f}", lg, center_confidence(lg)))
    best_m, best_lg, best_c = cands[0]
    for m, lg, c in cands[1:]:
        if c > best_c + MARGIN:
            best_m, best_lg, best_c = m, lg, c
    if verbose:
        print(f"  adopted: {best_m} (conf={best_c:.4f})")
    return (torch.sigmoid(best_lg) > 0.5).float(), {"method": best_m, "confidence": best_c}


def ber_from_pred(pred, wm_gt):
    pred = pred.float()
    if pred.dim() == 3:
        pred = pred[:, None]
    if pred.shape[-2:] != wm_gt.shape[-2:]:
        pred = F.interpolate(pred, size=wm_gt.shape[-2:], mode='nearest')
    return ((pred > 0.5).float() != wm_gt).float().mean().item()


@torch.no_grad()
def ber_decode(dec, image, wm_gt, search=False):
    """BER of one (attacked) image; search=False is plain decoding."""
    if search:
        pred, _ = extract_watermark(dec, image)
    else:
        pred, _ = extract_watermark(dec, image, rotation_angles=[], scale_candidates=[])
    return ber_from_pred(pred, wm_gt.expand(image.shape[0], -1, -1, -1))
