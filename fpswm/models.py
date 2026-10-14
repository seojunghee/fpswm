"""Encoder / decoder of FPS-WM.

Residual generators (paper, Table 1)
    fb   : wm_encoder_lh3 / wm_encoder_hl3 / wm_encoder_hh3   (level-3 high-frequency, per subband)
    fu   : wm_encoder_ll                                       (level-2 low-frequency, LL2)
    fmid : wm_encoder_mid                                      (mid-band, HL2 / LH2)

Module names are kept identical to the training notebook so that the released checkpoints
(``enc_best.pt`` / ``dec_best.pt``) load with ``strict=True``. ``wm_encoder_detail3`` and the
Stage-1 version of ``wm_encoder_mid`` are not used by the final forward pass but are part of
those checkpoints, so they are still created.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HP
from .dwt import HaarDWT
from .mask import compute_flatness_mask


def make_detail_encoder(in_ch=4, feat_ch=3):
    """fb: Conv3x3-ReLU-Conv3x3, 4 -> 16 -> 3 channels (one per level-3 detail subband)."""
    return nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(16, feat_ch, 3, padding=1))


def make_mid_encoder():
    """fmid: Conv3x3-ReLU-Conv3x3, 7 -> 16 -> 6 channels (HL2 and LH2)."""
    return nn.Sequential(nn.Conv2d(7, 16, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(16, 6, 3, padding=1))


class DWTWatermarkEncoder(nn.Module):
    """Three-level Haar DWT encoder with a content-adaptive hard clamp.

    mode = "hybrid"      : final model (Stage 2). fu + fmid + fb, clamp dmax * M.
    mode = "detail_only" : Stage-1 model. fb only (LH3/HL3/HH3).

    Runtime switches (plain attributes, not saved in the state dict)
        dmax           residual bound delta_max            (default 0.022)
        use_clamp      apply |Delta| <= dmax * M           (False: no clamp)            
        freeze_detail  run fb under no_grad                (Stage 2: True)
        mask_floor     floor f of the flatness mask        (default 0.20)
    """

    def __init__(self, base=32, strength=0.05):
        super().__init__()
        self.dwt = HaarDWT()
        # fu (level-2 low-frequency): Conv-BN-ReLU-Conv-BN-ReLU-Conv1x1, 4 -> 32 -> 32 -> 3
        self.wm_encoder_ll = nn.Sequential(
            nn.Conv2d(1 + 3, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 1),
        )
        # legacy modules (present in the checkpoints; replaced / unused by the final forward)
        self.wm_encoder_mid = nn.Sequential(
            nn.Conv2d(1 + 3 * 2, base // 2, 3, padding=1), nn.BatchNorm2d(base // 2), nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, 3 * 2, 1),
        )
        self.wm_encoder_detail3 = nn.Sequential(
            nn.Conv2d(1 + 3 * 3, base // 2, 3, padding=1), nn.BatchNorm2d(base // 2), nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, 3 * 3, 1),
        )
        self.strength = strength  # legacy, unused
        # runtime switches
        self.mode = "hybrid"
        self.dmax = HP.dmax
        self.use_clamp = True
        self.freeze_detail = True
        self.mask_floor = HP.mask_floor
        self.ll2_strength = HP.ll2_strength
        self.level3_strength = HP.level3_strength

    def add_detail_branches(self):
        self.wm_encoder_lh3 = make_detail_encoder(in_ch=4)
        self.wm_encoder_hl3 = make_detail_encoder(in_ch=4)
        self.wm_encoder_hh3 = make_detail_encoder(in_ch=4)
        return self

    def detail_parameters(self):
        return (list(self.wm_encoder_lh3.parameters()) + list(self.wm_encoder_hl3.parameters())
                + list(self.wm_encoder_hh3.parameters()))

    def flatness_mask(self, x):
        return compute_flatness_mask(x, floor=self.mask_floor)

    # ------------------------------------------------------------------ clamp
    def _clamp(self, y_raw, x, mask):
        if self.use_clamp:
            m = self.dmax * mask
            delta = torch.max(torch.min((y_raw - x).clamp(-1, 1), m), -m)
        else:
            delta = (y_raw - x).clamp(-1, 1)
        return delta

    def _detail_deltas(self, wm_l3, lh3, hl3, hh3, mask_l3):
        a3 = self.level3_strength
        d_lh3 = torch.tanh(self.wm_encoder_lh3(torch.cat([wm_l3, lh3], dim=1))) * a3 * mask_l3
        d_hl3 = torch.tanh(self.wm_encoder_hl3(torch.cat([wm_l3, hl3], dim=1))) * a3 * mask_l3
        d_hh3 = torch.tanh(self.wm_encoder_hh3(torch.cat([wm_l3, hh3], dim=1))) * a3 * mask_l3
        return d_lh3, d_hl3, d_hh3

    # ---------------------------------------------------------------- forward
    def forward(self, x, wm_map):
        if self.mode == "detail_only":
            return self._forward_detail_only(x, wm_map)
        return self._forward_hybrid(x, wm_map)

    def _forward_hybrid(self, x, wm_map):
        B, C, H, W = x.shape
        if wm_map.shape[-2:] != (H, W):
            wm_map = F.interpolate(wm_map, size=(H, W), mode='nearest')
        mask = self.flatness_mask(x)

        ll1, lh1, hl1, hh1 = self.dwt.forward_dwt(x)
        ll2, lh2, hl2, hh2 = self.dwt.forward_dwt(ll1)
        ll3, lh3, hl3, hh3 = self.dwt.forward_dwt(ll2)

        wm_l2 = F.interpolate(wm_map, size=ll2.shape[-2:], mode='nearest')
        wm_l3 = F.interpolate(wm_map, size=ll3.shape[-2:], mode='nearest')
        mask_l2 = F.interpolate(mask, size=ll2.shape[-2:], mode='bilinear', align_corners=False)
        mask_l3 = F.interpolate(mask, size=ll3.shape[-2:], mode='bilinear', align_corners=False)

        a2 = self.ll2_strength
        # fu: LL2
        delta_ll2 = torch.tanh(self.wm_encoder_ll(torch.cat([wm_l2, ll2], dim=1))) * a2 * mask_l2
        ll2_wm = ll2 + delta_ll2
        # fmid: HL2 / LH2
        delta_mid = torch.tanh(self.wm_encoder_mid(torch.cat([wm_l2, hl2, lh2], dim=1))) * a2 * 0.3 * mask_l2
        hl2_wm = hl2 + delta_mid[:, :3]
        lh2_wm = lh2 + delta_mid[:, 3:]
        # fb: LH3 / HL3 / HH3 (frozen in Stage 2)
        if self.freeze_detail:
            with torch.no_grad():
                d_lh3, d_hl3, d_hh3 = self._detail_deltas(wm_l3, lh3, hl3, hh3, mask_l3)
        else:
            d_lh3, d_hl3, d_hh3 = self._detail_deltas(wm_l3, lh3, hl3, hh3, mask_l3)

        ll2_from_detail = self.dwt.inverse_dwt(ll3, lh3 + d_lh3, hl3 + d_hl3, hh3 + d_hh3)
        if ll2_from_detail.shape[-2:] != ll2.shape[-2:]:
            ll2_from_detail = F.interpolate(ll2_from_detail, size=ll2.shape[-2:], mode='bilinear',
                                            align_corners=False)
        ll2_final = ll2_wm + (ll2_from_detail - ll2)

        ll1_wm = self.dwt.inverse_dwt(ll2_final, lh2_wm, hl2_wm, hh2)
        if ll1_wm.shape[-2:] != ll1.shape[-2:]:
            ll1_wm = F.interpolate(ll1_wm, size=ll1.shape[-2:], mode='bilinear', align_corners=False)
        y_raw = self.dwt.inverse_dwt(ll1_wm, lh1, hl1, hh1)
        if y_raw.shape[-2:] != (H, W):
            # sides that are not multiples of 8 are resampled to the input size here
            y_raw = F.interpolate(y_raw, size=(H, W), mode='bilinear', align_corners=False)

        delta = self._clamp(y_raw, x, mask)
        y = (x + delta).clamp(0, 1)
        if self.training:
            return y, delta, mask
        return y

    def _forward_detail_only(self, x, wm_map):
        """Stage-1 forward: embed only in LH3 / HL3 / HH3; LL3 and level-1/2 bands untouched."""
        B, C, H, W = x.shape
        if wm_map.shape[-2:] != (H, W):
            wm_map = F.interpolate(wm_map, size=(H, W), mode='nearest')
        mask = self.flatness_mask(x)

        ll1, lh1, hl1, hh1 = self.dwt.forward_dwt(x)
        ll2, lh2, hl2, hh2 = self.dwt.forward_dwt(ll1)
        ll3, lh3, hl3, hh3 = self.dwt.forward_dwt(ll2)
        wm_l3 = F.interpolate(wm_map, size=ll3.shape[-2:], mode='nearest')
        mask_l3 = F.interpolate(mask, size=ll3.shape[-2:], mode='bilinear', align_corners=False)

        d_lh3, d_hl3, d_hh3 = self._detail_deltas(wm_l3, lh3, hl3, hh3, mask_l3)
        ll2_wm = self.dwt.inverse_dwt(ll3, lh3 + d_lh3, hl3 + d_hl3, hh3 + d_hh3)
        if ll2_wm.shape[-2:] != ll2.shape[-2:]:
            ll2_wm = F.interpolate(ll2_wm, size=ll2.shape[-2:], mode='bilinear', align_corners=False)
        ll1_wm = self.dwt.inverse_dwt(ll2_wm, lh2, hl2, hh2)
        if ll1_wm.shape[-2:] != ll1.shape[-2:]:
            ll1_wm = F.interpolate(ll1_wm, size=ll1.shape[-2:], mode='bilinear', align_corners=False)
        y_raw = self.dwt.inverse_dwt(ll1_wm, lh1, hl1, hh1)
        if y_raw.shape[-2:] != (H, W):
            y_raw = F.interpolate(y_raw, size=(H, W), mode='bilinear', align_corners=False)

        delta = self._clamp(y_raw, x, mask)
        y = (x + delta).clamp(0, 1)
        if self.training:
            return y, delta, mask
        return y


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class FreqWatermarkDecoder(nn.Module):
    """Four-path decoder: spatial, LL1, LL2 and level-3 detail (LH3, HL3, HH3), fused to a 64x64 logit map."""

    def __init__(self, out_hw=HP.wm_hw, base=64):
        super().__init__()
        self.out_hw = (out_hw, out_hw) if isinstance(out_hw, int) else tuple(out_hw)
        self.dwt = HaarDWT()
        self.spatial_stem = nn.Sequential(nn.Conv2d(3, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.spatial_res1 = ResBlock(base)
        self.spatial_res2 = ResBlock(base)

        def stem(cin):
            return nn.Sequential(
                nn.Conv2d(cin, base // 2, 3, padding=1), nn.BatchNorm2d(base // 2), nn.ReLU(inplace=True),
                nn.Conv2d(base // 2, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.freq_stem = stem(3)
        self.freq2_stem = stem(3)
        self.detail3_stem = stem(9)
        self.fusion = nn.Sequential(
            nn.Conv2d(base * 4, base, 1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base // 2, 3, padding=1), nn.BatchNorm2d(base // 2), nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, 1, 1),
        )

    def forward(self, y):
        up = lambda f: F.interpolate(f, size=self.out_hw, mode='bilinear', align_corners=False)
        f_sp = up(self.spatial_res2(self.spatial_res1(self.spatial_stem(y))))
        ll1, _, _, _ = self.dwt.forward_dwt(y)
        f_freq = up(self.freq_stem(ll1))
        ll2, _, _, _ = self.dwt.forward_dwt(ll1)
        f_freq2 = up(self.freq2_stem(ll2))
        _, lh3, hl3, hh3 = self.dwt.forward_dwt(ll2)
        f_detail3 = up(self.detail3_stem(torch.cat([lh3, hl3, hh3], dim=1)))
        return self.fusion(torch.cat([f_sp, f_freq, f_freq2, f_detail3], dim=1))


# ---------------------------------------------------------------------------- builders
def build_models(kind="hybrid", device="cpu"):
    """Create encoder / decoder in the same module order as the training notebook.

    kind = "stage1" : detail-only model (Stage-1 checkpoint layout)
    kind = "hybrid" : final model (Stage-2 checkpoint layout: fmid replaced by the 2-layer CNN)
    """
    enc = DWTWatermarkEncoder(base=32)
    dec = FreqWatermarkDecoder(out_hw=HP.wm_hw, base=64)
    enc.add_detail_branches()
    if kind == "stage1":
        enc.mode = "detail_only"
        enc.mask_floor = HP.mask_floor_stage1
        enc.dmax = HP.dmax_stage1
    elif kind == "hybrid":
        enc.wm_encoder_mid = make_mid_encoder()
        enc.mode = "hybrid"
    else:
        raise ValueError(kind)
    return enc.to(device), dec.to(device)


def _state(path, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    if isinstance(sd, dict):
        for k in ("enc", "dec", "encoder", "decoder", "state_dict", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd


def load_models(ckpt_dir, kind="hybrid", device="cpu", eval_mode=True):
    """Load ``enc_best.pt`` / ``dec_best.pt`` from ``ckpt_dir`` (strict)."""
    enc, dec = build_models(kind, device)
    enc.load_state_dict(_state(os.path.join(ckpt_dir, "enc_best.pt"), device), strict=True)
    dec.load_state_dict(_state(os.path.join(ckpt_dir, "dec_best.pt"), device), strict=True)
    if eval_mode:
        enc.eval(); dec.eval()
    return enc, dec
