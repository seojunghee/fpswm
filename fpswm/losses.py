"""Training losses (paper, Eq. (12) and Table 2)."""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim


def confidence_penalty_loss(logits, wm_bits, margin=2.0):
    """Hinge-type margin loss pushing logits beyond +-margin."""
    pos = F.relu(margin - logits) * wm_bits
    neg = F.relu(margin + logits) * (1 - wm_bits)
    return (pos + neg).mean()


# attack-adaptive coefficients (wm, wf, wc); crop/rotate also disable L_stab
ATTACK_WEIGHTS = {
    "jpeg": (2.0, 3.5, 2.0),
    "noise": (2.0, 4.0, 3.0),
    "crop": (2.5, 0.5, 1.0),
    "rotate": (2.5, 0.5, 1.0),
    "resize": (1.5, 2.0, 1.0),
    "blur": (1.5, 2.0, 1.0),
}


def compute_loss_v3(y, y_att, x, logits, wm_bits, atk_name, lambda_img, lambda_msg,
                    lambda_freq=2.0, lambda_stab=0.0, lambda_conf=0.5):
    l_img = F.mse_loss(y, x) + (1.0 - ssim(y, x, data_range=1.0, size_average=True))
    l_msg = F.binary_cross_entropy_with_logits(logits, wm_bits)
    l_stab = F.mse_loss(y_att, y) if lambda_stab > 0 else torch.tensor(0.0, device=y.device)
    l_freq = F.mse_loss(TF.gaussian_blur(y, kernel_size=7, sigma=2.0),
                        TF.gaussian_blur(x, kernel_size=7, sigma=2.0))
    l_conf = confidence_penalty_loss(logits, wm_bits)
    msg_w, freq_w, conf_w = ATTACK_WEIGHTS.get(atk_name, (1.0, 1.0, 1.0))
    if atk_name in ("crop", "rotate"):
        lambda_stab = 0.0
    total = (lambda_img * l_img + lambda_msg * msg_w * l_msg + lambda_stab * l_stab
             + lambda_freq * freq_w * l_freq + lambda_conf * conf_w * l_conf)
    return total, l_img, l_msg, l_stab, l_freq, l_conf


def flat_visibility_loss(delta, mask):
    """L_flat = mean( channel-mean |Delta| * (1 - M) )."""
    return (delta.abs().mean(dim=1, keepdim=True) * (1 - mask)).mean()
