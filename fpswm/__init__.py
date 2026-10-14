"""FPS-WM: frequency-path separation and inference-time search for high-capacity blind image watermarking."""
from .config import HP
from .dwt import HaarDWT
from .mask import compute_flatness_mask
from .models import (DWTWatermarkEncoder, FreqWatermarkDecoder, build_models, load_models,
                     make_detail_encoder, make_mid_encoder)
from .watermark import load_watermark_bits
from .attacks import attack_val_v2, real_jpeg_attack_eval, diffjpeg_attack, differentiable_attack_v2
from .search import extract_watermark
from .utils import get_device, set_all_seeds

__version__ = "1.0.0"
