"""In-process chest X-ray denoising via TorchXRayVision's ResNetAE.

Why this model:
- Pretrained autoencoder (31.1 M params) trained jointly on PadChest + NIH + CheXpert + MIMIC
  (~800 k chest X-rays).
- Weights auto-download from the mlmed/torchxrayvision GitHub release on first use and
  land in ``~/.torchxrayvision/models_data/``.
- Autoencoder reconstruction acts as a learned denoising prior — the encoder maps the
  noisy X-ray to a 512×3×3 latent, the decoder reconstructs the clean manifold estimate.

This replaces the SharpXR path for ``@xray_denoise`` because SharpXR's authors did not
publish pretrained weights. ResNetAE is the de-facto open-source CXR prior.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


_CACHE: dict[str, tuple[torch.nn.Module, torch.device]] = {}


def _pick_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _load(weights: str = "101-elastic") -> tuple[torch.nn.Module, torch.device]:
    if weights in _CACHE:
        return _CACHE[weights]
    import torchxrayvision as xrv  # type: ignore

    ae = xrv.autoencoders.ResNetAE(weights=weights)
    device = _pick_device()
    ae.to(device).eval()
    _CACHE[weights] = (ae, device)
    return _CACHE[weights]


def _prepare(arr: np.ndarray, size: int = 224) -> tuple[torch.Tensor, float, float]:
    """Match the torchxrayvision preprocessing: 1-channel, [-1024, 1024] pixel range,
    224x224 input. We normalize to that range so the autoencoder sees the expected stats."""
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    span = max(hi - lo, 1e-6)
    # Map [lo, hi] -> [-1024, 1024] (the xrv normalization convention).
    norm = (arr - lo) / span * 2048.0 - 1024.0
    t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t, lo, hi


def _denormalize(t: torch.Tensor, lo: float, hi: float) -> np.ndarray:
    arr = t.squeeze().cpu().numpy()
    # Map [-1024, 1024] back to [lo, hi]
    arr01 = (arr + 1024.0) / 2048.0
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * (hi - lo) + lo).astype(np.float32)


def torchxrayvision_denoise(arr: np.ndarray, weights: str = "101-elastic") -> tuple[np.ndarray, dict[str, object]]:
    ae, device = _load(weights)
    x, lo, hi = _prepare(arr, size=224)
    x = x.to(device)
    with torch.no_grad():
        z = ae.encode(x)
        y = ae.decode(z)
    out = _denormalize(y, lo, hi)
    info: dict[str, object] = {
        "device": str(device),
        "ckpt": weights,
        "backend_name": "torchxrayvision-ResNetAE",
        "latent_shape": tuple(z.shape)[1:],
        "vram_used_mb": (f"{torch.cuda.memory_allocated(device) / 1024**2:.1f}"
                        if device.type == "cuda" else "n/a"),
    }
    return out, info
