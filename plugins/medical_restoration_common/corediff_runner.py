"""In-process CoreDiff inference.

Loads the official CoreDiff Diffusion wrapper from
external_backends/CoreDiff/models/corediff/ and applies the EMA checkpoint
(ema_model-150000) downloaded into external_weights/CoreDiff/checkpoint/.

The checkpoint is a 10-step DDIM diffusion over 3-channel context input
(prev / current / next CT slice). For single-slice demos we reuse the
current slice as both neighbors.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .paths import COREDIFF_BACKEND, COREDIFF_WEIGHTS


def _load_abs(module_name: str, file_path: Path) -> types.ModuleType:
    """Load a module from an absolute file path under a unique name, bypassing sys.path
    collisions between backends that share top-level package names (models/functions)."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_COREDIFF_WEIGHT_FILE = COREDIFF_WEIGHTS / "checkpoint" / "ema_model-150000"
_CACHE: dict[str, tuple[torch.nn.Module, torch.device]] = {}


def _ensure_on_path() -> None:
    """Prime sys.path + purge stale `models` modules so CoreDiff's package wins."""
    p = str(COREDIFF_BACKEND)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    for stale in [k for k in list(sys.modules.keys())
                  if k == "models" or k.startswith("models.")
                  or k == "functions" or k.startswith("functions.")]:
        sys.modules.pop(stale, None)


def _pick_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _load() -> tuple[torch.nn.Module, torch.device]:
    if "diffusion" in _CACHE:
        return _CACHE["diffusion"]
    if not _COREDIFF_WEIGHT_FILE.exists():
        raise FileNotFoundError(f"CoreDiff checkpoint missing: {_COREDIFF_WEIGHT_FILE}")

    wrapper_mod = _load_abs("corediff_wrapper_abs",
                            COREDIFF_BACKEND / "models" / "corediff" / "corediff_wrapper.py")
    diffusion_mod = _load_abs("corediff_diffusion_abs",
                              COREDIFF_BACKEND / "models" / "corediff" / "diffusion_modules.py")
    Network = wrapper_mod.Network
    Diffusion = diffusion_mod.Diffusion

    denoise_fn = Network(in_channels=3, out_channels=1, context=True)
    diffusion = Diffusion(
        denoise_fn=denoise_fn,
        image_size=256,
        timesteps=10,
        context=True,
    )
    state = torch.load(str(_COREDIFF_WEIGHT_FILE), map_location="cpu", weights_only=False)
    diffusion.load_state_dict(state, strict=False)
    device = _pick_device()
    diffusion.to(device).eval()
    _CACHE["diffusion"] = (diffusion, device)
    return _CACHE["diffusion"]


def _prepare_slice(arr: np.ndarray, size: int = 256) -> tuple[torch.Tensor, float, float]:
    """Match CoreDiff's training preprocessing from ``utils/dataset.py`` lines 146-151.

    CoreDiff was trained on Mayo 2016 DICOMs normalized to [0, 1] via:
        img = img - 1024
        img = clip(img, -1024, 3072)
        img = (img + 1024) / 4096
    i.e. [0, 1] where 0 = HU -1024 (air) and 1 = HU 3072 (dense bone).

    For a normalized PNG we just map [min, max] -> [0, 1] directly and save
    the lo/hi to invert at the end. The model output is also in [0, 1]."""
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    span = max(hi - lo, 1e-6)
    norm = (arr - lo) / span  # [0, 1] — matches CoreDiff's training distribution
    t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t, lo, hi


def _denormalize(t: torch.Tensor, lo: float, hi: float) -> np.ndarray:
    """CoreDiff.sample() returns img.clamp(0., 1.), so we map [0, 1] back to [lo, hi]."""
    t = t.clamp(0.0, 1.0)
    return (t.squeeze().cpu().numpy() * (hi - lo) + lo).astype(np.float32)


def corediff_infer(
    arr: np.ndarray,
    *,
    mode: Literal["denoise", "artifact_reduction"] = "denoise",
) -> tuple[np.ndarray, dict[str, str]]:
    """Run CoreDiff's 10-step DDIM on a 2D CT slice. Returns (output, info)."""
    diffusion, device = _load()
    current, lo, hi = _prepare_slice(arr, size=256)
    # Context expects 3 channels (prev, curr, next). With only one slice we
    # duplicate it — the denoise_fn uses channel 1 as the target.
    img = torch.cat([current, current, current], dim=1).to(device)

    with torch.no_grad():
        # Diffusion.sample returns (final_img, stacked_direct_recons, stacked_imstep_imgs)
        final_img, _, _ = diffusion.sample(batch_size=1, img=img, t=10, sampling_routine="ddim")
    arr_out = _denormalize(final_img, lo, hi)

    info = {
        "device": str(device),
        "timesteps": "10",
        "ckpt": "ema_model-150000",
        "mode": mode,
        "vram_used_mb": (f"{torch.cuda.memory_allocated(device) / 1024**2:.1f}"
                        if device.type == "cuda" else "n/a"),
    }
    return arr_out, info
