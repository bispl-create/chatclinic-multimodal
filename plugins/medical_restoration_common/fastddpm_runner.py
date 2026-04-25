"""In-process Fast-DDPM inference.

Loads the official Fast-DDPM UNet (external_backends/Fast-DDPM/models/diffusion.py),
applies the matching checkpoint from external_weights/FastDDPM/, and runs the
10-step `generalized_steps` sampler on a single slice. This actually loads the
model onto an RTX 3090 — `nvidia-smi` should show VRAM usage while the adapter
runs.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .paths import FASTDDPM_BACKEND, FASTDDPM_WEIGHTS


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

# Cache {task: (model, betas, device)} so subsequent calls don't reload the ckpt.
_MODEL_CACHE: dict[str, tuple[torch.nn.Module, torch.Tensor, torch.device]] = {}


def _ensure_backend_on_path() -> None:
    """Prime sys.path + sys.modules so `from models.diffusion import Model` resolves
    to Fast-DDPM's models package (not CoreDiff's, which uses the same top-level name).

    Both CoreDiff and Fast-DDPM declare top-level ``models`` and ``functions``
    packages. Without isolation, whichever was imported first wins. We purge any
    stale copies from ``sys.modules`` before placing Fast-DDPM first on ``sys.path``.
    """
    p = str(FASTDDPM_BACKEND)
    # Put Fast-DDPM first
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    # Drop any stale `models` / `functions` modules loaded by other backends.
    for stale in [k for k in list(sys.modules.keys())
                  if k == "models" or k.startswith("models.")
                  or k == "functions" or k.startswith("functions.")]:
        sys.modules.pop(stale, None)


def _config_namespace(task: str) -> types.SimpleNamespace:
    """Build a minimal config that matches the ckpt's architecture."""
    # Shared base
    data = types.SimpleNamespace(image_size=256, channels=1)
    diffusion = types.SimpleNamespace(
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000,
    )
    if task == "sr_mri":
        model = types.SimpleNamespace(
            type="sr",
            in_channels=3,
            out_ch=1,
            ch=128,
            ch_mult=(1, 1, 2, 2, 4, 4),
            num_res_blocks=2,
            attn_resolutions=(16,),
            dropout=0.0,
            var_type="fixedsmall",
            ema_rate=0.999,
            ema=True,
            resamp_with_conv=True,
        )
    else:  # denoise_ct, translation
        model = types.SimpleNamespace(
            type="sg",
            in_channels=2,
            out_ch=1,
            ch=128,
            ch_mult=(1, 1, 2, 2, 4, 4),
            num_res_blocks=2,
            attn_resolutions=(16,),
            dropout=0.0,
            var_type="fixedsmall",
            ema_rate=0.999,
            ema=True,
            resamp_with_conv=True,
        )
    return types.SimpleNamespace(data=data, model=model, diffusion=diffusion)


def _linear_betas(n: int = 1000) -> torch.Tensor:
    return torch.linspace(0.0001, 0.02, n, dtype=torch.float32)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        # Respect CUDA_VISIBLE_DEVICES if set; otherwise grab device 0.
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_model(task: str) -> tuple[torch.nn.Module, torch.Tensor, torch.device]:
    if task in _MODEL_CACHE:
        return _MODEL_CACHE[task]

    ckpt_map = {"sr_mri": "ckpt_PMUB.pth", "denoise_ct": "ckpt_LDFDCT.pth", "translation": "ckpt_BRATS.pth"}
    ckpt_path = FASTDDPM_WEIGHTS / ckpt_map[task]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fast-DDPM checkpoint missing: {ckpt_path}")

    diffusion_mod = _load_abs("fastddpm_models_diffusion", FASTDDPM_BACKEND / "models" / "diffusion.py")
    Model = diffusion_mod.Model

    cfg = _config_namespace(task)
    net = Model(cfg)
    raw = torch.load(str(ckpt_path), map_location="cpu")
    # Fast-DDPM saves either a list [state_dict, optim, ...] or a dict
    if isinstance(raw, (list, tuple)):
        state = raw[0]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError:
        # Allow slight key mismatches (e.g. EMA prefix)
        cleaned = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        net.load_state_dict(cleaned, strict=False)
    device = _pick_device()
    net.to(device).eval()
    betas = _linear_betas().to(device)
    _MODEL_CACHE[task] = (net, betas, device)
    return _MODEL_CACHE[task]


def _prepare_slice(arr: np.ndarray, size: int = 256) -> tuple[torch.Tensor, float, float]:
    """Normalize a 2D array to [-1, 1], resize to (size, size), return tensor + lo/hi."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.squeeze()
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = max(hi - lo, 1e-6)
    norm = (arr - lo) / span  # [0, 1]
    norm = norm * 2.0 - 1.0  # [-1, 1]
    t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t, lo, hi


def _denormalize(t: torch.Tensor, lo: float, hi: float) -> np.ndarray:
    t = t.clamp(-1, 1)
    span = hi - lo
    arr = (t.squeeze().cpu().numpy() + 1.0) / 2.0 * span + lo
    return arr.astype(np.float32)


def _generalized_steps(x_t: torch.Tensor, cond: torch.Tensor, model: torch.nn.Module,
                       betas: torch.Tensor, timesteps: int = 10, eta: float = 0.0) -> torch.Tensor:
    """DDIM-style sampler matching Fast-DDPM's generalized_steps loop."""
    device = x_t.device
    num_timesteps = betas.shape[0]
    skip = num_timesteps // timesteps
    seq = list(range(-1, num_timesteps, skip))
    seq[0] = 0
    seq_next = [-1] + list(seq[:-1])

    alphas = (1.0 - betas).cumprod(dim=0)

    def compute_alpha(t_idx: torch.Tensor) -> torch.Tensor:
        mask = (t_idx >= 0).float()
        safe = t_idx.clamp(min=0)
        a = alphas[safe] * mask + (1.0 - mask)  # a=1 when t<0
        return a.view(-1, 1, 1, 1)

    x = x_t
    n = x.size(0)
    with torch.no_grad():
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            next_t = torch.full((n,), j, device=device, dtype=torch.long)
            at = compute_alpha(t)
            at_next = compute_alpha(next_t)
            et = model(torch.cat([cond, x], dim=1), t.float())
            x0 = (x - et * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            x = at_next.sqrt() * x0 + c1 * torch.randn_like(x) + c2 * et
    return x


def fastddpm_infer(
    arr: np.ndarray,
    *,
    task: Literal["sr_mri", "denoise_ct", "translation"],
    timesteps: int = 10,
) -> tuple[np.ndarray, dict[str, str]]:
    """Run real Fast-DDPM inference on a 2D slice. Returns (output_array, info)."""
    model, betas, device = _load_model(task)
    cond, lo, hi = _prepare_slice(arr, size=256)
    cond = cond.to(device)
    if task == "sr_mri":
        # PMUB SR expects 2 neighbor slices + noise. For single-slice use we
        # duplicate the slice as both neighbors (in-context self-prior).
        cond_pair = torch.cat([cond, cond], dim=1)  # [1,2,H,W]
    else:
        cond_pair = cond  # [1,1,H,W]

    noise = torch.randn_like(cond)
    out = _generalized_steps(noise, cond_pair, model, betas, timesteps=timesteps, eta=0.0)
    arr_out = _denormalize(out, lo, hi)

    info = {
        "device": str(device),
        "timesteps": str(timesteps),
        "ckpt": {"sr_mri": "ckpt_PMUB.pth", "denoise_ct": "ckpt_LDFDCT.pth", "translation": "ckpt_BRATS.pth"}[task],
        "vram_used_mb": (f"{torch.cuda.memory_allocated(device) / 1024**2:.1f}"
                        if device.type == "cuda" else "n/a"),
    }
    return arr_out, info
