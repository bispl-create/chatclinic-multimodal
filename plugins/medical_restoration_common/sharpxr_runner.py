"""In-process SharpXR inference (pediatric chest X-ray denoising).

SharpXR (MIRASOL/MICCAI 2025) is a **dual-decoder U-Net** with a noise-suppression
decoder and a Laplacian-edge decoder fused by a learnable attention block.

Status notes:
- Architecture is in ``external_backends/SharpXR/models/dual_decoder.py``.
- The SharpXR authors did **not** publish pretrained weights — the repo only ships
  a training-state pickle (`denoising_training.pkl`, 2 KB) and an unrelated pneumonia
  classifier (`cls_checkpoint/best_classifier.pth`). To run the real model you'd
  need to train on the Pediatric Pneumonia Chest X-ray dataset yourself.
- This runner looks for a user-supplied checkpoint at any of:
    external_weights/SharpXR/sharpxr_best.pth
    external_backends/SharpXR/checkpoints/sharpxr_best.pth
  and loads it if present. Otherwise it instantiates the architecture with random
  weights (NOT useful for inference) and the adapter redirects to Fast-DDPM —
  see ``adapters.run_sharpxr``.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .paths import EXTERNAL_BACKENDS, EXTERNAL_WEIGHTS


def _load_package(pkg_name: str, pkg_dir: Path, submodules: list[str]) -> types.ModuleType:
    """Register a directory as a package under a unique name so relative imports
    (`from .components import ...`) inside it resolve correctly, bypassing
    collisions with same-named packages from sibling backends."""
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create package spec for {pkg_dir}")
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)
    for sub in submodules:
        sub_name = f"{pkg_name}.{sub}"
        sub_spec = importlib.util.spec_from_file_location(sub_name, str(pkg_dir / f"{sub}.py"))
        if sub_spec is None or sub_spec.loader is None:
            raise ImportError(f"Could not load {sub} from {pkg_dir}")
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[sub_name] = sub_mod
        sub_spec.loader.exec_module(sub_mod)
        setattr(pkg, sub, sub_mod)
    return pkg


SHARPXR_BACKEND = EXTERNAL_BACKENDS / "SharpXR"
_CANDIDATE_WEIGHTS = [
    EXTERNAL_WEIGHTS / "SharpXR" / "sharpxr_best.pth",
    SHARPXR_BACKEND / "checkpoints" / "sharpxr_best.pth",
]

_CACHE: dict[str, tuple[torch.nn.Module, torch.device, Path | None]] = {}


def find_weights() -> Path | None:
    for c in _CANDIDATE_WEIGHTS:
        if c.exists() and c.stat().st_size > 1_000_000:  # >1 MB = actual weights
            return c
    return None


def _pick_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _load() -> tuple[torch.nn.Module, torch.device, Path | None]:
    if "model" in _CACHE:
        return _CACHE["model"]
    pkg = _load_package(
        "sharpxr_models",
        SHARPXR_BACKEND / "models",
        submodules=["components", "dual_decoder"],
    )
    DualDecoderHybrid = pkg.dual_decoder.DualDecoderHybrid  # type: ignore[attr-defined]
    net = DualDecoderHybrid(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
    ckpt = find_weights()
    if ckpt is not None:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        try:
            net.load_state_dict(cleaned, strict=True)
        except RuntimeError:
            net.load_state_dict(cleaned, strict=False)
    device = _pick_device()
    net.to(device).eval()
    _CACHE["model"] = (net, device, ckpt)
    return _CACHE["model"]


def _prepare_slice(arr: np.ndarray, size: int = 256) -> tuple[torch.Tensor, float, float]:
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    span = max(hi - lo, 1e-6)
    norm = (arr - lo) / span  # [0, 1]
    t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t, lo, hi


def _denormalize(t: torch.Tensor, lo: float, hi: float) -> np.ndarray:
    arr = t.squeeze().cpu().numpy().clip(0.0, 1.0)
    return (arr * (hi - lo) + lo).astype(np.float32)


def sharpxr_infer(arr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    model, device, ckpt = _load()
    x, lo, hi = _prepare_slice(arr, size=256)
    x = x.to(device)
    with torch.no_grad():
        y = model(x)
    out = _denormalize(y, lo, hi)
    info: dict[str, object] = {
        "device": str(device),
        "ckpt": str(ckpt) if ckpt else None,
        "ckpt_found": ckpt is not None,
        "vram_used_mb": (f"{torch.cuda.memory_allocated(device) / 1024**2:.1f}"
                        if device.type == "cuda" else "n/a"),
    }
    return out, info
