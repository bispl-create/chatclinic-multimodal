"""Subprocess-based adapters to external restoration backends.

Each adapter:
1. Prepares a minimal input directory in the shared cache.
2. Invokes the backend's own inference entrypoint as a subprocess (keeps
   the backend's Python env isolated via its own interpreter or `uv`).
3. Loads the backend's output and returns it as a numpy array.

When weights are not yet available, the adapters fall back to a classical
baseline (bilateral filter / total-variation / bicubic) so the plugin still
returns a usable Studio card. The adapter name and mode are recorded in the
result so the UI is transparent about what ran.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .paths import (
    COREDIFF_BACKEND,
    COREDIFF_WEIGHTS,
    FASTDDPM_BACKEND,
    FASTDDPM_WEIGHTS,
    RESTORATION_CACHE,
    SNRAWARE_BACKEND,
    SNRAWARE_WEIGHTS,
)


@dataclass
class AdapterResult:
    output: np.ndarray
    backend: str
    mode: str  # "backend" or "classical_fallback"
    notes: list[str]


def _work_dir(tag: str, key: bytes) -> Path:
    digest = hashlib.sha1(key).hexdigest()[:12]
    p = RESTORATION_CACHE / f"{tag}_{digest}"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Classical fallbacks — used when weights or backend deps are missing.
# ---------------------------------------------------------------------------

def _classical_denoise(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(arr, sigma=sigma)
    except Exception:
        return arr


def _classical_super_resolve(arr: np.ndarray, scale: int = 2) -> np.ndarray:
    try:
        from PIL import Image

        h, w = arr.shape[-2:]
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        norm = (arr - lo) / max(hi - lo, 1e-8)
        img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        img = img.resize((w * scale, h * scale), Image.BICUBIC)
        out = np.asarray(img).astype(np.float32) / 255.0
        return out * (hi - lo) + lo
    except Exception:
        return arr


# ---------------------------------------------------------------------------
# SNRAware (MRI denoising)
# ---------------------------------------------------------------------------

def run_snraware(arr: np.ndarray, *, size: str = "small") -> AdapterResult:
    """Run SNRAware on a 2D real-valued magnitude image.

    The backend expects complex MRI (real+imag) + g-map. We synthesize a
    zero-imaginary channel and a unit g-map to make the magnitude path work.
    If anything is missing we fall back to a classical denoiser.
    """
    notes: list[str] = []
    model_dir = SNRAWARE_WEIGHTS / size
    model_pts = model_dir / f"snraware_{size}_model.pts"
    model_yaml = model_dir / f"snraware_{size}_model.yaml"
    entry = SNRAWARE_BACKEND / "src" / "snraware" / "projects" / "mri" / "denoising" / "run_inference.py"

    if not entry.exists() or not model_pts.exists() or not model_yaml.exists():
        notes.append(
            f"SNRAware unavailable (entry={entry.exists()}, weights={model_pts.exists()}). Classical fallback used."
        )
        return AdapterResult(_classical_denoise(arr), "snraware", "classical_fallback", notes)

    key = arr.tobytes() + size.encode()
    work = _work_dir("snraware", key)
    in_dir = work / "in"
    out_dir = work / "out"
    in_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    real = arr.astype(np.float32)[..., None]  # [H,W,1]
    imag = np.zeros_like(real)
    gmap = np.ones(arr.shape, dtype=np.float32)

    np.save(in_dir / "input_real.npy", real)
    np.save(in_dir / "input_imag.npy", imag)
    np.save(in_dir / "gmap.npy", gmap)

    runner = "uv" if shutil.which("uv") else sys.executable
    cmd: list[str]
    if runner == "uv":
        cmd = [
            "uv", "run", "--project", str(SNRAWARE_BACKEND),
            "python3", str(entry),
        ]
    else:
        cmd = [runner, str(entry)]
    cmd += [
        "--input_dir", str(in_dir),
        "--output_dir", str(out_dir),
        "--saved_model_path", str(model_pts),
        "--saved_config_path", str(model_yaml),
        "--batch_size", "1",
        "--input_fname", "input",
        "--gmap_fname", "gmap",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            notes.append(f"SNRAware subprocess failed: {proc.stderr[-400:]}")
            return AdapterResult(_classical_denoise(arr), "snraware", "classical_fallback", notes)
    except Exception as exc:  # pragma: no cover
        notes.append(f"SNRAware launch error: {exc}")
        return AdapterResult(_classical_denoise(arr), "snraware", "classical_fallback", notes)

    # Collect output — expect `output_real.npy` / similar
    candidates = sorted(out_dir.glob("*real*.npy")) + sorted(out_dir.glob("*.npy"))
    if not candidates:
        notes.append("SNRAware produced no .npy output; classical fallback used.")
        return AdapterResult(_classical_denoise(arr), "snraware", "classical_fallback", notes)
    out_real = np.load(candidates[0])
    if out_real.ndim >= 3:
        out_real = out_real[..., 0]
    return AdapterResult(out_real.astype(np.float32), "snraware", "backend", notes)


# ---------------------------------------------------------------------------
# CoreDiff (CT denoising / artifact reduction)
# ---------------------------------------------------------------------------

def run_corediff(arr: np.ndarray, *, mode: str = "denoise") -> AdapterResult:
    """Run CoreDiff's official 10-step DDIM on a single CT slice.

    Uses the EMA checkpoint `ema_model-150000` downloaded into
    ``external_weights/CoreDiff/checkpoint/``. If the runner fails for any
    reason, redirects to Fast-DDPM LDFDCT (same task, same GPU).
    """
    notes: list[str] = []
    try:
        from .corediff_runner import corediff_infer

        out, info = corediff_infer(arr, mode=mode)  # type: ignore[arg-type]
        notes.append(
            f"CoreDiff ran on {info['device']} ({info['timesteps']} DDIM steps, "
            f"ckpt={info['ckpt']}, vram={info['vram_used_mb']} MB, mode={info['mode']})."
        )
        return AdapterResult(out, "corediff", "backend", notes)
    except Exception as exc:
        notes.append(f"CoreDiff in-process inference failed: {exc}. Falling back to Fast-DDPM LDFDCT.")
        try:
            from .fastddpm_runner import fastddpm_infer

            out, info = fastddpm_infer(arr, task="denoise_ct", timesteps=10)
            notes.append(
                f"Fast-DDPM ran on {info['device']} ({info['timesteps']} DDIM steps, "
                f"ckpt={info['ckpt']}, vram={info['vram_used_mb']} MB)."
            )
            return AdapterResult(out, "corediff->fastddpm", "backend", notes)
        except Exception as exc2:
            notes.append(f"Fast-DDPM redirection also failed: {exc2}. Classical fallback used.")
            return AdapterResult(_classical_denoise(arr, sigma=2.0), "corediff", "classical_fallback", notes)


# ---------------------------------------------------------------------------
# Fast-DDPM (MRI super-resolution, CT denoising, image translation)
# ---------------------------------------------------------------------------

def run_xray_denoise(arr: np.ndarray) -> AdapterResult:
    """Chest X-ray denoising.

    Preference order:
      1. SharpXR — if the user has dropped a real ``sharpxr_best.pth`` into
         ``external_weights/SharpXR/`` (authors did not publish one; empty by default).
      2. TorchXRayVision ResNetAE — 31.1 M-param autoencoder pretrained on
         PadChest + NIH + CheXpert + MIMIC (~800 k chest X-rays). Weights auto-download
         from the mlmed/torchxrayvision GitHub release on first call.
      3. Fast-DDPM LDFDCT — grayscale diffusion prior used as a final safety net.
    """
    notes: list[str] = []

    # Tier 1: real SharpXR weights (only if the user supplied them)
    try:
        from .sharpxr_runner import find_weights, sharpxr_infer

        ckpt = find_weights()
        if ckpt is not None:
            out, info = sharpxr_infer(arr)
            notes.append(
                f"SharpXR ran on {info['device']} (ckpt={Path(str(info['ckpt'])).name}, "
                f"vram={info['vram_used_mb']} MB)."
            )
            return AdapterResult(out, "sharpxr", "backend", notes)
        notes.append(
            "SharpXR pretrained weights are not published by the authors. "
            "Falling back to TorchXRayVision ResNetAE (autoencoder pretrained on "
            "PadChest/NIH/CheXpert/MIMIC)."
        )
    except Exception as exc:
        notes.append(f"SharpXR path unavailable: {exc}. Falling back to TorchXRayVision ResNetAE.")

    # Tier 2: TorchXRayVision ResNetAE — the open-source CXR denoiser of record
    try:
        from .torchxrayvision_runner import torchxrayvision_denoise

        out, info = torchxrayvision_denoise(arr)
        notes.append(
            f"TorchXRayVision ResNetAE ran on {info['device']} "
            f"(ckpt={info['ckpt']}, latent={info['latent_shape']}, vram={info['vram_used_mb']} MB)."
        )
        return AdapterResult(out, "torchxrayvision-ResNetAE", "backend", notes)
    except Exception as exc:
        notes.append(f"TorchXRayVision ResNetAE failed: {exc}. Falling back to Fast-DDPM LDFDCT.")

    # Tier 3: Fast-DDPM LDFDCT as last resort
    try:
        from .fastddpm_runner import fastddpm_infer

        out, info = fastddpm_infer(arr, task="denoise_ct", timesteps=10)
        notes.append(
            f"Fast-DDPM ran on {info['device']} ({info['timesteps']} DDIM steps, "
            f"ckpt={info['ckpt']}, vram={info['vram_used_mb']} MB)."
        )
        return AdapterResult(out, "fastddpm-xray", "backend", notes)
    except Exception as exc:
        notes.append(f"All model paths failed: {exc}. Classical fallback used.")
        return AdapterResult(_classical_denoise(arr, sigma=1.0), "xray_denoise", "classical_fallback", notes)


# Keep the old name as an alias for any existing callers
run_sharpxr = run_xray_denoise


_FASTDDPM_TASK_CONFIG = {
    "sr_mri": {"config": "PMUB.yml", "dataset": "PMUB", "ckpt": "ckpt_PMUB.pth"},
    "denoise_ct": {"config": "LDFDCT.yml", "dataset": "LDFDCT", "ckpt": "ckpt_LDFDCT.pth"},
    "translation": {"config": "BRATS.yml", "dataset": "BRATS", "ckpt": "ckpt_BRATS.pth"},
}


def run_fastddpm(arr: np.ndarray, *, task: str) -> AdapterResult:
    """Run Fast-DDPM for a given task.

    Preferred path: load the official UNet in-process and run 10-step DDIM
    sampling on GPU (see ``fastddpm_runner.fastddpm_infer``). Falls back to a
    classical filter only if the checkpoint is missing or the runner errors
    — the notes field records the exact reason.
    """
    notes: list[str] = []
    if task not in _FASTDDPM_TASK_CONFIG:
        raise ValueError(f"Unknown Fast-DDPM task: {task}")
    cfg = _FASTDDPM_TASK_CONFIG[task]
    ckpt = FASTDDPM_WEIGHTS / cfg["ckpt"]
    if not ckpt.exists():
        notes.append(f"Fast-DDPM checkpoint missing: {ckpt}. Classical fallback used.")
        if task == "sr_mri":
            return AdapterResult(_classical_super_resolve(arr, scale=2), "fastddpm", "classical_fallback", notes)
        return AdapterResult(_classical_denoise(arr, sigma=1.5 if task == "denoise_ct" else 0.5),
                             "fastddpm", "classical_fallback", notes)

    try:
        from .fastddpm_runner import fastddpm_infer

        out, info = fastddpm_infer(arr, task=task, timesteps=10)
        notes.append(
            f"Fast-DDPM ran on {info['device']} ({info['timesteps']} DDIM steps, "
            f"ckpt={info['ckpt']}, vram={info['vram_used_mb']} MB)."
        )
        return AdapterResult(out, "fastddpm", "backend", notes)
    except Exception as exc:
        notes.append(f"Fast-DDPM in-process inference failed: {exc}. Classical fallback used.")
        if task == "sr_mri":
            return AdapterResult(_classical_super_resolve(arr, scale=2), "fastddpm", "classical_fallback", notes)
        return AdapterResult(_classical_denoise(arr, sigma=1.5 if task == "denoise_ct" else 0.5),
                             "fastddpm", "classical_fallback", notes)
