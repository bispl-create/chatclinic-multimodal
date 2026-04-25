"""Pre-demo health check. Run from chatclinic-multimodal/ with:

    python -m plugins.medical_restoration_common.check_health

Prints:
- GPU availability (torch + nvidia-smi)
- Backend directory / weights presence
- One real Fast-DDPM forward pass on GPU for each task
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .paths import backend_status


def _line(s: str = "") -> None:
    print(s, flush=True)


def main() -> int:
    _line("=" * 70)
    _line("ChatClinic medical-restoration health check")
    _line("=" * 70)

    _line("\n[1] GPU / torch")
    try:
        import torch

        _line(f"  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                _line(f"  cuda:{i}  {name}  {total:.1f} GB")
        else:
            _line("  WARNING: torch reports no CUDA. Activate the `chatclinic` conda env.")
    except Exception as exc:
        _line(f"  ERROR: torch import failed: {exc}")

    _line("\n[2] nvidia-smi snapshot")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
            text=True, timeout=10,
        )
        for row in out.strip().splitlines():
            _line(f"  {row}")
    except Exception as exc:
        _line(f"  nvidia-smi unavailable: {exc}")

    _line("\n[3] Backend + weights presence")
    status = backend_status()
    for name, info in status.items():
        ok_b = "OK" if info["backend_exists"] else "MISSING"
        ok_w = "OK" if info["weights_exist"] else "MISSING"
        _line(f"  {name:10s}  backend={ok_b:7s} weights={ok_w:7s}  {info['weights_dir']}")

    _line("\n[4] Fast-DDPM in-process inference (loads checkpoint -> runs 10 DDIM steps)")
    try:
        from .fastddpm_runner import fastddpm_infer

        arr = np.random.rand(256, 256).astype("float32")
        for task in ("denoise_ct", "sr_mri", "translation"):
            t0 = time.time()
            out, info = fastddpm_infer(arr, task=task, timesteps=10)
            dt = time.time() - t0
            _line(
                f"  {task:12s}  device={info['device']:6s} ckpt={info['ckpt']:16s} "
                f"vram={info['vram_used_mb']:>6s} MB  time={dt:4.1f}s  out={out.shape}"
            )
    except Exception as exc:
        _line(f"  Fast-DDPM inference FAILED: {exc}")
        return 2

    _line("\n[4b] CoreDiff in-process inference")
    try:
        from .corediff_runner import corediff_infer

        import time as _t
        t0 = _t.time()
        out, info = corediff_infer(np.random.rand(256, 256).astype("float32"))
        _line(
            f"  device={info['device']:6s} ckpt={info['ckpt']:20s} "
            f"vram={info['vram_used_mb']:>6s} MB  time={_t.time()-t0:4.1f}s  out={out.shape}"
        )
    except FileNotFoundError as exc:
        _line(f"  CoreDiff weights missing: {exc}")
    except Exception as exc:
        _line(f"  CoreDiff inference FAILED: {exc}")

    _line("\n[4c] Chest X-ray denoising (TorchXRayVision ResNetAE)")
    try:
        from .torchxrayvision_runner import torchxrayvision_denoise

        out, info = torchxrayvision_denoise(np.random.rand(224, 224).astype("float32"))
        _line(
            f"  device={info['device']}  ckpt={info['ckpt']}  "
            f"latent={info['latent_shape']}  vram={info['vram_used_mb']} MB  out={out.shape}"
        )
    except Exception as exc:
        _line(f"  TorchXRayVision ResNetAE FAILED: {exc}")

    _line("\n[4d] SharpXR arch status (user-supplied weights)")
    try:
        from .sharpxr_runner import find_weights

        ckpt = find_weights()
        if ckpt is None:
            _line("  SharpXR weights not supplied (authors did not publish). "
                  "Drop `sharpxr_best.pth` into external_weights/SharpXR/ to prefer over ResNetAE.")
        else:
            _line(f"  SharpXR weights present at: {ckpt} (adapter will prefer these)")
    except Exception as exc:
        _line(f"  SharpXR check failed: {exc}")

    _line("\n[5] Plugin registration")
    try:
        from app.services.tool_runner import manifest_for_alias

        for alias in ("ct_denoise", "ct_artifact", "mri_denoise", "mri_sr", "med_translate", "xray_denoise"):
            m = manifest_for_alias(alias)
            _line(f"  @{alias:<14s} -> {(m or {}).get('name', 'MISSING')}")
    except Exception as exc:
        _line(f"  plugin registry check FAILED: {exc}")
        return 3

    _line("\n" + "=" * 70)
    _line("All checks complete. If you saw cuda:N lines in [1] and device=cuda in [4],")
    _line("the demo will show real GPU inference.")
    _line("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
