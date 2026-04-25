from __future__ import annotations

from typing import Any

from plugins.medical_restoration_common.adapters import run_fastddpm, run_snraware
from plugins.medical_restoration_common.base_logic import run_restoration


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    # Default: SNRAware — the Microsoft reference MRI denoiser (multi field-strength,
    # runs via uv venv on GPU). Fast-DDPM is available via backend=fastddpm but
    # note its only denoising checkpoint is trained on CT (LDFDCT), not MRI —
    # useful as a grayscale prior only.
    backend = str(payload.get("backend") or "snraware").strip().lower()
    size = str(payload.get("size") or "small").strip().lower()
    if size not in ("small", "medium", "large"):
        size = "small"
    if backend == "fastddpm":
        runner = lambda arr: run_fastddpm(arr, task="denoise_ct")
    else:
        runner = lambda arr: run_snraware(arr, size=size)
    return run_restoration(
        tool_name="mri_denoise_tool",
        task_label="MRI Denoising",
        payload=payload,
        runner=runner,
    )
