from __future__ import annotations

from typing import Any

from plugins.medical_restoration_common.adapters import run_corediff, run_fastddpm
from plugins.medical_restoration_common.base_logic import run_restoration


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    # Default: CoreDiff — official low-dose CT denoising model (Mayo 2016 25% dose,
    # 4.83 M params, 10-step DDIM). Weights at external_weights/CoreDiff/checkpoint/.
    # Switch to Fast-DDPM's LDFDCT checkpoint with backend=fastddpm.
    backend = str(payload.get("backend") or "corediff").strip().lower()
    if backend == "fastddpm":
        runner = lambda arr: run_fastddpm(arr, task="denoise_ct")
    else:
        runner = lambda arr: run_corediff(arr, mode="denoise")
    return run_restoration(
        tool_name="ct_denoise_tool",
        task_label="CT Denoising",
        payload=payload,
        runner=runner,
    )
