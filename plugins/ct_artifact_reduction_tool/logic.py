from __future__ import annotations

from typing import Any

from plugins.medical_restoration_common.adapters import run_corediff, run_fastddpm
from plugins.medical_restoration_common.base_logic import run_restoration


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    # Default: CoreDiff — artifact reduction uses the same low-dose CT diffusion
    # model with mode=artifact_reduction. Fast-DDPM LDFDCT is available via backend=fastddpm.
    backend = str(payload.get("backend") or "corediff").strip().lower()
    if backend == "fastddpm":
        runner = lambda arr: run_fastddpm(arr, task="denoise_ct")
    else:
        runner = lambda arr: run_corediff(arr, mode="artifact_reduction")
    return run_restoration(
        tool_name="ct_artifact_reduction_tool",
        task_label="CT Artifact Reduction",
        payload=payload,
        runner=runner,
    )
