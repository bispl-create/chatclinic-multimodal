from __future__ import annotations

from typing import Any

from plugins.medical_restoration_common.adapters import run_sharpxr
from plugins.medical_restoration_common.base_logic import run_restoration


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    return run_restoration(
        tool_name="xray_denoise_tool",
        task_label="Chest X-ray Denoising",
        payload=payload,
        runner=lambda arr: run_sharpxr(arr),
    )
