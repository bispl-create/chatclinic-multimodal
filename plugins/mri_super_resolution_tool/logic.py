from __future__ import annotations

from typing import Any

from plugins.medical_restoration_common.adapters import run_fastddpm
from plugins.medical_restoration_common.base_logic import run_restoration


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    return run_restoration(
        tool_name="mri_super_resolution_tool",
        task_label="MRI Super-Resolution",
        payload=payload,
        runner=lambda arr: run_fastddpm(arr, task="sr_mri"),
    )
