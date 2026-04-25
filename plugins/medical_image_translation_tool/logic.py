from __future__ import annotations

from typing import Any

from plugins.medical_restoration_common.adapters import run_fastddpm
from plugins.medical_restoration_common.base_logic import run_restoration


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    return run_restoration(
        tool_name="medical_image_translation_tool",
        task_label="Medical Image Translation",
        payload=payload,
        runner=lambda arr: run_fastddpm(arr, task="translation"),
    )
