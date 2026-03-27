from __future__ import annotations

from pathlib import Path

from app.services.tool_runner import discover_tools
from plugins.fastqc_execution_tool.logic import run_fastqc_local


def execute(payload: dict[str, object]) -> dict[str, object]:
    raw_path = str(payload["raw_path"])
    original_name = str(payload.get("original_name") or Path(raw_path).name)
    analysis = run_fastqc_local(raw_path, original_name, discover_tools())
    return {"analysis": analysis.model_dump()}
