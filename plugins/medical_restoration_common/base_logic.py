"""Shared execute() builder for medical restoration plugins."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.models import ImageSourceResponse
from app.services.tool_runner import discover_tools

from .adapters import AdapterResult
from .io_utils import array_to_data_url, compute_metrics, load_input_as_2d, side_by_side


def run_restoration(
    *,
    tool_name: str,
    task_label: str,
    payload: dict[str, Any],
    runner: Callable[[Any], AdapterResult],
) -> dict[str, Any]:
    """Common glue: load input slice, call adapter, build response.

    ``payload`` accepts any of: ``image_path``, ``nifti_path``, ``dicom_path``,
    ``file_path``. ``runner`` takes a 2D float numpy array and returns
    ``AdapterResult``.
    """
    path_str = (
        str(payload.get("image_path")
            or payload.get("nifti_path")
            or payload.get("dicom_path")
            or payload.get("file_path")
            or "").strip()
    )
    if not path_str:
        raise ValueError("One of `image_path`, `nifti_path`, `dicom_path`, `file_path` is required.")
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    before = load_input_as_2d(path)
    result = runner(before)
    after = result.output

    before_url = array_to_data_url(before)
    after_url = array_to_data_url(after)
    compare_url = side_by_side(before, after)
    metrics = compute_metrics(before, after)

    restoration_artifact = {
        "tool": tool_name,
        "task": task_label,
        "backend": result.backend,
        "mode": result.mode,
        "notes": result.notes,
        "metrics": metrics,
        "before_preview_data_url": before_url,
        "after_preview_data_url": after_url,
        "compare_preview_data_url": compare_url,
        "input_path": str(path),
    }

    studio_cards = [
        {
            "id": tool_name,
            "title": task_label,
            "subtitle": f"Backend: {result.backend} ({result.mode})",
        }
    ]

    draft_lines = [
        f"{task_label} completed via `{result.backend}` ({result.mode}).",
        f"- Input: `{path.name}`",
    ]
    if metrics:
        snr_b = metrics.get("snr_before_db")
        snr_a = metrics.get("snr_after_db")
        delta = metrics.get("snr_improvement_db")
        sign = "+" if delta is not None and delta >= 0 else ""
        draft_lines.append(
            f"- SNR: {snr_b} dB → {snr_a} dB ({sign}{delta} dB); "
            f"Noise σ: {metrics.get('noise_sigma_before')} → {metrics.get('noise_sigma_after')}"
        )
    if result.notes:
        draft_lines.append("- Notes: " + "; ".join(result.notes))
    draft_lines.append("The Studio card shows a before/after comparison. Use `$studio` for grounded follow-up.")

    response = ImageSourceResponse(
        analysis_id="",
        source_image_path=str(path),
        file_name=path.name,
        file_kind="RESTORATION",
        width=int(before.shape[-1]),
        height=int(before.shape[-2]),
        format_name=path.suffix.lstrip(".").upper() or "NPY",
        color_mode="L",
        bit_depth=None,
        exif_data={},
        metadata_items=[{
            "file_name": path.name,
            "backend": result.backend,
            "mode": result.mode,
            "task": task_label,
        }],
        studio_cards=studio_cards,
        artifacts={
            tool_name: restoration_artifact,
            "restoration": restoration_artifact,
        },
        warnings=result.notes,
        preview_data_url=compare_url or after_url,
        draft_answer="\n".join(draft_lines),
        used_tools=[tool_name],
        tool_registry=discover_tools(),
    )
    return {"analysis": response.model_dump()}
