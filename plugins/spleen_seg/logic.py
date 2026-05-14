"""Plugin entrypoint for the UNet spleen CT segmentation tool."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf

from .infer import run_inference

_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_NAME = "spleen_seg"
_TOOL_VERSION = "0.1.0"


def _local_paths() -> Dict[str, str]:
    return {
        "tool_dir": str(_THIS_DIR),
        "weights_dir": str(_THIS_DIR / "weights"),
        "dataset_dir": str(_THIS_DIR / "dataset"),
        "bundle_dir": str(_THIS_DIR / "weights"),
        "msd_data_dir": str(_THIS_DIR / "dataset"),
    }


def _resolve_input_path(payload: Dict[str, Any]) -> str:
    for key in ("source_nifti_path", "nifti_path", "image_path", "dicom_path", "input_path"):
        raw = payload.get(key)
        if raw:
            return str(raw).strip()
    raise ValueError(
        "Payload must include one of: source_nifti_path, nifti_path, image_path, dicom_path, input_path."
    )


def _build_cfg(payload: Dict[str, Any], input_path: str):
    tool_cfg = OmegaConf.load(_THIS_DIR / "config.yaml")
    cfg = OmegaConf.create(
        {
            "device_name": "cuda:0",
            "save_output": True,
            "tool": tool_cfg,
            "paths": _local_paths(),
        }
    )

    device = payload.get("device")
    if device == "cpu":
        cfg.device_name = "cpu"
    elif device == "gpu":
        cfg.device_name = "cuda:0"

    cfg.paths.output_dir = payload.get("output_dir") or str(
        _THIS_DIR / "results" / time.strftime("%Y-%m-%d_%H-%M-%S")
    )
    cfg.paths.input_image = input_path

    for extra in payload.get("overrides") or []:
        key, _, val = str(extra).partition("=")
        if key:
            OmegaConf.update(cfg, key.lstrip("+"), val, merge=False)

    return cfg


def execute(payload: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []

    input_path = _resolve_input_path(payload)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    file_name = str(payload.get("file_name") or Path(input_path).name)
    cfg = _build_cfg(payload, input_path)
    result = run_inference(cfg)

    num_samples = int(result.get("num_samples", 1))
    elapsed = float(result.get("elapsed_sec", 0.0))

    analysis = {
        "file_name": file_name,
        "segmentation_tool": _PLUGIN_NAME,
        "display_name": str(cfg.tool.display_name),
        "num_classes": int(result.get("num_classes", cfg.tool.num_classes)),
        "num_samples": num_samples,
        "elapsed_sec": elapsed,
        "targets": list(cfg.tool.targets) if "targets" in cfg.tool else [],
    }

    artifacts: Dict[str, Any] = {"mask_path": result.get("mask_path", "")}
    if "mask_paths" in result:
        artifacts["mask_paths"] = list(result["mask_paths"])
    if result.get("image_path"):
        artifacts["image_path"] = result["image_path"]
    if result.get("label_path"):
        artifacts["label_path"] = result["label_path"]

    summary = (
        f"{cfg.tool.display_name}: segmented {num_samples} sample(s) from "
        f"'{file_name}' in {elapsed:.1f}s."
    )

    return {
        "tool": _PLUGIN_NAME,
        "summary": summary,
        "analysis": analysis,
        "artifacts": artifacts,
        "warnings": warnings,
        "provenance": {
            "tool_version": _TOOL_VERSION,
            "received_keys": sorted(payload.keys()),
        },
    }