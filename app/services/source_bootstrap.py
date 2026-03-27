from __future__ import annotations

import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.models import AnalysisResponse, RawQcResponse, SummaryStatsResponse
from app.services.workflows import (
    analyze_raw_qc_workflow,
    analyze_summary_stats_workflow,
    analyze_vcf_workflow,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
BOOTSTRAP_DIR = ROOT_DIR / "skills" / "chatgenome-orchestrator" / "bootstrap"
UPLOAD_ROOT = ROOT_DIR / "uploads"


@lru_cache(maxsize=1)
def load_bootstrap_manifests() -> dict[str, dict[str, object]]:
    manifests: dict[str, dict[str, object]] = {}
    for manifest_path in sorted(BOOTSTRAP_DIR.glob("*.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        source_type = str(payload.get("source_type") or "").strip().lower()
        if not source_type:
            continue
        manifests[source_type] = payload
    return manifests


def load_bootstrap_manifest(source_type: str) -> dict[str, object] | None:
    return load_bootstrap_manifests().get(source_type.strip().lower())


def persist_uploaded_source_bytes(source_type: str, file_name: str, data: bytes) -> Path:
    manifest = load_bootstrap_manifest(source_type)
    if manifest is None:
        raise ValueError(f"No bootstrap manifest is registered for source type: {source_type}")
    upload_subdir = str(manifest.get("upload_subdir") or source_type).strip()
    upload_dir = UPLOAD_ROOT / upload_subdir
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(Path(file_name).suffixes) or Path(file_name).suffix or ".dat"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in Path(file_name).stem)
    durable_path = upload_dir / f"{uuid.uuid4().hex[:12]}_{safe_stem}{suffixes}"
    durable_path.write_bytes(data)
    return durable_path


BOOTSTRAP_RUNNERS: dict[str, Any] = {
    "vcf": analyze_vcf_workflow,
    "raw_qc": analyze_raw_qc_workflow,
    "summary_stats": analyze_summary_stats_workflow,
}


def run_bootstrap_analysis(
    source_type: str,
    source_path: str,
    file_name: str,
    **kwargs: Any,
) -> AnalysisResponse | RawQcResponse | SummaryStatsResponse:
    manifest = load_bootstrap_manifest(source_type)
    if manifest is None:
        raise ValueError(f"No bootstrap manifest is registered for source type: {source_type}")
    runner_name = str(manifest.get("runner") or "").strip().lower()
    runner = BOOTSTRAP_RUNNERS.get(runner_name or source_type.strip().lower())
    if runner is None:
        raise NotImplementedError(f"No bootstrap runner is registered for source type: {source_type}")
    if source_type == "vcf":
        return runner(
            source_path,
            annotation_scope=kwargs.get("annotation_scope", "representative"),
            annotation_limit=kwargs.get("annotation_limit"),
        )
    if source_type == "raw_qc":
        return runner(source_path, file_name)
    if source_type == "summary_stats":
        return runner(
            source_path,
            file_name,
            genome_build=kwargs.get("genome_build", "unknown"),
            trait_type=kwargs.get("trait_type", "unknown"),
        )
    raise NotImplementedError(f"Unsupported bootstrap source type: {source_type}")
