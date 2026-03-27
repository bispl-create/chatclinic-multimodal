from __future__ import annotations

import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from app.models import AnalysisResponse, PrsPrepResponse, RawQcResponse, SummaryStatsResponse, SymbolicAltSummary, ToolInfo
from app.services.fastqc import FASTQC_OUTPUT_DIR
from app.services.prs_prep import analyze_prs_prep
from app.services.summary_stats import analyze_summary_stats
from app.services.tool_runner import discover_tools, run_tool
from app.services.workflow_fallbacks import compute_vcf_fallback_value
from app.services.workflow_hooks import (
    apply_vcf_postprocess_hook,
    apply_vcf_preprocess_hook,
)
from app.services.workflow_internal_steps import (
    execute_raw_qc_internal_step,
    execute_summary_stats_internal_step,
)
from app.services.workflow_responses import (
    assemble_analysis_response_from_vcf_context,
    build_analysis_workflow_result,
    build_raw_qc_workflow_result,
    build_summary_stats_workflow_result,
)
from app.services.workflow_transforms import transform_bound_value


ROOT_DIR = Path(__file__).resolve().parents[2]
PLUGINS_DIR = ROOT_DIR / "plugins"
WORKFLOWS_DIR = ROOT_DIR / "skills" / "chatgenome-orchestrator" / "workflows"


def _normalize_workflow_step(step: object, workflow_name: str) -> dict[str, object]:
    if not isinstance(step, dict):
        raise ValueError(f"Workflow {workflow_name} contains a non-object step.")
    tool_name = str(step.get("tool") or "").strip()
    bind_name = str(step.get("bind") or "").strip()
    if not tool_name:
        raise ValueError(f"Workflow {workflow_name} contains a step without `tool`.")
    if not bind_name:
        raise ValueError(f"Workflow {workflow_name} step `{tool_name}` is missing `bind`.")
    needs = [str(item).strip() for item in step.get("needs", []) if str(item).strip()]
    normalized: dict[str, object] = {
        "tool": tool_name,
        "bind": bind_name,
        "needs": needs,
    }
    on_fail = str(step.get("on_fail") or "").strip().lower()
    if on_fail:
        normalized["on_fail"] = on_fail
    return normalized


def _normalize_workflow_manifest(payload: dict[str, object]) -> dict[str, object]:
    workflow_name = str(payload.get("name") or "").strip()
    source_type = str(payload.get("source_type") or "").strip().lower()
    requested_view = str(payload.get("requested_view") or payload.get("default_view") or "").strip()
    response_kind = str(payload.get("response_kind") or "").strip().lower()
    answer_template = str(payload.get("answer_template") or "").strip()
    steps = payload.get("steps")

    if not workflow_name:
        raise ValueError("Workflow manifest is missing `name`.")
    if not source_type:
        raise ValueError(f"Workflow {workflow_name} is missing `source_type`.")
    if not requested_view:
        raise ValueError(f"Workflow {workflow_name} is missing `requested_view`.")
    if not response_kind:
        raise ValueError(f"Workflow {workflow_name} is missing `response_kind`.")
    if not answer_template:
        raise ValueError(f"Workflow {workflow_name} is missing `answer_template`.")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Workflow {workflow_name} does not define a valid non-empty step list.")

    normalized_steps = [_normalize_workflow_step(step, workflow_name) for step in steps]
    normalized_manifest = dict(payload)
    normalized_manifest["name"] = workflow_name
    normalized_manifest["source_type"] = source_type
    normalized_manifest["requested_view"] = requested_view
    normalized_manifest["response_kind"] = response_kind
    normalized_manifest["answer_template"] = answer_template
    normalized_manifest["steps"] = normalized_steps
    normalized_manifest.setdefault("requires", [])
    normalized_manifest.setdefault("produces", [])
    normalized_manifest.setdefault("notes", [])
    normalized_manifest["default_view"] = requested_view
    return normalized_manifest


@lru_cache(maxsize=1)
def load_workflow_manifests() -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    for manifest in sorted(WORKFLOWS_DIR.glob("*.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            try:
                manifests.append(_normalize_workflow_manifest(payload))
            except Exception:
                continue
    return manifests


def list_workflow_manifests(source_type: str | None = None) -> list[dict[str, object]]:
    manifests = load_workflow_manifests()
    if source_type is None:
        return manifests
    normalized = source_type.strip().lower()
    return [
        item
        for item in manifests
        if str(item.get("source_type") or "").strip().lower() == normalized
    ]


def load_workflow_manifest(name: str | None) -> dict[str, object] | None:
    if not name:
        return None
    normalized = str(name).strip().lower()
    for manifest in load_workflow_manifests():
        if str(manifest.get("name") or "").strip().lower() == normalized:
            return manifest
    return None


@lru_cache(maxsize=1)
def _load_tool_workflow_bindings() -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(PLUGINS_DIR.glob("*/tool.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        tool_name = str(payload.get("name") or "").strip()
        workflow_binding = payload.get("workflow_binding")
        if tool_name and isinstance(workflow_binding, dict):
            bindings[tool_name] = workflow_binding
    return bindings


def _workflow_binding_for_tool(tool_name: str, source_type: str | None = None) -> dict[str, Any] | None:
    binding = _load_tool_workflow_bindings().get(tool_name)
    if not isinstance(binding, dict):
        return None
    if source_type is not None:
        binding_source_type = str(binding.get("source_type") or "").strip().lower()
        if binding_source_type and binding_source_type != source_type.strip().lower():
            return None
    return binding


def _serialize_binding_input(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_serialize_binding_input(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_binding_input(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_binding_input(item) for key, item in value.items()}
    return value


def _resolve_binding_reference(reference: Any, context: dict[str, Any]) -> Any:
    if not isinstance(reference, str) or not reference.startswith("$"):
        return reference
    path = reference[1:].strip()
    value: Any = context
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            break
    return value


def _build_tool_payload_from_binding(binding: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    input_map = binding.get("input_map")
    if not isinstance(input_map, dict):
        return {}
    payload: dict[str, Any] = {}
    for payload_key, reference in input_map.items():
        payload[str(payload_key)] = _serialize_binding_input(_resolve_binding_reference(reference, context))
    return payload


def _extract_tool_result_value(result: dict[str, Any], binding: dict[str, Any]) -> Any:
    result_path = str(binding.get("result_path") or "").strip()
    if not result_path:
        return result
    return result.get(result_path)


def _execute_generic_vcf_bound_tool(tool_name: str, context: dict[str, Any], bind_name: str) -> None:
    binding = _workflow_binding_for_tool(tool_name, source_type="vcf")
    if binding is None:
        raise NotImplementedError(f"No generic VCF workflow binding is registered for {tool_name}.")
    payload = _build_tool_payload_from_binding(binding, context)
    preprocess_hook = str(binding.get("preprocess") or "").strip()
    if preprocess_hook:
        payload = apply_vcf_preprocess_hook(preprocess_hook, context, payload)
    result = run_tool(tool_name, payload)
    value = _extract_tool_result_value(result, binding)
    transformed_value = transform_bound_value(str(binding.get("transform") or "identity"), value)
    append_used_tool = True
    postprocess_hook = str(binding.get("postprocess") or "").strip()
    if postprocess_hook:
        transformed_value, append_used_tool = apply_vcf_postprocess_hook(
            postprocess_hook,
            context,
            result,
            transformed_value,
        )
    context[bind_name] = transformed_value
    used_tools_label = str(binding.get("used_tools_label") or tool_name).strip()
    if append_used_tool and used_tools_label:
        context["used_tools"].append(used_tools_label)


def _apply_generic_vcf_fallback(tool_name: str, context: dict[str, Any], bind_name: str) -> bool:
    binding = _workflow_binding_for_tool(tool_name, source_type="vcf")
    if binding is None:
        return False
    fallback_transform = str(binding.get("fallback_transform") or "").strip()
    if not fallback_transform:
        return False
    context[bind_name] = compute_vcf_fallback_value(fallback_transform, context)
    return True


def _vcf_workflow_context(
    path: str,
    annotation_scope: str,
    annotation_limit: int | None,
) -> dict[str, Any]:
    return {
        "source_vcf_path": path,
        "annotation_scope": annotation_scope,
        "annotation_limit": annotation_limit,
        "max_examples": int(os.getenv("MAX_EXAMPLE_VARIANTS", "8")),
        "used_tools": [],
        "tool_registry": discover_tools(),
        "facts": None,
        "annotations": [],
        "roh_segments": [],
        "snpeff_result": None,
        "candidate_variants": [],
        "clinvar_summary": [],
        "consequence_summary": [],
        "clinical_coverage_summary": [],
        "filtering_summary": [],
        "symbolic_alt_summary": SymbolicAltSummary(count=0, examples=[]),
        "references": [],
        "recommendations": [],
        "ui_cards": [],
        "draft_answer": "",
    }

VCF_CUSTOM_STEP_EXECUTORS: dict[str, Any] = {}


def _run_vcf_workflow_step(step: dict[str, Any], context: dict[str, Any]) -> None:
    tool_name = str(step.get("tool") or "").strip()
    bind_name = str(step.get("bind") or "").strip()
    needs = [str(item).strip() for item in step.get("needs", []) if str(item).strip()]
    binding = _workflow_binding_for_tool(tool_name, source_type="vcf")
    default_on_fail = str(binding.get("on_fail_default") or "raise").strip().lower() if binding else "raise"
    on_fail = str(step.get("on_fail") or default_on_fail).strip().lower()

    for need in needs:
        value = context.get(need)
        if value in (None, "", []):
            raise RuntimeError(f"VCF workflow step `{tool_name}` is missing required context `{need}`.")

    try:
        if binding is not None:
            _execute_generic_vcf_bound_tool(tool_name, context, bind_name)
            return
        executors = VCF_CUSTOM_STEP_EXECUTORS.get(tool_name)
        if executors is None:
            raise NotImplementedError(f"Unsupported VCF workflow step tool: {tool_name}")
        primary_executor, _ = executors
        primary_executor(context, bind_name)
        return
    except Exception:
        if binding is not None and _apply_generic_vcf_fallback(tool_name, context, bind_name):
            return
        executors = VCF_CUSTOM_STEP_EXECUTORS.get(tool_name)
        fallback_executor = executors[1] if executors is not None else None
        if fallback_executor is not None and on_fail == "continue":
            fallback_executor(context, bind_name)
            return
        if on_fail != "continue":
            raise


def _summary_stats_workflow_context(
    analysis: SummaryStatsResponse,
) -> dict[str, Any]:
    return {
        "source_stats_path": analysis.source_stats_path,
        "file_name": analysis.file_name,
        "genome_build": analysis.genome_build,
        "trait_type": analysis.trait_type,
        "analysis": analysis,
        "prs_prep_result": analysis.prs_prep_result,
        "draft_answer": analysis.draft_answer,
    }


def _run_summary_stats_workflow_step(step: dict[str, Any], context: dict[str, Any]) -> None:
    tool_name = str(step.get("tool") or "").strip()
    bind_name = str(step.get("bind") or "").strip()
    needs = [str(item).strip() for item in step.get("needs", []) if str(item).strip()]

    for need in needs:
        value = context.get(need)
        if value in (None, "", []):
            raise RuntimeError(f"Summary-statistics workflow step `{tool_name}` is missing required context `{need}`.")
    binding = _workflow_binding_for_tool(tool_name, source_type="summary_stats")
    if binding is not None:
        payload = _build_tool_payload_from_binding(binding, context)
        result = run_tool(tool_name, payload)
        value = _extract_tool_result_value(result, binding)
        transformed_value = transform_bound_value(str(binding.get("transform") or "identity"), value)
        context[bind_name] = transformed_value
        used_tools_label = str(binding.get("used_tools_label") or tool_name).strip()
        analysis_obj = context.get("analysis")
        if used_tools_label and isinstance(analysis_obj, SummaryStatsResponse):
            analysis_obj.used_tools.append(used_tools_label)
        return
    execute_summary_stats_internal_step(
        tool_name,
        context,
        bind_name,
        analyze_summary_stats_workflow=analyze_summary_stats_workflow,
        analyze_prs_prep_workflow=analyze_prs_prep_workflow,
    )


def _run_registered_summary_stats_workflow_from_manifest(
    analysis: SummaryStatsResponse,
    manifest: dict[str, object],
) -> dict[str, object]:
    context = _summary_stats_workflow_context(analysis)
    steps = manifest.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"Workflow {manifest.get('name')} does not define a valid step list.")
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(f"Workflow {manifest.get('name')} contains a non-object step.")
        _run_summary_stats_workflow_step(step, context)
    return build_summary_stats_workflow_result(analysis, manifest, context)


def _raw_qc_workflow_context(
    analysis: RawQcResponse,
) -> dict[str, Any]:
    return {
        "source_raw_path": analysis.source_raw_path,
        "file_name": analysis.facts.file_name,
        "analysis": analysis,
    }


def _run_raw_qc_workflow_step(step: dict[str, Any], context: dict[str, Any]) -> None:
    tool_name = str(step.get("tool") or "").strip()
    bind_name = str(step.get("bind") or "").strip()
    needs = [str(item).strip() for item in step.get("needs", []) if str(item).strip()]

    for need in needs:
        value = context.get(need)
        if value in (None, "", []):
            raise RuntimeError(f"Raw-QC workflow step `{tool_name}` is missing required context `{need}`.")
    execute_raw_qc_internal_step(
        tool_name,
        context,
        bind_name,
        analyze_raw_qc_workflow=analyze_raw_qc_workflow,
    )


def _run_registered_raw_qc_workflow_from_manifest(
    analysis: RawQcResponse,
    manifest: dict[str, object],
) -> dict[str, object]:
    context = _raw_qc_workflow_context(analysis)
    steps = manifest.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"Workflow {manifest.get('name')} does not define a valid step list.")
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(f"Workflow {manifest.get('name')} contains a non-object step.")
        _run_raw_qc_workflow_step(step, context)
    return build_raw_qc_workflow_result(manifest, context)


def _run_registered_vcf_workflow_from_manifest(
    path: str,
    manifest: dict[str, object],
    annotation_scope: str = "representative",
    annotation_limit: int | None = None,
) -> AnalysisResponse:
    context = _vcf_workflow_context(path, annotation_scope=annotation_scope, annotation_limit=annotation_limit)
    steps = manifest.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"Workflow {manifest.get('name')} does not define a valid step list.")
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(f"Workflow {manifest.get('name')} contains a non-object step.")
        _run_vcf_workflow_step(step, context)
    return assemble_analysis_response_from_vcf_context(context)


def analyze_vcf_workflow(
    path: str,
    annotation_scope: str = "representative",
    annotation_limit: int | None = None,
) -> AnalysisResponse:
    manifest = load_workflow_manifest("representative_vcf_review")
    if manifest is None:
        raise ValueError("The representative_vcf_review workflow manifest is not available.")
    return _run_registered_vcf_workflow_from_manifest(
        path,
        manifest,
        annotation_scope=annotation_scope,
        annotation_limit=annotation_limit,
    )


def analyze_raw_qc_workflow(path: str, original_name: str) -> RawQcResponse:
    try:
        result = run_tool(
            "fastqc_execution_tool",
            {
                "raw_path": path,
                "original_name": original_name,
            },
        )
        return RawQcResponse(**result)
    except Exception as exc:
        raise RuntimeError(f"Raw QC failed: {exc}") from exc


def run_registered_analysis_workflow(
    workflow_name: str,
    analysis: AnalysisResponse,
) -> dict[str, object]:
    manifest = load_workflow_manifest(workflow_name)
    if manifest is None:
        raise ValueError(f"Unknown analysis workflow: {workflow_name}")
    source_type = str(manifest.get("source_type") or "").strip().lower()
    if source_type != "vcf":
        raise ValueError(f"Workflow {workflow_name} is not registered for VCF sources.")

    requires = [str(item).strip() for item in manifest.get("requires", []) if str(item).strip()]
    if "source_vcf_path" in requires and not analysis.source_vcf_path:
        raise RuntimeError(
            "The active analysis does not expose a source VCF path, so this workflow cannot be rerun from chat."
        )

    refreshed = _run_registered_vcf_workflow_from_manifest(
        analysis.source_vcf_path or "",
        manifest,
        annotation_scope="representative",
        annotation_limit=None,
    )
    return build_analysis_workflow_result(manifest, refreshed)


def run_registered_raw_qc_workflow(
    workflow_name: str,
    analysis: RawQcResponse,
) -> dict[str, object]:
    manifest = load_workflow_manifest(workflow_name)
    if manifest is None:
        raise ValueError(f"Unknown raw-QC workflow: {workflow_name}")
    source_type = str(manifest.get("source_type") or "").strip().lower()
    if source_type != "raw_qc":
        raise ValueError(f"Workflow {workflow_name} is not registered for raw-QC sources.")

    requires = [str(item).strip() for item in manifest.get("requires", []) if str(item).strip()]
    if "source_raw_path" in requires and not analysis.source_raw_path:
        raise RuntimeError(
            "The active raw-QC session does not expose a durable source file path, so this workflow cannot be rerun from chat."
        )

    return _run_registered_raw_qc_workflow_from_manifest(analysis, manifest)


def analyze_summary_stats_workflow(
    path: str,
    original_name: str,
    genome_build: str = "unknown",
    trait_type: str = "unknown",
) -> SummaryStatsResponse:
    result = analyze_summary_stats(path, original_name, genome_build=genome_build, trait_type=trait_type)
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_prs_prep_workflow(
    path: str,
    original_name: str,
    genome_build: str = "unknown",
) -> PrsPrepResponse:
    result = analyze_prs_prep(path, original_name, genome_build=genome_build)
    result.analysis_id = str(uuid.uuid4())
    return result


def run_registered_summary_stats_workflow(
    workflow_name: str,
    analysis: SummaryStatsResponse,
) -> dict[str, object]:
    manifest = load_workflow_manifest(workflow_name)
    if manifest is None:
        raise ValueError(f"Unknown summary-statistics workflow: {workflow_name}")
    source_type = str(manifest.get("source_type") or "").strip().lower()
    if source_type != "summary_stats":
        raise ValueError(f"Workflow {workflow_name} is not registered for summary-statistics sources.")

    requires = [str(item).strip() for item in manifest.get("requires", []) if str(item).strip()]
    if "source_stats_path" in requires and not analysis.source_stats_path:
        raise RuntimeError(
            "The active summary-statistics session does not expose a durable source file path, so this workflow cannot be rerun from chat."
        )

    return _run_registered_summary_stats_workflow_from_manifest(analysis, manifest)
