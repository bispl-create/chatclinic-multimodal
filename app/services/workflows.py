from __future__ import annotations

import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from app.models import (
    AnalysisFacts,
    AnalysisResponse,
    CountSummaryItem,
    DetailedCountSummaryItem,
    PrsPrepResponse,
    RawQcResponse,
    RankedCandidate,
    RohSegment,
    SummaryStatsResponse,
    SymbolicAltSummary,
    ToolInfo,
    VariantAnnotation,
)
from app.services.annotation import build_draft_answer, build_ui_cards
from app.services.candidate_ranking import build_ranked_candidates
from app.services.fastqc import FASTQC_OUTPUT_DIR
from app.services.recommendation import build_recommendations
from app.services.references import build_reference_bundle
from app.services.roh_analysis import run_roh_analysis
from app.services.prs_prep import analyze_prs_prep
from app.services.summary_stats import analyze_summary_stats
from app.services.tool_runner import discover_tools, run_tool
from app.services.variant_annotation import annotate_variants
from app.services.vcf_summary import summarize_vcf
from app.services.workflow_fallbacks import compute_vcf_fallback_value
from app.services.workflow_hooks import (
    annotation_key,
    apply_vcf_postprocess_hook,
    apply_vcf_preprocess_hook,
    snpeff_genome_from_build,
)
from app.services.workflow_internal_steps import (
    execute_raw_qc_internal_step,
    execute_summary_stats_internal_step,
)
from app.services.workflow_transforms import transform_bound_value


ROOT_DIR = Path(__file__).resolve().parents[2]
PLUGINS_DIR = ROOT_DIR / "plugins"
WORKFLOWS_DIR = ROOT_DIR / "skills" / "chatgenome-orchestrator" / "workflows"


class _SafeFormatDict(dict[str, object]):
    def __missing__(self, key: str) -> str:
        return ""


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


def _execute_vcf_qc_step(context: dict[str, Any], bind_name: str) -> None:
    path = str(context["source_vcf_path"])
    qc_result = run_tool(
        "vcf_qc_tool",
        {
            "vcf_path": path,
            "max_examples": context["max_examples"],
        },
    )
    context[bind_name] = AnalysisFacts(**qc_result["facts"])
    context["used_tools"].append("vcf_qc_tool")


def _fallback_vcf_qc_step(context: dict[str, Any], bind_name: str) -> None:
    path = str(context["source_vcf_path"])
    context[bind_name] = summarize_vcf(path, max_examples=context["max_examples"])


def _execute_annotation_step(context: dict[str, Any], bind_name: str) -> None:
    path = str(context["source_vcf_path"])
    facts: AnalysisFacts = context["facts"]
    annotation_result = run_tool(
        "annotation_tool",
        {
            "vcf_path": path,
            "facts": facts.model_dump(),
            "scope": context["annotation_scope"],
            "limit": context["annotation_limit"],
        },
    )
    context[bind_name] = [VariantAnnotation(**item) for item in annotation_result["annotations"]]
    context["used_tools"].append("annotation_tool")


def _fallback_annotation_step(context: dict[str, Any], bind_name: str) -> None:
    path = str(context["source_vcf_path"])
    facts: AnalysisFacts = context["facts"]
    context[bind_name] = annotate_variants(
        path,
        facts,
        scope=context["annotation_scope"],
        limit=context["annotation_limit"],
    )


def _execute_roh_step(context: dict[str, Any], bind_name: str) -> None:
    path = str(context["source_vcf_path"])
    roh_result = run_tool("roh_analysis_tool", {"vcf_path": path})
    context[bind_name] = [RohSegment(**item) for item in roh_result["roh_segments"]]
    context["used_tools"].append("roh_analysis_tool")


def _fallback_roh_step(context: dict[str, Any], bind_name: str) -> None:
    path = str(context["source_vcf_path"])
    context[bind_name] = run_roh_analysis(path)


def _execute_candidate_ranking_step(context: dict[str, Any], bind_name: str) -> None:
    candidate_result = run_tool(
        "candidate_ranking_tool",
        {
            "annotations": [item.model_dump() for item in context["annotations"]],
            "roh_segments": [item.model_dump() for item in context["roh_segments"]],
            "limit": 8,
        },
    )
    context[bind_name] = [RankedCandidate(**item) for item in candidate_result["candidate_variants"]]
    context["used_tools"].append("candidate_ranking_tool")


def _fallback_candidate_ranking_step(context: dict[str, Any], bind_name: str) -> None:
    context[bind_name] = build_ranked_candidates(context["annotations"], context["roh_segments"], limit=8)


def _execute_clinvar_review_step(context: dict[str, Any], bind_name: str) -> None:
    clinvar_result = run_tool(
        "clinvar_review_tool",
        {"annotations": [item.model_dump() for item in context["annotations"]]},
    )
    context[bind_name] = [CountSummaryItem(**item) for item in clinvar_result["clinvar_summary"]]
    context["used_tools"].append("clinvar_review_tool")


def _fallback_clinvar_review_step(context: dict[str, Any], bind_name: str) -> None:
    counts: dict[str, int] = {}
    for item in context["annotations"]:
        key = (
            item.clinical_significance.strip()
            if item.clinical_significance and item.clinical_significance != "."
            else "Unreviewed"
        )
        counts[key] = counts.get(key, 0) + 1
    context[bind_name] = [
        CountSummaryItem(label=label, count=count)
        for label, count in sorted(counts.items(), key=lambda part: part[1], reverse=True)
    ]


def _execute_vep_consequence_step(context: dict[str, Any], bind_name: str) -> None:
    consequence_result = run_tool(
        "vep_consequence_tool",
        {
            "annotations": [item.model_dump() for item in context["annotations"]],
            "limit": 10,
        },
    )
    context[bind_name] = [CountSummaryItem(**item) for item in consequence_result["consequence_summary"]]
    context["used_tools"].append("vep_consequence_tool")


def _fallback_vep_consequence_step(context: dict[str, Any], bind_name: str) -> None:
    counts: dict[str, int] = {}
    for item in context["annotations"]:
        key = item.consequence.strip() if item.consequence and item.consequence != "." else "Unclassified"
        counts[key] = counts.get(key, 0) + 1
    context[bind_name] = [
        CountSummaryItem(label=label, count=count)
        for label, count in sorted(counts.items(), key=lambda part: part[1], reverse=True)[:10]
    ]


def _execute_clinical_coverage_step(context: dict[str, Any], bind_name: str) -> None:
    coverage_result = run_tool(
        "clinical_coverage_tool",
        {"annotations": [item.model_dump() for item in context["annotations"]]},
    )
    context[bind_name] = [DetailedCountSummaryItem(**item) for item in coverage_result["clinical_coverage_summary"]]
    context["used_tools"].append("clinical_coverage_tool")


def _fallback_clinical_coverage_step(context: dict[str, Any], bind_name: str) -> None:
    annotations = list(context["annotations"])
    total = len(annotations)

    def detail(label: str, count: int) -> DetailedCountSummaryItem:
        percent = round((count / total) * 100) if total else 0
        return DetailedCountSummaryItem(label=label, count=count, detail=f"{count}/{total} annotated ({percent}%)")

    context[bind_name] = [
        detail(
            "ClinVar coverage",
            sum(
                1
                for item in annotations
                if (item.clinical_significance and item.clinical_significance != ".")
                or (item.clinvar_conditions and item.clinvar_conditions != ".")
            ),
        ),
        detail("gnomAD coverage", sum(1 for item in annotations if item.gnomad_af and item.gnomad_af != ".")),
        detail("Gene mapping", sum(1 for item in annotations if item.gene and item.gene != ".")),
        detail(
            "HGVS coverage",
            sum(1 for item in annotations if (item.hgvsc and item.hgvsc != ".") or (item.hgvsp and item.hgvsp != "."))
        ),
        detail("Protein change", sum(1 for item in annotations if item.hgvsp and item.hgvsp != ".")),
    ]


def _execute_filtering_view_step(context: dict[str, Any], bind_name: str) -> None:
    filtering_result = run_tool(
        "filtering_view_tool",
        {"annotations": [item.model_dump() for item in context["annotations"]]},
    )
    context[bind_name] = [DetailedCountSummaryItem(**item) for item in filtering_result["filtering_summary"]]
    context["used_tools"].append("filtering_view_tool")


def _fallback_filtering_view_step(context: dict[str, Any], bind_name: str) -> None:
    annotations = list(context["annotations"])
    unique_genes = {item.gene.strip() for item in annotations if item.gene and item.gene.strip() not in {"", "."}}
    clinvar_labeled = sum(1 for item in annotations if item.clinical_significance and item.clinical_significance != ".")
    symbolic = sum(1 for item in annotations if any(alt.startswith("<") and alt.endswith(">") for alt in item.alts))
    context[bind_name] = [
        DetailedCountSummaryItem(
            label="Annotated rows", count=len(annotations), detail=f"{len(annotations)} rows currently available in the triage table"
        ),
        DetailedCountSummaryItem(
            label="Distinct genes", count=len(unique_genes), detail=f"{len(unique_genes)} genes represented in the annotated subset"
        ),
        DetailedCountSummaryItem(
            label="ClinVar-labeled rows",
            count=clinvar_labeled,
            detail=f"{clinvar_labeled} rows contain a ClinVar-style significance label",
        ),
        DetailedCountSummaryItem(
            label="Symbolic ALT rows",
            count=symbolic,
            detail=f"{symbolic} rows are symbolic ALT records that may need separate handling",
        ),
    ]


def _execute_symbolic_alt_step(context: dict[str, Any], bind_name: str) -> None:
    symbolic_result = run_tool(
        "symbolic_alt_tool",
        {"annotations": [item.model_dump() for item in context["annotations"]]},
    )
    context[bind_name] = SymbolicAltSummary(**symbolic_result["symbolic_alt_summary"])
    context["used_tools"].append("symbolic_alt_tool")


def _fallback_symbolic_alt_step(context: dict[str, Any], bind_name: str) -> None:
    symbolic_items = [item for item in context["annotations"] if any(alt.startswith("<") and alt.endswith(">") for alt in item.alts)]
    context[bind_name] = SymbolicAltSummary(
        count=len(symbolic_items),
        examples=[
            {
                "locus": f"{item.contig}:{item.pos_1based}",
                "gene": item.gene or "",
                "alts": item.alts,
                "consequence": item.consequence or "",
                "genotype": item.genotype or "",
            }
            for item in symbolic_items[:5]
        ],
    )


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


def _assemble_analysis_response_from_vcf_context(context: dict[str, Any]) -> AnalysisResponse:
    facts: AnalysisFacts = context["facts"]
    annotations = list(context["annotations"])
    if not context["references"]:
        context["references"] = build_reference_bundle(facts, annotations[: min(len(annotations), 20)])
    if not context["recommendations"]:
        context["recommendations"] = build_recommendations(facts)
    if not context["ui_cards"]:
        context["ui_cards"] = build_ui_cards(facts, annotations)
    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        facts=facts,
        annotations=annotations,
        roh_segments=list(context["roh_segments"]),
        source_vcf_path=str(context["source_vcf_path"]),
        snpeff_result=context.get("snpeff_result"),
        candidate_variants=list(context["candidate_variants"]),
        clinvar_summary=list(context["clinvar_summary"]),
        consequence_summary=list(context["consequence_summary"]),
        clinical_coverage_summary=list(context["clinical_coverage_summary"]),
        filtering_summary=list(context["filtering_summary"]),
        symbolic_alt_summary=context["symbolic_alt_summary"],
        references=list(context["references"]),
        recommendations=list(context["recommendations"]),
        ui_cards=list(context["ui_cards"]),
        draft_answer=str(context["draft_answer"]),
        used_tools=list(context["used_tools"]),
        tool_registry=list(context["tool_registry"]),
    )


def _stringify_logged_tools(value: Any) -> str:
    items = [str(item).strip() for item in list(value or []) if str(item).strip()]
    return ", ".join(items) if items else "none"


def _workflow_answer_tokens(
    source_type: str,
    workflow_name: str,
    requested_view: str,
    analysis: object,
    context: dict[str, Any],
) -> dict[str, object]:
    tokens: dict[str, object] = {
        "workflow_name": workflow_name,
        "requested_view": requested_view,
        "logged_tools": "none",
        "active_file": "",
        "candidate_count": 0,
        "module_count": 0,
        "row_count": 0,
        "auto_mapped_count": 0,
        "score_file_ready": "",
        "kept_rows": 0,
        "dropped_rows": 0,
        "build_check": "",
    }

    if source_type == "vcf" and isinstance(analysis, AnalysisResponse):
        tokens.update(
            {
                "active_file": analysis.facts.file_name,
                "logged_tools": _stringify_logged_tools(analysis.used_tools),
                "candidate_count": len(analysis.candidate_variants or []),
            }
        )
        return tokens

    if source_type == "raw_qc" and isinstance(analysis, RawQcResponse):
        tokens.update(
            {
                "active_file": analysis.facts.file_name,
                "logged_tools": _stringify_logged_tools(analysis.used_tools),
                "module_count": len(analysis.modules),
            }
        )
        return tokens

    if source_type == "summary_stats" and isinstance(analysis, SummaryStatsResponse):
        auto_mapped_count = sum(1 for value in analysis.mapped_fields.model_dump().values() if value)
        tokens.update(
            {
                "active_file": analysis.file_name,
                "logged_tools": _stringify_logged_tools(getattr(analysis, "used_tools", []) or []),
                "row_count": analysis.row_count,
                "auto_mapped_count": auto_mapped_count,
            }
        )
        prs_prep_result = context.get("prs_prep_result")
        if isinstance(prs_prep_result, PrsPrepResponse):
            tokens.update(
                {
                    "active_file": prs_prep_result.file_name,
                    "score_file_ready": "yes" if prs_prep_result.score_file_ready else "no",
                    "kept_rows": prs_prep_result.kept_rows,
                    "dropped_rows": prs_prep_result.dropped_rows,
                    "build_check": (
                        f"{prs_prep_result.build_check.inferred_build} "
                        f"({prs_prep_result.build_check.build_confidence})"
                    ),
                }
            )
        return tokens

    return tokens


def _format_workflow_answer(
    manifest: dict[str, object],
    source_type: str,
    analysis: object,
    context: dict[str, Any],
) -> str:
    workflow_name = str(manifest.get("name") or "")
    requested_view = str(manifest.get("requested_view") or "")
    template = str(manifest.get("answer_template") or "").strip()
    tokens = _workflow_answer_tokens(source_type, workflow_name, requested_view, analysis, context)
    return template.format_map(_SafeFormatDict(tokens))


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

    requested_view = str(manifest.get("requested_view") or "sumstats")
    prs_prep_result = context.get("prs_prep_result")
    refreshed = (
        analysis.model_copy(update={"prs_prep_result": prs_prep_result})
        if isinstance(prs_prep_result, PrsPrepResponse)
        else context["analysis"]
    )
    answer = _format_workflow_answer(manifest, "summary_stats", refreshed, context)
    result: dict[str, object] = {
        "answer": answer,
        "analysis": refreshed,
        "requested_view": requested_view,
    }
    if isinstance(prs_prep_result, PrsPrepResponse):
        result["prs_prep_result"] = prs_prep_result
    return result


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

    refreshed: RawQcResponse = context["analysis"]
    requested_view = str(manifest.get("requested_view") or "rawqc")
    answer = _format_workflow_answer(manifest, "raw_qc", refreshed, context)
    return {
        "answer": answer,
        "analysis": refreshed,
        "requested_view": requested_view,
    }


def _analyze_vcf_workflow_legacy(
    path: str,
    annotation_scope: str = "representative",
    annotation_limit: int | None = None,
) -> AnalysisResponse:
    max_examples = int(os.getenv("MAX_EXAMPLE_VARIANTS", "8"))
    used_tools: list[str] = []
    tool_registry = discover_tools()

    try:
        qc_result = run_tool(
            "vcf_qc_tool",
            {
                "vcf_path": path,
                "max_examples": max_examples,
            },
        )
        facts = AnalysisFacts(**qc_result["facts"])
        used_tools.append("vcf_qc_tool")
    except Exception:
        facts = summarize_vcf(path, max_examples=max_examples)

    try:
        annotation_result = run_tool(
            "annotation_tool",
            {
                "vcf_path": path,
                "facts": facts.model_dump(),
                "scope": annotation_scope,
                "limit": annotation_limit,
            },
        )
        annotations = [VariantAnnotation(**item) for item in annotation_result["annotations"]]
        used_tools.append("annotation_tool")
    except Exception:
        annotations = annotate_variants(
            path,
            facts,
            scope=annotation_scope,
            limit=annotation_limit,
        )

    snpeff_result = None
    try:
        snpeff_payload = run_tool(
            "snpeff_execution_tool",
            {
                "vcf_path": path,
                "genome": snpeff_genome_from_build(facts.genome_build_guess),
                "output_prefix": f"{Path(path).stem}.aux",
                "parse_limit": 10,
            },
        )
        from app.models import SnpEffResponse

        snpeff_result = SnpEffResponse(**snpeff_payload)
        used_tools.append("snpeff_execution_tool")
    except Exception:
        snpeff_result = None

    try:
        roh_result = run_tool("roh_analysis_tool", {"vcf_path": path})
        roh_segments = [RohSegment(**item) for item in roh_result["roh_segments"]]
        used_tools.append("roh_analysis_tool")
    except Exception:
        roh_segments = run_roh_analysis(path)

    preliminary_candidates = build_ranked_candidates(annotations, roh_segments, limit=24)
    shortlisted_annotations = [entry.item for entry in preliminary_candidates]
    try:
        cadd_result = run_tool(
            "cadd_lookup_tool",
            {
                "annotations": [item.model_dump() for item in shortlisted_annotations],
                "genome_build_guess": facts.genome_build_guess,
            },
        )
        enriched_shortlisted_annotations = [VariantAnnotation(**item) for item in cadd_result["annotations"]]
        if bool(cadd_result.get("lookup_performed")):
            used_tools.append("cadd_lookup_tool")
        enriched_by_key = {annotation_key(item): item for item in enriched_shortlisted_annotations}
        annotations = [enriched_by_key.get(annotation_key(item), item) for item in annotations]
    except Exception:
        enriched_shortlisted_annotations = shortlisted_annotations

    try:
        revel_result = run_tool(
            "revel_lookup_tool",
            {
                "annotations": [item.model_dump() for item in enriched_shortlisted_annotations],
                "genome_build_guess": facts.genome_build_guess,
            },
        )
        revel_enriched_annotations = [VariantAnnotation(**item) for item in revel_result["annotations"]]
        if bool(revel_result.get("lookup_performed")):
            used_tools.append("revel_lookup_tool")
        revel_by_key = {annotation_key(item): item for item in revel_enriched_annotations}
        annotations = [revel_by_key.get(annotation_key(item), item) for item in annotations]
        enriched_shortlisted_annotations = [
            revel_by_key.get(annotation_key(item), item) for item in enriched_shortlisted_annotations
        ]
    except Exception:
        pass

    try:
        candidate_result = run_tool(
            "candidate_ranking_tool",
            {
                "annotations": [item.model_dump() for item in enriched_shortlisted_annotations],
                "roh_segments": [item.model_dump() for item in roh_segments],
                "limit": 8,
            },
        )
        candidate_variants = [RankedCandidate(**item) for item in candidate_result["candidate_variants"]]
        used_tools.append("candidate_ranking_tool")
    except Exception:
        candidate_variants = build_ranked_candidates(enriched_shortlisted_annotations, roh_segments, limit=8)

    try:
        clinvar_result = run_tool(
            "clinvar_review_tool",
            {"annotations": [item.model_dump() for item in annotations]},
        )
        clinvar_summary = [CountSummaryItem(**item) for item in clinvar_result["clinvar_summary"]]
        used_tools.append("clinvar_review_tool")
    except Exception:
        counts: dict[str, int] = {}
        for item in annotations:
            key = (
                item.clinical_significance.strip()
                if item.clinical_significance and item.clinical_significance != "."
                else "Unreviewed"
            )
            counts[key] = counts.get(key, 0) + 1
        clinvar_summary = [
            CountSummaryItem(label=label, count=count)
            for label, count in sorted(counts.items(), key=lambda part: part[1], reverse=True)
        ]

    try:
        consequence_result = run_tool(
            "vep_consequence_tool",
            {
                "annotations": [item.model_dump() for item in annotations],
                "limit": 10,
            },
        )
        consequence_summary = [CountSummaryItem(**item) for item in consequence_result["consequence_summary"]]
        used_tools.append("vep_consequence_tool")
    except Exception:
        counts = {}
        for item in annotations:
            key = item.consequence.strip() if item.consequence and item.consequence != "." else "Unclassified"
            counts[key] = counts.get(key, 0) + 1
        consequence_summary = [
            CountSummaryItem(label=label, count=count)
            for label, count in sorted(counts.items(), key=lambda part: part[1], reverse=True)[:10]
        ]

    try:
        coverage_result = run_tool(
            "clinical_coverage_tool",
            {"annotations": [item.model_dump() for item in annotations]},
        )
        clinical_coverage_summary = [
            DetailedCountSummaryItem(**item) for item in coverage_result["clinical_coverage_summary"]
        ]
        used_tools.append("clinical_coverage_tool")
    except Exception:
        total = len(annotations)

        def detail(label: str, count: int) -> DetailedCountSummaryItem:
            percent = round((count / total) * 100) if total else 0
            return DetailedCountSummaryItem(label=label, count=count, detail=f"{count}/{total} annotated ({percent}%)")

        clinical_coverage_summary = [
            detail(
                "ClinVar coverage",
                sum(
                    1
                    for item in annotations
                    if (item.clinical_significance and item.clinical_significance != ".")
                    or (item.clinvar_conditions and item.clinvar_conditions != ".")
                ),
            ),
            detail("gnomAD coverage", sum(1 for item in annotations if item.gnomad_af and item.gnomad_af != ".")),
            detail("Gene mapping", sum(1 for item in annotations if item.gene and item.gene != ".")),
            detail(
                "HGVS coverage",
                sum(1 for item in annotations if (item.hgvsc and item.hgvsc != ".") or (item.hgvsp and item.hgvsp != ".")),
            ),
            detail("Protein change", sum(1 for item in annotations if item.hgvsp and item.hgvsp != ".")),
        ]

    try:
        filtering_result = run_tool(
            "filtering_view_tool",
            {"annotations": [item.model_dump() for item in annotations]},
        )
        filtering_summary = [DetailedCountSummaryItem(**item) for item in filtering_result["filtering_summary"]]
        used_tools.append("filtering_view_tool")
    except Exception:
        unique_genes = {item.gene.strip() for item in annotations if item.gene and item.gene.strip() not in {"", "."}}
        clinvar_labeled = sum(1 for item in annotations if item.clinical_significance and item.clinical_significance != ".")
        symbolic = sum(1 for item in annotations if any(alt.startswith("<") and alt.endswith(">") for alt in item.alts))
        filtering_summary = [
            DetailedCountSummaryItem(
                label="Annotated rows", count=len(annotations), detail=f"{len(annotations)} rows currently available in the triage table"
            ),
            DetailedCountSummaryItem(
                label="Distinct genes", count=len(unique_genes), detail=f"{len(unique_genes)} genes represented in the annotated subset"
            ),
            DetailedCountSummaryItem(
                label="ClinVar-labeled rows",
                count=clinvar_labeled,
                detail=f"{clinvar_labeled} rows contain a ClinVar-style significance label",
            ),
            DetailedCountSummaryItem(
                label="Symbolic ALT rows",
                count=symbolic,
                detail=f"{symbolic} rows are symbolic ALT records that may need separate handling",
            ),
        ]

    try:
        symbolic_result = run_tool(
            "symbolic_alt_tool",
            {"annotations": [item.model_dump() for item in annotations]},
        )
        symbolic_alt_summary = SymbolicAltSummary(**symbolic_result["symbolic_alt_summary"])
        used_tools.append("symbolic_alt_tool")
    except Exception:
        symbolic_items = [item for item in annotations if any(alt.startswith("<") and alt.endswith(">") for alt in item.alts)]
        symbolic_alt_summary = SymbolicAltSummary(
            count=len(symbolic_items),
            examples=[
                {
                    "locus": f"{item.contig}:{item.pos_1based}",
                    "gene": item.gene or "",
                    "alts": item.alts,
                    "consequence": item.consequence or "",
                    "genotype": item.genotype or "",
                }
                for item in symbolic_items[:5]
            ],
        )

    reference_annotations = annotations[: min(len(annotations), 20)]
    references = build_reference_bundle(facts, reference_annotations)
    recommendations = build_recommendations(facts)
    ui_cards = build_ui_cards(facts, annotations)
    try:
        summary_result = run_tool(
            "grounded_summary_tool",
            {
                "facts": facts.model_dump(),
                "annotations": [item.model_dump() for item in annotations],
                "references": [item.model_dump() for item in references],
                "recommendations": [item.model_dump() for item in recommendations],
            },
        )
        draft_answer = str(summary_result["draft_answer"])
        used_tools.append("grounded_summary_tool")
    except Exception:
        draft_answer = build_draft_answer(
            facts,
            annotations,
            [item.id for item in references],
            [item.id for item in recommendations],
        )

    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        facts=facts,
        annotations=annotations,
        roh_segments=roh_segments,
        source_vcf_path=path,
        snpeff_result=snpeff_result,
        candidate_variants=candidate_variants,
        clinvar_summary=clinvar_summary,
        consequence_summary=consequence_summary,
        clinical_coverage_summary=clinical_coverage_summary,
        filtering_summary=filtering_summary,
        symbolic_alt_summary=symbolic_alt_summary,
        references=references,
        recommendations=recommendations,
        ui_cards=ui_cards,
        draft_answer=draft_answer,
        used_tools=used_tools,
        tool_registry=tool_registry,
    )


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
    return _assemble_analysis_response_from_vcf_context(context)


def analyze_vcf_workflow(
    path: str,
    annotation_scope: str = "representative",
    annotation_limit: int | None = None,
) -> AnalysisResponse:
    manifest = load_workflow_manifest("representative_vcf_review")
    structured_steps = isinstance(manifest, dict) and isinstance(manifest.get("steps"), list) and all(
        isinstance(step, dict) for step in manifest.get("steps", [])
    )
    if annotation_scope == "representative" and annotation_limit is None and structured_steps:
        return _run_registered_vcf_workflow_from_manifest(
            path,
            manifest,
            annotation_scope=annotation_scope,
            annotation_limit=annotation_limit,
        )
    return _analyze_vcf_workflow_legacy(
        path,
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
    requested_view = str(manifest.get("requested_view") or "summary")
    answer = _format_workflow_answer(manifest, "vcf", refreshed, {})
    return {
        "answer": answer,
        "analysis": refreshed,
        "requested_view": requested_view,
    }


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
