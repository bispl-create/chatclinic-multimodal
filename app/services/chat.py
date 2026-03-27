from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

from app.models import (
    AnalysisChatRequest,
    AnalysisChatResponse,
    QqmanAssociationRequest,
    RawQcChatRequest,
    RawQcChatResponse,
    SummaryStatsChatRequest,
    SummaryStatsChatResponse,
    TextChatRequest,
    TextChatResponse,
)
from app.models import (
    GatkLiftoverVcfRequest,
    LDBlockShowRequest,
    LDBlockShowResponse,
    PlinkRequest,
    SamtoolsRequest,
    SnpEffRequest,
)
from app.services.tool_runner import (
    load_tool_manifests,
    manifest_for_alias,
    tool_aliases,
    tool_chat_metadata,
    tool_direct_chat_metadata,
)
from app.services.workflow_responses import workflow_studio_metadata
from app.services.workflows import (
    run_registered_analysis_workflow,
    list_workflow_manifests,
    load_workflow_manifest,
    run_registered_raw_qc_workflow,
    run_registered_summary_stats_workflow,
    run_registered_text_workflow,
)
from plugins.gatk_liftover_vcf_tool.logic import (
    DEFAULT_CHAIN_FILE,
    DEFAULT_TARGET_FASTA,
    run_gatk_liftover_vcf,
)
from plugins.ldblockshow_execution_tool.logic import run_ldblockshow
from plugins.plink_execution_tool.logic import run_plink
from plugins.qqman_execution_tool.logic import run_qqman_association
from plugins.samtools_execution_tool.logic import run_samtools
from plugins.snpeff_execution_tool.logic import run_snpeff

OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))


def _parse_at_tool_request(question: str) -> dict[str, object] | None:
    stripped = question.strip()
    match = re.match(r"^@([A-Za-z0-9_-]+)(?:\s+(.*))?$", stripped, flags=re.DOTALL)
    if not match:
        return None
    raw_alias = match.group(1).strip().lower()
    if raw_alias == "skill":
        return None
    remainder = (match.group(2) or "").strip()
    lowered = remainder.lower()
    manifest = manifest_for_alias(raw_alias)
    if manifest is not None:
        registry_entry = tool_chat_metadata(manifest)
        aliases = [str(item).strip().lower() for item in registry_entry.get("aliases", []) if str(item).strip()]
        canonical_alias = aliases[0] if aliases else raw_alias
        return {
            "manifest": manifest,
            "alias": canonical_alias or raw_alias,
            "input_alias": raw_alias,
            "registry_entry": registry_entry,
            "remainder": remainder,
            "is_help": lowered in {"help", "--help", "-h"} or lowered.startswith("help "),
        }
    return {
        "manifest": None,
        "alias": raw_alias,
        "input_alias": raw_alias,
        "registry_entry": None,
        "remainder": remainder,
        "is_help": False,
    }


def _parse_skill_request(question: str) -> dict[str, object] | None:
    stripped = question.strip()
    match = re.match(r"^@skill(?:\s+(.*))?$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    remainder = (match.group(1) or "").strip()
    lowered = remainder.lower()
    if not remainder or lowered in {"help", "--help", "-h"}:
        return {
            "name": None,
            "input_name": None,
            "manifest": None,
            "remainder": remainder,
            "is_help": True,
        }
    input_name = remainder.split()[0].strip()
    is_help = remainder.lower().endswith(" help")
    manifest = load_workflow_manifest(input_name)
    canonical_name = str(manifest.get("name") or input_name) if isinstance(manifest, dict) else input_name
    return {
        "name": canonical_name,
        "input_name": input_name,
        "manifest": manifest,
        "remainder": remainder,
        "is_help": is_help,
    }


def _render_tool_help(manifest: dict[str, object]) -> str:
    name = str(manifest.get("name") or "tool")
    registry_entry = tool_chat_metadata(manifest)
    help_block = manifest.get("help")
    if not isinstance(help_block, dict):
        aliases = ", ".join(f"@{item}" for item in tool_aliases(manifest)[:4])
        return (
            f"`{name}` is registered, but no curated help metadata is available yet.\n\n"
            f"- Try one of these aliases: {aliases or '@tool'}"
        )
    input_hint = "active source"
    if registry_entry:
        source_types = [str(item).lower() for item in registry_entry.get("source_types", [])]
        if "vcf" in source_types:
            input_hint = "active VCF source"
        elif "raw_qc" in source_types:
            input_hint = "active BAM/SAM/CRAM source"
        elif "summary_stats" in source_types:
            input_hint = "active summary-statistics source"
    else:
        orchestration = manifest.get("orchestration") if isinstance(manifest.get("orchestration"), dict) else {}
        consumes = orchestration.get("consumes") if isinstance(orchestration, dict) else []
        if isinstance(consumes, list):
            lowered = [str(item).lower() for item in consumes]
            if "vcf_path" in lowered:
                input_hint = "active VCF source"
            elif "alignment_file" in lowered:
                input_hint = "active BAM/SAM/CRAM source"
            elif "summary_stats_path" in lowered:
                input_hint = "active summary-statistics source"

    alias_candidates = (
        [str(item).strip() for item in registry_entry.get("aliases", []) if str(item).strip()]
        if registry_entry
        else tool_aliases(manifest)[:4]
    )
    alias_list = [f"@{item}" for item in alias_candidates[:4]]
    primary_alias = alias_list[0] if alias_list else "@tool"
    lines: list[str] = [f"**{primary_alias}**", ""]
    summary = str(help_block.get("summary") or "").strip()
    if summary:
        lines.append(summary)
    else:
        lines.append(f"`{name}` is available in this build.")
    lines.append("")
    lines.append("Quick use")
    lines.append(f"- Run on: {input_hint}")
    lines.append(f"- Start with: `{primary_alias}`")
    lines.append(f"- Help: `{primary_alias} help`")
    modes = help_block.get("modes") or []
    if isinstance(modes, list) and modes:
        lines.append("")
        lines.append("Available modes")
        for mode in modes:
            if isinstance(mode, dict):
                mode_name = str(mode.get("name") or "").strip()
                mode_description = str(mode.get("description") or "").strip()
                if mode_name:
                    lines.append(f"- `{mode_name}`: {mode_description}")
    options = help_block.get("options") or []
    if isinstance(options, list) and options:
        lines.append("")
        lines.append("Options")
        for option in options:
            if not isinstance(option, dict):
                continue
            option_name = str(option.get("name") or "").strip()
            option_type = str(option.get("type") or "").strip()
            option_description = str(option.get("description") or "").strip()
            default = option.get("default")
            default_suffix = f" Default: `{default}`." if default not in (None, "") else ""
            if option_name:
                label = f"`{option_name}`"
                if option_type:
                    label += f" ({option_type})"
                lines.append(f"- {label}: {option_description}{default_suffix}")
    examples = help_block.get("examples") or []
    if isinstance(examples, list) and examples:
        lines.append("")
        lines.append("Examples")
        for example in examples:
            lines.append(f"- `{example}`")
    notes = help_block.get("notes") or []
    if isinstance(notes, list) and notes:
        lines.append("")
        lines.append("Notes")
        for note in notes:
            lines.append(f"- {note}")
    if alias_list:
        lines.append("")
        lines.append(f"Aliases: {', '.join(alias_list)}")
    return "\n".join(lines).strip()


def _resolve_tool_help_response(tool_request: dict[str, object] | None) -> str | None:
    if not tool_request or not bool(tool_request.get("is_help")):
        return None
    manifest = tool_request.get("manifest")
    alias = str(tool_request.get("alias") or tool_request.get("input_alias") or "tool").strip()
    if isinstance(manifest, dict):
        return _render_tool_help(manifest)
    return f"`@{alias}` is not a registered ChatGenome tool."


def _describe_source_type(source_type: str) -> str:
    normalized = source_type.strip().lower()
    if normalized == "vcf":
        return "VCF session"
    if normalized == "raw_qc":
        return "raw-QC/alignment session"
    if normalized == "summary_stats":
        return "summary-statistics session"
    if normalized == "text":
        return "text-note session"
    return "current session"


def _tool_input_hint(source_type: str) -> str:
    normalized = source_type.strip().lower()
    if normalized == "vcf":
        return "active VCF source"
    if normalized == "raw_qc":
        return "active BAM/SAM/CRAM source"
    if normalized == "summary_stats":
        return "active summary-statistics source"
    if normalized == "text":
        return "active text source"
    return "active source"


def _resolve_tool_source_mismatch_response(
    tool_request: dict[str, object] | None, current_source_type: str
) -> str | None:
    if not tool_request:
        return None
    alias = str(tool_request.get("alias") or tool_request.get("input_alias") or "tool").strip()
    registry_entry = tool_request.get("registry_entry")
    if not isinstance(registry_entry, dict):
        manifest = tool_request.get("manifest")
        if isinstance(manifest, dict):
            registry_entry = tool_chat_metadata(manifest)
    allowed_source_types = [
        str(item).strip().lower()
        for item in (registry_entry.get("source_types", []) if isinstance(registry_entry, dict) else [])
        if str(item).strip()
    ]
    current = current_source_type.strip().lower()
    if not allowed_source_types or current in allowed_source_types:
        return None
    if len(allowed_source_types) == 1:
        target_source_type = allowed_source_types[0]
        return (
            f"`@{alias}` uses the {_tool_input_hint(target_source_type)}. "
            f"Run it from a {_describe_source_type(target_source_type)} rather than a {_describe_source_type(current)}."
        )
    expected = ", ".join(_describe_source_type(item) for item in allowed_source_types)
    return (
        f"`@{alias}` is not available for the current source type. "
        f"It expects one of: {expected}."
    )


def _unknown_tool_answer(tool_request: dict[str, object] | None) -> str:
    alias = "tool"
    if tool_request:
        alias = str(tool_request.get("alias") or tool_request.get("input_alias") or "tool").strip() or "tool"
    return f"`@{alias}` is not a registered ChatGenome tool."


def _source_response_class(source_type: str) -> type[AnalysisChatResponse] | type[RawQcChatResponse] | type[SummaryStatsChatResponse] | type[TextChatResponse]:
    normalized = source_type.strip().lower()
    if normalized == "vcf":
        return AnalysisChatResponse
    if normalized == "raw_qc":
        return RawQcChatResponse
    if normalized == "summary_stats":
        return SummaryStatsChatResponse
    if normalized == "text":
        return TextChatResponse
    raise NotImplementedError(f"Unsupported chat source type: {source_type}")


def _basic_source_response(
    source_type: str,
    answer: str,
    *,
    used_fallback: bool = False,
) -> AnalysisChatResponse | RawQcChatResponse | SummaryStatsChatResponse:
    response_cls = _source_response_class(source_type)
    return response_cls(answer=answer, citations=[], used_fallback=used_fallback)


def _unknown_workflow_response(
    source_type: str,
    skill_request: dict[str, object],
) -> AnalysisChatResponse | RawQcChatResponse | SummaryStatsChatResponse | TextChatResponse:
    name = str(skill_request.get("name") or "workflow")
    return _basic_source_response(
        source_type,
        f"`@skill {name}` is not a registered workflow for the current build.",
    )


def _workflow_template_context(source_type: str, workflow_name: str, workflow_result: dict[str, object]) -> dict[str, object]:
    source = source_type.strip().lower()
    context: dict[str, object] = {
        "workflow_name": workflow_name,
        "requested_view": str(workflow_result.get("requested_view") or ""),
    }
    if source == "vcf":
        analysis = workflow_result.get("analysis")
        context.update(
            {
                "active_file": getattr(getattr(analysis, "facts", None), "file_name", "unknown"),
                "logged_tools": ", ".join(getattr(analysis, "used_tools", []) or []) or "none",
                "candidate_count": len(getattr(analysis, "candidate_variants", []) or []),
            }
        )
    elif source == "raw_qc":
        analysis = workflow_result.get("analysis")
        context.update(
            {
                "active_file": getattr(getattr(analysis, "facts", None), "file_name", "unknown"),
                "logged_tools": ", ".join(getattr(analysis, "used_tools", []) or []) or "none",
                "module_count": len(getattr(analysis, "modules", []) or []),
            }
        )
    elif source == "summary_stats":
        analysis = workflow_result.get("analysis")
        prs_prep_result = workflow_result.get("prs_prep_result")
        context.update(
            {
                "active_file": getattr(analysis, "file_name", "unknown"),
                "logged_tools": ", ".join(getattr(analysis, "used_tools", []) or []) or "none",
                "row_count": getattr(analysis, "row_count", "unknown"),
                "score_file_ready": getattr(prs_prep_result, "score_file_ready", "unknown"),
                "kept_rows": getattr(prs_prep_result, "kept_rows", "unknown"),
                "dropped_rows": getattr(prs_prep_result, "dropped_rows", "unknown"),
            }
        )
    elif source == "text":
        analysis = workflow_result.get("analysis")
        context.update(
            {
                "active_file": getattr(analysis, "file_name", "unknown"),
                "logged_tools": ", ".join(getattr(analysis, "used_tools", []) or []) or "none",
                "line_count": getattr(analysis, "line_count", "unknown"),
                "word_count": getattr(analysis, "word_count", "unknown"),
            }
        )
    return context


def _render_workflow_answer(manifest: dict[str, object], workflow_result: dict[str, object]) -> str:
    template = str(manifest.get("answer_template") or "").strip()
    if not template:
        return str(workflow_result.get("answer") or "").strip()
    workflow_name = str(manifest.get("name") or "workflow")
    source_type = str(manifest.get("source_type") or "").strip().lower()
    context = _workflow_template_context(source_type, workflow_name, workflow_result)
    try:
        return template.format(**context)
    except Exception:
        return str(workflow_result.get("answer") or "").strip()


def _run_registered_workflow_for_source(source_type: str, workflow_name: str, analysis: object) -> dict[str, object]:
    source = source_type.strip().lower()
    if source == "vcf":
        return run_registered_analysis_workflow(workflow_name, analysis)
    if source == "raw_qc":
        return run_registered_raw_qc_workflow(workflow_name, analysis)
    if source == "summary_stats":
        return run_registered_summary_stats_workflow(workflow_name, analysis)
    if source == "text":
        return run_registered_text_workflow(workflow_name, analysis)
    raise NotImplementedError(f"Unsupported workflow source type: {source_type}")


def _assemble_workflow_chat_response(
    manifest: dict[str, object], workflow_result: dict[str, object]
) -> AnalysisChatResponse | RawQcChatResponse | SummaryStatsChatResponse | TextChatResponse:
    response_kind = str(manifest.get("response_kind") or "").strip().lower()
    requested_view = str(workflow_result.get("requested_view") or manifest.get("requested_view") or "").strip() or None
    studio = workflow_result.get("studio")
    if not isinstance(studio, dict):
        studio = workflow_studio_metadata(manifest)
    answer = _render_workflow_answer(manifest, workflow_result)
    if response_kind == "analysis_chat":
        return AnalysisChatResponse(
            answer=answer,
            citations=[],
            used_fallback=False,
            studio=studio,
            requested_view=requested_view,
            analysis=workflow_result.get("analysis"),
        )
    if response_kind == "raw_qc_chat":
        return RawQcChatResponse(
            answer=answer,
            citations=[],
            used_fallback=False,
            studio=studio,
            requested_view=requested_view,
            analysis=workflow_result.get("analysis"),
        )
    if response_kind == "summary_stats_chat":
        return SummaryStatsChatResponse(
            answer=answer,
            citations=[],
            used_fallback=False,
            studio=studio,
            requested_view=requested_view,
            analysis=workflow_result.get("analysis"),
            prs_prep_result=workflow_result.get("prs_prep_result"),
        )
    if response_kind == "text_chat":
        return TextChatResponse(
            answer=answer,
            citations=[],
            used_fallback=False,
            studio=studio,
            requested_view=requested_view,
            analysis=workflow_result.get("analysis"),
        )
    raise NotImplementedError(f"Unsupported workflow response kind: {response_kind or 'unknown'}")


def _with_result_field(result_kind: str | None, result: object, **kwargs: Any) -> dict[str, Any]:
    payload = dict(kwargs)
    if result_kind:
        payload["result_kind"] = result_kind
    if result_kind and result is not None:
        payload[result_kind] = result
    return payload


def _analysis_tool_response(
    answer: str,
    *,
    result_kind: str | None = None,
    result: object = None,
    citations: list[str] | None = None,
    used_fallback: bool = False,
    used_tools: list[str] | None = None,
    requested_view: str | None = None,
    studio: dict[str, Any] | None = None,
    analysis: object = None,
) -> AnalysisChatResponse:
    return AnalysisChatResponse(
        **_with_result_field(
            result_kind,
            result,
            answer=answer,
            citations=citations or [],
            used_fallback=used_fallback,
            used_tools=used_tools or [],
            requested_view=requested_view,
            studio=studio,
            analysis=analysis,
        )
    )


def _raw_qc_tool_response(
    answer: str,
    *,
    result_kind: str | None = None,
    result: object = None,
    citations: list[str] | None = None,
    used_fallback: bool = False,
    requested_view: str | None = None,
    studio: dict[str, Any] | None = None,
    analysis: object = None,
) -> RawQcChatResponse:
    return RawQcChatResponse(
        **_with_result_field(
            result_kind,
            result,
            answer=answer,
            citations=citations or [],
            used_fallback=used_fallback,
            requested_view=requested_view,
            studio=studio,
            analysis=analysis,
        )
    )


def _summary_stats_tool_response(
    answer: str,
    *,
    result_kind: str | None = None,
    result: object = None,
    citations: list[str] | None = None,
    used_fallback: bool = False,
    requested_view: str | None = None,
    studio: dict[str, Any] | None = None,
    analysis: object = None,
) -> SummaryStatsChatResponse:
    return SummaryStatsChatResponse(
        **_with_result_field(
            result_kind,
            result,
            answer=answer,
            citations=citations or [],
            used_fallback=used_fallback,
            requested_view=requested_view,
            studio=studio,
            analysis=analysis,
        )
    )


def _text_tool_response(
    answer: str,
    *,
    result_kind: str | None = None,
    result: object = None,
    citations: list[str] | None = None,
    used_fallback: bool = False,
    requested_view: str | None = None,
    studio: dict[str, Any] | None = None,
    analysis: object = None,
) -> TextChatResponse:
    return TextChatResponse(
        **_with_result_field(
            result_kind,
            result,
            answer=answer,
            citations=citations or [],
            used_fallback=used_fallback,
            requested_view=requested_view,
            studio=studio,
            analysis=analysis,
        )
    )


def _render_skill_help(source_type: str | None = None, selected: dict[str, object] | None = None) -> str:
    manifests = list_workflow_manifests(source_type=source_type)
    if selected is not None:
        manifests = [selected]
    if not manifests:
        return "No workflow registry entries are available for the current source."
    tool_lookup = {
        str(item.get("name") or "").strip(): item
        for item in load_tool_manifests()
        if isinstance(item, dict)
    }
    lines: list[str] = ["**Workflow registry**", ""]
    for item in manifests:
        name = str(item.get("name") or "workflow")
        description = str(item.get("description") or "").strip()
        lines.append(f"- `@skill {name}`: {description}")
        if selected is not None:
            steps = item.get("steps") or []
            if isinstance(steps, list) and steps:
                lines.append("")
                lines.append("Steps")
                for step in steps:
                    if isinstance(step, dict):
                        step_name = str(step.get("tool") or "").strip()
                        bind_name = str(step.get("bind") or "").strip()
                        needs = [
                            str(value).strip()
                            for value in step.get("needs", [])
                            if str(value).strip()
                        ]
                        on_fail = str(step.get("on_fail") or "").strip().lower()
                    else:
                        step_name = str(step).strip()
                        bind_name = ""
                        needs = []
                        on_fail = ""
                    manifest = tool_lookup.get(step_name)
                    if manifest is not None:
                        step_description = str(manifest.get("description") or "").strip()
                        detail_parts: list[str] = []
                        if bind_name:
                            detail_parts.append(f"binds `{bind_name}`")
                        if needs:
                            detail_parts.append(f"needs `{', '.join(needs)}`")
                        if on_fail == "continue":
                            detail_parts.append("continues on failure")
                        detail_text = f" ({'; '.join(detail_parts)})" if detail_parts else ""
                        lines.append(f"- `{step_name}`: {step_description}{detail_text}")
                    else:
                        detail_parts = []
                        if bind_name:
                            detail_parts.append(f"binds `{bind_name}`")
                        if needs:
                            detail_parts.append(f"needs `{', '.join(needs)}`")
                        if on_fail == "continue":
                            detail_parts.append("continues on failure")
                        detail_text = f" ({'; '.join(detail_parts)})" if detail_parts else ""
                        lines.append(f"- `{step_name}`{detail_text}")
    lines.append("")
    lines.append("Examples")
    if source_type == "vcf":
        lines.append("- `@skill representative_vcf_review`")
    elif source_type == "raw_qc":
        lines.append("- `@skill raw_qc_review`")
    elif source_type == "summary_stats":
        lines.append("- `@skill summary_stats_review`")
        lines.append("- `@skill prs_prep`")
    elif source_type == "text":
        lines.append("- `@skill text_review`")
    else:
        lines.append("- `@skill help`")
        lines.append("- `@skill representative_vcf_review`")
    return "\n".join(lines)


def _resolve_skill_help_response(
    skill_request: dict[str, object] | None, source_type: str
) -> str | None:
    if not skill_request or not bool(skill_request.get("is_help")):
        return None
    manifest = skill_request.get("manifest")
    if isinstance(manifest, dict):
        return _render_skill_help(source_type, selected=manifest)
    return _render_skill_help(source_type)


def _is_korean(text: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7a3]", text))


def _flatten_studio_context(studio_context: dict) -> dict[str, object]:
    return {
        "active_view": studio_context.get("active_view"),
        "qc_summary": studio_context.get("qc_summary"),
        "clinical_coverage": studio_context.get("clinical_coverage"),
        "symbolic_alt_review": studio_context.get("symbolic_alt_review"),
        "roh_review": studio_context.get("roh_review"),
        "candidate_variants": studio_context.get("candidate_variants"),
        "clinvar_review": studio_context.get("clinvar_review"),
        "vep_consequence": studio_context.get("vep_consequence"),
        "snpeff_preview": studio_context.get("snpeff_preview"),
        "selected_annotation": studio_context.get("selected_annotation"),
    }


def _has_studio_trigger(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in ("$studio", "$current analysis", "$current card", "$grounded"))


def _strip_studio_triggers(question: str) -> str:
    cleaned = question
    for token in ("$studio", "$current analysis", "$current card", "$grounded"):
        cleaned = re.sub(re.escape(token), " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _needs_grounded_clarification(question: str) -> bool:
    if not _has_studio_trigger(question):
        return False
    stripped = _strip_studio_triggers(question)
    if not stripped:
        return True
    tokens = stripped.split()
    return len(tokens) < 2 and len(stripped) < 16


def _grounded_clarification_text() -> str:
    return (
        "Grounded mode is on.\n\n"
        "- Tell me which Studio result you want me to use.\n"
        "- Example: `$studio candidate card 설명해줘`\n"
        "- Example: `$studio ROH 결과를 요약해줘`\n"
        "- Example: `$studio summary statistics review를 정리해줘`"
    )


def _compact_analysis_context(payload: AnalysisChatRequest) -> dict[str, object]:
    analysis = payload.analysis
    context = {
        "analysis_id": analysis.analysis_id,
        "draft_answer": analysis.draft_answer,
        "used_tools": analysis.used_tools,
        "tool_registry": [
            {
                "name": item.name,
                "task": item.task,
                "source": item.source,
            }
            for item in analysis.tool_registry
        ],
        "facts": {
            "file_name": analysis.facts.file_name,
            "genome_build_guess": analysis.facts.genome_build_guess,
            "record_count": analysis.facts.record_count,
            "samples": analysis.facts.samples,
            "variant_types": analysis.facts.variant_types,
            "warnings": analysis.facts.warnings,
        },
        "annotations": [
            {
                "pos": item.pos_1based,
                "gene": item.gene,
                "consequence": item.consequence,
                "rsid": item.rsid,
                "clinical_significance": item.clinical_significance,
                "condition": item.clinvar_conditions,
                "gnomad_af": item.gnomad_af,
            }
            for item in payload.analysis.annotations[:6]
        ],
        "snpeff_result": (
            {
                "tool": payload.analysis.snpeff_result.tool,
                "genome": payload.analysis.snpeff_result.genome,
                "output_path": payload.analysis.snpeff_result.output_path,
                "parsed_records": [
                    {
                        "contig": item.contig,
                        "pos_1based": item.pos_1based,
                        "ref": item.ref,
                        "alt": item.alt,
                        "ann": [
                            {
                                "annotation": ann.annotation,
                                "impact": ann.impact,
                                "gene_name": ann.gene_name,
                                "hgvs_c": ann.hgvs_c,
                                "hgvs_p": ann.hgvs_p,
                            }
                            for ann in item.ann[:3]
                        ],
                    }
                    for item in payload.analysis.snpeff_result.parsed_records[:5]
                ],
            }
            if payload.analysis.snpeff_result
            else None
        ),
        "ldblockshow_result": (
            {
                "tool": payload.analysis.ldblockshow_result.tool,
                "region": payload.analysis.ldblockshow_result.region,
                "svg_path": payload.analysis.ldblockshow_result.svg_path,
                "png_path": payload.analysis.ldblockshow_result.png_path,
                "pdf_path": payload.analysis.ldblockshow_result.pdf_path,
                "warnings": payload.analysis.ldblockshow_result.warnings,
            }
            if payload.analysis.ldblockshow_result
            else None
        ),
        "roh_segments": [
            {
                "sample": item.sample,
                "contig": item.contig,
                "start_1based": item.start_1based,
                "end_1based": item.end_1based,
                "length_bp": item.length_bp,
                "marker_count": item.marker_count,
                "quality": item.quality,
            }
            for item in payload.analysis.roh_segments[:6]
        ],
        "references": [
            {"id": item.id, "title": item.title, "url": item.url}
            for item in payload.analysis.references[:8]
        ],
        "recommendations": [
            {"id": item.id, "title": item.title, "action": item.action}
            for item in payload.analysis.recommendations[:6]
        ],
    }
    if payload.studio_context:
        context["studio_context"] = _flatten_studio_context(payload.studio_context)
    return context

def _compact_raw_qc_context(payload: RawQcChatRequest) -> dict[str, object]:
    context = {
        "analysis_id": payload.analysis.analysis_id,
        "draft_answer": payload.analysis.draft_answer,
        "facts": payload.analysis.facts.model_dump(),
        "modules": [item.model_dump() for item in payload.analysis.modules[:12]],
    }
    if payload.studio_context:
        context["studio_context"] = _flatten_studio_context(payload.studio_context)
    return context


def _compact_summary_stats_context(payload: SummaryStatsChatRequest) -> dict[str, object]:
    context = {
        "analysis_id": payload.analysis.analysis_id,
        "file_name": payload.analysis.file_name,
        "genome_build": payload.analysis.genome_build,
        "trait_type": payload.analysis.trait_type,
        "delimiter": payload.analysis.delimiter,
        "detected_columns": payload.analysis.detected_columns,
        "mapped_fields": payload.analysis.mapped_fields.model_dump(),
        "row_count": payload.analysis.row_count,
        "warnings": payload.analysis.warnings[:12],
        "preview_rows": payload.analysis.preview_rows[:8],
        "draft_answer": payload.analysis.draft_answer,
        "used_tools": payload.analysis.used_tools,
    }
    if payload.studio_context:
        context["studio_context"] = _flatten_studio_context(payload.studio_context)
    return context


def _compact_text_context(payload: TextChatRequest) -> dict[str, object]:
    context = {
        "analysis_id": payload.analysis.analysis_id,
        "file_name": payload.analysis.file_name,
        "media_type": payload.analysis.media_type,
        "char_count": payload.analysis.char_count,
        "word_count": payload.analysis.word_count,
        "line_count": payload.analysis.line_count,
        "warnings": payload.analysis.warnings[:12],
        "preview_lines": payload.analysis.preview_lines[:12],
        "draft_answer": payload.analysis.draft_answer,
        "used_tools": payload.analysis.used_tools,
    }
    if payload.studio_context:
        context["studio_context"] = _flatten_studio_context(payload.studio_context)
    return context


CHAT_OPENAI_CONFIG: dict[str, dict[str, Any]] = {
    "vcf": {
        "context_label": "Analysis context JSON",
        "compact_context_builder": _compact_analysis_context,
        "grounded_system_prompt": (
            "You are a genomics analysis copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer from the provided VCF analysis context and do not invent variant facts. "
            "Treat analysis.used_tools as the authoritative record of which deterministic tools were actually run in the current analysis. "
            "Do not claim that a tool was used unless it appears in analysis.used_tools or the user explicitly requests a new run. "
            "Do not infer tool execution only from card names such as VEP consequence summaries. "
            "Treat studio_context as part of the trusted analysis state, including ROH, coverage, candidate, ClinVar, and consequence summaries when present. "
            "When possible, cite reference ids like REF1 or REF4 inline. "
            "Format the answer in clean Markdown with short sections or bullet points when helpful. "
            "Label the answer as analysis-grounded."
        ),
        "general_system_prompt": (
            "You are a helpful general assistant. "
            "The user did not request analysis grounding. "
            "Answer from general knowledge only and ignore any uploaded analysis/studio context. "
            "Do not mention analysis context, Studio cards, or uploaded-file facts unless the user explicitly asks with a grounding trigger such as $studio."
        ),
    },
    "raw_qc": {
        "context_label": "FastQC context JSON",
        "compact_context_builder": _compact_raw_qc_context,
        "grounded_system_prompt": (
            "You are a sequencing QC copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer from the provided FastQC context and do not invent QC metrics. "
            "Be concise, grounded, and practical. "
            "If there are WARN or FAIL modules, explain why they matter for downstream genomics workflows."
        ),
        "general_system_prompt": (
            "You are a helpful general assistant. "
            "The user did not request analysis grounding. "
            "Answer from general knowledge only and ignore any uploaded FastQC/raw-QC context unless the user explicitly asks with a grounding trigger such as $studio."
        ),
    },
    "summary_stats": {
        "context_label": "Summary statistics context JSON",
        "compact_context_builder": _compact_summary_stats_context,
        "grounded_system_prompt": (
            "You are a post-GWAS and summary-statistics copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer from the provided summary statistics context. "
            "When the answer is grounded in the uploaded file, distinguish that clearly from general knowledge. "
            "Do not claim that a downstream tool has already been run unless it appears in used_tools."
        ),
        "general_system_prompt": (
            "You are a helpful general assistant. "
            "The user did not request analysis grounding. "
            "Answer from general knowledge only and ignore any uploaded summary-statistics context unless the user explicitly asks with a grounding trigger such as $studio."
        ),
    },
    "text": {
        "context_label": "Text-note context JSON",
        "compact_context_builder": _compact_text_context,
        "grounded_system_prompt": (
            "You are a document and note-reading copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer only from the provided text-note context and do not invent unseen document details. "
            "Be concise and explicit when the preview is partial."
        ),
        "general_system_prompt": (
            "You are a helpful general assistant. "
            "The user did not request grounded note reasoning. "
            "Answer from general knowledge only and ignore any uploaded text-note context unless the user explicitly asks with a grounding trigger such as $studio."
        ),
    },
}


def _fallback_chat_answer(
    source_type: str,
    question: str,
) -> AnalysisChatResponse | RawQcChatResponse | SummaryStatsChatResponse | TextChatResponse:
    if _has_studio_trigger(question):
        answer = (
            "I could not complete the grounded Studio response right now.\n\n"
            "- The request was recognized as `$studio` grounded chat.\n"
            "- Please retry the question, or ask again after the backend model connection is restored."
        )
    else:
        answer = (
            "I could not complete the general GPT response right now.\n\n"
            "- Please retry the question.\n"
            "- If you want the answer grounded in the current Studio state, add `$studio` to the message."
        )
    return _basic_source_response(source_type, answer, used_fallback=True)


def _extract_openai_output_text(result: dict[str, Any]) -> str:
    output_text = result.get("output_text")
    if output_text:
        return str(output_text).strip()
    output = result.get("output", [])
    texts: list[str] = []
    for item in output:
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "\n".join(texts).strip()


def _call_openai_for_source(
    source_type: str,
    payload: AnalysisChatRequest | RawQcChatRequest | SummaryStatsChatRequest | TextChatRequest,
) -> AnalysisChatResponse | RawQcChatResponse | SummaryStatsChatResponse | TextChatResponse:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    if not api_key:
        return _fallback_chat_answer(source_type, payload.question)

    config = CHAT_OPENAI_CONFIG[source_type]
    grounded = _has_studio_trigger(payload.question)
    if grounded:
        compact_context = config["compact_context_builder"](payload)
        system_prompt = str(config["grounded_system_prompt"])
        user_content = (
            "Question:\n"
            f"{payload.question}\n\n"
            f"{config['context_label']}:\n"
            f"{json.dumps(compact_context, ensure_ascii=False)}"
        )
    else:
        system_prompt = str(config["general_system_prompt"])
        user_content = payload.question

    history_lines = [{"role": turn.role, "content": turn.content} for turn in payload.history[-6:]]
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            *history_lines,
            {"role": "user", "content": user_content},
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT_SECONDS) as response:
        result = json.loads(response.read().decode("utf-8"))

    output_text = _extract_openai_output_text(result)
    citations = sorted(set(re.findall(r"\bREF\d+\b", output_text or "")))
    response_cls = _source_response_class(source_type)
    fallback_answer = _fallback_chat_answer(source_type, payload.question).answer
    return response_cls(
        answer=output_text or fallback_answer,
        citations=citations,
        used_fallback=False,
    )


def _extract_ldblockshow_region(question: str) -> str | None:
    match = re.search(r"((?:chr)?[A-Za-z0-9_]+):(\d+)(?:[:-])(\d+)", question, flags=re.IGNORECASE)
    if not match:
        return None
    chrom, start, end = match.group(1), match.group(2), match.group(3)
    return f"{chrom}:{start}:{end}"


def _snpeff_genome_from_build(build_guess: str | None) -> str:
    guess = (build_guess or "").lower()
    if any(token in guess for token in ("38", "hg38", "grch38")):
        return "GRCh38.99"
    return "GRCh37.75"


def _extract_liftover_target_build(question: str, build_guess: str | None) -> tuple[str, str]:
    lowered = question.lower()
    if any(token in lowered for token in ("hg38", "grch38", "38")):
        return "GRCh38", "hg38"
    if any(token in lowered for token in ("hg19", "grch37", "37")):
        return "GRCh37", "hg19"
    current_guess = (build_guess or "").lower()
    if any(token in current_guess for token in ("37", "hg19", "grch37")):
        return "GRCh38", "hg38"
    return "GRCh38", "hg38"


def _extract_key_value_options(text: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_-]*)=([A-Za-z0-9._:-]+)", text):
        options[key.lower()] = value
    return options


def _parse_direct_tool_options(remainder: str, argument_mode: str) -> dict[str, str]:
    mode = argument_mode.strip().lower()
    if mode in {"", "none"}:
        return {}
    if mode == "key_value":
        return _extract_key_value_options(remainder)
    if mode == "region_or_key_value":
        options = _extract_key_value_options(remainder)
        if "region" not in options:
            region = _extract_ldblockshow_region(remainder)
            if region:
                options["region"] = region
        return options
    if mode == "mode_or_key_value":
        options = _extract_key_value_options(remainder)
        stripped = remainder.strip().lower()
        if "mode" not in options and stripped:
            first_token = stripped.split()[0]
            if re.fullmatch(r"[a-z0-9_-]+", first_token):
                options["mode"] = first_token
        return options
    return {}


def _tool_request_direct_chat_metadata(tool_request: dict[str, object]) -> dict[str, Any]:
    manifest = tool_request.get("manifest")
    if not isinstance(manifest, dict):
        return {}
    return tool_direct_chat_metadata(manifest)


def _execute_analysis_direct_liftover(
    payload: AnalysisChatRequest,
    tool_request: dict[str, object],
    direct_chat: dict[str, Any],
    options: dict[str, str],
) -> AnalysisChatResponse:
    source_vcf_path = payload.analysis.source_vcf_path
    result_kind = str(direct_chat.get("result_kind") or "liftover_result")
    if not source_vcf_path:
        return _analysis_tool_response(
            "The current analysis context does not include a source VCF path, so liftover cannot be run from this chat turn.",
            used_fallback=True,
            used_tools=["gatk_liftover_vcf_tool"],
        )

    target_build, target_label = _extract_liftover_target_build(
        options.get("target") or str(tool_request.get("remainder") or payload.question),
        payload.analysis.facts.genome_build_guess,
    )
    source_build = options.get("source_build") or payload.analysis.facts.genome_build_guess
    output_prefix = options.get("output_prefix") or f"{payload.analysis.analysis_id}-liftover-{target_label}"
    existing = payload.analysis.liftover_result
    if existing is not None and (existing.target_build or "").lower() == target_build.lower():
        return _analysis_tool_response(
            (
                f"Liftover results are already available for the current VCF.\n\n"
                f"- Source build: `{existing.source_build or 'unknown'}`\n"
                f"- Target build: `{existing.target_build or target_build}`\n"
                f"- Lifted VCF: `{existing.output_path}`\n"
                f"- Reject VCF: `{existing.reject_path}`\n\n"
                "The existing LiftOver Review card has been reused instead of rerunning the tool."
            ),
            result_kind=result_kind,
            result=existing,
            used_tools=["gatk_liftover_vcf_tool"],
        )

    result = run_gatk_liftover_vcf(
        GatkLiftoverVcfRequest(
            vcf_path=source_vcf_path,
            target_reference_fasta=str(DEFAULT_TARGET_FASTA),
            chain_file=str(DEFAULT_CHAIN_FILE),
            source_build=source_build or None,
            target_build=target_build,
            output_prefix=output_prefix,
            parse_limit=8,
        )
    )
    return _analysis_tool_response(
        (
            f"GATK LiftoverVcf was run for the current VCF.\n\n"
            f"- Source build: `{result.source_build or 'unknown'}`\n"
            f"- Target build: `{result.target_build or target_build}`\n"
            f"- Lifted records: {result.lifted_record_count if result.lifted_record_count is not None else 'unknown'}\n"
            f"- Rejected records: {result.rejected_record_count if result.rejected_record_count is not None else 'unknown'}\n"
            f"- Lifted VCF: `{result.output_path}`\n"
            f"- Reject VCF: `{result.reject_path}`\n\n"
            "The Studio card has been updated with the latest liftover result."
        ),
        result_kind=result_kind,
        result=result,
        used_tools=["gatk_liftover_vcf_tool"],
        requested_view=str(direct_chat.get("requested_view") or None) or None,
    )


def _execute_raw_qc_direct_samtools(
    payload: RawQcChatRequest,
    tool_request: dict[str, object],
    direct_chat: dict[str, Any],
    options: dict[str, str],
) -> RawQcChatResponse:
    del tool_request, options
    alignment_kind = (payload.analysis.facts.file_kind or "").upper()
    if alignment_kind not in {"BAM", "SAM", "CRAM", "ALIGNMENT"}:
        return _raw_qc_tool_response(
            (
                "samtools is intended for alignment files such as BAM, SAM, or CRAM. "
                f"The current active source is `{payload.analysis.facts.file_name}` ({payload.analysis.facts.file_kind})."
            ),
        )
    raw_path = payload.analysis.source_raw_path
    if not raw_path:
        return _raw_qc_tool_response(
            "The active raw-QC session does not include a durable alignment-file path, so `@samtools` cannot run yet."
        )
    result = run_samtools(
        SamtoolsRequest(
            raw_path=raw_path,
            original_name=payload.analysis.facts.file_name,
        )
    )
    idx_lines = [
        f"- {item.contig}: mapped {item.mapped}, unmapped {item.unmapped}, length {item.length_bp}"
        for item in result.idxstats_rows[:5]
    ]
    answer = (
        f"samtools reviewed the active source `{result.display_name}` ({result.file_kind}).\n\n"
        f"- Quickcheck: {'PASS' if result.quickcheck_ok else 'issue detected'}\n"
        f"- Total reads: {result.total_reads if result.total_reads is not None else 'unknown'}\n"
        f"- Mapped reads: {result.mapped_reads if result.mapped_reads is not None else 'unknown'}"
        f"{f' ({result.mapped_rate:.2f}%)' if result.mapped_rate is not None else ''}\n"
        f"- Properly paired reads: {result.properly_paired_reads if result.properly_paired_reads is not None else 'unknown'}"
        f"{f' ({result.properly_paired_rate:.2f}%)' if result.properly_paired_rate is not None else ''}\n"
        f"- Index path: {result.index_path or 'none'}\n\n"
        "Top idxstats rows:\n"
        f"{chr(10).join(idx_lines) if idx_lines else '- idxstats rows are not available for this input.'}"
    )
    if result.warnings:
        answer += "\n\nWarnings:\n" + "\n".join(f"- {warning}" for warning in result.warnings[:5])
    return _raw_qc_tool_response(
        answer,
        result_kind=str(direct_chat.get("result_kind") or "samtools_result"),
        result=result,
        requested_view=str(direct_chat.get("requested_view") or None) or None,
    )


def _execute_summary_stats_direct_qqman(
    payload: SummaryStatsChatRequest,
    tool_request: dict[str, object],
    direct_chat: dict[str, Any],
    options: dict[str, str],
) -> SummaryStatsChatResponse:
    del tool_request
    source_stats_path = payload.analysis.source_stats_path
    if not source_stats_path:
        return _summary_stats_tool_response(
            "The active summary-statistics session does not expose a durable source file path, so `@qqman` cannot run yet."
        )
    result = run_qqman_association(
        QqmanAssociationRequest(
            association_path=source_stats_path,
            output_prefix=options.get("output_prefix") or f"{payload.analysis.analysis_id}-qqman",
        )
    )
    return _summary_stats_tool_response(
        (
            "qqman plots were generated for the active summary-statistics source.\n\n"
            f"- Output directory: `{result.output_dir}`\n"
            f"- Plot artifacts: {len(result.artifacts)}\n"
            f"- Warnings: {len(result.warnings)}\n\n"
            "The Studio card has been updated with the latest qqman result."
        ),
        result_kind=str(direct_chat.get("result_kind") or "qqman_result"),
        result=result,
        requested_view=str(direct_chat.get("requested_view") or None) or None,
    )


def _execute_analysis_direct_snpeff(
    payload: AnalysisChatRequest,
    tool_request: dict[str, object],
    direct_chat: dict[str, Any],
    options: dict[str, str],
) -> AnalysisChatResponse:
    del tool_request
    source_vcf_path = payload.analysis.source_vcf_path
    if not source_vcf_path:
        return _analysis_tool_response(
            "The current analysis context does not include a source VCF path, so SnpEff cannot be run from this chat turn.",
            used_fallback=True,
            used_tools=["snpeff_execution_tool"],
        )

    existing = payload.analysis.snpeff_result
    if existing is not None:
        return _analysis_tool_response(
            (
                f"SnpEff results are already available for the current VCF using genome `{existing.genome}`.\n\n"
                f"- Output path: `{existing.output_path}`\n"
                f"- Preview records: {len(existing.parsed_records)}\n\n"
                "The existing SnpEff Review card has been reused instead of rerunning the tool."
            ),
            result_kind=str(direct_chat.get("result_kind") or "snpeff_result"),
            result=existing,
            used_tools=["snpeff_execution_tool"],
            requested_view=str(direct_chat.get("requested_view") or None) or None,
        )

    result = run_snpeff(
        SnpEffRequest(
            vcf_path=source_vcf_path,
            genome=options.get("genome") or _snpeff_genome_from_build(payload.analysis.facts.genome_build_guess),
            output_prefix=options.get("output_prefix") or f"{payload.analysis.analysis_id}-snpeff",
            parse_limit=10,
        )
    )
    return _analysis_tool_response(
        (
            f"SnpEff was run on the current VCF using genome `{result.genome}`.\n\n"
            f"- Output path: `{result.output_path}`\n"
            f"- Preview records: {len(result.parsed_records)}\n\n"
            "The SnpEff Review card has been updated with the latest result."
        ),
        result_kind=str(direct_chat.get("result_kind") or "snpeff_result"),
        result=result,
        used_tools=["snpeff_execution_tool"],
        requested_view=str(direct_chat.get("requested_view") or None) or None,
    )


def _execute_analysis_direct_ldblockshow(
    payload: AnalysisChatRequest,
    tool_request: dict[str, object],
    direct_chat: dict[str, Any],
    options: dict[str, str],
) -> AnalysisChatResponse:
    source_vcf_path = payload.analysis.source_vcf_path
    region = options.get("region") or _extract_ldblockshow_region(str(tool_request.get("remainder") or payload.question))

    if not source_vcf_path:
        return AnalysisChatResponse(
            answer="The current analysis context does not include a source VCF path, so LDBlockShow cannot be run from this chat turn.",
            citations=[],
            used_fallback=True,
            used_tools=["ldblockshow_execution_tool"],
            ldblockshow_result=LDBlockShowResponse(
                tool="ldblockshow",
                input_path="",
                region=region or "unknown",
                output_prefix="",
                command_preview="LDBlockShow -InVCF <source.vcf.gz> -Region chr:start:end ...",
                warnings=["The current analysis context does not include a source VCF path."],
            ),
        )

    if not region:
        return AnalysisChatResponse(
            answer="LDBlockShow needs a concrete region in `chr:start:end` format. Example: `Run LDBlockShow on chr11:24100000:24200000`.",
            citations=[],
            used_fallback=True,
            used_tools=["ldblockshow_execution_tool"],
            ldblockshow_result=LDBlockShowResponse(
                tool="ldblockshow",
                input_path=source_vcf_path,
                region="unknown",
                output_prefix="",
                command_preview="LDBlockShow -InVCF <source.vcf.gz> -Region chr:start:end ...",
                warnings=["No valid region was found in the request."],
            ),
        )

    result = run_ldblockshow(
        LDBlockShowRequest(
            vcf_path=source_vcf_path,
            region=region,
            sele_var=2,
            block_type=5,
            out_png=False,
            out_pdf=False,
        )
    )
    answer = (
        f"LDBlockShow was run on the current VCF for region `{result.region}`.\n\n"
        f"- Primary artifact: `{result.svg_path or result.png_path or result.pdf_path or 'not available'}`\n"
        f"- Block table: `{result.block_path or 'not available'}`\n"
        f"- Warnings: {len(result.warnings)}\n\n"
        "The Studio card has been updated with the latest LD block result."
    )
    return _analysis_tool_response(
        answer,
        result_kind=str(direct_chat.get("result_kind") or "ldblockshow_result"),
        result=result,
        used_tools=["ldblockshow_execution_tool"],
        requested_view=str(direct_chat.get("requested_view") or None) or None,
    )


def _execute_analysis_direct_plink(
    payload: AnalysisChatRequest,
    tool_request: dict[str, object],
    direct_chat: dict[str, Any],
    options: dict[str, str],
) -> AnalysisChatResponse:
    del tool_request
    source_vcf_path = payload.analysis.source_vcf_path
    if not source_vcf_path:
        return AnalysisChatResponse(
            answer="The current analysis context does not include a source VCF path, so PLINK cannot be run from this chat turn.",
            citations=[],
            used_fallback=True,
            used_tools=["plink_execution_tool"],
            requested_view=str(direct_chat.get("requested_view") or "plink"),
        )

    requested_mode = str(
        options.get("mode")
        or direct_chat.get("default_mode")
        or "qc"
    ).strip().lower()
    if requested_mode not in {"qc", "score"}:
        requested_mode = "qc"

    existing = payload.analysis.plink_result
    if requested_mode == "score":
        answer = (
            "The PLINK score card is ready in Studio.\n\n"
            "- This mode uses the latest PRS prep score file.\n"
            "- Run `@skill prs_prep` first on a summary-statistics source if the score file is not ready.\n"
            "- You can review the score configuration and run PLINK from the Studio card."
        )
        if existing is not None and existing.mode == "score":
            answer += (
                f"\n\nAn existing PLINK score result is already present for this analysis:\n"
                f"- Output prefix: `{existing.output_prefix}`\n"
                f"- Scored samples: {existing.scored_sample_count if existing.scored_sample_count is not None else 'unknown'}\n"
                f"- Score rows preview: {len(existing.score_rows)}"
            )
    else:
        answer = (
            "The PLINK card is ready in Studio.\n\n"
            "- This ChatGenome build currently exposes PLINK as a deterministic QC workflow.\n"
            "- You can choose the QC options, review the command preview, and run PLINK from the card.\n"
            "- The result will appear in the same card after execution."
        )
        if existing is not None:
            answer += (
                f"\n\nAn existing PLINK result is already present for this analysis:\n"
                f"- Output prefix: `{existing.output_prefix}`\n"
                f"- Frequency rows: {len(existing.freq_rows)}\n"
                f"- Missingness rows: {len(existing.missing_rows)}\n"
                f"- Hardy rows: {len(existing.hardy_rows)}"
            )
    return _analysis_tool_response(
        answer,
        result_kind=str(direct_chat.get("result_kind") or "plink_result"),
        result=existing,
        used_tools=payload.analysis.used_tools or [],
        requested_view=str(direct_chat.get("requested_view") or None) or None,
    )


DIRECT_TOOL_ENDPOINT_EXECUTORS: dict[str, dict[str, Any]] = {
    "vcf": {
        "liftover": _execute_analysis_direct_liftover,
        "snpeff": _execute_analysis_direct_snpeff,
        "ldblockshow": _execute_analysis_direct_ldblockshow,
        "plink": _execute_analysis_direct_plink,
    },
    "raw_qc": {
        "samtools": _execute_raw_qc_direct_samtools,
    },
    "summary_stats": {
        "qqman": _execute_summary_stats_direct_qqman,
    },
    "text": {},
}


def _run_direct_tool_for_source(
    source_type: str,
    payload: AnalysisChatRequest | RawQcChatRequest | SummaryStatsChatRequest | TextChatRequest,
    tool_request: dict[str, object],
) -> AnalysisChatResponse | RawQcChatResponse | SummaryStatsChatResponse | TextChatResponse | None:
    direct_chat = _tool_request_direct_chat_metadata(tool_request)
    endpoint = str(direct_chat.get("endpoint") or "").strip().lower()
    executor = DIRECT_TOOL_ENDPOINT_EXECUTORS.get(source_type, {}).get(endpoint)
    if executor is None:
        return None
    options = _parse_direct_tool_options(
        str(tool_request.get("remainder") or ""),
        str(direct_chat.get("argument_mode") or ""),
    )
    return executor(payload, tool_request, direct_chat, options)


def _handle_at_tool_request_for_source(
    source_type: str,
    payload: AnalysisChatRequest | RawQcChatRequest | SummaryStatsChatRequest | TextChatRequest,
    tool_request: dict[str, object],
) -> AnalysisChatResponse | RawQcChatResponse | SummaryStatsChatResponse | TextChatResponse:
    manifest = tool_request.get("manifest")
    help_text = _resolve_tool_help_response(tool_request)
    if help_text is not None:
        return _basic_source_response(source_type, help_text)
    mismatch_text = _resolve_tool_source_mismatch_response(tool_request, source_type)
    if mismatch_text is not None:
        return _basic_source_response(source_type, mismatch_text)
    if manifest is None:
        return _basic_source_response(source_type, _unknown_tool_answer(tool_request))
    direct_response = _run_direct_tool_for_source(source_type, payload, tool_request)
    if direct_response is not None:
        return direct_response
    return _basic_source_response(source_type, _unknown_tool_answer(tool_request))


def _handle_skill_request_for_source(
    source_type: str,
    payload: AnalysisChatRequest | RawQcChatRequest | SummaryStatsChatRequest | TextChatRequest,
    skill_request: dict[str, object],
) -> AnalysisChatResponse | RawQcChatResponse | SummaryStatsChatResponse | TextChatResponse:
    manifest = skill_request.get("manifest")
    help_text = _resolve_skill_help_response(skill_request, source_type)
    if help_text is not None:
        return _basic_source_response(source_type, help_text)
    if manifest is None:
        return _unknown_workflow_response(source_type, skill_request)
    workflow_name = str(manifest.get("name") or "")
    workflow_result = _run_registered_workflow_for_source(source_type, workflow_name, payload.analysis)
    return _assemble_workflow_chat_response(manifest, workflow_result)

def answer_analysis_chat(payload: AnalysisChatRequest) -> AnalysisChatResponse:
    if _needs_grounded_clarification(payload.question):
        return AnalysisChatResponse(
            answer=_grounded_clarification_text(),
            citations=[],
            used_fallback=False,
        )

    skill_request = _parse_skill_request(payload.question)
    if skill_request:
        try:
            response = _handle_skill_request_for_source("vcf", payload, skill_request)
            assert isinstance(response, AnalysisChatResponse)
            return response
        except Exception as exc:
            name = str(skill_request.get("name") or "workflow")
            return AnalysisChatResponse(
                answer=f"`@skill {name}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    at_tool_request = _parse_at_tool_request(payload.question)
    if at_tool_request:
        try:
            response = _handle_at_tool_request_for_source("vcf", payload, at_tool_request)
            assert isinstance(response, AnalysisChatResponse)
            return response
        except Exception as exc:
            alias = str(at_tool_request.get("alias") or "tool")
            return AnalysisChatResponse(
                answer=f"`@{alias}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    lowered_question = payload.question.lower()

    if "opencravat" in lowered_question or "open cravat" in lowered_question:
        return AnalysisChatResponse(
            answer=(
                "OpenCRAVAT is not available in this ChatGenome build.\n\n"
                "- The OpenCRAVAT plugin and Studio card have been removed because the runtime was unstable.\n"
                "- So this request did not run OpenCRAVAT.\n"
                "- If you want additional deterministic annotation, please use the currently supported tools such as SnpEff, CADD lookup, or REVEL lookup."
            ),
            citations=[],
            used_fallback=False,
            used_tools=[],
        )

    try:
        response = _call_openai_for_source("vcf", payload)
        assert isinstance(response, AnalysisChatResponse)
        return response
    except Exception:
        response = _fallback_chat_answer("vcf", payload.question)
        assert isinstance(response, AnalysisChatResponse)
        return response


def answer_raw_qc_chat(payload: RawQcChatRequest) -> RawQcChatResponse:
    if _needs_grounded_clarification(payload.question):
        return RawQcChatResponse(
            answer=_grounded_clarification_text(),
            citations=[],
            used_fallback=False,
        )

    skill_request = _parse_skill_request(payload.question)
    if skill_request:
        try:
            response = _handle_skill_request_for_source("raw_qc", payload, skill_request)
            assert isinstance(response, RawQcChatResponse)
            return response
        except Exception as exc:
            name = str(skill_request.get("name") or "workflow")
            return RawQcChatResponse(
                answer=f"`@skill {name}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    at_tool_request = _parse_at_tool_request(payload.question)
    if at_tool_request:
        try:
            response = _handle_at_tool_request_for_source("raw_qc", payload, at_tool_request)
            assert isinstance(response, RawQcChatResponse)
            return response
        except Exception as exc:
            alias = str(at_tool_request.get("alias") or "tool")
            return RawQcChatResponse(
                answer=f"`@{alias}` execution failed: {exc}",
                citations=[],
                used_fallback=False,
            )

    try:
        response = _call_openai_for_source("raw_qc", payload)
        assert isinstance(response, RawQcChatResponse)
        return response
    except Exception:
        response = _fallback_chat_answer("raw_qc", payload.question)
        assert isinstance(response, RawQcChatResponse)
        return response


def answer_summary_stats_chat(payload: SummaryStatsChatRequest) -> SummaryStatsChatResponse:
    if _needs_grounded_clarification(payload.question):
        return SummaryStatsChatResponse(
            answer=_grounded_clarification_text(),
            citations=[],
            used_fallback=False,
        )

    at_tool_request = _parse_at_tool_request(payload.question)
    if at_tool_request:
        try:
            response = _handle_at_tool_request_for_source("summary_stats", payload, at_tool_request)
            assert isinstance(response, SummaryStatsChatResponse)
            return response
        except Exception as exc:
            alias = str(at_tool_request.get("alias") or "tool")
            return SummaryStatsChatResponse(
                answer=f"`@{alias}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    skill_request = _parse_skill_request(payload.question)
    if skill_request:
        try:
            response = _handle_skill_request_for_source("summary_stats", payload, skill_request)
            assert isinstance(response, SummaryStatsChatResponse)
            return response
        except Exception as exc:
            name = str(skill_request.get("name") or "workflow")
            return SummaryStatsChatResponse(
                answer=f"`@skill {name}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    try:
        response = _call_openai_for_source("summary_stats", payload)
        assert isinstance(response, SummaryStatsChatResponse)
        return response
    except Exception:
        response = _fallback_chat_answer("summary_stats", payload.question)
        assert isinstance(response, SummaryStatsChatResponse)
        return response


def answer_text_chat(payload: TextChatRequest) -> TextChatResponse:
    if _needs_grounded_clarification(payload.question):
        return TextChatResponse(
            answer=_grounded_clarification_text(),
            citations=[],
            used_fallback=False,
        )

    skill_request = _parse_skill_request(payload.question)
    if skill_request:
        try:
            response = _handle_skill_request_for_source("text", payload, skill_request)
            assert isinstance(response, TextChatResponse)
            return response
        except Exception as exc:
            name = str(skill_request.get("name") or "workflow")
            return TextChatResponse(
                answer=f"`@skill {name}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    at_tool_request = _parse_at_tool_request(payload.question)
    if at_tool_request:
        try:
            response = _handle_at_tool_request_for_source("text", payload, at_tool_request)
            assert isinstance(response, TextChatResponse)
            return response
        except Exception as exc:
            alias = str(at_tool_request.get("alias") or "tool")
            return TextChatResponse(
                answer=f"`@{alias}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    try:
        response = _call_openai_for_source("text", payload)
        assert isinstance(response, TextChatResponse)
        return response
    except Exception:
        response = _fallback_chat_answer("text", payload.question)
        assert isinstance(response, TextChatResponse)
        return response
