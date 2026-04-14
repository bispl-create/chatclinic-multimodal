from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from app.models import (
    AnalysisResponse,
    AnalysisChatRequest,
    AnalysisChatResponse,
    DicomChatRequest,
    DicomChatResponse,
    DicomSourceResponse,
    FhirChatRequest,
    FhirChatResponse,
    FhirSourceResponse,
    ImageChatRequest,
    ImageChatResponse,
    ImageSourceResponse,
    NiftiChatRequest,
    NiftiChatResponse,
    NiftiSourceResponse,
    QqmanAssociationRequest,
    RawQcResponse,
    RawQcChatRequest,
    RawQcChatResponse,
    SourceChatRequest,
    SourceChatResponse,
    SpreadsheetChatRequest,
    SpreadsheetChatResponse,
    SpreadsheetSourceResponse,
    StudioContextPayload,
    SummaryStatsResponse,
    SummaryStatsChatRequest,
    SummaryStatsChatResponse,
    TextSourceResponse,
    TextChatRequest,
    TextChatResponse,
    MultimodalChatRequest,
    MultimodalChatResponse,
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
    run_tool,
    tool_aliases,
    tool_chat_metadata,
    tool_direct_chat_metadata,
)
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))


def _parse_at_tool_request(question: str) -> dict[str, object] | None:
    stripped = question.strip()
    match = re.match(r"^@([A-Za-z0-9_-]+)(?:\s+(.*))?$", stripped, flags=re.DOTALL)
    if not match:
        return None
    raw_alias = match.group(1).strip().lower()
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
        elif "image" in source_types:
            input_hint = "active image source"
        elif "dicom" in source_types:
            input_hint = "active DICOM source"
        elif "nifti" in source_types:
            input_hint = "active NIfTI source"
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
    if normalized == "spreadsheet":
        return "spreadsheet session"
    if normalized == "dicom":
        return "DICOM imaging session"
    if normalized == "image":
        return "image session"
    if normalized == "nifti":
        return "NIfTI session"
    if normalized == "fhir":
        return "FHIR session"
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
    if normalized == "spreadsheet":
        return "active spreadsheet source"
    if normalized == "dicom":
        return "active DICOM source"
    if normalized == "image":
        return "active image source"
    if normalized == "nifti":
        return "active NIfTI source"
    if normalized == "fhir":
        return "active FHIR source"
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


SOURCE_CHAT_RESPONSE_CLASS: dict[str, type] = {
    "vcf": AnalysisChatResponse,
    "raw_qc": RawQcChatResponse,
    "summary_stats": SummaryStatsChatResponse,
    "dicom": DicomChatResponse,
    "text": TextChatResponse,
    "spreadsheet": SpreadsheetChatResponse,
    "image": ImageChatResponse,
    "nifti": NiftiChatResponse,
    "fhir": FhirChatResponse,
}


def _source_response_class(source_type: str) -> type:
    cls = SOURCE_CHAT_RESPONSE_CLASS.get(source_type.strip().lower())
    if cls is None:
        raise NotImplementedError(f"Unsupported chat source type: {source_type}")
    return cls


def _basic_source_response(
    source_type: str,
    answer: str,
    *,
    used_fallback: bool = False,
) -> AnalysisChatResponse | DicomChatResponse | RawQcChatResponse | SpreadsheetChatResponse | SummaryStatsChatResponse | TextChatResponse | ImageChatResponse | NiftiChatResponse | FhirChatResponse:
    response_cls = _source_response_class(source_type)
    return response_cls(source_type=source_type, answer=answer, citations=[], used_fallback=used_fallback)



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
            source_type="vcf",
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
            source_type="raw_qc",
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
            source_type="summary_stats",
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
            source_type="text",
            answer=answer,
            citations=citations or [],
            used_fallback=used_fallback,
            requested_view=requested_view,
            studio=studio,
            analysis=analysis,
        )
    )


def _image_tool_response(
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
) -> ImageChatResponse:
    return ImageChatResponse(
        **_with_result_field(
            result_kind,
            result,
            source_type="image",
            answer=answer,
            citations=citations or [],
            used_fallback=used_fallback,
            used_tools=used_tools or [],
            requested_view=requested_view,
            studio=studio,
            analysis=analysis,
        )
    )



def _is_korean(text: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7a3]", text))


def _flatten_studio_context(studio_context: dict) -> dict[str, object]:
    if isinstance(studio_context, StudioContextPayload):
        base = studio_context.model_dump(exclude_none=True)
        extra = getattr(studio_context, "model_extra", None) or {}
        studio_context = {**base, **extra}
    # In multimodal mode, VCF-specific keys live inside extra (not top-level).
    # Fall back to extra for keys not found at the top level.
    extra_dict = studio_context.get("extra") or {}
    if not isinstance(extra_dict, dict):
        extra_dict = {}

    def _get(key: str) -> object:
        val = studio_context.get(key)
        if val is not None:
            return val
        return extra_dict.get(key)

    flattened = {
        "active_view": studio_context.get("active_view"),
        "qc_summary": _get("qc_summary"),
        "clinical_coverage": _get("clinical_coverage"),
        "symbolic_alt_review": _get("symbolic_alt_review"),
        "roh_review": _get("roh_review"),
        "candidate_variants": _get("candidate_variants"),
        "clinvar_review": _get("clinvar_review"),
        "vep_consequence": _get("vep_consequence"),
        "snpeff_preview": _get("snpeff_preview"),
        "selected_annotation": _get("selected_annotation"),
    }
    for key in (
        "current_card",
        "current_summary",
        "current_schema",
        "current_preview",
        "current_warnings",
        "extra",
        "sheet_count",
        "selected_sheet",
        "sheet_names",
        "sheet_details",
        "current_sheet",
    ):
        if key in studio_context:
            flattened[key] = studio_context.get(key)
    return flattened


def _serialize_source_chat_response(
    source_type: str,
    response: AnalysisChatResponse | DicomChatResponse | RawQcChatResponse | SpreadsheetChatResponse | SummaryStatsChatResponse | TextChatResponse,
) -> SourceChatResponse:
    data = response.model_dump(exclude_none=True)
    artifact_payload: dict[str, Any] = {}
    analysis_payload = data.pop("analysis", None)
    for key in (
        "plink_result",
        "liftover_result",
        "ldblockshow_result",
        "samtools_result",
        "qqman_result",
        "prs_prep_result",
    ):
        value = data.pop(key, None)
        if value is not None:
            artifact_payload[key] = value
    return SourceChatResponse(
        source_type=source_type,  # type: ignore[arg-type]
        answer=str(data.get("answer") or ""),
        citations=list(data.get("citations") or []),
        used_fallback=bool(data.get("used_fallback")),
        result_kind=data.get("result_kind"),
        requested_view=data.get("requested_view"),
        studio=data.get("studio"),
        analysis_payload=analysis_payload if isinstance(analysis_payload, dict) else None,
        artifact_payload=artifact_payload,
    )


def answer_source_chat(payload: SourceChatRequest) -> SourceChatResponse:
    studio_context = StudioContextPayload(**payload.studio_context.model_dump(exclude_none=True), **(getattr(payload.studio_context, "model_extra", None) or {}))
    source_type = payload.source_type
    if source_type == "vcf":
        response = answer_analysis_chat(
            AnalysisChatRequest(
                question=payload.question,
                analysis=AnalysisResponse(**payload.analysis_payload),
                history=payload.history,
                studio_context=studio_context,
            )
        )
        return _serialize_source_chat_response("vcf", response)
    if source_type == "raw_qc":
        response = answer_raw_qc_chat(
            RawQcChatRequest(
                question=payload.question,
                analysis=RawQcResponse(**payload.analysis_payload),
                history=payload.history,
                studio_context=studio_context,
            )
        )
        return _serialize_source_chat_response("raw_qc", response)
    if source_type == "summary_stats":
        response = answer_summary_stats_chat(
            SummaryStatsChatRequest(
                question=payload.question,
                analysis=SummaryStatsResponse(**payload.analysis_payload),
                history=payload.history,
                studio_context=studio_context,
            )
        )
        return _serialize_source_chat_response("summary_stats", response)
    if source_type == "dicom":
        response = answer_dicom_chat(
            DicomChatRequest(
                question=payload.question,
                analysis=DicomSourceResponse(**payload.analysis_payload),
                history=payload.history,
                studio_context=studio_context,
            )
        )
        return _serialize_source_chat_response("dicom", response)
    if source_type == "text":
        response = answer_text_chat(
            TextChatRequest(
                question=payload.question,
                analysis=TextSourceResponse(**payload.analysis_payload),
                history=payload.history,
                studio_context=studio_context,
            )
        )
        return _serialize_source_chat_response("text", response)
    if source_type == "spreadsheet":
        response = answer_spreadsheet_chat(
            SpreadsheetChatRequest(
                question=payload.question,
                analysis=SpreadsheetSourceResponse(**payload.analysis_payload),
                history=payload.history,
                studio_context=studio_context,
            )
        )
        return _serialize_source_chat_response("spreadsheet", response)
    if source_type == "image":
        response = answer_image_chat(
            ImageChatRequest(
                question=payload.question,
                analysis=ImageSourceResponse(**payload.analysis_payload),
                history=payload.history,
                studio_context=studio_context,
            )
        )
        return _serialize_source_chat_response("image", response)
    raise NotImplementedError(f"Unsupported source chat type: {source_type}")


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


def _compact_dicom_context(payload: DicomChatRequest) -> dict[str, object]:
    first_item = payload.analysis.metadata_items[0] if payload.analysis.metadata_items else {}
    context = {
        "analysis_id": payload.analysis.analysis_id,
        "file_name": payload.analysis.file_name,
        "file_kind": payload.analysis.file_kind,
        "modality": first_item.get("modality"),
        "patient_id": first_item.get("patient_id"),
        "study_description": first_item.get("study_description"),
        "series_description": first_item.get("series_description"),
        "rows": first_item.get("rows"),
        "columns": first_item.get("columns"),
        "series_count": len(payload.analysis.series),
        "warnings": payload.analysis.warnings[:12],
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


def _compact_spreadsheet_context(payload: SpreadsheetChatRequest) -> dict[str, object]:
    context = {
        "analysis_id": payload.analysis.analysis_id,
        "file_name": payload.analysis.file_name,
        "workbook_format": payload.analysis.workbook_format,
        "sheet_count": payload.analysis.sheet_count,
        "sheet_names": payload.analysis.sheet_names[:20],
        "selected_sheet": payload.analysis.selected_sheet,
        "sheet_details": payload.analysis.sheet_details[:12],
        "draft_answer": payload.analysis.draft_answer,
        "used_tools": payload.analysis.used_tools,
    }
    if payload.studio_context:
        context["studio_context"] = _flatten_studio_context(payload.studio_context)
    return context


def _compact_image_context(payload: ImageChatRequest) -> dict[str, object]:
    exif_summary = {}
    for key in ("Make", "Model", "DateTime", "Software", "ImageWidth", "ImageLength", "GPS"):
        if key in payload.analysis.exif_data:
            exif_summary[key] = payload.analysis.exif_data[key]
    unsb_summary = None
    if isinstance(payload.analysis.artifacts, dict):
        artifact = payload.analysis.artifacts.get("unsb_enhancement")
        if isinstance(artifact, dict):
            unsb_summary = {
                "output_image_path": artifact.get("output_image_path"),
                "epoch": artifact.get("epoch"),
                "device": artifact.get("device"),
                "num_timesteps": artifact.get("num_timesteps"),
                "tau": artifact.get("tau"),
                "crop_size": artifact.get("crop_size"),
                "deterministic": artifact.get("deterministic"),
            }
    context: dict[str, object] = {
        "analysis_id": payload.analysis.analysis_id,
        "file_name": payload.analysis.file_name,
        "file_kind": payload.analysis.file_kind,
        "width": payload.analysis.width,
        "height": payload.analysis.height,
        "format_name": payload.analysis.format_name,
        "color_mode": payload.analysis.color_mode,
        "bit_depth": payload.analysis.bit_depth,
        "exif_summary": exif_summary,
        "warnings": payload.analysis.warnings[:12],
        "draft_answer": payload.analysis.draft_answer,
        "used_tools": payload.analysis.used_tools,
    }
    if unsb_summary is not None:
        context["unsb_enhancement"] = unsb_summary
    if payload.studio_context:
        context["studio_context"] = _flatten_studio_context(payload.studio_context)
    return context


def _compact_fhir_context(payload: FhirChatRequest, **kwargs: Any) -> dict[str, object]:
    analysis = payload.analysis
    parts: list[str] = [f"FHIR source: {analysis.file_name}"]
    parts.append(f"Resource type: {analysis.resource_type}, count: {analysis.resource_count}")
    patient = analysis.patient_summary
    if patient:
        parts.append(f"Patient: {patient.get('full_name', 'n/a')}, {patient.get('gender', 'n/a')}, DOB {patient.get('birth_date', 'n/a')}")
    obs = analysis.artifacts.get("observations", {})
    if obs.get("count"):
        parts.append(f"Observations: {obs['count']} total")
        for item in (obs.get("items") or [])[:5]:
            parts.append(f"  - {item.get('code', 'n/a')}: {item.get('value', 'n/a')} ({item.get('effective', 'n/a')})")
    meds = analysis.artifacts.get("medications", {})
    if meds.get("count"):
        parts.append(f"Medications: {meds['count']} total")
        for item in (meds.get("items") or [])[:5]:
            parts.append(f"  - {item.get('medication', 'n/a')}: {item.get('status', 'n/a')}")
    allergies = analysis.artifacts.get("allergies", {})
    if allergies.get("count"):
        parts.append(f"Allergies: {allergies['count']} total")
        for item in (allergies.get("items") or [])[:4]:
            parts.append(f"  - {item.get('substance', 'n/a')}: {item.get('criticality', 'n/a')}")
    if analysis.warnings:
        parts.append(f"Warnings: {'; '.join(analysis.warnings[:5])}")
    context: dict[str, object] = {
        "summary": "\n".join(parts),
        "analysis_id": analysis.analysis_id,
        "file_name": analysis.file_name,
        "resource_type": analysis.resource_type,
        "resource_count": analysis.resource_count,
        "patient_summary": analysis.patient_summary,
        "artifacts": {k: v for k, v in analysis.artifacts.items()},
        "warnings": analysis.warnings[:12],
        "draft_answer": analysis.draft_answer,
        "used_tools": analysis.used_tools,
    }
    if payload.studio_context:
        context["studio_context"] = _flatten_studio_context(payload.studio_context)
    return context


def _compact_nifti_context(payload: NiftiChatRequest) -> dict[str, object]:
    a = payload.analysis
    context: dict[str, object] = {
        "analysis_id": a.analysis_id,
        "file_name": a.file_name,
        "file_kind": a.file_kind,
        "shape": a.shape,
        "voxel_dims": a.voxel_dims,
        "fov_mm": a.fov_mm,
        "orientation": a.orientation,
        "datatype": a.datatype,
        "is_4d": a.is_4d,
        "metadata_items": a.metadata_items[:12] if a.metadata_items else [],
        "warnings": a.warnings[:12] if a.warnings else [],
        "draft_answer": a.draft_answer,
        "used_tools": a.used_tools,
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
    "dicom": {
        "context_label": "DICOM review context JSON",
        "compact_context_builder": _compact_dicom_context,
        "grounded_system_prompt": (
            "You are a DICOM imaging review copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer only from the provided DICOM metadata, preview state, and current Studio card context. "
            "Do not invent pixel findings or diagnoses that are not present in the provided context. "
            "Be explicit when the answer is based only on metadata or preview state."
        ),
        "general_system_prompt": (
            "You are a helpful general assistant. "
            "The user did not request grounded DICOM reasoning. "
            "Answer from general knowledge only and ignore any uploaded DICOM context unless the user explicitly asks with a grounding trigger such as $studio."
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
    "spreadsheet": {
        "context_label": "Spreadsheet cohort context JSON",
        "compact_context_builder": _compact_spreadsheet_context,
        "grounded_system_prompt": (
            "You are a spreadsheet and cohort-browser copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer only from the provided workbook context and do not invent unseen sheet contents. "
            "Be explicit about sheet names, cohort counts, missingness, and whether the answer is based on preview rows or summarized sheet metadata."
        ),
        "general_system_prompt": (
            "You are a helpful general assistant. "
            "The user did not request grounded spreadsheet reasoning. "
            "Answer from general knowledge only and ignore any uploaded workbook context unless the user explicitly asks with a grounding trigger such as $studio."
        ),
    },
    "image": {
        "context_label": "Image metadata context JSON",
        "compact_context_builder": _compact_image_context,
        "grounded_system_prompt": (
            "You are an image metadata analysis copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer only from the provided image metadata (dimensions, format, EXIF, color mode) and do not invent information. "
            "When EXIF data includes GPS coordinates, camera make/model, or timestamps, mention them explicitly."
        ),
        "general_system_prompt": (
            "You are a helpful general assistant. "
            "The user did not request grounded image reasoning. "
            "Answer from general knowledge only."
        ),
    },
    "nifti": {
        "context_label": "NIfTI volume context JSON",
        "compact_context_builder": _compact_nifti_context,
        "grounded_system_prompt": (
            "You are a neuroimaging and NIfTI volume analysis copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer only from the provided NIfTI context including shape, voxel dimensions, FOV, orientation, and data type. "
            "Do not invent imaging findings or clinical interpretations that are not present in the provided context. "
            "Be concise and cite volume properties when relevant."
        ),
        "general_system_prompt": (
            "You are a helpful general assistant. "
            "The user did not request grounded NIfTI reasoning. "
            "Answer from general knowledge only and ignore any uploaded NIfTI context unless the user explicitly asks with a grounding trigger such as $studio."
        ),
    },
    "fhir": {
        "context_label": "FHIR clinical context JSON",
        "compact_context_builder": _compact_fhir_context,
        "grounded_system_prompt": (
            "You are a clinical data copilot for FHIR resources. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer only from the provided FHIR context including patient demographics, observations, medications, allergies, vitals, labs, timeline events, and care team. "
            "Do not invent clinical findings or lab values that are not present in the provided context. "
            "Be concise and cite resource types when relevant."
        ),
        "general_system_prompt": (
            "You are a helpful general assistant. "
            "The user did not request grounded FHIR reasoning. "
            "Answer from general knowledge only and ignore any uploaded FHIR context unless the user explicitly asks with a grounding trigger such as $studio."
        ),
    },
}


def _fallback_chat_answer(
    source_type: str,
    question: str,
) -> AnalysisChatResponse | DicomChatResponse | ImageChatResponse | RawQcChatResponse | SpreadsheetChatResponse | SummaryStatsChatResponse | TextChatResponse:
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
    payload: AnalysisChatRequest | DicomChatRequest | RawQcChatRequest | SpreadsheetChatRequest | SummaryStatsChatRequest | TextChatRequest,
) -> AnalysisChatResponse | DicomChatResponse | RawQcChatResponse | SpreadsheetChatResponse | SummaryStatsChatResponse | TextChatResponse:
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
    from plugins.gatk_liftover_vcf_tool.logic import (
        DEFAULT_CHAIN_FILE,
        DEFAULT_TARGET_FASTA,
        run_gatk_liftover_vcf,
    )

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
    from plugins.samtools_execution_tool.logic import run_samtools

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
    from plugins.qqman_execution_tool.logic import run_qqman_association

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
    from plugins.snpeff_execution_tool.logic import run_snpeff

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
    from plugins.ldblockshow_execution_tool.logic import run_ldblockshow

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


def _execute_analysis_direct_vcf_interpretation(
    payload: AnalysisChatRequest,
    tool_request: dict[str, object],
    direct_chat: dict[str, Any],
    options: dict[str, str],
) -> AnalysisChatResponse:
    from plugins.vcf_interpretation_tool.logic import execute as run_vcf_interpretation

    del tool_request, options
    source_vcf_path = payload.analysis.source_vcf_path
    if not source_vcf_path:
        return AnalysisChatResponse(
            answer="The current analysis context does not include a source VCF path, so interpretation cannot be run.",
            citations=[],
            used_fallback=True,
            used_tools=["vcf_interpretation_tool"],
        )
    result = run_vcf_interpretation({
        "vcf_path": source_vcf_path,
        "facts": payload.analysis.facts.model_dump(),
        "scope": "representative",
        "ranking_limit": 8,
    })
    from app.models import RankedCandidate, RohSegment, VariantAnnotation
    annotations = [VariantAnnotation(**a) for a in result.get("annotations", [])]
    roh_segments = [RohSegment(**r) for r in result.get("roh_segments", [])]
    candidate_variants = [RankedCandidate(**c) for c in result.get("candidate_variants", [])]
    updated_analysis = payload.analysis.model_copy(update={
        "annotations": annotations,
        "roh_segments": roh_segments,
        "candidate_variants": candidate_variants,
        "used_tools": list(payload.analysis.used_tools or []) + ["vcf_interpretation_tool"],
    })
    return AnalysisChatResponse(
        answer=(
            f"VCF interpretation complete.\n\n"
            f"- Annotations: {result.get('annotation_count', 0)}\n"
            f"- ROH segments: {result.get('roh_segment_count', 0)}\n"
            f"- Ranked candidates: {result.get('candidate_count', 0)}\n"
            f"- CADD enrichment: {'yes' if result.get('cadd_lookup_performed') else 'no'} ({result.get('cadd_matched_count', 0)} matched)\n"
            f"- REVEL enrichment: {'yes' if result.get('revel_lookup_performed') else 'no'} ({result.get('revel_matched_count', 0)} matched)\n\n"
            "The Candidates card in Studio has been updated."
        ),
        citations=[],
        used_fallback=False,
        used_tools=["vcf_interpretation_tool"],
        analysis=updated_analysis,
        requested_view="candidates",
        studio={"renderer": payload.analysis.studio.get("renderer", "qc") if isinstance(payload.analysis.studio, dict) else "qc"},
    )


def _execute_analysis_direct_vcf_qc(
    payload: AnalysisChatRequest,
    tool_request: dict[str, object],
    direct_chat: dict[str, Any],
    options: dict[str, str],
) -> AnalysisChatResponse:
    from plugins.vcf_qc_tool.logic import summarize_vcf

    del tool_request, options
    source_vcf_path = payload.analysis.source_vcf_path
    if not source_vcf_path:
        return AnalysisChatResponse(
            answer="The current analysis context does not include a source VCF path, so VCF QC cannot be re-run.",
            citations=[],
            used_fallback=True,
            used_tools=["vcf_qc_tool"],
        )
    max_examples = int(os.getenv("MAX_EXAMPLE_VARIANTS", "8"))
    facts = summarize_vcf(source_vcf_path, max_examples=max_examples)
    updated_analysis = payload.analysis.model_copy(update={"facts": facts})
    return AnalysisChatResponse(
        answer=(
            f"VCF QC re-run complete.\n\n"
            f"- Records: {facts.record_count}\n"
            f"- Build: {facts.genome_build_guess or 'unknown'}\n"
            f"- Ti/Tv: {facts.qc.transition_transversion_ratio if facts.qc else 'n/a'}\n"
            f"- Pass rate: {facts.qc.pass_rate if facts.qc else 'n/a'}\n\n"
            "The QC Summary card in Studio has been updated."
        ),
        citations=[],
        used_fallback=False,
        used_tools=["vcf_qc_tool"],
        analysis=updated_analysis,
        requested_view=str(direct_chat.get("requested_view") or "qc"),
        studio={"renderer": "qc"},
    )


def _execute_analysis_direct_vcf_review(
    payload: AnalysisChatRequest,
    tool_request: dict[str, object],
    direct_chat: dict[str, Any],
    options: dict[str, str],
) -> AnalysisChatResponse:
    from plugins.vcf_review_tool.logic import execute as run_vcf_review

    del tool_request, options
    from app.models import CountSummaryItem, DetailedCountSummaryItem, SymbolicAltSummary
    result = run_vcf_review({
        "facts": payload.analysis.facts.model_dump(),
        "annotations": [a.model_dump() for a in payload.analysis.annotations],
        "candidate_variants": [c.model_dump() for c in payload.analysis.candidate_variants],
        "references": [r.model_dump() for r in payload.analysis.references],
    })
    clinvar_summary = [CountSummaryItem(**item) for item in result.get("clinvar_summary", [])]
    consequence_summary = [CountSummaryItem(**item) for item in result.get("consequence_summary", [])]
    clinical_coverage_summary = [DetailedCountSummaryItem(**item) for item in result.get("clinical_coverage_summary", [])]
    symbolic_alt_raw = result.get("symbolic_alt_summary")
    symbolic_alt_summary = (
        SymbolicAltSummary(**symbolic_alt_raw)
        if isinstance(symbolic_alt_raw, dict) and "count" in symbolic_alt_raw
        else payload.analysis.symbolic_alt_summary
    )
    updated_analysis = payload.analysis.model_copy(update={
        "clinvar_summary": clinvar_summary,
        "consequence_summary": consequence_summary,
        "clinical_coverage_summary": clinical_coverage_summary,
        "symbolic_alt_summary": symbolic_alt_summary,
        "draft_answer": result.get("draft_answer", "") or payload.analysis.draft_answer,
        "used_tools": list(payload.analysis.used_tools or []) + ["vcf_review_tool"],
    })
    return AnalysisChatResponse(
        answer=(
            f"VCF review complete.\n\n"
            f"- ClinVar buckets: {len(clinvar_summary)}\n"
            f"- Consequence buckets: {len(consequence_summary)}\n"
            f"- Coverage rows: {len(clinical_coverage_summary)}\n"
            f"- Symbolic ALT count: {symbolic_alt_summary.count if symbolic_alt_summary else 0}\n\n"
            "The Clinical Review cards in Studio have been updated."
        ),
        citations=[],
        used_fallback=False,
        used_tools=["vcf_review_tool"],
        analysis=updated_analysis,
        requested_view="clinvar",
        studio={"renderer": "clinvar"},
    )


def _execute_image_direct_unsb(
    payload: ImageChatRequest,
    tool_request: dict[str, object],
    direct_chat: dict[str, Any],
    options: dict[str, str],
) -> ImageChatResponse:
    del tool_request
    source_image_path = payload.analysis.source_image_path
    if not source_image_path:
        return _image_tool_response(
            "The active image session does not include a durable image path, so `@unsb` cannot run yet.",
            used_fallback=True,
            used_tools=["unsb_tool"],
        )

    request_payload: dict[str, object] = {
        "image_path": source_image_path,
        "file_name": payload.analysis.file_name,
        "analysis": payload.analysis.model_dump(),
    }
    for key in ("ckpt_dir", "epoch", "num_timesteps", "tau", "crop_size", "deterministic", "gpu", "save_channel", "output_prefix"):
        value = options.get(key)
        if value not in (None, ""):
            request_payload[key] = value

    result = run_tool("unsb_tool", request_payload)
    analysis_payload = result.get("analysis")
    if not isinstance(analysis_payload, dict):
        raise RuntimeError("UNSB tool did not return an updated image analysis payload.")

    updated_analysis = ImageSourceResponse(**analysis_payload)
    answer = str(result.get("summary") or updated_analysis.draft_answer or "UNSB enhancement is ready.")
    return _image_tool_response(
        answer,
        used_tools=["unsb_tool"],
        result_kind=str(direct_chat.get("result_kind") or "image_analysis"),
        requested_view=str(direct_chat.get("requested_view") or "image_review"),
        studio=direct_chat.get("studio") if isinstance(direct_chat.get("studio"), dict) else {"renderer": "image_review"},
        analysis=updated_analysis,
    )


DIRECT_TOOL_ENDPOINT_EXECUTORS: dict[str, dict[str, Any]] = {
    "vcf": {
        "liftover": _execute_analysis_direct_liftover,
        "snpeff": _execute_analysis_direct_snpeff,
        "ldblockshow": _execute_analysis_direct_ldblockshow,
        "plink": _execute_analysis_direct_plink,
        "vcf-qc": _execute_analysis_direct_vcf_qc,
        "vcf-interpretation": _execute_analysis_direct_vcf_interpretation,
        "vcf-review": _execute_analysis_direct_vcf_review,
    },
    "raw_qc": {
        "samtools": _execute_raw_qc_direct_samtools,
    },
    "summary_stats": {
        "qqman": _execute_summary_stats_direct_qqman,
    },
    "text": {},
    "spreadsheet": {},
    "image": {
        "unsb": _execute_image_direct_unsb,
    },
    "dicom": {},
    "nifti": {},
    "fhir": {},
}


def _run_direct_tool_for_source(
    source_type: str,
    payload: AnalysisChatRequest | RawQcChatRequest | SpreadsheetChatRequest | SummaryStatsChatRequest | TextChatRequest | ImageChatRequest,
    tool_request: dict[str, object],
) -> AnalysisChatResponse | RawQcChatResponse | SpreadsheetChatResponse | SummaryStatsChatResponse | TextChatResponse | ImageChatResponse | None:
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
    payload: AnalysisChatRequest | RawQcChatRequest | SpreadsheetChatRequest | SummaryStatsChatRequest | TextChatRequest | ImageChatRequest,
    tool_request: dict[str, object],
) -> AnalysisChatResponse | RawQcChatResponse | SpreadsheetChatResponse | SummaryStatsChatResponse | TextChatResponse | ImageChatResponse:
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



_SOURCE_PRE_CHAT_HOOKS: dict[str, Any] = {}


def _answer_source_chat(source_type: str, payload: Any) -> Any:
    """Generic chat handler for all source types."""
    response_cls = _source_response_class(source_type)

    if _needs_grounded_clarification(payload.question):
        return response_cls(
            source_type=source_type,
            answer=_grounded_clarification_text(),
            citations=[],
            used_fallback=False,
        )

    at_tool_request = _parse_at_tool_request(payload.question)
    if at_tool_request:
        try:
            return _handle_at_tool_request_for_source(source_type, payload, at_tool_request)
        except Exception as exc:
            alias = str(at_tool_request.get("alias") or "tool")
            return response_cls(
                source_type=source_type,
                answer=f"`@{alias}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    pre_hook = _SOURCE_PRE_CHAT_HOOKS.get(source_type)
    if pre_hook is not None:
        hook_result = pre_hook(payload)
        if hook_result is not None:
            return hook_result

    try:
        return _call_openai_for_source(source_type, payload)
    except Exception:
        return _fallback_chat_answer(source_type, payload.question)


def answer_analysis_chat(payload: AnalysisChatRequest) -> AnalysisChatResponse:
    return _answer_source_chat("vcf", payload)


def answer_raw_qc_chat(payload: RawQcChatRequest) -> RawQcChatResponse:
    return _answer_source_chat("raw_qc", payload)


def answer_summary_stats_chat(payload: SummaryStatsChatRequest) -> SummaryStatsChatResponse:
    return _answer_source_chat("summary_stats", payload)


def answer_text_chat(payload: TextChatRequest) -> TextChatResponse:
    return _answer_source_chat("text", payload)


def answer_dicom_chat(payload: DicomChatRequest) -> DicomChatResponse:
    return _answer_source_chat("dicom", payload)


def answer_spreadsheet_chat(payload: SpreadsheetChatRequest) -> SpreadsheetChatResponse:
    return _answer_source_chat("spreadsheet", payload)


def answer_image_chat(payload: ImageChatRequest) -> ImageChatResponse:
    return _answer_source_chat("image", payload)


def answer_nifti_chat(payload: NiftiChatRequest) -> NiftiChatResponse:
    return _answer_source_chat("nifti", payload)


def answer_fhir_chat(payload: FhirChatRequest) -> FhirChatResponse:
    return _answer_source_chat("fhir", payload)


def answer_multimodal_chat(payload: MultimodalChatRequest) -> MultimodalChatResponse:
    """Handle chat with merged context from all active sources."""
    # Determine primary source for @tool routing
    primary = str(payload.primary_source_type or "").strip()

    # For @tool requests, route to the primary source's handler
    at_tool_request = _parse_at_tool_request(payload.question)
    if at_tool_request:
        source_type, single_payload = _multimodal_to_single(payload, primary or None)
        if single_payload is not None:
            try:
                result = _handle_at_tool_request_for_source(source_type, single_payload, at_tool_request)
                return _single_response_to_multimodal(result)
            except Exception as exc:
                alias = str(at_tool_request.get("alias") or "tool")
                return MultimodalChatResponse(
                    answer=f"`@{alias}` execution failed: {exc}",
                    citations=[],
                    used_fallback=True,
                )
        return MultimodalChatResponse(
            answer="No active source to run this tool against.",
            citations=[],
            used_fallback=True,
        )

    if _needs_grounded_clarification(payload.question):
        return MultimodalChatResponse(
            answer=_grounded_clarification_text(),
            citations=[],
            used_fallback=False,
        )

    # For grounded/general chat, merge all source contexts into one prompt
    try:
        return _call_openai_multimodal(payload)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        return MultimodalChatResponse(
            answer=f"Multimodal chat error: {exc}\n\n```\n{tb}\n```",
            citations=[],
            used_fallback=True,
        )


def _multimodal_to_single(
    payload: MultimodalChatRequest,
    preferred: str | None,
) -> tuple[str, Any]:
    """Extract a single-source payload from multimodal request for @tool routing."""
    sources: dict[str, Any] = {}
    if payload.vcf_analysis:
        sources["vcf"] = AnalysisChatRequest(
            question=payload.question, analysis=payload.vcf_analysis,
            history=payload.history, studio_context=payload.studio_context,
        )
    if payload.raw_qc_analysis:
        sources["raw_qc"] = RawQcChatRequest(
            question=payload.question, analysis=payload.raw_qc_analysis,
            history=payload.history, studio_context=payload.studio_context,
        )
    if payload.summary_stats_analysis:
        sources["summary_stats"] = SummaryStatsChatRequest(
            question=payload.question, analysis=payload.summary_stats_analysis,
            history=payload.history, studio_context=payload.studio_context,
        )
    if payload.text_analysis:
        sources["text"] = TextChatRequest(
            question=payload.question, analysis=payload.text_analysis,
            history=payload.history, studio_context=payload.studio_context,
        )
    if payload.dicom_analysis:
        sources["dicom"] = DicomChatRequest(
            question=payload.question, analysis=payload.dicom_analysis,
            history=payload.history, studio_context=payload.studio_context,
        )
    if payload.spreadsheet_analysis:
        sources["spreadsheet"] = SpreadsheetChatRequest(
            question=payload.question, analysis=payload.spreadsheet_analysis,
            history=payload.history, studio_context=payload.studio_context,
        )
    if payload.image_analysis:
        sources["image"] = ImageChatRequest(
            question=payload.question, analysis=payload.image_analysis,
            history=payload.history, studio_context=payload.studio_context,
        )
    if payload.nifti_analysis:
        sources["nifti"] = NiftiChatRequest(
            question=payload.question, analysis=payload.nifti_analysis,
            history=payload.history, studio_context=payload.studio_context,
        )
    if payload.fhir_analysis:
        sources["fhir"] = FhirChatRequest(
            question=payload.question, analysis=payload.fhir_analysis,
            history=payload.history, studio_context=payload.studio_context,
        )
    if preferred and preferred in sources:
        return preferred, sources[preferred]
    if sources:
        key = next(iter(sources))
        return key, sources[key]
    return "vcf", None


def _single_response_to_multimodal(resp: Any) -> MultimodalChatResponse:
    """Convert any single-source chat response to MultimodalChatResponse."""
    return MultimodalChatResponse(
        source_type=getattr(resp, "source_type", None),
        answer=resp.answer,
        citations=resp.citations,
        used_fallback=resp.used_fallback,
        used_tools=getattr(resp, "used_tools", []),
        result_kind=getattr(resp, "result_kind", None),
        requested_view=getattr(resp, "requested_view", None),
        studio=getattr(resp, "studio", None),
        analysis=getattr(resp, "analysis", None),
        plink_result=getattr(resp, "plink_result", None),
        liftover_result=getattr(resp, "liftover_result", None),
        ldblockshow_result=getattr(resp, "ldblockshow_result", None),
        samtools_result=getattr(resp, "samtools_result", None),
        qqman_result=getattr(resp, "qqman_result", None),
        prs_prep_result=getattr(resp, "prs_prep_result", None),
    )


def _call_openai_multimodal(payload: MultimodalChatRequest) -> MultimodalChatResponse:
    """Call OpenAI with merged context from all active sources."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    if not api_key:
        return MultimodalChatResponse(
            answer=_fallback_chat_answer("vcf", payload.question).answer,
            citations=[],
            used_fallback=True,
        )

    grounded = _has_studio_trigger(payload.question)
    context_sections: list[str] = []

    # Build real Pydantic request objects for each active source so that
    # the compact context builders can access typed attributes correctly.
    _REQUEST_CLS: dict[str, tuple[type, Any]] = {
        "vcf": (AnalysisChatRequest, payload.vcf_analysis),
        "raw_qc": (RawQcChatRequest, payload.raw_qc_analysis),
        "summary_stats": (SummaryStatsChatRequest, payload.summary_stats_analysis),
        "text": (TextChatRequest, payload.text_analysis),
        "spreadsheet": (SpreadsheetChatRequest, payload.spreadsheet_analysis),
        "dicom": (DicomChatRequest, payload.dicom_analysis),
        "image": (ImageChatRequest, payload.image_analysis),
        "nifti": (NiftiChatRequest, payload.nifti_analysis),
        "fhir": (FhirChatRequest, payload.fhir_analysis),
    }

    # Build per-source studio contexts from the merged studioContext.
    # The frontend puts VCF keys at extra top-level, and source-specific
    # data under extra.spreadsheet, extra.dicom, etc.
    raw_sc = {}
    if payload.studio_context:
        if isinstance(payload.studio_context, StudioContextPayload):
            raw_sc = {**payload.studio_context.model_dump(exclude_none=True), **(getattr(payload.studio_context, "model_extra", None) or {})}
        elif isinstance(payload.studio_context, dict):
            raw_sc = dict(payload.studio_context)
    merged_extra = raw_sc.get("extra") or {}
    if not isinstance(merged_extra, dict):
        merged_extra = {}

    def _build_source_studio_context(source_type: str) -> dict | None:
        """Reconstruct a source-specific studio context from the merged one."""
        base = {"active_view": raw_sc.get("active_view")}
        if source_type == "vcf":
            # VCF keys live at merged_extra top level
            for k in ("qc_summary", "clinical_coverage", "symbolic_alt_review",
                       "roh_review", "candidate_variants", "clinvar_review",
                       "vep_consequence", "snpeff_preview", "selected_annotation",
                       "filtering_summary", "liftover_preview", "ldblockshow_preview",
                       "plink_preview"):
                if k in merged_extra:
                    base[k] = merged_extra[k]
            for k in ("current_card", "current_summary", "current_preview", "current_warnings"):
                if k in raw_sc:
                    base[k] = raw_sc[k]
            return base if len(base) > 1 else None
        elif source_type == "spreadsheet":
            src = merged_extra.get("spreadsheet") or {}
            if not isinstance(src, dict):
                return None
            for k in ("sheet_count", "selected_sheet", "sheet_names", "sheet_details", "current_sheet"):
                if k in src:
                    base[k] = src[k]
            # Reconstruct current_card/current_summary from sheet data
            if src.get("current_sheet"):
                base["current_card"] = src["current_sheet"]
                base["current_summary"] = {
                    "selected_sheet": src.get("selected_sheet"),
                    "overview": src["current_sheet"].get("overview"),
                    "intake": src["current_sheet"].get("intake"),
                    "composition": src["current_sheet"].get("composition"),
                }
                base["current_preview"] = {
                    "columns": src["current_sheet"].get("preview_columns", []),
                    "rows": src["current_sheet"].get("preview_rows", []),
                }
            return base if len(base) > 1 else None
        elif source_type == "dicom":
            src = merged_extra.get("dicom") or {}
            if not isinstance(src, dict):
                return None
            base["current_card"] = src.get("current_card")
            base["current_summary"] = src.get("current_summary")
            base["extra"] = {"metadata_items": src.get("metadata_items"), "series": src.get("series"), "preview": src.get("preview")}
            return base if len(base) > 1 else None
        # For other source types, pass the raw context as-is
        return raw_sc if raw_sc else None

    for source_type, (req_cls, analysis_obj) in _REQUEST_CLS.items():
        if analysis_obj is None:
            continue
        config = CHAT_OPENAI_CONFIG.get(source_type)
        if not config:
            continue
        try:
            source_sc = _build_source_studio_context(source_type)
            kwargs: dict[str, Any] = dict(
                question=payload.question,
                analysis=analysis_obj,
                history=payload.history,
            )
            if source_sc:
                kwargs["studio_context"] = source_sc
            single = req_cls(**kwargs)
            compact = config["compact_context_builder"](single)
            label = config["context_label"]
            section_json = json.dumps(compact, ensure_ascii=False)
            # Cap each source section to ~8k chars to keep total payload manageable
            if len(section_json) > 8000:
                section_json = section_json[:8000] + "..."
            context_sections.append(f"{label} ({source_type}):\n{section_json}")
        except Exception as e:
            context_sections.append(f"[{source_type} context build failed: {e}]")

    if grounded and context_sections:
        system_prompt = (
            "You are a multimodal biomedical analysis copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Multiple source types are loaded simultaneously. "
            "Answer by synthesizing ALL provided source contexts together — cross-reference and summarize findings across all modalities. "
            "Every loaded source is equally important; do not ignore any source context. "
            "Do not invent facts not present in the provided contexts. "
            "When possible, cite reference ids like REF1 or REF4 inline. "
            "Format the answer in clean Markdown."
        )
        user_content = (
            "Question:\n"
            f"{payload.question}\n\n"
            + "\n\n".join(context_sections)
        )
    elif grounded:
        system_prompt = (
            "You are a multimodal biomedical analysis copilot. "
            "The user requested grounded reasoning but no analysis contexts are available yet. "
            "Ask the user to upload source files first."
        )
        user_content = payload.question
    else:
        system_prompt = (
            "You are a helpful general assistant. "
            "The user did not request analysis grounding. "
            "Answer from general knowledge only."
        )
        user_content = payload.question

    # Truncate user_content to avoid exceeding token limits
    max_chars = 40_000
    if len(user_content) > max_chars:
        user_content = user_content[:max_chars] + "\n\n[... context truncated for length ...]"

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
    try:
        with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        error_body = http_err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI {http_err.code}: {error_body}") from http_err

    output_text = _extract_openai_output_text(result)
    citations = sorted(set(re.findall(r"\bREF\d+\b", output_text or "")))
    return MultimodalChatResponse(
        answer=output_text or "Could not generate a response.",
        citations=citations,
        used_fallback=False,
    )
