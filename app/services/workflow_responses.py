from __future__ import annotations

import uuid
from typing import Any

from app.models import (
    AnalysisFacts,
    AnalysisResponse,
    PrsPrepResponse,
    RawQcResponse,
    SummaryStatsResponse,
    TextSourceResponse,
)
from app.services.annotation import build_ui_cards
from app.services.recommendation import build_recommendations
from app.services.references import build_reference_bundle


class _SafeFormatDict(dict[str, object]):
    def __missing__(self, key: str) -> str:
        return ""


def workflow_studio_metadata(manifest: dict[str, object]) -> dict[str, object]:
    studio = manifest.get("studio")
    if isinstance(studio, dict):
        payload = dict(studio)
    else:
        payload = {}
    renderer = str(payload.get("renderer") or manifest.get("requested_view") or "").strip()
    if renderer:
        payload["renderer"] = renderer
    return payload


def assemble_analysis_response_from_vcf_context(context: dict[str, Any]) -> AnalysisResponse:
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


def workflow_answer_tokens(
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

    if source_type == "text" and isinstance(analysis, TextSourceResponse):
        tokens.update(
            {
                "active_file": analysis.file_name,
                "logged_tools": _stringify_logged_tools(getattr(analysis, "used_tools", []) or []),
                "line_count": analysis.line_count,
                "word_count": analysis.word_count,
                "char_count": analysis.char_count,
            }
        )
        return tokens

    return tokens


def format_workflow_answer(
    manifest: dict[str, object],
    source_type: str,
    analysis: object,
    context: dict[str, Any],
) -> str:
    workflow_name = str(manifest.get("name") or "")
    requested_view = str(manifest.get("requested_view") or "")
    template = str(manifest.get("answer_template") or "").strip()
    tokens = workflow_answer_tokens(source_type, workflow_name, requested_view, analysis, context)
    return template.format_map(_SafeFormatDict(tokens))


def build_summary_stats_workflow_result(
    analysis: SummaryStatsResponse,
    manifest: dict[str, object],
    context: dict[str, Any],
) -> dict[str, object]:
    requested_view = str(manifest.get("requested_view") or "sumstats")
    prs_prep_result = context.get("prs_prep_result")
    refreshed = (
        analysis.model_copy(update={"prs_prep_result": prs_prep_result})
        if isinstance(prs_prep_result, PrsPrepResponse)
        else context["analysis"]
    )
    answer = format_workflow_answer(manifest, "summary_stats", refreshed, context)
    result: dict[str, object] = {
        "answer": answer,
        "analysis": refreshed,
        "requested_view": requested_view,
        "studio": workflow_studio_metadata(manifest),
    }
    if isinstance(prs_prep_result, PrsPrepResponse):
        result["prs_prep_result"] = prs_prep_result
    return result


def build_raw_qc_workflow_result(
    manifest: dict[str, object],
    context: dict[str, Any],
) -> dict[str, object]:
    refreshed: RawQcResponse = context["analysis"]
    requested_view = str(manifest.get("requested_view") or "rawqc")
    answer = format_workflow_answer(manifest, "raw_qc", refreshed, context)
    return {
        "answer": answer,
        "analysis": refreshed,
        "requested_view": requested_view,
        "studio": workflow_studio_metadata(manifest),
    }


def build_analysis_workflow_result(
    manifest: dict[str, object],
    analysis: AnalysisResponse,
) -> dict[str, object]:
    requested_view = str(manifest.get("requested_view") or "summary")
    answer = format_workflow_answer(manifest, "vcf", analysis, {})
    return {
        "answer": answer,
        "analysis": analysis,
        "requested_view": requested_view,
        "studio": workflow_studio_metadata(manifest),
    }


def build_text_workflow_result(
    manifest: dict[str, object],
    context: dict[str, Any],
) -> dict[str, object]:
    refreshed: TextSourceResponse = context["analysis"]
    requested_view = str(manifest.get("requested_view") or "text")
    answer = format_workflow_answer(manifest, "text", refreshed, context)
    return {
        "answer": answer,
        "analysis": refreshed,
        "requested_view": requested_view,
        "studio": workflow_studio_metadata(manifest),
    }
