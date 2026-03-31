from __future__ import annotations

import uuid
from typing import Any

from app.models import AnalysisFacts, AnalysisResponse
from app.services.annotation import build_ui_cards
from app.services.recommendation import build_recommendations
from app.services.references import build_reference_bundle
from app.services.source_registry import source_response_metadata


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
        **source_response_metadata("vcf"),
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
