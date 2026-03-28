from __future__ import annotations

from plugins.clinical_coverage_tool.logic import execute as execute_clinical_coverage_review
from plugins.clinvar_review_tool.logic import execute as execute_clinvar_review
from plugins.grounded_summary_tool.logic import execute as execute_grounded_summary
from plugins.symbolic_alt_tool.logic import execute as execute_symbolic_alt_review
from plugins.vep_consequence_tool.logic import execute as execute_vep_consequence_review


def execute(payload: dict[str, object]) -> dict[str, object]:
    facts = payload["facts"]
    annotations = list(payload.get("annotations", []))
    candidate_variants = list(payload.get("candidate_variants", []))
    references = list(payload.get("references", []))
    recommendations = list(payload.get("recommendations", []))

    clinvar_summary = execute_clinvar_review({"annotations": annotations}).get("clinvar_summary", [])
    consequence_summary = execute_vep_consequence_review({"annotations": annotations, "limit": 10}).get(
        "consequence_summary",
        [],
    )
    clinical_coverage_summary = execute_clinical_coverage_review({"annotations": annotations}).get(
        "clinical_coverage_summary",
        [],
    )
    symbolic_alt_summary = execute_symbolic_alt_review({"annotations": annotations}).get("symbolic_alt_summary", {})
    draft_answer = execute_grounded_summary(
        {
            "facts": facts,
            "annotations": annotations,
            "references": references,
            "recommendations": recommendations,
        }
    ).get("draft_answer", "")

    return {
        "tool": "vcf_review_tool",
        "clinvar_summary": clinvar_summary,
        "consequence_summary": consequence_summary,
        "clinical_coverage_summary": clinical_coverage_summary,
        "symbolic_alt_summary": symbolic_alt_summary,
        "draft_answer": draft_answer,
        "candidate_variants": candidate_variants,
        "review_annotation_count": len(annotations),
        "candidate_count": len(candidate_variants),
        "summary": (
            f"Reviewed {len(annotations)} annotation(s): "
            f"{len(candidate_variants)} candidate(s), "
            f"{len(clinvar_summary)} ClinVar bucket(s), "
            f"{len(consequence_summary)} consequence bucket(s)."
        ),
        "omitted_tools": ["filtering_view_tool"],
    }
