from __future__ import annotations

import os
import uuid
from typing import Any

from app.models import (
    AnalysisFacts,
    AnalysisResponse,
    DicomSourceResponse,
    FhirSourceResponse,
    ImageSourceResponse,
    PrsPrepResponse,
    RawQcResponse,
    SpreadsheetSourceResponse,
    SummaryStatsResponse,
    SymbolicAltSummary,
    TextSourceResponse,
)
from app.services.tool_runner import discover_tools, run_tool
from app.services.workflow_responses import assemble_analysis_response_from_vcf_context
from plugins.fastqc_execution_tool.logic import FASTQC_OUTPUT_DIR
from plugins.cohort_sheet_browser_tool.logic import analyze_spreadsheet_source
from plugins.dicom_review_tool.logic import analyze_dicom_source
from plugins.fhir_browser_tool.logic import analyze_fhir_source
from plugins.image_review_tool.logic import analyze_image_source
from plugins.prs_prep_tool.logic import analyze_prs_prep
from plugins.summary_stats_review_tool.logic import analyze_summary_stats
from plugins.text_review_tool.logic import analyze_text_source


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


def analyze_vcf_workflow(
    path: str,
    annotation_scope: str = "representative",
    annotation_limit: int | None = None,
) -> AnalysisResponse:
    max_examples = int(os.getenv("MAX_EXAMPLE_VARIANTS", "8"))
    result = run_tool("vcf_qc_tool", {"vcf_path": path, "max_examples": max_examples})
    facts = AnalysisFacts(**dict(result.get("facts") or {}))
    context = _vcf_workflow_context(path, annotation_scope=annotation_scope, annotation_limit=annotation_limit)
    context["facts"] = facts
    context["used_tools"].append("vcf_qc_tool")
    response = assemble_analysis_response_from_vcf_context(context)
    response.studio = {"renderer": "qc"}
    response.requested_view = "qc"
    return response


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


def analyze_text_workflow(path: str, original_name: str) -> TextSourceResponse:
    result = analyze_text_source(path, original_name)
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_spreadsheet_workflow(path: str, original_name: str) -> SpreadsheetSourceResponse:
    result = analyze_spreadsheet_source(path, original_name)
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_dicom_workflow(path: str, original_name: str) -> DicomSourceResponse:
    result = analyze_dicom_source(path, original_name)
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_fhir_workflow(path: str, original_name: str) -> FhirSourceResponse:
    result = analyze_fhir_source(path, original_name)
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_image_workflow(path: str, original_name: str) -> ImageSourceResponse:
    result = analyze_image_source(path, original_name)
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
