from __future__ import annotations

from typing import Any, Callable

from app.models import PrsPrepResponse, RawQcResponse, SummaryStatsResponse


def execute_summary_stats_internal_step(
    tool_name: str,
    context: dict[str, Any],
    bind_name: str,
    *,
    analyze_summary_stats_workflow: Callable[[str, str, str, str], SummaryStatsResponse],
    analyze_prs_prep_workflow: Callable[[str, str, str], PrsPrepResponse],
) -> None:
    if tool_name == "summary_stats_review_engine":
        context[bind_name] = analyze_summary_stats_workflow(
            str(context["source_stats_path"] or ""),
            str(context["file_name"]),
            str(context["genome_build"]),
            str(context["trait_type"]),
        )
        return

    if tool_name == "summary_stats_draft_answer":
        analysis: SummaryStatsResponse = context["analysis"]
        context[bind_name] = analysis.draft_answer
        return

    if tool_name == "prs_prep_engine":
        context[bind_name] = analyze_prs_prep_workflow(
            str(context["source_stats_path"] or ""),
            str(context["file_name"]),
            str(context["genome_build"]),
        )
        return

    if tool_name == "prs_score_file_status":
        prs_prep_result: PrsPrepResponse = context["prs_prep_result"]
        context[bind_name] = prs_prep_result.score_file_ready
        return

    raise NotImplementedError(f"Unsupported summary-statistics workflow step tool: {tool_name}")


def execute_raw_qc_internal_step(
    tool_name: str,
    context: dict[str, Any],
    bind_name: str,
    *,
    analyze_raw_qc_workflow: Callable[[str, str], RawQcResponse],
) -> None:
    if tool_name == "raw_qc_review_engine":
        context[bind_name] = analyze_raw_qc_workflow(
            str(context["source_raw_path"] or ""),
            str(context["file_name"]),
        )
        return

    raise NotImplementedError(f"Unsupported raw-QC workflow step tool: {tool_name}")
