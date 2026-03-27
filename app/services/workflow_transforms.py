from __future__ import annotations

from typing import Any

from app.models import (
    AnalysisFacts,
    CountSummaryItem,
    DetailedCountSummaryItem,
    PrsPrepResponse,
    RawQcResponse,
    RankedCandidate,
    RohSegment,
    SnpEffResponse,
    SummaryStatsResponse,
    SymbolicAltSummary,
    VariantAnnotation,
)


def transform_bound_value(transform: str, value: Any) -> Any:
    normalized = transform.strip().lower()
    if normalized in {"", "identity"}:
        return value
    if normalized == "analysis_facts":
        return AnalysisFacts(**dict(value or {}))
    if normalized == "variant_annotation_list":
        return [VariantAnnotation(**item) for item in list(value or [])]
    if normalized == "roh_segment_list":
        return [RohSegment(**item) for item in list(value or [])]
    if normalized == "ranked_candidate_list":
        return [RankedCandidate(**item) for item in list(value or [])]
    if normalized == "count_summary_list":
        return [CountSummaryItem(**item) for item in list(value or [])]
    if normalized == "detailed_count_summary_list":
        return [DetailedCountSummaryItem(**item) for item in list(value or [])]
    if normalized == "symbolic_alt_summary":
        return SymbolicAltSummary(**dict(value or {}))
    if normalized == "snpeff_response":
        return SnpEffResponse(**dict(value or {}))
    if normalized == "summary_stats_response":
        return SummaryStatsResponse(**dict(value or {}))
    if normalized == "prs_prep_response":
        return PrsPrepResponse(**dict(value or {}))
    if normalized == "raw_qc_response":
        return RawQcResponse(**dict(value or {}))
    raise NotImplementedError(f"Unsupported workflow binding transform: {transform}")
