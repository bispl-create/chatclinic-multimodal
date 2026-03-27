from __future__ import annotations

from pathlib import Path
from typing import Any

from app.models import AnalysisFacts, VariantAnnotation
from app.services.annotation import build_ui_cards
from app.services.recommendation import build_recommendations
from app.services.references import build_reference_bundle
from plugins.candidate_ranking_tool.logic import build_ranked_candidates


def snpeff_genome_from_build(genome_build_guess: str | None) -> str:
    value = (genome_build_guess or "").lower()
    if any(token in value for token in ("38", "hg38", "grch38")):
        return "GRCh38.99"
    return "GRCh37.75"


def annotation_key(item: VariantAnnotation) -> tuple[str, int, str, tuple[str, ...]]:
    return (item.contig, item.pos_1based, item.ref, tuple(item.alts))


def _vcf_context_shortlisted_annotations(context: dict[str, Any]) -> list[VariantAnnotation]:
    annotations = list(context.get("annotations") or [])
    roh_segments = list(context.get("roh_segments") or [])
    preliminary_candidates = build_ranked_candidates(annotations, roh_segments, limit=24)
    return [entry.item for entry in preliminary_candidates]


def apply_vcf_preprocess_hook(hook_name: str, context: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    normalized = hook_name.strip().lower()
    if normalized == "vcf_prepare_snpeff_payload":
        prepared = dict(payload)
        genome_build_guess = str(prepared.pop("genome_build_guess", "") or "")
        prepared["genome"] = snpeff_genome_from_build(genome_build_guess)
        prepared["output_prefix"] = f"{Path(str(context['source_vcf_path'])).stem}.aux"
        prepared["parse_limit"] = 10
        return prepared
    if normalized == "vcf_shortlist_annotations":
        prepared = dict(payload)
        prepared["annotations"] = [item.model_dump() for item in _vcf_context_shortlisted_annotations(context)]
        return prepared
    if normalized == "vcf_prepare_grounded_summary_inputs":
        facts: AnalysisFacts = context["facts"]
        annotations = list(context["annotations"])
        references = build_reference_bundle(facts, annotations[: min(len(annotations), 20)])
        recommendations = build_recommendations(facts)
        ui_cards = build_ui_cards(facts, annotations)
        context["references"] = references
        context["recommendations"] = recommendations
        context["ui_cards"] = ui_cards

        prepared = dict(payload)
        prepared["references"] = [item.model_dump() for item in references]
        prepared["recommendations"] = [item.model_dump() for item in recommendations]
        return prepared
    raise NotImplementedError(f"Unsupported VCF workflow preprocess hook: {hook_name}")


def apply_vcf_postprocess_hook(
    hook_name: str,
    context: dict[str, Any],
    result: dict[str, Any],
    transformed_value: Any,
) -> tuple[Any, bool]:
    normalized = hook_name.strip().lower()
    if normalized == "vcf_merge_lookup_annotations":
        annotations = list(context["annotations"])
        enriched_annotations = list(transformed_value or [])
        enriched_by_key = {annotation_key(item): item for item in enriched_annotations}
        merged_annotations = [enriched_by_key.get(annotation_key(item), item) for item in annotations]
        return merged_annotations, bool(result.get("lookup_performed"))
    raise NotImplementedError(f"Unsupported VCF workflow postprocess hook: {hook_name}")
