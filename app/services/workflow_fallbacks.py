from __future__ import annotations

from typing import Any

from app.models import AnalysisFacts, CountSummaryItem, DetailedCountSummaryItem, SymbolicAltSummary
from app.services.annotation import build_draft_answer, build_ui_cards
from app.services.recommendation import build_recommendations
from app.services.references import build_reference_bundle
from plugins.annotation_tool.logic import annotate_variants
from plugins.candidate_ranking_tool.logic import build_ranked_candidates
from plugins.roh_analysis_tool.logic import run_roh_analysis
from plugins.vcf_qc_tool.logic import summarize_vcf


def compute_vcf_fallback_value(transform: str, context: dict[str, Any]) -> Any:
    normalized = transform.strip().lower()
    if normalized == "vcf_qc_summary":
        return summarize_vcf(str(context["source_vcf_path"]), max_examples=context["max_examples"])
    if normalized == "annotation_local":
        return annotate_variants(
            str(context["source_vcf_path"]),
            context["facts"],
            scope=context["annotation_scope"],
            limit=context["annotation_limit"],
        )
    if normalized == "roh_local":
        return run_roh_analysis(str(context["source_vcf_path"]))
    if normalized == "candidate_ranking_local":
        return build_ranked_candidates(context["annotations"], context["roh_segments"], limit=8)
    if normalized == "clinvar_summary_local":
        counts: dict[str, int] = {}
        for item in context["annotations"]:
            key = (
                item.clinical_significance.strip()
                if item.clinical_significance and item.clinical_significance != "."
                else "Unreviewed"
            )
            counts[key] = counts.get(key, 0) + 1
        return [
            CountSummaryItem(label=label, count=count)
            for label, count in sorted(counts.items(), key=lambda part: part[1], reverse=True)
        ]
    if normalized == "vep_consequence_local":
        counts: dict[str, int] = {}
        for item in context["annotations"]:
            key = item.consequence.strip() if item.consequence and item.consequence != "." else "Unclassified"
            counts[key] = counts.get(key, 0) + 1
        return [
            CountSummaryItem(label=label, count=count)
            for label, count in sorted(counts.items(), key=lambda part: part[1], reverse=True)[:10]
        ]
    if normalized == "clinical_coverage_local":
        annotations = list(context["annotations"])
        total = len(annotations)

        def detail(label: str, count: int) -> DetailedCountSummaryItem:
            percent = round((count / total) * 100) if total else 0
            return DetailedCountSummaryItem(label=label, count=count, detail=f"{count}/{total} annotated ({percent}%)")

        return [
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
                sum(
                    1
                    for item in annotations
                    if (item.hgvsc and item.hgvsc != ".") or (item.hgvsp and item.hgvsp != ".")
                ),
            ),
            detail("Protein change", sum(1 for item in annotations if item.hgvsp and item.hgvsp != ".")),
        ]
    if normalized == "filtering_view_local":
        annotations = list(context["annotations"])
        unique_genes = {item.gene.strip() for item in annotations if item.gene and item.gene.strip() not in {"", "."}}
        clinvar_labeled = sum(1 for item in annotations if item.clinical_significance and item.clinical_significance != ".")
        symbolic = sum(1 for item in annotations if any(alt.startswith("<") and alt.endswith(">") for alt in item.alts))
        return [
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
    if normalized == "symbolic_alt_local":
        symbolic_items = [item for item in context["annotations"] if any(alt.startswith("<") and alt.endswith(">") for alt in item.alts)]
        return SymbolicAltSummary(
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
    if normalized == "grounded_summary_local":
        facts: AnalysisFacts = context["facts"]
        annotations = list(context["annotations"])
        references = build_reference_bundle(facts, annotations[: min(len(annotations), 20)])
        recommendations = build_recommendations(facts)
        ui_cards = build_ui_cards(facts, annotations)
        context["references"] = references
        context["recommendations"] = recommendations
        context["ui_cards"] = ui_cards
        return build_draft_answer(
            facts,
            annotations,
            [item.id for item in references],
            [item.id for item in recommendations],
        )
    raise NotImplementedError(f"Unsupported VCF fallback transform: {transform}")
