from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import pysam

from app.models import AnalysisFacts, QualityControlMetrics, VariantExample


def _classify_variant(ref: str, alt: str) -> str:
    if alt.startswith("<") and alt.endswith(">"):
        return "symbolic"
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    if len(ref) < len(alt):
        return "INS"
    if len(ref) > len(alt):
        return "DEL"
    return "MNV_or_complex"


def _guess_build(contigs: list[dict[str, int | None]]) -> str | None:
    contig_map = {item["name"]: item["length"] for item in contigs}
    if contig_map.get("1") == 249250621:
        return "GRCh37 (inferred from chr1 length)"
    if contig_map.get("chr1") == 249250621:
        return "GRCh37 (inferred from chr1 length)"
    if contig_map.get("1") == 248956422:
        return "GRCh38 (inferred from chr1 length)"
    if contig_map.get("chr1") == 248956422:
        return "GRCh38 (inferred from chr1 length)"
    return None


def _is_transition(ref: str, alt: str) -> bool:
    return (ref, alt) in {
        ("A", "G"),
        ("G", "A"),
        ("C", "T"),
        ("T", "C"),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


def _safe_mean(total: int | float, count: int) -> float | None:
    if not count:
        return None
    return total / count


def summarize_vcf(path: str, max_examples: int = 8) -> AnalysisFacts:
    vcf_path = Path(path)
    warnings: list[str] = []
    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF not found: {path}")

    index_candidates = [f"{path}.tbi", f"{path}.csi"]
    if not any(Path(candidate).exists() for candidate in index_candidates):
        warnings.append(
            "No tabix/CSI index was found. Whole-file parsing works, but region fetches will be unavailable."
        )

    with pysam.VariantFile(path) as vcf:
        sample_names = list(vcf.header.samples)
        first_sample = sample_names[0] if sample_names else None
        contigs = [
            {"name": name, "length": vcf.header.contigs[name].length}
            for name in vcf.header.contigs
        ]

        chrom_counts = Counter()
        variant_types = Counter()
        genotype_counts = Counter()
        filter_counts = Counter()
        multi_allelic_count = 0
        symbolic_record_count = 0
        snv_record_count = 0
        indel_record_count = 0
        transition_count = 0
        transversion_count = 0
        missing_gt_count = 0
        het_count = 0
        hom_alt_count = 0
        dp_total = 0
        dp_count = 0
        gq_total = 0
        gq_count = 0
        positions: list[int] = []
        examples: list[VariantExample] = []
        representative_examples: dict[str, VariantExample] = {}
        recent_examples: deque[VariantExample] = deque(maxlen=3)

        def build_example() -> VariantExample:
            return VariantExample(
                contig=rec.contig,
                pos_1based=rec.pos,
                ref=rec.ref,
                alts=alts,
                genotype=genotype,
                variant_class=variant_class,
            )

        for rec in vcf:
            chrom_counts[rec.contig] += 1
            positions.append(rec.pos)

            filters = list(rec.filter.keys()) or ["."]
            for value in filters:
                filter_counts[value] += 1

            alts = list(rec.alts or [])
            if len(alts) > 1:
                multi_allelic_count += 1
            if not alts:
                variant_class = "no_alt"
            else:
                classes = sorted({_classify_variant(rec.ref, alt) for alt in alts})
                variant_class = ",".join(classes)
                if any(item == "symbolic" for item in classes):
                    symbolic_record_count += 1
                if any(item == "SNV" for item in classes):
                    snv_record_count += 1
                if any(item in {"INS", "DEL", "MNV_or_complex"} for item in classes):
                    indel_record_count += 1
                for alt in alts:
                    variant_type = _classify_variant(rec.ref, alt)
                    variant_types[variant_type] += 1
                    if variant_type == "SNV" and len(rec.ref) == 1 and len(alt) == 1:
                        if _is_transition(rec.ref, alt):
                            transition_count += 1
                        else:
                            transversion_count += 1

            genotype = "."
            if first_sample:
                sample_call = rec.samples[first_sample]
                gt = sample_call.get("GT")
                genotype = "." if gt is None else "/".join("." if x is None else str(x) for x in gt)
                genotype_counts[genotype] += 1
                gt_values = tuple(x for x in gt or () if x is not None)
                if not gt_values or any(x is None for x in (gt or ())):
                    missing_gt_count += 1
                elif gt_values == (0, 1) or gt_values == (1, 0):
                    het_count += 1
                elif gt_values == (1, 1):
                    hom_alt_count += 1

                dp_value = sample_call.get("DP")
                if isinstance(dp_value, int):
                    dp_total += dp_value
                    dp_count += 1
                gq_value = sample_call.get("GQ")
                if isinstance(gq_value, int):
                    gq_total += gq_value
                    gq_count += 1

            example_record = build_example()
            recent_examples.append(example_record)

            if len(examples) < min(4, max_examples):
                examples.append(example_record)

            if genotype == "1/1" and "first_homozygous_alt" not in representative_examples:
                previous_context = [item for item in list(recent_examples)[:-1] if item.genotype != "1/1"]
                if previous_context:
                    representative_examples["context_before_first_homozygous_alt"] = previous_context[-1]
                representative_examples["first_homozygous_alt"] = example_record
            if "symbolic" in variant_class and "first_symbolic" not in representative_examples:
                representative_examples["first_symbolic"] = example_record
            if variant_class != "SNV" and "first_non_snv" not in representative_examples:
                representative_examples["first_non_snv"] = example_record
            if genotype == "0/1" and "first_heterozygous" not in representative_examples:
                representative_examples["first_heterozygous"] = example_record
            if genotype == "0/0" and "first_homozygous_ref" not in representative_examples:
                representative_examples["first_homozygous_ref"] = example_record

        for example in representative_examples.values():
            if len(examples) >= max_examples:
                break
            if any(
                current.contig == example.contig
                and current.pos_1based == example.pos_1based
                and current.ref == example.ref
                for current in examples
            ):
                continue
            examples.append(example)

        if not positions:
            warnings.append("The file contained no variant records.")

        record_count = sum(chrom_counts.values())
        qc = QualityControlMetrics(
            pass_rate=_safe_ratio(filter_counts.get("PASS", 0), record_count),
            missing_gt_rate=_safe_ratio(missing_gt_count, record_count) if first_sample else None,
            multi_allelic_rate=_safe_ratio(multi_allelic_count, record_count),
            symbolic_alt_rate=_safe_ratio(symbolic_record_count, record_count),
            snv_fraction=_safe_ratio(snv_record_count, record_count),
            indel_fraction=_safe_ratio(indel_record_count, record_count),
            transition_transversion_ratio=_safe_ratio(transition_count, transversion_count),
            het_hom_alt_ratio=_safe_ratio(het_count, hom_alt_count) if first_sample else None,
            mean_dp=_safe_mean(dp_total, dp_count),
            mean_gq=_safe_mean(gq_total, gq_count),
            records_with_dp_rate=_safe_ratio(dp_count, record_count) if first_sample else None,
            records_with_gq_rate=_safe_ratio(gq_count, record_count) if first_sample else None,
        )

        if qc.pass_rate is not None and qc.pass_rate < 0.95:
            warnings.append("PASS rate is below 95%, so downstream interpretation should consider failing filters.")
        if qc.missing_gt_rate is not None and qc.missing_gt_rate > 0.05:
            warnings.append("Missing genotype rate is above 5%, which may indicate low sample callability.")
        if qc.transition_transversion_ratio is not None and qc.transition_transversion_ratio < 1.5:
            warnings.append("Ti/Tv is lower than expected for a typical germline SNV callset.")

        return AnalysisFacts(
            file_name=vcf_path.name,
            vcf_version=vcf.header.version,
            genome_build_guess=_guess_build(contigs),
            samples=sample_names,
            contigs=contigs[:25],
            record_count=record_count,
            chrom_counts=dict(chrom_counts),
            variant_types=dict(variant_types),
            genotype_counts=dict(genotype_counts),
            filter_counts=dict(filter_counts),
            qc=qc,
            position_range_1based=[min(positions), max(positions)] if positions else [],
            example_variants=examples,
            warnings=warnings,
        )


def execute(payload: dict[str, object]) -> dict[str, object]:
    vcf_path = str(payload["vcf_path"])
    max_examples = int(payload.get("max_examples", 8))
    facts = summarize_vcf(vcf_path, max_examples=max_examples)
    return {
        "tool": "vcf_qc_tool",
        "facts": facts.model_dump(),
        "summary": (
            f"Summarized {facts.file_name}: {facts.record_count} record(s), "
            f"{len(facts.samples)} sample(s), build guess {facts.genome_build_guess or 'unknown'}."
        ),
    }
