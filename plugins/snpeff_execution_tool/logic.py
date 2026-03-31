from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pysam

from app.models import (
    SnpEffAnnotatedRecord,
    SnpEffAnnEntry,
    SnpEffRequest,
    SnpEffResponse,
)


SNPEFF_OUTPUT_DIR = Path(
    os.getenv(
        "SNPEFF_OUTPUT_DIR",
        "/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/outputs/snpeff",
    )
)
SNPEFF_JAVA = Path(
    os.getenv(
        "SNPEFF_JAVA_BIN",
        "/Users/jongcye/Documents/Codex/.local/java/jdk-21.0.10+7/Contents/Home/bin/java",
    )
)
SNPEFF_JAR = Path(
    os.getenv(
        "SNPEFF_JAR",
        "/Users/jongcye/Documents/Codex/.local/snpeff/snpEff/snpEff.jar",
    )
)
SNPEFF_DATA_DIR = Path(
    os.getenv(
        "SNPEFF_DATA_DIR",
        "/Users/jongcye/Documents/Codex/.local/snpeff/snpEff/data",
    )
)


from app.utils.parsing import safe_prefix as _safe_prefix_base


def _safe_prefix(prefix: str | None, source_path: str) -> str:
    return _safe_prefix_base(prefix, source_path, tool="snpeff")


def _ensure_inputs(request: SnpEffRequest) -> tuple[Path, Path, Path]:
    input_path = Path(request.vcf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"VCF not found: {request.vcf_path}")
    if not SNPEFF_JAVA.exists():
        raise FileNotFoundError(f"SnpEff Java runtime not found: {SNPEFF_JAVA}")
    if not SNPEFF_JAR.exists():
        raise FileNotFoundError(f"SnpEff jar not found: {SNPEFF_JAR}")
    if not (SNPEFF_DATA_DIR / request.genome).exists():
        raise FileNotFoundError(f"SnpEff genome database not found: {request.genome}")

    SNPEFF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(request.output_prefix, request.vcf_path)
    plain_vcf = SNPEFF_OUTPUT_DIR / f"{prefix}.{request.genome}.ann.vcf"
    gz_vcf = SNPEFF_OUTPUT_DIR / f"{prefix}.{request.genome}.ann.vcf.gz"
    return input_path, plain_vcf, gz_vcf


def _parse_ann_value(value: str) -> SnpEffAnnEntry:
    fields = (value or "").split("|")
    padded = fields + [""] * (11 - len(fields))
    return SnpEffAnnEntry(
        allele=padded[0] or ".",
        annotation=padded[1] or ".",
        impact=padded[2] or ".",
        gene_name=padded[3] or ".",
        gene_id=padded[4] or ".",
        feature_type=padded[5] or ".",
        feature_id=padded[6] or ".",
        transcript_biotype=padded[7] or ".",
        rank=padded[8] or ".",
        hgvs_c=padded[9] or ".",
        hgvs_p=padded[10] or ".",
    )


def parse_snpeff_ann(vcf_path: str, limit: int = 25) -> list[SnpEffAnnotatedRecord]:
    records: list[SnpEffAnnotatedRecord] = []
    with pysam.VariantFile(vcf_path) as vcf:
        for rec in vcf:
            ann_values = rec.info.get("ANN")
            if not ann_values:
                continue
            if isinstance(ann_values, str):
                ann_list = [ann_values]
            else:
                ann_list = list(ann_values)
            first_alt = (rec.alts or ["."])[0]
            records.append(
                SnpEffAnnotatedRecord(
                    contig=rec.contig,
                    pos_1based=rec.pos,
                    ref=rec.ref,
                    alt=first_alt,
                    ann=[_parse_ann_value(value) for value in ann_list[:10]],
                )
            )
            if len(records) >= limit:
                break
    return records


def run_snpeff(request: SnpEffRequest) -> SnpEffResponse:
    input_path, plain_vcf, gz_vcf = _ensure_inputs(request)
    cmd = [
        str(SNPEFF_JAVA),
        "-jar",
        str(SNPEFF_JAR),
        "ann",
        "-dataDir",
        str(SNPEFF_DATA_DIR),
        request.genome,
        str(input_path),
    ]

    with plain_vcf.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.PIPE, text=True)

    pysam.tabix_compress(str(plain_vcf), str(gz_vcf), force=True)
    pysam.tabix_index(str(gz_vcf), preset="vcf", force=True)
    plain_vcf.unlink(missing_ok=True)

    return SnpEffResponse(
        tool="snpeff",
        genome=request.genome,
        input_path=str(input_path),
        output_path=str(gz_vcf),
        index_path=f"{gz_vcf}.tbi",
        command_preview=" ".join(cmd),
        parsed_records=parse_snpeff_ann(str(gz_vcf), limit=request.parse_limit),
    )


def execute(payload: dict[str, object]) -> SnpEffResponse:
    request = SnpEffRequest(**payload)
    return run_snpeff(request)
