from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pysam import bcftools

from app.models import FilterRequest, FilterResponse


FILTER_OUTPUT_DIR = Path(
    os.getenv(
        "FILTER_OUTPUT_DIR",
        "/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/outputs/filters",
    )
)
LOCAL_JAVA = Path(
    os.getenv(
        "LOCAL_JAVA_BIN",
        "/Users/jongcye/Documents/Codex/.local/java/jdk-17.0.18+8/Contents/Home/bin/java",
    )
)
LOCAL_GATK_JAR = Path(
    os.getenv(
        "LOCAL_GATK_JAR",
        "/Users/jongcye/Documents/Codex/.local/gatk/gatk-4.6.2.0/gatk-package-4.6.2.0-local.jar",
    )
)


from app.utils.parsing import safe_prefix as _safe_prefix


def _ensure_paths(request: FilterRequest) -> tuple[Path, Path]:
    input_path = Path(request.vcf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"VCF not found: {request.vcf_path}")

    FILTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(request.output_prefix, request.vcf_path, request.tool)
    output_path = FILTER_OUTPUT_DIR / f"{prefix}.{request.tool}.filtered.vcf.gz"
    return input_path, output_path


def _ensure_tabix_index(path: Path) -> str | None:
    tbi = Path(f"{path}.tbi")
    csi = Path(f"{path}.csi")
    if tbi.exists():
        return str(tbi)
    if csi.exists():
        return str(csi)
    bcftools.index("-f", "-t", str(path))
    return str(tbi) if tbi.exists() else (str(csi) if csi.exists() else None)


def run_bcftools_filter(request: FilterRequest) -> FilterResponse:
    input_path, output_path = _ensure_paths(request)

    if request.mode == "soft_filter":
        args = [
            "-s",
            request.filter_name,
            "-e",
            request.expression,
            "-m",
            "+",
            "-O",
            "z",
            "-o",
            str(output_path),
            str(input_path),
        ]
        command_preview = f"bcftools filter -s {request.filter_name} -e '{request.expression}' -m + -O z -o {output_path} {input_path}"
    elif request.mode == "include":
        args = [
            "-i",
            request.expression,
            "-O",
            "z",
            "-o",
            str(output_path),
            str(input_path),
        ]
        command_preview = f"bcftools filter -i '{request.expression}' -O z -o {output_path} {input_path}"
    else:
        args = [
            "-e",
            request.expression,
            "-O",
            "z",
            "-o",
            str(output_path),
            str(input_path),
        ]
        command_preview = f"bcftools filter -e '{request.expression}' -O z -o {output_path} {input_path}"

    result = bcftools.filter(*args)
    if isinstance(result, (bytes, bytearray)):
        output_path.write_bytes(result)
    bcftools.index("-f", "-t", str(output_path))

    return FilterResponse(
        tool="bcftools",
        input_path=str(input_path),
        output_path=str(output_path),
        index_path=f"{output_path}.tbi",
        command_preview=command_preview,
    )


def run_gatk_variant_filtration(request: FilterRequest) -> FilterResponse:
    if request.mode != "soft_filter":
        raise ValueError("GATK VariantFiltration currently supports only soft_filter mode in this app.")
    input_path, output_path = _ensure_paths(request)

    if not LOCAL_JAVA.exists():
        raise FileNotFoundError(f"Local Java runtime not found: {LOCAL_JAVA}")
    if not LOCAL_GATK_JAR.exists():
        raise FileNotFoundError(f"Local GATK jar not found: {LOCAL_GATK_JAR}")
    _ensure_tabix_index(input_path)

    cmd = [
        str(LOCAL_JAVA),
        "-jar",
        str(LOCAL_GATK_JAR),
        "VariantFiltration",
        "-V",
        str(input_path),
        "-O",
        str(output_path),
        "--filter-name",
        request.filter_name,
        "--filter-expression",
        request.expression,
        "--create-output-variant-index",
        "true",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    return FilterResponse(
        tool="gatk",
        input_path=str(input_path),
        output_path=str(output_path),
        index_path=f"{output_path}.tbi" if Path(f"{output_path}.tbi").exists() else None,
        command_preview=" ".join(cmd),
    )


def run_filter(request: FilterRequest) -> FilterResponse:
    if request.tool == "bcftools":
        return run_bcftools_filter(request)
    return run_gatk_variant_filtration(request)


def execute(payload: dict[str, object]) -> FilterResponse:
    request = FilterRequest(**payload)
    return run_filter(request)
