from __future__ import annotations

import gzip
import os
import subprocess
from pathlib import Path

from app.models import GatkLiftoverRecord, GatkLiftoverVcfRequest, GatkLiftoverVcfResponse


ROOT_DIR = Path(__file__).resolve().parents[2]
LIFTOVER_OUTPUT_DIR = ROOT_DIR / "outputs" / "liftover"
DEFAULT_CHAIN_FILE = Path(
    os.getenv(
        "GATK_LIFTOVER_CHAIN",
        str(ROOT_DIR / "references" / "liftover" / "chains" / "hg19ToHg38.over.chain.gz"),
    )
)
DEFAULT_TARGET_FASTA = Path(
    os.getenv(
        "GATK_LIFTOVER_TARGET_FASTA",
        str(ROOT_DIR / "references" / "liftover" / "GRCh38" / "hg38.fa"),
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


from app.utils.parsing import safe_prefix as _safe_prefix_base


def _safe_prefix(prefix: str | None, source_path: str) -> str:
    return _safe_prefix_base(prefix, source_path, tool="liftover")


def _count_vcf_records(path: Path) -> int:
    if not path.exists():
        return 0
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            count += 1
    return count


def _parse_preview_records(path: Path, limit: int) -> list[GatkLiftoverRecord]:
    if not path.exists() or limit <= 0:
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[GatkLiftoverRecord] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            rows.append(
                GatkLiftoverRecord(
                    contig=parts[0],
                    pos_1based=int(parts[1]),
                    ref=parts[3],
                    alts=parts[4].split(","),
                )
            )
            if len(rows) >= limit:
                break
    return rows


def run_gatk_liftover_vcf(request: GatkLiftoverVcfRequest) -> GatkLiftoverVcfResponse:
    input_path = Path(request.vcf_path)
    target_reference = Path(request.target_reference_fasta)
    chain_file = Path(request.chain_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input VCF not found: {request.vcf_path}")
    if not target_reference.exists():
        raise FileNotFoundError(f"Target reference FASTA not found: {request.target_reference_fasta}")
    if not chain_file.exists():
        raise FileNotFoundError(f"Chain file not found: {request.chain_file}")
    if not LOCAL_JAVA.exists():
        raise FileNotFoundError(f"Java runtime not found: {LOCAL_JAVA}")
    if not LOCAL_GATK_JAR.exists():
        raise FileNotFoundError(f"GATK jar not found: {LOCAL_GATK_JAR}")

    LIFTOVER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(request.output_prefix, request.vcf_path)
    output_path = LIFTOVER_OUTPUT_DIR / f"{prefix}.{request.target_build or 'target'}.lifted.vcf.gz"
    reject_path = LIFTOVER_OUTPUT_DIR / f"{prefix}.{request.target_build or 'target'}.reject.vcf.gz"

    cmd = [
        str(LOCAL_JAVA),
        "-jar",
        str(LOCAL_GATK_JAR),
        "LiftoverVcf",
        "-I",
        str(input_path),
        "-O",
        str(output_path),
        "-CHAIN",
        str(chain_file),
        "-REJECT",
        str(reject_path),
        "-R",
        str(target_reference),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)

    warnings = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
    output_index = output_path.with_suffix(output_path.suffix + ".tbi")
    reject_index = reject_path.with_suffix(reject_path.suffix + ".tbi")

    return GatkLiftoverVcfResponse(
        tool="gatk-liftover-vcf",
        input_path=str(input_path),
        source_build=request.source_build,
        target_build=request.target_build,
        target_reference_fasta=str(target_reference),
        chain_file=str(chain_file),
        output_path=str(output_path),
        output_index_path=str(output_index) if output_index.exists() else None,
        reject_path=str(reject_path),
        reject_index_path=str(reject_index) if reject_index.exists() else None,
        command_preview=" ".join(cmd),
        lifted_record_count=_count_vcf_records(output_path),
        rejected_record_count=_count_vcf_records(reject_path),
        parsed_records=_parse_preview_records(output_path, request.parse_limit),
        warnings=warnings,
    )


def execute(payload: dict[str, object]) -> GatkLiftoverVcfResponse:
    request = GatkLiftoverVcfRequest(**payload)
    return run_gatk_liftover_vcf(request)
