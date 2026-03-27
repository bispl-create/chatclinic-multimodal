from __future__ import annotations

from functools import lru_cache
from pathlib import Path


SOURCE_REGISTRY: dict[str, dict[str, object]] = {
    "raw_qc": {
        "upload_label": "raw sequencing file",
        "suffixes": [
            ".fastq.gz",
            ".fq.gz",
            ".fastq",
            ".fq",
            ".bam",
            ".sam",
        ],
        "file_kind_map": {
            ".fastq.gz": "FASTQ",
            ".fq.gz": "FASTQ",
            ".fastq": "FASTQ",
            ".fq": "FASTQ",
            ".bam": "BAM",
            ".sam": "SAM",
        },
    },
    "summary_stats": {
        "upload_label": "summary statistics file",
        "suffixes": [
            ".sumstats.gz",
            ".tsv.gz",
            ".txt.gz",
            ".csv.gz",
            ".sumstats",
            ".tsv",
            ".txt",
            ".csv",
        ],
    },
    "vcf": {
        "upload_label": "VCF file",
        "suffixes": [
            ".vcf.gz",
            ".vcf",
        ],
        "file_kind_map": {
            ".vcf.gz": "VCF",
            ".vcf": "VCF",
        },
    },
}


@lru_cache(maxsize=1)
def list_registered_source_types() -> tuple[str, ...]:
    return tuple(SOURCE_REGISTRY.keys())


def load_source_registration(source_type: str) -> dict[str, object] | None:
    return SOURCE_REGISTRY.get(source_type.strip().lower())


def detect_source_registration(file_name: str) -> tuple[str, dict[str, object], str] | None:
    lowered = file_name.strip().lower()
    if not lowered:
        return None
    for source_type, registration in SOURCE_REGISTRY.items():
        suffixes = registration.get("suffixes") or []
        if not isinstance(suffixes, list):
            continue
        for suffix in suffixes:
            suffix_text = str(suffix).strip().lower()
            if suffix_text and lowered.endswith(suffix_text):
                return source_type, registration, suffix_text
    return None


def detect_source_type(file_name: str) -> str | None:
    detected = detect_source_registration(file_name)
    return detected[0] if detected else None


def infer_source_file_kind(file_name: str, source_type: str, matched_suffix: str | None = None) -> str | None:
    registration = load_source_registration(source_type)
    if registration is None:
        return None
    suffix = matched_suffix or "".join(Path(file_name).suffixes).lower() or Path(file_name).suffix.lower()
    file_kind_map = registration.get("file_kind_map") or {}
    if isinstance(file_kind_map, dict):
        matched = file_kind_map.get(suffix)
        if isinstance(matched, str) and matched.strip():
            return matched.strip()
    if source_type == "raw_qc":
        simple = Path(file_name).suffix.lower().lstrip(".")
        return simple.upper() if simple else "RAW"
    return None
