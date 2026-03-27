from __future__ import annotations

from functools import lru_cache
from pathlib import Path


SOURCE_REGISTRY: dict[str, dict[str, object]] = {
    "text": {
        "upload_label": "text note",
        "dedicated_upload_detail": "Only Markdown and plain-text note uploads such as .md, .markdown, .text, .note, and .log are supported.",
        "bootstrap_source_type": "text",
        "chat_response_kind": "text",
        "workflow_names": ["text_review"],
        "capabilities": ["source_upload", "bootstrap_analysis", "workflow", "grounded_chat"],
        "suffixes": [
            ".markdown",
            ".md",
            ".text",
            ".note",
            ".log",
        ],
        "file_kind_map": {
            ".markdown": "TEXT",
            ".md": "TEXT",
            ".text": "TEXT",
            ".note": "TEXT",
            ".log": "TEXT",
        },
    },
    "raw_qc": {
        "upload_label": "raw sequencing file",
        "dedicated_upload_detail": "Only FASTQ, FASTQ.gz, FQ, FQ.gz, BAM, and SAM uploads are supported.",
        "bootstrap_source_type": "raw_qc",
        "chat_response_kind": "raw_qc",
        "workflow_names": ["raw_qc_review"],
        "capabilities": ["source_upload", "bootstrap_analysis", "direct_tool", "workflow"],
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
        "dedicated_upload_detail": "Only TSV/TXT/CSV summary statistics uploads are supported.",
        "bootstrap_source_type": "summary_stats",
        "chat_response_kind": "summary_stats",
        "workflow_names": ["summary_stats_review", "prs_prep"],
        "capabilities": ["source_upload", "bootstrap_analysis", "direct_tool", "workflow"],
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
        "dedicated_upload_detail": "Only .vcf and .vcf.gz uploads are supported.",
        "bootstrap_source_type": "vcf",
        "chat_response_kind": "analysis",
        "workflow_names": ["representative_vcf_review"],
        "capabilities": ["source_upload", "bootstrap_analysis", "direct_tool", "workflow"],
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


def source_upload_detail(source_type: str) -> str | None:
    registration = load_source_registration(source_type)
    if registration is None:
        return None
    detail = registration.get("dedicated_upload_detail")
    return str(detail).strip() if isinstance(detail, str) and str(detail).strip() else None


def source_bootstrap_type(source_type: str) -> str:
    registration = load_source_registration(source_type)
    if registration is None:
        return source_type
    bootstrap_source_type = registration.get("bootstrap_source_type")
    if isinstance(bootstrap_source_type, str) and bootstrap_source_type.strip():
        return bootstrap_source_type.strip().lower()
    return source_type


def source_workflow_names(source_type: str) -> tuple[str, ...]:
    registration = load_source_registration(source_type)
    if registration is None:
        return ()
    names = registration.get("workflow_names") or []
    if not isinstance(names, list):
        return ()
    return tuple(str(name).strip() for name in names if str(name).strip())


def source_capabilities(source_type: str) -> tuple[str, ...]:
    registration = load_source_registration(source_type)
    if registration is None:
        return ()
    capabilities = registration.get("capabilities") or []
    if not isinstance(capabilities, list):
        return ()
    return tuple(str(item).strip() for item in capabilities if str(item).strip())
