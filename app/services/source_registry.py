from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Source type registry — single source of truth for all source types.
#
# To add a new source type (e.g. "fhir"), add an entry here.  Everything
# else (upload routing, initial-tool dispatch, chat routing) derives from
# this table automatically.
# ---------------------------------------------------------------------------

SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "dicom": {
        "upload_label": "DICOM file",
        "dedicated_upload_detail": "Only DICOM uploads such as .dcm and .dicom are supported.",
        "upload_endpoint": "dicom",
        "bootstrap_source_type": "dicom",
        "chat_response_kind": "dicom",
        "default_result_kind": "dicom_analysis",
        "default_requested_view": "dicom_review",
        "studio_renderer": "dicom_review",
        "studio_card_kind": "dicom_browser",
        "studio_preview_kind": "image_preview",
        "initial_tools": ["dicom_review_tool"],
        "capabilities": ["source_upload", "bootstrap_analysis", "grounded_chat"],
        "suffixes": [".dcm", ".dicom"],
        "file_kind_map": {".dcm": "DICOM", ".dicom": "DICOM"},
    },
    "spreadsheet": {
        "upload_label": "spreadsheet workbook",
        "dedicated_upload_detail": "Only Excel workbook uploads such as .xlsx and .xlsm are supported.",
        "upload_endpoint": "spreadsheet",
        "bootstrap_source_type": "spreadsheet",
        "chat_response_kind": "spreadsheet",
        "default_result_kind": "spreadsheet_analysis",
        "default_requested_view": "cohort_browser",
        "studio_renderer": "cohort_browser",
        "studio_card_kind": "tabular_browser",
        "studio_preview_kind": "sheet_grid",
        "initial_tools": ["cohort_sheet_browser_tool"],
        "capabilities": ["source_upload", "bootstrap_analysis", "grounded_chat"],
        "suffixes": [".xlsx", ".xlsm"],
        "file_kind_map": {".xlsx": "XLSX", ".xlsm": "XLSM"},
    },
    "text": {
        "upload_label": "text note",
        "dedicated_upload_detail": "Only Markdown and plain-text note uploads such as .md, .markdown, .text, .note, and .log are supported.",
        "upload_endpoint": "text",
        "bootstrap_source_type": "text",
        "chat_response_kind": "text",
        "default_result_kind": "text_analysis",
        "default_requested_view": "text",
        "studio_renderer": "text",
        "studio_card_kind": "document_viewer",
        "studio_preview_kind": "markdown",
        "initial_tools": ["text_review_tool"],
        "capabilities": ["source_upload", "bootstrap_analysis", "grounded_chat"],
        "suffixes": [".markdown", ".md", ".text", ".note", ".log"],
        "file_kind_map": {
            ".markdown": "TEXT", ".md": "TEXT", ".text": "TEXT",
            ".note": "TEXT", ".log": "TEXT",
        },
    },
    "raw_qc": {
        "upload_label": "raw sequencing file",
        "dedicated_upload_detail": "Only FASTQ, FASTQ.gz, FQ, FQ.gz, BAM, and SAM uploads are supported.",
        "upload_endpoint": "raw-qc",
        "bootstrap_source_type": "raw_qc",
        "chat_response_kind": "raw_qc",
        "default_result_kind": "raw_qc_analysis",
        "default_requested_view": "rawqc",
        "studio_renderer": "rawqc",
        "studio_card_kind": "qc_review",
        "studio_preview_kind": "qc_modules",
        "initial_tools": ["fastqc_execution_tool"],
        "capabilities": ["source_upload", "bootstrap_analysis", "direct_tool", "grounded_chat"],
        "suffixes": [".fastq.gz", ".fq.gz", ".fastq", ".fq", ".bam", ".sam"],
        "file_kind_map": {
            ".fastq.gz": "FASTQ", ".fq.gz": "FASTQ",
            ".fastq": "FASTQ", ".fq": "FASTQ",
            ".bam": "BAM", ".sam": "SAM",
        },
    },
    "summary_stats": {
        "upload_label": "summary statistics file",
        "dedicated_upload_detail": "Only TSV/TXT/CSV summary statistics uploads are supported.",
        "upload_endpoint": "summary-stats",
        "bootstrap_source_type": "summary_stats",
        "chat_response_kind": "summary_stats",
        "default_result_kind": "summary_stats_analysis",
        "default_requested_view": "sumstats",
        "studio_renderer": "sumstats",
        "studio_card_kind": "tabular_summary",
        "studio_preview_kind": "preview_rows",
        "initial_tools": ["summary_stats_review_tool"],
        "capabilities": ["source_upload", "bootstrap_analysis", "direct_tool"],
        "suffixes": [
            ".sumstats.gz", ".tsv.gz", ".txt.gz", ".csv.gz",
            ".sumstats", ".tsv", ".txt", ".csv",
        ],
    },
    "vcf": {
        "upload_label": "VCF file",
        "dedicated_upload_detail": "Only .vcf and .vcf.gz uploads are supported.",
        "upload_endpoint": "analysis",
        "bootstrap_source_type": "vcf",
        "chat_response_kind": "analysis",
        "default_result_kind": "analysis",
        "default_requested_view": "qc",
        "studio_renderer": "qc",
        "studio_card_kind": "qc_review",
        "studio_preview_kind": "qc_metrics",
        "initial_tools": ["vcf_qc_tool"],
        "capabilities": ["source_upload", "bootstrap_analysis", "direct_tool"],
        "suffixes": [".vcf.gz", ".vcf"],
        "file_kind_map": {".vcf.gz": "VCF", ".vcf": "VCF"},
    },
}


# ---------------------------------------------------------------------------
# SourceRegistry — class-based access to the registry.
# ---------------------------------------------------------------------------

class SourceRegistry:
    """Centralised access to the source-type registry."""

    @staticmethod
    def types() -> tuple[str, ...]:
        return tuple(SOURCE_REGISTRY.keys())

    @staticmethod
    def get(source_type: str) -> dict[str, Any] | None:
        return SOURCE_REGISTRY.get(source_type.strip().lower())

    @staticmethod
    def detect(file_name: str) -> tuple[str, dict[str, Any], str] | None:
        """Return (source_type, registration, matched_suffix) or None."""
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

    @staticmethod
    def detect_type(file_name: str) -> str | None:
        detected = SourceRegistry.detect(file_name)
        return detected[0] if detected else None

    @staticmethod
    def upload_endpoint(source_type: str) -> str | None:
        reg = SourceRegistry.get(source_type)
        return str(reg["upload_endpoint"]) if reg and "upload_endpoint" in reg else None

    @staticmethod
    def initial_tools(source_type: str) -> list[str]:
        reg = SourceRegistry.get(source_type)
        if reg is None:
            return []
        tools = reg.get("initial_tools") or []
        return [str(t) for t in tools if str(t).strip()]

    @staticmethod
    def upload_detail(source_type: str) -> str | None:
        reg = SourceRegistry.get(source_type)
        if reg is None:
            return None
        detail = reg.get("dedicated_upload_detail")
        return str(detail).strip() if isinstance(detail, str) and str(detail).strip() else None

    @staticmethod
    def bootstrap_type(source_type: str) -> str:
        reg = SourceRegistry.get(source_type)
        if reg is None:
            return source_type
        val = reg.get("bootstrap_source_type")
        return str(val).strip().lower() if isinstance(val, str) and str(val).strip() else source_type

    @staticmethod
    def capabilities(source_type: str) -> tuple[str, ...]:
        reg = SourceRegistry.get(source_type)
        if reg is None:
            return ()
        caps = reg.get("capabilities") or []
        if not isinstance(caps, list):
            return ()
        return tuple(str(item).strip() for item in caps if str(item).strip())

    @staticmethod
    def file_kind(file_name: str, source_type: str, matched_suffix: str | None = None) -> str | None:
        reg = SourceRegistry.get(source_type)
        if reg is None:
            return None
        suffix = matched_suffix or "".join(Path(file_name).suffixes).lower() or Path(file_name).suffix.lower()
        file_kind_map = reg.get("file_kind_map") or {}
        if isinstance(file_kind_map, dict):
            matched = file_kind_map.get(suffix)
            if isinstance(matched, str) and matched.strip():
                return matched.strip()
        if source_type == "raw_qc":
            simple = Path(file_name).suffix.lower().lstrip(".")
            return simple.upper() if simple else "RAW"
        return None

    @staticmethod
    def response_metadata(source_type: str) -> dict[str, Any]:
        reg = SourceRegistry.get(source_type)
        if reg is None:
            return {"source_type": source_type}
        studio_renderer = str(reg.get("studio_renderer") or "").strip()
        studio_card_kind = str(reg.get("studio_card_kind") or "").strip()
        studio_preview_kind = str(reg.get("studio_preview_kind") or "").strip()
        requested_view = str(reg.get("default_requested_view") or "").strip() or None
        result_kind = str(reg.get("default_result_kind") or "").strip() or None
        studio = (
            {
                "renderer": studio_renderer,
                "card_kind": studio_card_kind or None,
                "preview_kind": studio_preview_kind or None,
            }
            if studio_renderer or studio_card_kind or studio_preview_kind
            else None
        )
        return {
            "source_type": source_type,
            "result_kind": result_kind,
            "requested_view": requested_view,
            "studio": studio,
        }


# ---------------------------------------------------------------------------
# Legacy function API — thin wrappers for backward compatibility.
# Will be removed once all callers migrate to SourceRegistry.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def list_registered_source_types() -> tuple[str, ...]:
    return SourceRegistry.types()


def load_source_registration(source_type: str) -> dict[str, Any] | None:
    return SourceRegistry.get(source_type)


def detect_source_registration(file_name: str) -> tuple[str, dict[str, Any], str] | None:
    return SourceRegistry.detect(file_name)


def detect_source_type(file_name: str) -> str | None:
    return SourceRegistry.detect_type(file_name)


def infer_source_file_kind(file_name: str, source_type: str, matched_suffix: str | None = None) -> str | None:
    return SourceRegistry.file_kind(file_name, source_type, matched_suffix)


def source_upload_detail(source_type: str) -> str | None:
    return SourceRegistry.upload_detail(source_type)


def source_bootstrap_type(source_type: str) -> str:
    return SourceRegistry.bootstrap_type(source_type)


def source_workflow_names(source_type: str) -> tuple[str, ...]:
    reg = SourceRegistry.get(source_type)
    if reg is None:
        return ()
    names = reg.get("workflow_names") or []
    if not isinstance(names, list):
        return ()
    return tuple(str(name).strip() for name in names if str(name).strip())


def source_capabilities(source_type: str) -> tuple[str, ...]:
    return SourceRegistry.capabilities(source_type)


def source_response_metadata(source_type: str) -> dict[str, Any]:
    return SourceRegistry.response_metadata(source_type)
