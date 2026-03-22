from __future__ import annotations

import csv
import gzip
from pathlib import Path

from app.models import SummaryStatsFieldMapping, SummaryStatsResponse


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _detect_delimiter(sample_line: str) -> str:
    if "\t" in sample_line:
        return "\t"
    if "," in sample_line:
        return ","
    return None  # type: ignore[return-value]


def _canonical_delimiter_label(delimiter: str | None) -> str:
    if delimiter == "\t":
        return "tab"
    if delimiter == ",":
        return "comma"
    return "whitespace"


def _normalize(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {_normalize(column): column for column in columns}
    for alias in aliases:
        match = normalized.get(_normalize(alias))
        if match:
            return match
    return None


def _infer_mapping(columns: list[str]) -> SummaryStatsFieldMapping:
    return SummaryStatsFieldMapping(
        chrom=_find_column(columns, ("chr", "chrom", "chromosome")),
        pos=_find_column(columns, ("bp", "pos", "position", "base_pair_location")),
        rsid=_find_column(columns, ("snp", "rsid", "markername", "variant_id")),
        effect_allele=_find_column(columns, ("a1", "ea", "effect_allele", "alt", "tested_allele")),
        other_allele=_find_column(columns, ("a2", "nea", "other_allele", "ref", "non_effect_allele")),
        beta_or=_find_column(columns, ("beta", "or", "effect", "estimate", "beta_or")),
        standard_error=_find_column(columns, ("se", "stderr", "standard_error")),
        p_value=_find_column(columns, ("p", "pvalue", "p_value", "pval")),
        n=_find_column(columns, ("n", "n_total", "samplesize", "sample_size")),
        eaf=_find_column(columns, ("eaf", "maf", "af", "effect_allele_frequency")),
    )


def analyze_summary_stats(path: str, original_name: str, genome_build: str = "unknown", trait_type: str = "unknown") -> SummaryStatsResponse:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Summary statistics file not found: {path}")

    with _open_text(file_path) as handle:
        first_line = ""
        for line in handle:
            if line.strip():
                first_line = line.rstrip("\n")
                break
        if not first_line:
            raise ValueError("Summary statistics file appears to be empty.")

    delimiter = _detect_delimiter(first_line)
    preview_rows: list[dict[str, str]] = []
    row_count = 0
    warnings: list[str] = []

    with _open_text(file_path) as handle:
        if delimiter:
            reader = csv.DictReader(handle, delimiter=delimiter)
            columns = reader.fieldnames or []
            for row in reader:
                if not row:
                    continue
                compact = {str(key): str(value or "") for key, value in row.items() if key is not None}
                if not any(value.strip() for value in compact.values()):
                    continue
                row_count += 1
                if len(preview_rows) < 100:
                    preview_rows.append(compact)
        else:
            header = first_line.strip().split()
            columns = header
            with _open_text(file_path) as whitespace_handle:
                next(whitespace_handle)
                for raw_line in whitespace_handle:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    row_count += 1
                    if len(preview_rows) < 100:
                        preview_rows.append({columns[idx]: parts[idx] if idx < len(parts) else "" for idx in range(len(columns))})

    mapping = _infer_mapping(columns)
    mapped_count = sum(
        1
        for value in (
            mapping.chrom,
            mapping.pos,
            mapping.rsid,
            mapping.effect_allele,
            mapping.other_allele,
            mapping.beta_or,
            mapping.standard_error,
            mapping.p_value,
            mapping.n,
            mapping.eaf,
        )
        if value
    )
    if mapping.p_value is None:
        warnings.append("P-value column was not confidently detected.")
    if mapping.chrom is None and mapping.rsid is None:
        warnings.append("Neither genomic position nor rsID column was confidently detected.")
    if mapped_count < 5:
        warnings.append("Only a small subset of expected GWAS summary-stat columns was detected automatically.")

    draft_answer = (
        f"Summary statistics file `{original_name}` was loaded.\n\n"
        f"- Rows detected: {row_count}\n"
        f"- Genome build: {genome_build}\n"
        f"- Trait type: {trait_type}\n"
        f"- Columns detected: {len(columns)}\n"
        f"- Auto-mapped fields: {mapped_count}\n\n"
        "Review the Summary Stats card to confirm column mapping before post-GWAS analysis."
    )

    return SummaryStatsResponse(
        analysis_id=file_path.stem,
        source_stats_path=str(file_path),
        file_name=original_name,
        genome_build=genome_build,
        trait_type=trait_type,
        delimiter=_canonical_delimiter_label(delimiter),
        detected_columns=columns,
        mapped_fields=mapping,
        row_count=row_count,
        preview_rows=preview_rows,
        warnings=warnings,
        draft_answer=draft_answer,
    )


def load_summary_stats_rows(path: str, offset: int = 0, limit: int = 200) -> tuple[list[dict[str, str]], bool]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Summary statistics file not found: {path}")

    if offset < 0:
        offset = 0
    if limit <= 0:
        limit = 200

    with _open_text(file_path) as handle:
        first_line = ""
        for line in handle:
            if line.strip():
                first_line = line.rstrip("\n")
                break
        if not first_line:
            return [], False

    delimiter = _detect_delimiter(first_line)
    rows: list[dict[str, str]] = []
    seen = 0
    has_more = False

    with _open_text(file_path) as handle:
        if delimiter:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                if not row:
                    continue
                compact = {str(key): str(value or "") for key, value in row.items() if key is not None}
                if not any(value.strip() for value in compact.values()):
                    continue
                if seen < offset:
                    seen += 1
                    continue
                if len(rows) < limit:
                    rows.append(compact)
                    seen += 1
                    continue
                has_more = True
                break
        else:
            columns = first_line.strip().split()
            with _open_text(file_path) as whitespace_handle:
                next(whitespace_handle)
                for raw_line in whitespace_handle:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    if seen < offset:
                        seen += 1
                        continue
                    if len(rows) < limit:
                        rows.append({columns[idx]: parts[idx] if idx < len(parts) else "" for idx in range(len(columns))})
                        seen += 1
                        continue
                    has_more = True
                    break

    return rows, has_more
