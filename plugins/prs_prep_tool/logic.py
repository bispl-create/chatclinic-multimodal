from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path

from app.models import (
    PrsPrepBuildCheck,
    PrsPrepHarmonizationResult,
    PrsPrepResponse,
    SummaryStatsFieldMapping,
)
from plugins.summary_stats_review_tool.logic import _detect_delimiter, _infer_mapping

ROOT_DIR = Path(__file__).resolve().parents[2]
PRS_PREP_OUTPUT_DIR = ROOT_DIR / "outputs" / "prs_prep"


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _load_preview_rows(path: Path, limit: int = 500) -> tuple[list[str], list[dict[str, str]]]:
    with _open_text(path) as handle:
        first_line = ""
        for line in handle:
            if line.strip():
                first_line = line.rstrip("\n")
                break
        if not first_line:
            raise ValueError("Summary statistics file appears to be empty.")

    delimiter = _detect_delimiter(first_line)
    rows: list[dict[str, str]] = []
    with _open_text(path) as handle:
        if delimiter:
            reader = csv.DictReader(handle, delimiter=delimiter)
            columns = reader.fieldnames or []
            for row in reader:
                if not row:
                    continue
                compact = {str(key): str(value or "") for key, value in row.items() if key is not None}
                if not any(value.strip() for value in compact.values()):
                    continue
                rows.append(compact)
                if len(rows) >= limit:
                    break
        else:
            columns = first_line.strip().split()
            with _open_text(path) as whitespace_handle:
                next(whitespace_handle)
                for raw_line in whitespace_handle:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    rows.append({columns[idx]: parts[idx] if idx < len(parts) else "" for idx in range(len(columns))})
                    if len(rows) >= limit:
                        break
    return columns, rows


def _infer_build(mapping: SummaryStatsFieldMapping, rows: list[dict[str, str]], genome_build: str) -> PrsPrepBuildCheck:
    warnings: list[str] = []
    source_build = genome_build or "unknown"
    if source_build.lower() not in {"unknown", ""}:
        inferred = source_build
        confidence = "user"
    else:
        inferred = "unknown"
        confidence = "low"
        if mapping.chrom and mapping.pos and rows:
            inferred = "unknown"
            warnings.append("Genome build could not be inferred confidently from the summary-statistics header alone.")
    target_build = inferred if inferred.lower() not in {"unknown", ""} else "unknown"
    build_match = None if target_build == "unknown" or source_build == "unknown" else source_build.lower() == target_build.lower()
    if source_build == "unknown":
        warnings.append("Genome build is currently unknown. Confirm GRCh37 vs GRCh38 before PRS scoring.")
    return PrsPrepBuildCheck(
        inferred_build=inferred,
        build_confidence=confidence,
        source_build=source_build,
        target_build=target_build,
        build_match=build_match,
        warnings=warnings,
    )


def _detect_effect_size_kind(mapping: SummaryStatsFieldMapping) -> str:
    column = (mapping.beta_or or "").lower()
    if not column:
        return "unknown"
    if "or" == column or column.endswith("_or") or "odds" in column:
        return "odds_ratio"
    if "beta" in column or "effect" in column or "estimate" in column:
        return "beta"
    return "unknown"


def _is_ambiguous_snp(effect_allele: str, other_allele: str) -> bool:
    pair = {effect_allele.upper().strip(), other_allele.upper().strip()}
    return pair in ({"A", "T"}, {"C", "G"})


def _build_harmonization_result(mapping: SummaryStatsFieldMapping, rows: list[dict[str, str]]) -> PrsPrepHarmonizationResult:
    required = {
        "chrom": mapping.chrom,
        "pos": mapping.pos,
        "effect_allele": mapping.effect_allele,
        "other_allele": mapping.other_allele,
        "beta_or": mapping.beta_or,
    }
    missing_fields = [field for field, value in required.items() if not value]
    required_fields_present = not missing_fields
    warnings: list[str] = []
    ambiguous_snp_count = 0
    harmonizable_preview_rows = 0
    if required_fields_present:
        for row in rows:
            effect = str(row.get(mapping.effect_allele or "", "")).strip()
            other = str(row.get(mapping.other_allele or "", "")).strip()
            chrom = str(row.get(mapping.chrom or "", "")).strip()
            pos = str(row.get(mapping.pos or "", "")).strip()
            score = str(row.get(mapping.beta_or or "", "")).strip()
            if chrom and pos and effect and other and score:
                harmonizable_preview_rows += 1
                if _is_ambiguous_snp(effect, other):
                    ambiguous_snp_count += 1
    else:
        warnings.append("Required columns for PRS harmonization are not all available yet.")
    if ambiguous_snp_count:
        warnings.append("Ambiguous A/T or C/G SNPs were detected in the preview and will need an explicit handling policy.")
    return PrsPrepHarmonizationResult(
        required_fields_present=required_fields_present,
        effect_size_kind=_detect_effect_size_kind(mapping),
        ambiguous_snp_count=ambiguous_snp_count,
        harmonizable_preview_rows=harmonizable_preview_rows,
        missing_fields=missing_fields,
        warnings=warnings,
    )


def _build_variant_id(mapping: SummaryStatsFieldMapping, row: dict[str, str]) -> str:
    rsid_value = str(row.get(mapping.rsid or "", "")).strip() if mapping.rsid else ""
    if rsid_value and rsid_value.upper() not in {"NA", ".", "NAN"}:
        return rsid_value
    chrom = str(row.get(mapping.chrom or "", "")).strip()
    pos = str(row.get(mapping.pos or "", "")).strip()
    effect = str(row.get(mapping.effect_allele or "", "")).strip().upper()
    other = str(row.get(mapping.other_allele or "", "")).strip().upper()
    return f"{chrom}:{pos}:{other}:{effect}"


def _normalize_score(raw_value: str, effect_size_kind: str) -> float | None:
    try:
        value = float(raw_value)
    except Exception:
        return None
    if effect_size_kind == "odds_ratio":
        if value <= 0:
            return None
        value = math.log(value)
    if not math.isfinite(value):
        return None
    return value


def _write_score_file(
    file_path: Path,
    mapping: SummaryStatsFieldMapping,
    effect_size_kind: str,
) -> tuple[str | None, list[str], list[dict[str, str]], int, int]:
    required_columns = [mapping.chrom, mapping.pos, mapping.effect_allele, mapping.other_allele, mapping.beta_or]
    if any(not item for item in required_columns):
        return None, ["ID", "A1", "BETA"], [], 0, 0

    PRS_PREP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PRS_PREP_OUTPUT_DIR / f"{file_path.stem}.score.tsv"
    preview_rows: list[dict[str, str]] = []
    kept_rows = 0
    dropped_rows = 0

    with _open_text(file_path) as handle, output_path.open("w", encoding="utf-8", newline="") as writer_handle:
        delimiter = _detect_delimiter(next((line.rstrip("\n") for line in handle if line.strip()), ""))
        handle.seek(0)
        output_writer = csv.DictWriter(writer_handle, fieldnames=["ID", "A1", "BETA"], delimiter="\t")
        output_writer.writeheader()

        if delimiter:
            reader = csv.DictReader(handle, delimiter=delimiter)
            iterable = reader
        else:
            header_line = next((line for line in handle if line.strip()), "")
            columns = header_line.strip().split()

            def whitespace_rows():
                for raw_line in handle:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    yield {columns[idx]: parts[idx] if idx < len(parts) else "" for idx in range(len(columns))}

            iterable = whitespace_rows()

        for row in iterable:
            if not row:
                continue
            effect = str(row.get(mapping.effect_allele or "", "")).strip().upper()
            other = str(row.get(mapping.other_allele or "", "")).strip().upper()
            chrom = str(row.get(mapping.chrom or "", "")).strip()
            pos = str(row.get(mapping.pos or "", "")).strip()
            raw_score = str(row.get(mapping.beta_or or "", "")).strip()
            if not chrom or not pos or not effect or not other or not raw_score:
                dropped_rows += 1
                continue
            if _is_ambiguous_snp(effect, other):
                dropped_rows += 1
                continue
            score_value = _normalize_score(raw_score, effect_size_kind)
            if score_value is None:
                dropped_rows += 1
                continue
            output_row = {
                "ID": _build_variant_id(mapping, row),
                "A1": effect,
                "BETA": f"{score_value:.10g}",
            }
            output_writer.writerow(output_row)
            kept_rows += 1
            if len(preview_rows) < 12:
                preview_rows.append(output_row)

    return str(output_path), ["ID", "A1", "BETA"], preview_rows, kept_rows, dropped_rows


def analyze_prs_prep(path: str, original_name: str, genome_build: str = "unknown") -> PrsPrepResponse:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Summary statistics file not found: {path}")

    columns, rows = _load_preview_rows(file_path)
    mapping = _infer_mapping(columns)
    build_check = _infer_build(mapping, rows, genome_build)
    harmonization = _build_harmonization_result(mapping, rows)
    score_file_path, score_file_columns, score_file_preview_rows, kept_rows, dropped_rows = _write_score_file(
        file_path,
        mapping,
        harmonization.effect_size_kind,
    )
    score_file_ready = bool(score_file_path and kept_rows > 0 and harmonization.required_fields_present)

    draft_answer = (
        f"PRS preparation review was initialized for `{original_name}`.\n\n"
        f"- Build check: {build_check.inferred_build} ({build_check.build_confidence})\n"
        f"- Harmonization-ready preview rows: {harmonization.harmonizable_preview_rows}\n"
        f"- Ambiguous SNPs in preview: {harmonization.ambiguous_snp_count}\n"
        f"- Score-file rows kept: {kept_rows}\n"
        f"- Score-file rows dropped: {dropped_rows}\n"
        f"- Score file ready: {'yes' if score_file_ready else 'no'}\n\n"
        "This stage establishes build check, harmonization readiness, and a PLINK-compatible score file preview."
    )

    return PrsPrepResponse(
        analysis_id=f"{file_path.stem}-prs-prep",
        source_stats_path=str(file_path),
        file_name=original_name,
        build_check=build_check,
        harmonization=harmonization,
        score_file_path=score_file_path,
        score_file_columns=score_file_columns,
        score_file_preview_rows=score_file_preview_rows,
        kept_rows=kept_rows,
        dropped_rows=dropped_rows,
        score_file_ready=score_file_ready,
        draft_answer=draft_answer,
    )


def execute(payload: dict[str, object]) -> dict[str, object]:
    result = analyze_prs_prep(
        str(payload["source_stats_path"]),
        str(payload["file_name"]),
        genome_build=str(payload.get("genome_build", "unknown")),
    )
    return {"prs_prep_result": result.model_dump()}
