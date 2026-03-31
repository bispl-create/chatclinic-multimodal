from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

from app.models import (
    PlinkFreqRow,
    PlinkHardyRow,
    PlinkMissingRow,
    PlinkRequest,
    PlinkResponse,
    PlinkScoreRow,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
PLINK_OUTPUT_DIR = ROOT_DIR / "outputs" / "plink"
PLINK_BIN = Path(
    os.getenv(
        "PLINK2_BIN",
        str(ROOT_DIR / "third_party" / "plink2" / "plink2"),
    )
)


from app.utils.parsing import maybe_float as _maybe_float
from app.utils.parsing import maybe_int as _maybe_int
from app.utils.parsing import safe_prefix as _safe_prefix_base


def _safe_prefix(prefix: str | None, source_path: str) -> str:
    return _safe_prefix_base(prefix, source_path, tool="plink")


def _parse_freq_rows(path: Path, limit: int) -> list[PlinkFreqRow]:
    if not path.exists() or limit <= 0:
        return []
    rows: list[PlinkFreqRow] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                PlinkFreqRow(
                    chrom=str(row.get("#CHROM") or row.get("CHROM") or ""),
                    variant_id=str(row.get("ID") or row.get("SNP") or ""),
                    ref_allele=str(row.get("REF") or ""),
                    alt_allele=str(row.get("ALT1") or row.get("ALT") or ""),
                    alt_freq=_maybe_float(row.get("ALT1_FREQ") or row.get("ALT_FREQS")),
                    observation_count=_maybe_int(row.get("OBS_CT")),
                )
            )
            if len(rows) >= limit:
                break
    return rows


def _parse_missing_rows(path: Path, limit: int) -> list[PlinkMissingRow]:
    if not path.exists() or limit <= 0:
        return []
    rows: list[PlinkMissingRow] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                PlinkMissingRow(
                    sample_id=str(row.get("IID") or row.get("FID") or ""),
                    missing_genotype_count=_maybe_int(row.get("MISSING_CT")) or 0,
                    observation_count=_maybe_int(row.get("OBS_CT")) or 0,
                    missing_rate=_maybe_float(row.get("F_MISS")) or 0.0,
                )
            )
            if len(rows) >= limit:
                break
    return rows


def _parse_hardy_rows(path: Path, limit: int) -> list[PlinkHardyRow]:
    if not path.exists() or limit <= 0:
        return []
    rows: list[PlinkHardyRow] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                PlinkHardyRow(
                    chrom=str(row.get("#CHROM") or row.get("CHROM") or ""),
                    variant_id=str(row.get("ID") or row.get("SNP") or ""),
                    observed_hets=_maybe_int(row.get("O(HET_A1)")),
                    expected_hets=_maybe_float(row.get("E(HET_A1)")),
                    p_value=_maybe_float(row.get("P")),
                )
            )
            if len(rows) >= limit:
                break
    return rows


def _extract_count_from_log(log_path: Path, label: str) -> int | None:
    if not log_path.exists():
        return None
    pattern = re.compile(rf"(\d+)\s+{re.escape(label)}")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def _parse_score_rows(path: Path, limit: int = 12) -> tuple[list[PlinkScoreRow], float | None, float | None, float | None]:
    if not path.exists() or limit <= 0:
        return [], None, None, None
    rows: list[PlinkScoreRow] = []
    score_values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score_sum = _maybe_float(
                row.get("SCORE1_SUM")
                or row.get("SCORE_SUM")
                or row.get("SCORE_AVG")
                or row.get("BETA_SUM")
            )
            if score_sum is not None:
                score_values.append(score_sum)
            rows.append(
                PlinkScoreRow(
                    sample_id=str(row.get("#IID") or row.get("IID") or row.get("FID") or ""),
                    allele_ct=_maybe_float(row.get("ALLELE_CT")),
                    named_allele_dosage_sum=_maybe_float(row.get("DENOM")),
                    score_sum=score_sum,
                )
            )
            if len(rows) >= limit:
                break
    if not score_values:
        return rows, None, None, None
    return rows, sum(score_values) / len(score_values), min(score_values), max(score_values)


def run_plink(request: PlinkRequest) -> PlinkResponse:
    input_path = Path(request.vcf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"VCF input not found: {request.vcf_path}")
    if not PLINK_BIN.exists():
        raise FileNotFoundError(f"PLINK 2 binary not found: {PLINK_BIN}")

    PLINK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = _safe_prefix(request.output_prefix, request.vcf_path)
    output_base = PLINK_OUTPUT_DIR / output_prefix

    cmd = [str(PLINK_BIN), "--vcf", str(input_path)]
    if request.allow_extra_chr:
        cmd.append("--allow-extra-chr")

    score_path: Path | None = None
    if request.mode == "score":
        if not request.score_file_path:
            raise ValueError("score_file_path is required for PLINK score mode.")
        score_path = Path(request.score_file_path)
        if not score_path.exists():
            raise FileNotFoundError(f"PLINK score file not found: {request.score_file_path}")
        cmd.extend(
            [
                "--set-all-var-ids",
                "@:#:$r:$a",
                "--new-id-max-allele-len",
                "200",
                "--score",
                str(score_path),
                "1",
                "2",
                "3",
                "header-read",
                "no-mean-imputation",
                "cols=scoresums",
            ]
        )
    else:
        cmd.append("dosage=DS")
        if request.freq_limit > 0:
            cmd.append("--freq")
        if request.missing_limit > 0:
            cmd.append("--missing")
        if request.hardy_limit > 0:
            cmd.append("--hardy")

    cmd.extend(["--out", str(output_base)])

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)

    warnings: list[str] = [
        "This ChatGenome build currently bundles the macOS Apple Silicon PLINK 2 binary. "
        "If you are running on Linux, Windows, or Intel macOS, download the matching PLINK 2 binary and update `PLINK2_BIN`."
    ]
    if completed.stderr.strip():
        warnings.extend(line.strip() for line in completed.stderr.splitlines() if line.strip())

    log_path = Path(f"{output_base}.log")
    freq_path = Path(f"{output_base}.afreq")
    missing_path = Path(f"{output_base}.smiss")
    hardy_path = Path(f"{output_base}.hardy")
    score_output_path = Path(f"{output_base}.sscore")
    score_rows, score_mean, score_min, score_max = _parse_score_rows(score_output_path)

    parsed_sample_count = _extract_count_from_log(log_path, "samples (")
    if parsed_sample_count is None and score_rows:
        parsed_sample_count = len(score_rows)

    return PlinkResponse(
        tool="plink2-qc" if request.mode == "qc" else "plink2-score",
        mode=request.mode,
        input_path=str(input_path),
        command_preview=" ".join(cmd),
        output_prefix=str(output_base),
        log_path=str(log_path) if log_path.exists() else None,
        freq_path=str(freq_path) if freq_path.exists() else None,
        missing_path=str(missing_path) if missing_path.exists() else None,
        hardy_path=str(hardy_path) if hardy_path.exists() else None,
        score_file_path=str(score_path) if score_path else None,
        score_output_path=str(score_output_path) if score_output_path.exists() else None,
        variant_count=_extract_count_from_log(log_path, "variants loaded from"),
        sample_count=parsed_sample_count,
        freq_rows=_parse_freq_rows(freq_path, request.freq_limit),
        missing_rows=_parse_missing_rows(missing_path, request.missing_limit),
        hardy_rows=_parse_hardy_rows(hardy_path, request.hardy_limit),
        score_rows=score_rows,
        score_mean=score_mean,
        score_min=score_min,
        score_max=score_max,
        warnings=warnings,
    )


def execute(payload: dict[str, object]) -> PlinkResponse:
    request = PlinkRequest(**payload)
    return run_plink(request)
