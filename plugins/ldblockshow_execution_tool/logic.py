from __future__ import annotations

import gzip
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from app.models import LDBlockShowRequest, LDBlockShowResponse


ROOT_DIR = Path(__file__).resolve().parents[2]
LDBLOCKSHOW_BUNDLE_DIR = ROOT_DIR / "third_party" / "LDBlockShow"
LDBLOCKSHOW_DEFAULT_EXECUTABLE = LDBLOCKSHOW_BUNDLE_DIR / "bin" / "LDBlockShow"
LDBLOCKSHOW_OUTPUT_DIR = ROOT_DIR / "outputs" / "ldblockshow"
MAX_REGION_SPAN_BP = int(os.getenv("LDBLOCKSHOW_MAX_REGION_SPAN_BP", "1000000"))
DEFAULT_REGION_WINDOW_BP = int(os.getenv("LDBLOCKSHOW_DEFAULT_REGION_WINDOW_BP", "200000"))
FALLBACK_REGION_WINDOWS_BP = [
    DEFAULT_REGION_WINDOW_BP,
    min(DEFAULT_REGION_WINDOW_BP, 100000),
    min(DEFAULT_REGION_WINDOW_BP, 50000),
]


def _resolve_ldblockshow_executable() -> Path:
    configured = Path(os.getenv("LDBLOCKSHOW_EXECUTABLE", str(LDBLOCKSHOW_DEFAULT_EXECUTABLE)))
    if configured.exists():
        return configured
    fallback = shutil.which("LDBlockShow")
    if fallback:
        return Path(fallback)
    raise FileNotFoundError(
        "LDBlockShow executable not found. Build it under third_party/LDBlockShow or set LDBLOCKSHOW_EXECUTABLE."
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _default_output_prefix(vcf_path: str, region: str) -> str:
    return f"{Path(vcf_path).stem}.{_safe_name(region)}.{uuid.uuid4().hex[:8]}"


def run_ldblockshow(request: LDBlockShowRequest) -> LDBlockShowResponse:
    executable = _resolve_ldblockshow_executable()
    input_path = Path(request.vcf_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"VCF not found: {request.vcf_path}")

    adjusted_region, region_warnings = _normalize_region(request.region)
    output_dir = LDBLOCKSHOW_OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    attempted_regions = _fallback_regions(adjusted_region)
    last_error = "LDBlockShow finished without producing a viewable figure artifact."
    response_payload: dict[str, object] | None = None
    attempted_history: list[str] = []

    for attempt_index, candidate_region in enumerate(attempted_regions):
        attempted_history.append(candidate_region)
        prefix_name = _safe_name(
            request.output_prefix or _default_output_prefix(request.vcf_path, f"{candidate_region}.attempt{attempt_index+1}")
        )
        output_prefix = output_dir / prefix_name
        command = [
            str(executable),
            "-InVCF",
            str(input_path),
            "-OutPut",
            str(output_prefix),
            "-Region",
            candidate_region,
            "-SeleVar",
            str(request.sele_var),
            "-BlockType",
            str(request.block_type),
        ]
        if request.subgroup_path:
            command.extend(["-SubPop", request.subgroup_path])
        if request.gwas_path:
            command.extend(["-InGWAS", request.gwas_path])
        if request.gff_path:
            command.extend(["-InGFF", request.gff_path])
        if request.out_png:
            command.append("-OutPng")
        if request.out_pdf:
            command.append("-OutPdf")

        env = dict(os.environ)
        env.setdefault("LANG", "C")
        env.setdefault("LC_ALL", "C")
        completed = subprocess.run(
            command,
            cwd=str(LDBLOCKSHOW_BUNDLE_DIR),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            last_error = completed.stderr.strip() or completed.stdout.strip() or "LDBlockShow exited with a non-zero status."
            continue

        warnings = region_warnings + _collect_warnings(completed.stdout, completed.stderr)
        if candidate_region != adjusted_region:
            warnings.insert(
                0,
                f"LDBlockShow did not produce a figure for the broader request, so it was retried with a narrower window {candidate_region}.",
            )
        svg_path = _detect_file(output_prefix, ".svg")
        png_path = _detect_file(output_prefix, ".png")
        pdf_path = _detect_file(output_prefix, ".pdf")
        block_path = _detect_file(output_prefix, ".blocks.gz")
        site_path = _detect_file(output_prefix, ".site.gz")
        triangle_path = _detect_file(output_prefix, ".TriangleV.gz")
        site_row_count = _count_gzip_data_rows(site_path)
        block_row_count = _count_gzip_data_rows(block_path)
        triangle_pair_count = _count_gzip_data_rows(triangle_path, comment_prefix="#")

        if svg_path is None and png_path is None and pdf_path is None:
            if site_row_count == 0 and block_row_count == 0 and triangle_pair_count == 0:
                last_error = (
                    "No sufficient SNP pairs were available to draw an LD heatmap in the requested locus. "
                    f"Tried regions: {', '.join(attempted_history)}."
                )
            else:
                last_error = (
                    "LDBlockShow ran but did not produce a viewable figure artifact. "
                    f"Tried regions: {', '.join(attempted_history)}."
                )
            continue

        response_payload = {
            "region": candidate_region,
            "output_prefix": str(output_prefix),
            "command_preview": " ".join(command),
            "svg_path": svg_path,
            "png_path": png_path,
            "pdf_path": pdf_path,
            "block_path": block_path,
            "site_path": site_path,
            "triangle_path": triangle_path,
            "attempted_regions": attempted_history.copy(),
            "site_row_count": site_row_count,
            "block_row_count": block_row_count,
            "triangle_pair_count": triangle_pair_count,
            "warnings": warnings,
        }
        break

    if response_payload is None:
        raise RuntimeError(last_error)

    return LDBlockShowResponse(
        tool="ldblockshow",
        input_path=str(input_path),
        region=str(response_payload["region"]),
        output_prefix=str(response_payload["output_prefix"]),
        command_preview=str(response_payload["command_preview"]),
        svg_path=response_payload["svg_path"],
        png_path=response_payload["png_path"],
        pdf_path=response_payload["pdf_path"],
        block_path=response_payload["block_path"],
        site_path=response_payload["site_path"],
        triangle_path=response_payload["triangle_path"],
        attempted_regions=list(response_payload["attempted_regions"]),
        site_row_count=int(response_payload["site_row_count"]),
        block_row_count=int(response_payload["block_row_count"]),
        triangle_pair_count=int(response_payload["triangle_pair_count"]),
        warnings=list(response_payload["warnings"]),
    )


def _detect_file(prefix: Path, suffix: str) -> str | None:
    path = Path(f"{prefix}{suffix}")
    if path.exists():
        return str(path)
    return None


def _count_gzip_data_rows(path_text: str | None, comment_prefix: str | None = None) -> int:
    if not path_text:
        return 0
    path = Path(path_text)
    if not path.exists():
        return 0
    count = 0
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if comment_prefix and stripped.startswith(comment_prefix):
                continue
            count += 1
    return count


def _collect_warnings(stdout: str, stderr: str) -> list[str]:
    joined = "\n".join(part for part in (stdout, stderr) if part).strip()
    if not joined:
        return []
    warnings: list[str] = []
    for line in joined.splitlines():
        lowered = line.lower()
        if "warning" in lowered or "can'" in lowered or "unable to locate a java runtime" in lowered:
            warnings.append(line.strip())
    return warnings[:12]


def _normalize_region(region: str) -> tuple[str, list[str]]:
    match = re.fullmatch(r"((?:chr)?[A-Za-z0-9_]+):(\d+):(\d+)", region)
    if not match:
        return region, []
    chrom, start_text, end_text = match.group(1), match.group(2), match.group(3)
    start = int(start_text)
    end = int(end_text)
    if end < start:
        start, end = end, start
    span = end - start
    if span <= MAX_REGION_SPAN_BP:
        return f"{chrom}:{start}:{end}", []
    center = (start + end) // 2
    half_window = DEFAULT_REGION_WINDOW_BP // 2
    new_start = max(1, center - half_window)
    new_end = center + half_window
    adjusted = f"{chrom}:{new_start}:{new_end}"
    warning = (
        f"Requested region span {span} bp exceeded the default LD window limit of {MAX_REGION_SPAN_BP} bp, "
        f"so LDBlockShow was automatically narrowed to {adjusted}."
    )
    return adjusted, [warning]


def _fallback_regions(region: str) -> list[str]:
    match = re.fullmatch(r"((?:chr)?[A-Za-z0-9_]+):(\d+):(\d+)", region)
    if not match:
        return [region]
    chrom, start_text, end_text = match.group(1), match.group(2), match.group(3)
    start = int(start_text)
    end = int(end_text)
    center = (start + end) // 2
    candidates: list[str] = [region]
    for window in FALLBACK_REGION_WINDOWS_BP:
        half = max(1, window // 2)
        candidate = f"{chrom}:{max(1, center - half)}:{center + half}"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def execute(payload: dict[str, object]) -> LDBlockShowResponse:
    request = LDBlockShowRequest(**payload)
    return run_ldblockshow(request)
