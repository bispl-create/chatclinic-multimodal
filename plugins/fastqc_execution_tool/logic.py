from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path

from app.models import RawQcFacts, RawQcModule, RawQcResponse, ToolInfo
from app.services.tool_runner import discover_tools


ROOT_DIR = Path(__file__).resolve().parents[2]
FASTQC_BUNDLE_DIR = ROOT_DIR / "third_party" / "FastQC"
FASTQC_DEFAULT_EXECUTABLE = FASTQC_BUNDLE_DIR / "fastqc"
FASTQC_OUTPUT_DIR = ROOT_DIR / "outputs" / "fastqc"


def detect_raw_qc_kind(file_name: str) -> str:
    lowered = file_name.lower()
    if lowered.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz")):
        return "FASTQ"
    if lowered.endswith(".bam"):
        return "BAM"
    if lowered.endswith(".sam"):
        return "SAM"
    return "raw-sequencing"


def run_fastqc_local(
    input_path: str,
    original_name: str,
    tool_registry: list[ToolInfo],
) -> RawQcResponse:
    executable = _resolve_fastqc_executable()
    output_dir = FASTQC_OUTPUT_DIR / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [str(executable), "--quiet", "-o", str(output_dir), input_path]
    env = dict(os.environ)
    env.setdefault("LANG", "C")
    env.setdefault("LC_ALL", "C")
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "FastQC exited with a non-zero status."
        if "Unable to locate a Java Runtime" in message:
            raise RuntimeError(
                "FastQC is installed locally, but Java Runtime is not available on this machine. "
                "Install a JRE/JDK or point JAVA_HOME to one before running raw sequencing QC."
            )
        raise RuntimeError(message)

    zip_path = _find_fastqc_zip(output_dir)
    html_path = _find_fastqc_html(output_dir)
    facts, modules = _parse_fastqc_zip(zip_path, original_name)
    draft_answer = _build_fastqc_summary(facts, modules)

    return RawQcResponse(
        analysis_id=f"fastqc-{uuid.uuid4().hex[:10]}",
        source_raw_path=input_path,
        facts=facts,
        modules=modules,
        draft_answer=draft_answer,
        report_html_path=str(html_path) if html_path.exists() else None,
        report_zip_path=str(zip_path) if zip_path.exists() else None,
        used_tools=["fastqc_execution_tool"],
        tool_registry=tool_registry,
    )


def _resolve_fastqc_executable() -> Path:
    configured = Path(os.getenv("FASTQC_EXECUTABLE", str(FASTQC_DEFAULT_EXECUTABLE)))
    if configured.exists():
        return configured
    fallback = shutil.which("fastqc")
    if fallback:
        return Path(fallback)
    raise FileNotFoundError(
        "FastQC executable not found. Set FASTQC_EXECUTABLE or place FastQC under third_party/FastQC."
    )


def _parse_fastqc_facts_text(text: str, original_name: str) -> RawQcFacts:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(">>"):
            continue
        if "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        values[key.strip()] = value.strip()

    return RawQcFacts(
        file_name=original_name,
        file_kind=detect_raw_qc_kind(original_name),
        total_sequences=_parse_int(values.get("Total Sequences")),
        filtered_sequences=_parse_int(values.get("Sequences flagged as poor quality")),
        poor_quality_sequences=_parse_int(values.get("Sequences flagged as poor quality")),
        sequence_length=values.get("Sequence length"),
        gc_content=_parse_float(values.get("%GC")),
        encoding=values.get("Encoding"),
    )


def _parse_fastqc_modules_text(text: str, original_name: str) -> list[RawQcModule]:
    modules: list[RawQcModule] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0].strip()
        name = parts[1].strip()
        detail = parts[2].strip() if len(parts) > 2 else ""
        if detail:
            detail = original_name
        modules.append(RawQcModule(name=name, status=status, detail=detail))
    return modules


def _parse_fastqc_zip(zip_path: Path, original_name: str) -> tuple[RawQcFacts, list[RawQcModule]]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        data_member = next((name for name in archive.namelist() if name.endswith("/fastqc_data.txt")), None)
        summary_member = next((name for name in archive.namelist() if name.endswith("/summary.txt")), None)
        if data_member is None or summary_member is None:
            raise RuntimeError(f"FastQC ZIP is missing expected files: {zip_path}")
        facts_text = archive.read(data_member).decode("utf-8", errors="replace")
        summary_text = archive.read(summary_member).decode("utf-8", errors="replace")
    return _parse_fastqc_facts_text(facts_text, original_name), _parse_fastqc_modules_text(summary_text, original_name)


def _find_fastqc_zip(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*_fastqc.zip"))
    if not candidates:
        raise RuntimeError(f"FastQC output ZIP was not created under {output_dir}")
    return candidates[0]


def _find_fastqc_html(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("*_fastqc.html"))
    return candidates[0] if candidates else None


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    return int(digits) if digits else None


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _build_fastqc_summary(facts: RawQcFacts, modules: list[RawQcModule]) -> str:
    pass_count = sum(1 for item in modules if item.status.upper() == "PASS")
    warn_count = sum(1 for item in modules if item.status.upper() == "WARN")
    fail_count = sum(1 for item in modules if item.status.upper() == "FAIL")
    total_sequences = f"{facts.total_sequences:,}" if facts.total_sequences is not None else "unknown"
    gc_text = f"{facts.gc_content:.1f}%" if facts.gc_content is not None else "unknown"
    return (
        f"FastQC completed for `{facts.file_name}` ({facts.file_kind}). "
        f"The file contains approximately {total_sequences} reads or records, "
        f"with sequence length `{facts.sequence_length or 'unknown'}` and GC content `{gc_text}`. "
        f"Module summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL. "
        "Review failed or warning modules first before downstream alignment or variant calling."
    )


def execute(payload: dict[str, object]) -> RawQcResponse:
    raw_path = str(payload["raw_path"])
    original_name = str(payload.get("original_name") or Path(raw_path).name)
    return run_fastqc_local(raw_path, original_name, discover_tools())
