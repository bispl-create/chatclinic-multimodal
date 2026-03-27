from __future__ import annotations

import os
import uuid
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.models import (
    AnalysisChatRequest,
    AnalysisChatResponse,
    AnalysisJobResponse,
    AnalysisResponse,
    GatkLiftoverVcfRequest,
    GatkLiftoverVcfResponse,
    LDBlockShowRequest,
    LDBlockShowResponse,
    CmplotAssociationRequest,
    FilterRequest,
    FilterResponse,
    FromPathRequest,
    PlinkRequest,
    PlinkResponse,
    PrsPrepRequest,
    PrsPrepResponse,
    QqmanAssociationRequest,
    RawQcChatRequest,
    RawQcChatResponse,
    RawQcResponse,
    RPlotRequest,
    RPlotResponse,
    SnpEffRequest,
    SnpEffResponse,
    SamtoolsRequest,
    SamtoolsResponse,
    SourceReadyResponse,
    SummaryStatsRowsRequest,
    SummaryStatsRowsResponse,
    SummaryStatsChatRequest,
    SummaryStatsChatResponse,
    SummaryStatsResponse,
    ToolInfo,
    WorkflowAgentResponse,
    WorkflowReplyRequest,
    WorkflowStartRequest,
)
from app.services.chat import answer_analysis_chat, answer_raw_qc_chat, answer_summary_stats_chat
from app.services.fastqc import FASTQC_OUTPUT_DIR
from app.services.filtering import run_filter
from app.services.gatk_liftover import run_gatk_liftover_vcf
from app.services.jobs import create_job, get_job, run_job
from app.services.ldblockshow import LDBLOCKSHOW_OUTPUT_DIR, run_ldblockshow
from app.services.plink import run_plink
from app.services.r_vcf_plots import RPLOT_OUTPUT_DIR, run_cmplot_association, run_qqman_association, run_r_vcf_plots
from app.services.samtools import run_samtools
from app.services.snpeff import run_snpeff
from app.services.summary_stats import load_summary_stats_rows
from app.services.source_bootstrap import (
    load_bootstrap_manifest,
    persist_uploaded_source_bytes,
    run_bootstrap_analysis,
)
from app.services.source_registry import detect_source_registration, infer_source_file_kind
from app.services.tool_runner import discover_tools
from app.services.workflow_agent import interpret_workflow_reply, start_workflow
from app.services.workflows import (
    analyze_prs_prep_workflow,
    analyze_vcf_workflow,
)


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_local_env()

app = FastAPI(title="Bioinformatics VCF Evidence MVP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3001",
        "http://localhost:3001",
        "http://127.0.0.1:3002",
        "http://localhost:3002",
        "http://127.0.0.1:3003",
        "http://localhost:3003",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PLUGINS_DIR = ROOT_DIR / "plugins"
WORKFLOWS_DIR = ROOT_DIR / "skills" / "chatgenome-orchestrator" / "workflows"


def _load_tool_manifests() -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    for manifest in sorted(PLUGINS_DIR.glob("*/tool.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            manifests.append(payload)
    return manifests


def _tool_aliases(manifest: dict[str, object]) -> list[str]:
    import re

    aliases: set[str] = set()
    name = str(manifest.get("name") or "").strip().lower()
    if name:
        aliases.add(name)
        simplified = re.sub(r"^(gatk_|bcftools_)", "", name)
        simplified = re.sub(r"_(execution|vcf|tool)$", "", simplified)
        simplified = re.sub(r"_+", "_", simplified).strip("_")
        if simplified:
            aliases.add(simplified)
            aliases.add(simplified.replace("_", ""))
            aliases.add(simplified.replace("_", "-"))
    routing = manifest.get("routing")
    if isinstance(routing, dict):
        for keyword in routing.get("trigger_keywords", []):
            text = str(keyword).strip().lower()
            if text:
                aliases.add(text)
    return sorted(aliases)


def _render_tool_help(manifest: dict[str, object]) -> str:
    name = str(manifest.get("name") or "tool")
    help_block = manifest.get("help")
    if not isinstance(help_block, dict):
        aliases = ", ".join(f"@{item}" for item in _tool_aliases(manifest)[:4])
        return (
            f"`{name}` is registered, but no curated help metadata is available yet.\n\n"
            f"- Try one of these aliases: {aliases or '@tool'}"
        )
    orchestration = manifest.get("orchestration") if isinstance(manifest.get("orchestration"), dict) else {}
    consumes = orchestration.get("consumes") if isinstance(orchestration, dict) else []
    input_hint = "active source"
    if isinstance(consumes, list):
        lowered = [str(item).lower() for item in consumes]
        if "vcf_path" in lowered:
            input_hint = "active VCF source"
        elif "alignment_file" in lowered:
            input_hint = "active BAM/SAM/CRAM source"
        elif "summary_stats_path" in lowered:
            input_hint = "active summary-statistics source"

    alias_list = [f"@{item}" for item in _tool_aliases(manifest)[:4]]
    primary_alias = alias_list[0] if alias_list else "@tool"
    lines: list[str] = [f"**{primary_alias}**", ""]
    summary = str(help_block.get("summary") or "").strip()
    if summary:
        lines.append(summary)
    else:
        lines.append(f"`{name}` is available in this build.")
    lines.append("")
    lines.append("Quick use")
    lines.append(f"- Run on: {input_hint}")
    lines.append(f"- Start with: `{primary_alias}`")
    lines.append(f"- Help: `{primary_alias} help`")
    modes = help_block.get("modes") or []
    if isinstance(modes, list) and modes:
        lines.append("")
        lines.append("Available modes")
        for mode in modes:
            if isinstance(mode, dict):
                mode_name = str(mode.get("name") or "").strip()
                mode_description = str(mode.get("description") or "").strip()
                if mode_name:
                    lines.append(f"- `{mode_name}`: {mode_description}")
    options = help_block.get("options") or []
    if isinstance(options, list) and options:
        lines.append("")
        lines.append("Options")
        for option in options:
            if not isinstance(option, dict):
                continue
            option_name = str(option.get("name") or "").strip()
            option_type = str(option.get("type") or "").strip()
            option_description = str(option.get("description") or "").strip()
            default = option.get("default")
            default_suffix = f" Default: `{default}`." if default not in (None, "") else ""
            if option_name:
                label = f"`{option_name}`"
                if option_type:
                    label += f" ({option_type})"
                lines.append(f"- {label}: {option_description}{default_suffix}")
    examples = help_block.get("examples") or []
    if isinstance(examples, list) and examples:
        lines.append("")
        lines.append("Examples")
        for example in examples:
            lines.append(f"- `{example}`")
    notes = help_block.get("notes") or []
    if isinstance(notes, list) and notes:
        lines.append("")
        lines.append("Notes")
        for note in notes:
            lines.append(f"- {note}")
    if alias_list:
        lines.append("")
        lines.append(f"Aliases: {', '.join(alias_list)}")
    return "\n".join(lines).strip()


def _safe_fastqc_artifact_path(path_str: str) -> Path:
    candidate = Path(path_str).resolve()
    root = FASTQC_OUTPUT_DIR.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access to the requested FastQC artifact is not allowed.") from exc
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"Output file not found: {candidate}")
    return candidate


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/tools", response_model=list[ToolInfo])
def list_registry_tools() -> list[ToolInfo]:
    return discover_tools()


@app.get("/api/v1/workflows")
def list_registry_workflows() -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    for manifest in sorted(WORKFLOWS_DIR.glob("*.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            manifests.append(payload)
    return manifests


@app.get("/api/v1/tools/help")
def get_tool_help(alias: str = Query(..., description="Tool alias such as snpeff, samtools, liftover, plink")) -> dict[str, object]:
    target = alias.strip().lower()
    for manifest in _load_tool_manifests():
        if target in _tool_aliases(manifest):
            return {
                "name": manifest.get("name"),
                "aliases": _tool_aliases(manifest),
                "help": _render_tool_help(manifest),
            }
    raise HTTPException(status_code=404, detail=f"Unknown tool alias: {alias}")


@app.get("/api/v1/files")
def get_output_file(path: str = Query(..., description="Absolute path to a generated output file")) -> FileResponse:
    file_path = Path(path).resolve()
    allowed_roots = [
        RPLOT_OUTPUT_DIR.resolve(),
        FASTQC_OUTPUT_DIR.resolve(),
        LDBLOCKSHOW_OUTPUT_DIR.resolve(),
    ]
    if not any(root == file_path or root in file_path.parents for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Access to the requested file is not allowed.")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Output file not found: {file_path}")
    return FileResponse(file_path)


@app.post("/api/v1/analysis/from-path", response_model=AnalysisResponse)
def analyze_from_path(request: FromPathRequest) -> AnalysisResponse:
    try:
        return analyze_vcf_workflow(
            request.vcf_path,
            annotation_scope=request.annotation_scope,
            annotation_limit=request.annotation_limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {exc}") from exc


@app.post("/api/v1/analysis/from-path/async", response_model=AnalysisJobResponse)
def analyze_from_path_async(request: FromPathRequest) -> AnalysisJobResponse:
    job_id = create_job()
    run_job(
        job_id,
        lambda: analyze_vcf_workflow(
            request.vcf_path,
            annotation_scope=request.annotation_scope,
            annotation_limit=request.annotation_limit,
        ).model_dump(),
    )
    job = get_job(job_id)
    return AnalysisJobResponse(job_id=job_id, status=job["status"])


@app.get("/api/v1/analysis/jobs/{job_id}", response_model=AnalysisJobResponse)
def get_analysis_job(job_id: str) -> AnalysisJobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    result = job["result"]
    parsed_result = AnalysisResponse(**result) if isinstance(result, dict) else None
    return AnalysisJobResponse(
        job_id=job_id,
        status=job["status"],
        result=parsed_result,
        error=job["error"],
    )


@app.post("/api/v1/chat/analysis", response_model=AnalysisChatResponse)
def chat_about_analysis(request: AnalysisChatRequest) -> AnalysisChatResponse:
    return answer_analysis_chat(request)


@app.post("/api/v1/chat/raw-qc", response_model=RawQcChatResponse)
def chat_about_raw_qc(request: RawQcChatRequest) -> RawQcChatResponse:
    return answer_raw_qc_chat(request)


@app.post("/api/v1/chat/summary-stats", response_model=SummaryStatsChatResponse)
def chat_about_summary_stats(request: SummaryStatsChatRequest) -> SummaryStatsChatResponse:
    return answer_summary_stats_chat(request)


@app.post("/api/v1/workflow/start", response_model=WorkflowAgentResponse)
def begin_workflow(request: WorkflowStartRequest) -> WorkflowAgentResponse:
    return start_workflow(request)


@app.post("/api/v1/workflow/reply", response_model=WorkflowAgentResponse)
def continue_workflow(request: WorkflowReplyRequest) -> WorkflowAgentResponse:
    return interpret_workflow_reply(request)


@app.post("/api/v1/filter/run", response_model=FilterResponse)
def run_filtering(request: FilterRequest) -> FilterResponse:
    try:
        return run_filter(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Filtering failed: {exc}") from exc


@app.post("/api/v1/snpeff/run", response_model=SnpEffResponse)
def run_snpeff_annotation(request: SnpEffRequest) -> SnpEffResponse:
    try:
        return run_snpeff(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"SnpEff failed: {exc}") from exc


@app.post("/api/v1/samtools/run", response_model=SamtoolsResponse)
def run_samtools_alignment_qc(request: SamtoolsRequest) -> SamtoolsResponse:
    try:
        return run_samtools(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"samtools failed: {exc}") from exc


@app.post("/api/v1/plink/run", response_model=PlinkResponse)
def run_plink_qc(request: PlinkRequest) -> PlinkResponse:
    try:
        return run_plink(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"PLINK 2 failed: {exc}") from exc


@app.post("/api/v1/liftover/run", response_model=GatkLiftoverVcfResponse)
def run_liftover_vcf(request: GatkLiftoverVcfRequest) -> GatkLiftoverVcfResponse:
    try:
        return run_gatk_liftover_vcf(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"GATK LiftoverVcf failed: {exc}") from exc


@app.post("/api/v1/ldblockshow/run", response_model=LDBlockShowResponse)
def run_ldblockshow_plot(request: LDBlockShowRequest) -> LDBlockShowResponse:
    try:
        return run_ldblockshow(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"LDBlockShow failed: {exc}") from exc


@app.post("/api/v1/r/plots", response_model=RPlotResponse)
def run_r_plots(request: RPlotRequest) -> RPlotResponse:
    try:
        return run_r_vcf_plots(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"R plotting failed: {exc}") from exc


@app.post("/api/v1/r/cmplot", response_model=RPlotResponse)
def run_cmplot(request: CmplotAssociationRequest) -> RPlotResponse:
    try:
        return run_cmplot_association(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CMplot association rendering failed: {exc}") from exc


@app.post("/api/v1/qqman/run", response_model=RPlotResponse)
def run_qqman(request: QqmanAssociationRequest) -> RPlotResponse:
    try:
        return run_qqman_association(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"qqman association rendering failed: {exc}") from exc


@app.post("/api/v1/analysis/upload", response_model=AnalysisResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    annotation_scope: str = Form("representative"),
    annotation_limit: Optional[int] = Form(None),
) -> AnalysisResponse:
    original_name = file.filename or "upload.vcf"
    detected = detect_source_registration(original_name)
    if detected is None or detected[0] != "vcf":
        raise HTTPException(status_code=400, detail="Only .vcf and .vcf.gz uploads are supported.")
    if load_bootstrap_manifest("vcf") is None:
        raise HTTPException(status_code=500, detail="The VCF bootstrap manifest is not available.")
    durable_path = persist_uploaded_source_bytes("vcf", original_name, await file.read())

    try:
        return run_bootstrap_analysis(
            "vcf",
            str(durable_path),
            original_name,
            annotation_scope=annotation_scope,
            annotation_limit=annotation_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {exc}") from exc


@app.post("/api/v1/raw-qc/upload", response_model=RawQcResponse)
async def analyze_raw_qc_upload(file: UploadFile = File(...)) -> RawQcResponse:
    filename = file.filename or "upload.fastq.gz"
    detected = detect_source_registration(filename)
    if detected is None or detected[0] != "raw_qc":
        raise HTTPException(
            status_code=400,
            detail="Only FASTQ, FASTQ.gz, FQ, FQ.gz, BAM, and SAM uploads are supported.",
        )
    if load_bootstrap_manifest("raw_qc") is None:
        raise HTTPException(status_code=500, detail="The raw-QC bootstrap manifest is not available.")
    durable_path = persist_uploaded_source_bytes("raw_qc", filename, await file.read())
    return run_bootstrap_analysis("raw_qc", str(durable_path), filename)


@app.post("/api/v1/summary-stats/upload", response_model=SummaryStatsResponse)
async def analyze_summary_stats_upload(
    file: UploadFile = File(...),
    genome_build: str = Form("unknown"),
    trait_type: str = Form("unknown"),
) -> SummaryStatsResponse:
    filename = file.filename or "summary_stats.tsv.gz"
    detected = detect_source_registration(filename)
    if detected is None or detected[0] != "summary_stats":
        raise HTTPException(
            status_code=400,
            detail="Only TSV/TXT/CSV summary statistics uploads are supported.",
        )
    if load_bootstrap_manifest("summary_stats") is None:
        raise HTTPException(status_code=500, detail="The summary-statistics bootstrap manifest is not available.")
    durable_path = persist_uploaded_source_bytes("summary_stats", filename, await file.read())
    try:
        return run_bootstrap_analysis(
            "summary_stats",
            str(durable_path),
            filename,
            genome_build=genome_build,
            trait_type=trait_type,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Summary statistics intake failed: {exc}") from exc


@app.post("/api/v1/source/upload", response_model=SourceReadyResponse)
async def upload_active_source(file: UploadFile = File(...)) -> SourceReadyResponse:
    filename = file.filename or "upload.dat"
    detected = detect_source_registration(filename)
    if detected is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported source type. Upload a VCF, raw sequencing file, or summary statistics file.",
        )
    source_type, _, matched_suffix = detected
    durable_path = persist_uploaded_source_bytes(source_type, filename, await file.read())
    response_payload: dict[str, object] = {
        "source_type": source_type,
        "file_name": filename,
        "source_path": str(durable_path),
    }
    file_kind = infer_source_file_kind(filename, source_type, matched_suffix)
    if file_kind:
        response_payload["file_kind"] = file_kind
    return SourceReadyResponse(**response_payload)


@app.post("/api/v1/summary-stats/rows", response_model=SummaryStatsRowsResponse)
async def analyze_summary_stats_rows(request: SummaryStatsRowsRequest) -> SummaryStatsRowsResponse:
    try:
        rows, has_more = load_summary_stats_rows(
            request.source_stats_path,
            offset=request.offset,
            limit=request.limit,
        )
        return SummaryStatsRowsResponse(
            rows=rows,
            offset=request.offset,
            limit=request.limit,
            returned=len(rows),
            has_more=has_more,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Summary statistics row loading failed: {exc}") from exc


@app.post("/api/v1/prs-prep/run", response_model=PrsPrepResponse)
async def run_prs_prep(request: PrsPrepRequest) -> PrsPrepResponse:
    try:
        return analyze_prs_prep_workflow(
            request.source_stats_path,
            request.file_name,
            genome_build=request.genome_build,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"PRS prep failed: {exc}") from exc


@app.get("/api/v1/raw-qc/report")
def get_raw_qc_report(path: str = Query(..., description="Absolute path to a FastQC artifact under outputs/fastqc")):
    artifact_path = _safe_fastqc_artifact_path(path)
    if artifact_path.suffix.lower() == ".html":
        return HTMLResponse(artifact_path.read_text(encoding="utf-8"))
    return FileResponse(artifact_path, media_type="application/zip", filename=artifact_path.name)
