from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.models import (
    AnalysisFacts,
    AnalysisChatRequest,
    AnalysisChatResponse,
    AnalysisJobResponse,
    AnalysisResponse,
    GatkLiftoverVcfRequest,
    GatkLiftoverVcfResponse,
    LDBlockShowRequest,
    LDBlockShowResponse,
    CountSummaryItem,
    DetailedCountSummaryItem,
    CmplotAssociationRequest,
    FilterRequest,
    FilterResponse,
    FromPathRequest,
    PlinkRequest,
    PlinkResponse,
    RankedCandidate,
    RawQcChatRequest,
    RawQcChatResponse,
    RawQcResponse,
    RohSegment,
    RPlotRequest,
    RPlotResponse,
    SnpEffRequest,
    SnpEffResponse,
    SamtoolsRequest,
    SamtoolsResponse,
    SummaryStatsRowsRequest,
    SummaryStatsRowsResponse,
    SummaryStatsChatRequest,
    SummaryStatsChatResponse,
    SummaryStatsResponse,
    SymbolicAltSummary,
    ToolInfo,
    VariantAnnotation,
    WorkflowAgentResponse,
    WorkflowReplyRequest,
    WorkflowStartRequest,
)
from app.services.annotation import build_draft_answer, build_ui_cards
from app.services.candidate_ranking import build_ranked_candidates
from app.services.chat import answer_analysis_chat, answer_raw_qc_chat, answer_summary_stats_chat
from app.services.fastqc import FASTQC_OUTPUT_DIR
from app.services.filtering import run_filter
from app.services.gatk_liftover import run_gatk_liftover_vcf
from app.services.jobs import create_job, get_job, run_job
from app.services.ldblockshow import LDBLOCKSHOW_OUTPUT_DIR, run_ldblockshow
from app.services.plink import run_plink
from app.services.samtools import run_samtools
from app.services.recommendation import build_recommendations
from app.services.references import build_reference_bundle
from app.services.r_vcf_plots import RPLOT_OUTPUT_DIR, run_cmplot_association, run_r_vcf_plots
from app.services.roh_analysis import run_roh_analysis
from app.services.snpeff import run_snpeff
from app.services.summary_stats import analyze_summary_stats, load_summary_stats_rows
from app.services.tool_runner import discover_tools, run_tool
from app.services.variant_annotation import annotate_variants
from app.services.vcf_summary import summarize_vcf
from app.services.workflow_agent import interpret_workflow_reply, start_workflow


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
ANALYSIS_UPLOAD_DIR = ROOT_DIR / "uploads" / "analysis"
RAW_QC_UPLOAD_DIR = ROOT_DIR / "uploads" / "raw_qc"
SUMMARY_STATS_UPLOAD_DIR = ROOT_DIR / "uploads" / "summary_stats"
PLUGINS_DIR = ROOT_DIR / "plugins"


def _annotation_key(item: VariantAnnotation) -> tuple[str, int, str, tuple[str, ...]]:
    return (item.contig, item.pos_1based, item.ref, tuple(item.alts))


def _snpeff_genome_from_build(genome_build_guess: str | None) -> str:
    value = (genome_build_guess or "").lower()
    if any(token in value for token in ("38", "hg38", "grch38")):
        return "GRCh38.99"
    return "GRCh37.75"


def _analyze_vcf(
    path: str,
    annotation_scope: str = "representative",
    annotation_limit: int | None = None,
) -> AnalysisResponse:
    max_examples = int(os.getenv("MAX_EXAMPLE_VARIANTS", "8"))
    used_tools: list[str] = []
    tool_registry = discover_tools()

    try:
        qc_result = run_tool(
            "vcf_qc_tool",
            {
                "vcf_path": path,
                "max_examples": max_examples,
            },
        )
        facts = AnalysisFacts(**qc_result["facts"])
        used_tools.append("vcf_qc_tool")
    except Exception:
        facts = summarize_vcf(path, max_examples=max_examples)

    try:
        annotation_result = run_tool(
            "annotation_tool",
            {
                "vcf_path": path,
                "facts": facts.model_dump(),
                "scope": annotation_scope,
                "limit": annotation_limit,
            },
        )
        annotations = [VariantAnnotation(**item) for item in annotation_result["annotations"]]
        used_tools.append("annotation_tool")
    except Exception:
        annotations = annotate_variants(
            path,
            facts,
            scope=annotation_scope,
            limit=annotation_limit,
        )
    snpeff_result: SnpEffResponse | None = None
    try:
        snpeff_payload = run_tool(
            "snpeff_execution_tool",
            {
                "vcf_path": path,
                "genome": _snpeff_genome_from_build(facts.genome_build_guess),
                "output_prefix": f"{Path(path).stem}.aux",
                "parse_limit": 10,
            },
        )
        snpeff_result = SnpEffResponse(**snpeff_payload)
        used_tools.append("snpeff_execution_tool")
    except Exception:
        snpeff_result = None
    try:
        roh_result = run_tool("roh_analysis_tool", {"vcf_path": path})
        roh_segments = [RohSegment(**item) for item in roh_result["roh_segments"]]
        used_tools.append("roh_analysis_tool")
    except Exception:
        roh_segments = run_roh_analysis(path)

    preliminary_candidates = build_ranked_candidates(annotations, roh_segments, limit=24)
    shortlisted_annotations = [entry.item for entry in preliminary_candidates]
    try:
        cadd_result = run_tool(
            "cadd_lookup_tool",
            {
                "annotations": [item.model_dump() for item in shortlisted_annotations],
                "genome_build_guess": facts.genome_build_guess,
            },
        )
        enriched_shortlisted_annotations = [VariantAnnotation(**item) for item in cadd_result["annotations"]]
        if bool(cadd_result.get("lookup_performed")):
            used_tools.append("cadd_lookup_tool")
        enriched_by_key = {_annotation_key(item): item for item in enriched_shortlisted_annotations}
        annotations = [enriched_by_key.get(_annotation_key(item), item) for item in annotations]
    except Exception:
        enriched_shortlisted_annotations = shortlisted_annotations

    try:
        revel_result = run_tool(
            "revel_lookup_tool",
            {
                "annotations": [item.model_dump() for item in enriched_shortlisted_annotations],
                "genome_build_guess": facts.genome_build_guess,
            },
        )
        revel_enriched_annotations = [VariantAnnotation(**item) for item in revel_result["annotations"]]
        if bool(revel_result.get("lookup_performed")):
            used_tools.append("revel_lookup_tool")
        revel_by_key = {_annotation_key(item): item for item in revel_enriched_annotations}
        annotations = [revel_by_key.get(_annotation_key(item), item) for item in annotations]
        enriched_shortlisted_annotations = [revel_by_key.get(_annotation_key(item), item) for item in enriched_shortlisted_annotations]
    except Exception:
        pass

    try:
        candidate_result = run_tool(
            "candidate_ranking_tool",
            {
                "annotations": [item.model_dump() for item in enriched_shortlisted_annotations],
                "roh_segments": [item.model_dump() for item in roh_segments],
                "limit": 8,
            },
        )
        candidate_variants = [RankedCandidate(**item) for item in candidate_result["candidate_variants"]]
        used_tools.append("candidate_ranking_tool")
    except Exception:
        candidate_variants = build_ranked_candidates(enriched_shortlisted_annotations, roh_segments, limit=8)
    try:
        clinvar_result = run_tool(
            "clinvar_review_tool",
            {"annotations": [item.model_dump() for item in annotations]},
        )
        clinvar_summary = [CountSummaryItem(**item) for item in clinvar_result["clinvar_summary"]]
        used_tools.append("clinvar_review_tool")
    except Exception:
        counts: dict[str, int] = {}
        for item in annotations:
            key = item.clinical_significance.strip() if item.clinical_significance and item.clinical_significance != "." else "Unreviewed"
            counts[key] = counts.get(key, 0) + 1
        clinvar_summary = [CountSummaryItem(label=label, count=count) for label, count in sorted(counts.items(), key=lambda part: part[1], reverse=True)]
    try:
        consequence_result = run_tool(
            "vep_consequence_tool",
            {
                "annotations": [item.model_dump() for item in annotations],
                "limit": 10,
            },
        )
        consequence_summary = [CountSummaryItem(**item) for item in consequence_result["consequence_summary"]]
        used_tools.append("vep_consequence_tool")
    except Exception:
        counts = {}
        for item in annotations:
            key = item.consequence.strip() if item.consequence and item.consequence != "." else "Unclassified"
            counts[key] = counts.get(key, 0) + 1
        consequence_summary = [
            CountSummaryItem(label=label, count=count)
            for label, count in sorted(counts.items(), key=lambda part: part[1], reverse=True)[:10]
        ]
    try:
        coverage_result = run_tool(
            "clinical_coverage_tool",
            {"annotations": [item.model_dump() for item in annotations]},
        )
        clinical_coverage_summary = [
            DetailedCountSummaryItem(**item) for item in coverage_result["clinical_coverage_summary"]
        ]
        used_tools.append("clinical_coverage_tool")
    except Exception:
        total = len(annotations)

        def detail(label: str, count: int) -> DetailedCountSummaryItem:
            percent = round((count / total) * 100) if total else 0
            return DetailedCountSummaryItem(label=label, count=count, detail=f"{count}/{total} annotated ({percent}%)")

        clinical_coverage_summary = [
            detail("ClinVar coverage", sum(1 for item in annotations if (item.clinical_significance and item.clinical_significance != ".") or (item.clinvar_conditions and item.clinvar_conditions != "."))),
            detail("gnomAD coverage", sum(1 for item in annotations if item.gnomad_af and item.gnomad_af != ".")),
            detail("Gene mapping", sum(1 for item in annotations if item.gene and item.gene != ".")),
            detail("HGVS coverage", sum(1 for item in annotations if (item.hgvsc and item.hgvsc != ".") or (item.hgvsp and item.hgvsp != "."))),
            detail("Protein change", sum(1 for item in annotations if item.hgvsp and item.hgvsp != ".")),
        ]
    try:
        filtering_result = run_tool(
            "filtering_view_tool",
            {"annotations": [item.model_dump() for item in annotations]},
        )
        filtering_summary = [DetailedCountSummaryItem(**item) for item in filtering_result["filtering_summary"]]
        used_tools.append("filtering_view_tool")
    except Exception:
        unique_genes = {item.gene.strip() for item in annotations if item.gene and item.gene.strip() not in {"", "."}}
        clinvar_labeled = sum(1 for item in annotations if item.clinical_significance and item.clinical_significance != ".")
        symbolic = sum(1 for item in annotations if any(alt.startswith("<") and alt.endswith(">") for alt in item.alts))
        filtering_summary = [
            DetailedCountSummaryItem(label="Annotated rows", count=len(annotations), detail=f"{len(annotations)} rows currently available in the triage table"),
            DetailedCountSummaryItem(label="Distinct genes", count=len(unique_genes), detail=f"{len(unique_genes)} genes represented in the annotated subset"),
            DetailedCountSummaryItem(label="ClinVar-labeled rows", count=clinvar_labeled, detail=f"{clinvar_labeled} rows contain a ClinVar-style significance label"),
            DetailedCountSummaryItem(label="Symbolic ALT rows", count=symbolic, detail=f"{symbolic} rows are symbolic ALT records that may need separate handling"),
        ]
    try:
        symbolic_result = run_tool(
            "symbolic_alt_tool",
            {"annotations": [item.model_dump() for item in annotations]},
        )
        symbolic_alt_summary = SymbolicAltSummary(**symbolic_result["symbolic_alt_summary"])
        used_tools.append("symbolic_alt_tool")
    except Exception:
        symbolic_items = [item for item in annotations if any(alt.startswith("<") and alt.endswith(">") for alt in item.alts)]
        symbolic_alt_summary = SymbolicAltSummary(
            count=len(symbolic_items),
            examples=[
                {
                    "locus": f"{item.contig}:{item.pos_1based}",
                    "gene": item.gene or "",
                    "alts": item.alts,
                    "consequence": item.consequence or "",
                    "genotype": item.genotype or "",
                }
                for item in symbolic_items[:5]
            ],
        )
    reference_annotations = annotations[: min(len(annotations), 20)]
    references = build_reference_bundle(facts, reference_annotations)
    recommendations = build_recommendations(facts)
    ui_cards = build_ui_cards(facts, annotations)
    try:
        summary_result = run_tool(
            "grounded_summary_tool",
            {
                "facts": facts.model_dump(),
                "annotations": [item.model_dump() for item in annotations],
                "references": [item.model_dump() for item in references],
                "recommendations": [item.model_dump() for item in recommendations],
            },
        )
        draft_answer = str(summary_result["draft_answer"])
        used_tools.append("grounded_summary_tool")
    except Exception:
        draft_answer = build_draft_answer(
            facts,
            annotations,
            [item.id for item in references],
            [item.id for item in recommendations],
        )
    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        facts=facts,
        annotations=annotations,
        roh_segments=roh_segments,
        source_vcf_path=path,
        snpeff_result=snpeff_result,
        candidate_variants=candidate_variants,
        clinvar_summary=clinvar_summary,
        consequence_summary=consequence_summary,
        clinical_coverage_summary=clinical_coverage_summary,
        filtering_summary=filtering_summary,
        symbolic_alt_summary=symbolic_alt_summary,
        references=references,
        recommendations=recommendations,
        ui_cards=ui_cards,
        draft_answer=draft_answer,
        used_tools=used_tools,
        tool_registry=tool_registry,
    )


def _is_raw_qc_filename(file_name: str) -> bool:
    lowered = file_name.lower()
    return lowered.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz", ".bam", ".sam"))


def _is_summary_stats_filename(file_name: str) -> bool:
    lowered = file_name.lower()
    return lowered.endswith(
        (
            ".tsv",
            ".tsv.gz",
            ".txt",
            ".txt.gz",
            ".csv",
            ".csv.gz",
            ".sumstats",
            ".sumstats.gz",
        )
    )


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


def _analyze_raw_qc(path: str, original_name: str) -> RawQcResponse:
    try:
        result = run_tool(
            "fastqc_execution_tool",
            {
                "raw_path": path,
                "original_name": original_name,
            },
        )
        return RawQcResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Raw QC failed: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/tools", response_model=list[ToolInfo])
def list_registry_tools() -> list[ToolInfo]:
    return discover_tools()


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
        return _analyze_vcf(
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
        lambda: _analyze_vcf(
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


@app.post("/api/v1/analysis/upload", response_model=AnalysisResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    annotation_scope: str = Form("representative"),
    annotation_limit: Optional[int] = Form(None),
) -> AnalysisResponse:
    original_name = file.filename or "upload.vcf"
    suffixes = Path(original_name).suffixes
    combined_suffix = "".join(suffixes) or Path(original_name).suffix or ".vcf"
    if not (combined_suffix.endswith(".vcf") or combined_suffix.endswith(".vcf.gz")):
        raise HTTPException(status_code=400, detail="Only .vcf and .vcf.gz uploads are supported.")
    ANALYSIS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in Path(original_name).stem)
    durable_path = ANALYSIS_UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}_{safe_stem}{combined_suffix}"
    durable_path.write_bytes(await file.read())

    try:
        return _analyze_vcf(
            str(durable_path),
            annotation_scope=annotation_scope,
            annotation_limit=annotation_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {exc}") from exc


@app.post("/api/v1/raw-qc/upload", response_model=RawQcResponse)
async def analyze_raw_qc_upload(file: UploadFile = File(...)) -> RawQcResponse:
    filename = file.filename or "upload.fastq.gz"
    if not _is_raw_qc_filename(filename):
        raise HTTPException(
            status_code=400,
            detail="Only FASTQ, FASTQ.gz, FQ, FQ.gz, BAM, and SAM uploads are supported.",
        )

    suffixes = "".join(Path(filename).suffixes) or Path(filename).suffix or ".dat"
    RAW_QC_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in Path(filename).stem)
    durable_path = RAW_QC_UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}_{safe_stem}{suffixes}"
    durable_path.write_bytes(await file.read())
    return _analyze_raw_qc(str(durable_path), filename)


@app.post("/api/v1/summary-stats/upload", response_model=SummaryStatsResponse)
async def analyze_summary_stats_upload(
    file: UploadFile = File(...),
    genome_build: str = Form("unknown"),
    trait_type: str = Form("unknown"),
) -> SummaryStatsResponse:
    filename = file.filename or "summary_stats.tsv.gz"
    if not _is_summary_stats_filename(filename):
        raise HTTPException(
            status_code=400,
            detail="Only TSV/TXT/CSV summary statistics uploads are supported.",
        )

    suffixes = "".join(Path(filename).suffixes) or Path(filename).suffix or ".tsv"
    SUMMARY_STATS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in Path(filename).stem)
    durable_path = SUMMARY_STATS_UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}_{safe_stem}{suffixes}"
    durable_path.write_bytes(await file.read())
    try:
        result = analyze_summary_stats(str(durable_path), filename, genome_build=genome_build, trait_type=trait_type)
        result.analysis_id = str(uuid.uuid4())
        result.tool_registry = discover_tools()
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Summary statistics intake failed: {exc}") from exc


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


@app.get("/api/v1/raw-qc/report")
def get_raw_qc_report(path: str = Query(..., description="Absolute path to a FastQC artifact under outputs/fastqc")):
    artifact_path = _safe_fastqc_artifact_path(path)
    if artifact_path.suffix.lower() == ".html":
        return HTMLResponse(artifact_path.read_text(encoding="utf-8"))
    return FileResponse(artifact_path, media_type="application/zip", filename=artifact_path.name)
