from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from app.models import (
    AnalysisChatRequest,
    AnalysisChatResponse,
    RawQcChatRequest,
    RawQcChatResponse,
    SummaryStatsChatRequest,
    SummaryStatsChatResponse,
)
from app.models import (
    GatkLiftoverVcfRequest,
    LDBlockShowRequest,
    LDBlockShowResponse,
    PlinkRequest,
    SamtoolsRequest,
    SnpEffRequest,
)
from app.services.gatk_liftover import (
    DEFAULT_CHAIN_FILE,
    DEFAULT_TARGET_FASTA,
    run_gatk_liftover_vcf,
)
from app.services.ldblockshow import run_ldblockshow
from app.services.plink import run_plink
from app.services.samtools import run_samtools
from app.services.snpeff import run_snpeff

ROOT_DIR = Path(__file__).resolve().parents[2]
PLUGINS_DIR = ROOT_DIR / "plugins"


def _load_direct_tool_routing_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for tool_name in (
        "gatk_liftover_vcf_tool",
        "ldblockshow_execution_tool",
        "plink_execution_tool",
        "snpeff_execution_tool",
    ):
        manifest = PLUGINS_DIR / tool_name / "tool.json"
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        routing = payload.get("routing")
        if isinstance(routing, dict):
            specs.append({"name": payload.get("name", tool_name), "routing": routing})
    return specs


def _match_direct_tool_request(question: str) -> dict[str, object] | None:
    lowered = question.lower()
    for spec in _load_direct_tool_routing_specs():
        routing = spec.get("routing") or {}
        trigger_keywords = [str(item).lower() for item in routing.get("trigger_keywords", [])]
        execution_intents = [str(item).lower() for item in routing.get("execution_intents", [])]
        if not any(keyword in lowered for keyword in trigger_keywords):
            continue
        if execution_intents and not any(intent in lowered for intent in execution_intents):
            continue
        return spec
    return None


def _is_korean(text: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7a3]", text))


def _flatten_studio_context(studio_context: dict) -> dict[str, object]:
    return {
        "active_view": studio_context.get("active_view"),
        "qc_summary": studio_context.get("qc_summary"),
        "clinical_coverage": studio_context.get("clinical_coverage"),
        "symbolic_alt_review": studio_context.get("symbolic_alt_review"),
        "roh_review": studio_context.get("roh_review"),
        "candidate_variants": studio_context.get("candidate_variants"),
        "clinvar_review": studio_context.get("clinvar_review"),
        "vep_consequence": studio_context.get("vep_consequence"),
        "snpeff_preview": studio_context.get("snpeff_preview"),
        "selected_annotation": studio_context.get("selected_annotation"),
    }


def _has_studio_trigger(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in ("$studio", "$current analysis", "$current card", "$grounded"))


def _strip_studio_triggers(question: str) -> str:
    cleaned = question
    for token in ("$studio", "$current analysis", "$current card", "$grounded"):
        cleaned = re.sub(re.escape(token), " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _needs_grounded_clarification(question: str) -> bool:
    if not _has_studio_trigger(question):
        return False
    stripped = _strip_studio_triggers(question)
    if not stripped:
        return True
    tokens = stripped.split()
    return len(tokens) < 2 and len(stripped) < 16


def _grounded_clarification_text() -> str:
    return (
        "Grounded mode is on.\n\n"
        "- Tell me which Studio result you want me to use.\n"
        "- Example: `$studio candidate card 설명해줘`\n"
        "- Example: `$studio ROH 결과를 요약해줘`\n"
        "- Example: `$studio summary statistics review를 정리해줘`"
    )


def _compact_analysis_context(payload: AnalysisChatRequest) -> dict[str, object]:
    analysis = payload.analysis
    context = {
        "analysis_id": analysis.analysis_id,
        "draft_answer": analysis.draft_answer,
        "used_tools": analysis.used_tools,
        "tool_registry": [
            {
                "name": item.name,
                "task": item.task,
                "source": item.source,
            }
            for item in analysis.tool_registry
        ],
        "facts": {
            "file_name": analysis.facts.file_name,
            "genome_build_guess": analysis.facts.genome_build_guess,
            "record_count": analysis.facts.record_count,
            "samples": analysis.facts.samples,
            "variant_types": analysis.facts.variant_types,
            "warnings": analysis.facts.warnings,
        },
        "annotations": [
            {
                "pos": item.pos_1based,
                "gene": item.gene,
                "consequence": item.consequence,
                "rsid": item.rsid,
                "clinical_significance": item.clinical_significance,
                "condition": item.clinvar_conditions,
                "gnomad_af": item.gnomad_af,
            }
            for item in payload.analysis.annotations[:6]
        ],
        "snpeff_result": (
            {
                "tool": payload.analysis.snpeff_result.tool,
                "genome": payload.analysis.snpeff_result.genome,
                "output_path": payload.analysis.snpeff_result.output_path,
                "parsed_records": [
                    {
                        "contig": item.contig,
                        "pos_1based": item.pos_1based,
                        "ref": item.ref,
                        "alt": item.alt,
                        "ann": [
                            {
                                "annotation": ann.annotation,
                                "impact": ann.impact,
                                "gene_name": ann.gene_name,
                                "hgvs_c": ann.hgvs_c,
                                "hgvs_p": ann.hgvs_p,
                            }
                            for ann in item.ann[:3]
                        ],
                    }
                    for item in payload.analysis.snpeff_result.parsed_records[:5]
                ],
            }
            if payload.analysis.snpeff_result
            else None
        ),
        "ldblockshow_result": (
            {
                "tool": payload.analysis.ldblockshow_result.tool,
                "region": payload.analysis.ldblockshow_result.region,
                "svg_path": payload.analysis.ldblockshow_result.svg_path,
                "png_path": payload.analysis.ldblockshow_result.png_path,
                "pdf_path": payload.analysis.ldblockshow_result.pdf_path,
                "warnings": payload.analysis.ldblockshow_result.warnings,
            }
            if payload.analysis.ldblockshow_result
            else None
        ),
        "roh_segments": [
            {
                "sample": item.sample,
                "contig": item.contig,
                "start_1based": item.start_1based,
                "end_1based": item.end_1based,
                "length_bp": item.length_bp,
                "marker_count": item.marker_count,
                "quality": item.quality,
            }
            for item in payload.analysis.roh_segments[:6]
        ],
        "references": [
            {"id": item.id, "title": item.title, "url": item.url}
            for item in payload.analysis.references[:8]
        ],
        "recommendations": [
            {"id": item.id, "title": item.title, "action": item.action}
            for item in payload.analysis.recommendations[:6]
        ],
    }
    if payload.studio_context:
        context["studio_context"] = _flatten_studio_context(payload.studio_context)
    return context

def _fallback_answer(payload: AnalysisChatRequest) -> AnalysisChatResponse:
    if _has_studio_trigger(payload.question):
        answer = (
            "I could not complete the grounded Studio response right now.\n\n"
            "- The request was recognized as `$studio` grounded chat.\n"
            "- Please retry the question, or ask again after the backend model connection is restored."
        )
    else:
        answer = (
            "I could not complete the general GPT response right now.\n\n"
            "- Please retry the question.\n"
            "- If you want the answer grounded in the current Studio state, add `$studio` to the message."
        )
    return AnalysisChatResponse(answer=answer, citations=[], used_fallback=True)


def _call_openai(payload: AnalysisChatRequest) -> AnalysisChatResponse:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    if not api_key:
        return _fallback_answer(payload)

    grounded = _has_studio_trigger(payload.question)
    if grounded:
        compact_context = _compact_analysis_context(payload)
        system_prompt = (
            "You are a genomics analysis copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer from the provided VCF analysis context and do not invent variant facts. "
            "Treat analysis.used_tools as the authoritative record of which deterministic tools were actually run in the current analysis. "
            "Do not claim that a tool was used unless it appears in analysis.used_tools or the user explicitly requests a new run. "
            "Do not infer tool execution only from card names such as VEP consequence summaries. "
            "Treat studio_context as part of the trusted analysis state, including ROH, coverage, candidate, ClinVar, and consequence summaries when present. "
            "When possible, cite reference ids like REF1 or REF4 inline. "
            "Format the answer in clean Markdown with short sections or bullet points when helpful. "
            "Label the answer as analysis-grounded."
        )
        user_content = (
            "Question:\n"
            f"{payload.question}\n\n"
            "Analysis context JSON:\n"
            f"{json.dumps(compact_context, ensure_ascii=False)}"
        )
    else:
        system_prompt = (
            "You are a helpful general assistant. "
            "The user did not request analysis grounding. "
            "Answer from general knowledge only and ignore any uploaded analysis/studio context. "
            "Do not mention analysis context, Studio cards, or uploaded-file facts unless the user explicitly asks with a grounding trigger such as $studio."
        )
        user_content = payload.question
    history_lines = [{"role": turn.role, "content": turn.content} for turn in payload.history[-6:]]
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            *history_lines,
            {"role": "user", "content": user_content},
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))) as response:
        result = json.loads(response.read().decode("utf-8"))

    output_text = result.get("output_text")
    if not output_text:
        output = result.get("output", [])
        texts: list[str] = []
        for item in output:
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
        output_text = "\n".join(texts).strip()

    citations = sorted(set(re.findall(r"\bREF\d+\b", output_text or "")))
    return AnalysisChatResponse(
        answer=output_text or _fallback_answer(payload).answer,
        citations=citations,
        used_fallback=False,
    )


def _extract_ldblockshow_region(question: str) -> str | None:
    match = re.search(r"((?:chr)?[A-Za-z0-9_]+):(\d+)(?:[:-])(\d+)", question, flags=re.IGNORECASE)
    if not match:
        return None
    chrom, start, end = match.group(1), match.group(2), match.group(3)
    return f"{chrom}:{start}:{end}"


def _snpeff_genome_from_build(build_guess: str | None) -> str:
    guess = (build_guess or "").lower()
    if any(token in guess for token in ("38", "hg38", "grch38")):
        return "GRCh38.99"
    return "GRCh37.75"


def _extract_liftover_target_build(question: str, build_guess: str | None) -> tuple[str, str]:
    lowered = question.lower()
    if any(token in lowered for token in ("hg38", "grch38", "38")):
        return "GRCh38", "hg38"
    if any(token in lowered for token in ("hg19", "grch37", "37")):
        return "GRCh37", "hg19"
    current_guess = (build_guess or "").lower()
    if any(token in current_guess for token in ("37", "hg19", "grch37")):
        return "GRCh38", "hg38"
    return "GRCh38", "hg38"


def _handle_liftover_request(payload: AnalysisChatRequest) -> AnalysisChatResponse:
    source_vcf_path = payload.analysis.source_vcf_path
    if not source_vcf_path:
        return AnalysisChatResponse(
            answer="The current analysis context does not include a source VCF path, so liftover cannot be run from this chat turn.",
            citations=[],
            used_fallback=True,
            used_tools=["gatk_liftover_vcf_tool"],
        )

    target_build, target_label = _extract_liftover_target_build(
        payload.question,
        payload.analysis.facts.genome_build_guess,
    )
    existing = payload.analysis.liftover_result
    if existing is not None and (existing.target_build or "").lower() == target_build.lower():
        return AnalysisChatResponse(
            answer=(
                f"Liftover results are already available for the current VCF.\n\n"
                f"- Source build: `{existing.source_build or 'unknown'}`\n"
                f"- Target build: `{existing.target_build or target_build}`\n"
                f"- Lifted VCF: `{existing.output_path}`\n"
                f"- Reject VCF: `{existing.reject_path}`\n\n"
                "The existing LiftOver Review card has been reused instead of rerunning the tool."
            ),
            citations=[],
            used_fallback=False,
            used_tools=["gatk_liftover_vcf_tool"],
            liftover_result=existing,
        )

    result = run_gatk_liftover_vcf(
        GatkLiftoverVcfRequest(
            vcf_path=source_vcf_path,
            target_reference_fasta=str(DEFAULT_TARGET_FASTA),
            chain_file=str(DEFAULT_CHAIN_FILE),
            source_build=payload.analysis.facts.genome_build_guess or None,
            target_build=target_build,
            output_prefix=f"{payload.analysis.analysis_id}-liftover-{target_label}",
            parse_limit=8,
        )
    )
    return AnalysisChatResponse(
        answer=(
            f"GATK LiftoverVcf was run for the current VCF.\n\n"
            f"- Source build: `{result.source_build or 'unknown'}`\n"
            f"- Target build: `{result.target_build or target_build}`\n"
            f"- Lifted records: {result.lifted_record_count if result.lifted_record_count is not None else 'unknown'}\n"
            f"- Rejected records: {result.rejected_record_count if result.rejected_record_count is not None else 'unknown'}\n"
            f"- Lifted VCF: `{result.output_path}`\n"
            f"- Reject VCF: `{result.reject_path}`\n\n"
            "The Studio card has been updated with the latest liftover result."
        ),
        citations=[],
        used_fallback=False,
        used_tools=["gatk_liftover_vcf_tool"],
        liftover_result=result,
    )


def _handle_snpeff_request(payload: AnalysisChatRequest) -> AnalysisChatResponse:
    source_vcf_path = payload.analysis.source_vcf_path
    if not source_vcf_path:
        return AnalysisChatResponse(
            answer="The current analysis context does not include a source VCF path, so SnpEff cannot be run from this chat turn.",
            citations=[],
            used_fallback=True,
            used_tools=["snpeff_execution_tool"],
        )

    existing = payload.analysis.snpeff_result
    if existing is not None:
        return AnalysisChatResponse(
            answer=(
                f"SnpEff results are already available for the current VCF using genome `{existing.genome}`.\n\n"
                f"- Output path: `{existing.output_path}`\n"
                f"- Preview records: {len(existing.parsed_records)}\n\n"
                "The existing SnpEff Review card has been reused instead of rerunning the tool."
            ),
            citations=[],
            used_fallback=False,
            used_tools=["snpeff_execution_tool"],
        )

    result = run_snpeff(
        SnpEffRequest(
            vcf_path=source_vcf_path,
            genome=_snpeff_genome_from_build(payload.analysis.facts.genome_build_guess),
            output_prefix=f"{payload.analysis.analysis_id}-snpeff",
            parse_limit=10,
        )
    )
    return AnalysisChatResponse(
        answer=(
            f"SnpEff was run on the current VCF using genome `{result.genome}`.\n\n"
            f"- Output path: `{result.output_path}`\n"
            f"- Preview records: {len(result.parsed_records)}\n\n"
            "The SnpEff Review card has been updated with the latest result."
        ),
        citations=[],
        used_fallback=False,
        used_tools=["snpeff_execution_tool"],
    )


def _handle_plink_request(payload: AnalysisChatRequest) -> AnalysisChatResponse:
    source_vcf_path = payload.analysis.source_vcf_path
    if not source_vcf_path:
        return AnalysisChatResponse(
            answer="The current analysis context does not include a source VCF path, so PLINK cannot be run from this chat turn.",
            citations=[],
            used_fallback=True,
            used_tools=["plink_execution_tool"],
            requested_view="plink",
        )

    existing = payload.analysis.plink_result
    answer = (
        "The PLINK card is ready in Studio.\n\n"
        "- This ChatGenome build currently exposes PLINK as a deterministic QC workflow.\n"
        "- You can choose the QC options, review the command preview, and run PLINK from the card.\n"
        "- The result will appear in the same card after execution."
    )
    if existing is not None:
        answer += (
            f"\n\nAn existing PLINK result is already present for this analysis:\n"
            f"- Output prefix: `{existing.output_prefix}`\n"
            f"- Frequency rows: {len(existing.freq_rows)}\n"
            f"- Missingness rows: {len(existing.missing_rows)}\n"
            f"- Hardy rows: {len(existing.hardy_rows)}"
        )
    return AnalysisChatResponse(
        answer=answer,
        citations=[],
        used_fallback=False,
        used_tools=payload.analysis.used_tools or [],
        requested_view="plink",
        plink_result=existing,
    )


def _handle_ldblockshow_request(payload: AnalysisChatRequest) -> AnalysisChatResponse:
    source_vcf_path = payload.analysis.source_vcf_path
    region = _extract_ldblockshow_region(payload.question)

    if not source_vcf_path:
        return AnalysisChatResponse(
            answer="The current analysis context does not include a source VCF path, so LDBlockShow cannot be run from this chat turn.",
            citations=[],
            used_fallback=True,
            used_tools=["ldblockshow_execution_tool"],
            ldblockshow_result=LDBlockShowResponse(
                tool="ldblockshow",
                input_path="",
                region=region or "unknown",
                output_prefix="",
                command_preview="LDBlockShow -InVCF <source.vcf.gz> -Region chr:start:end ...",
                warnings=["The current analysis context does not include a source VCF path."],
            ),
        )

    if not region:
        return AnalysisChatResponse(
            answer="LDBlockShow needs a concrete region in `chr:start:end` format. Example: `Run LDBlockShow on chr11:24100000:24200000`.",
            citations=[],
            used_fallback=True,
            used_tools=["ldblockshow_execution_tool"],
            ldblockshow_result=LDBlockShowResponse(
                tool="ldblockshow",
                input_path=source_vcf_path,
                region="unknown",
                output_prefix="",
                command_preview="LDBlockShow -InVCF <source.vcf.gz> -Region chr:start:end ...",
                warnings=["No valid region was found in the request."],
            ),
        )

    result = run_ldblockshow(
        LDBlockShowRequest(
            vcf_path=source_vcf_path,
            region=region,
            sele_var=2,
            block_type=5,
            out_png=False,
            out_pdf=False,
        )
    )
    answer = (
        f"LDBlockShow was run on the current VCF for region `{result.region}`.\n\n"
        f"- Primary artifact: `{result.svg_path or result.png_path or result.pdf_path or 'not available'}`\n"
        f"- Block table: `{result.block_path or 'not available'}`\n"
        f"- Warnings: {len(result.warnings)}\n\n"
        "The Studio card has been updated with the latest LD block result."
    )
    return AnalysisChatResponse(
        answer=answer,
        citations=[],
        used_fallback=False,
        used_tools=["ldblockshow_execution_tool"],
        ldblockshow_result=result,
    )


def answer_analysis_chat(payload: AnalysisChatRequest) -> AnalysisChatResponse:
    if _needs_grounded_clarification(payload.question):
        return AnalysisChatResponse(
            answer=_grounded_clarification_text(),
            citations=[],
            used_fallback=False,
        )

    direct_tool = _match_direct_tool_request(payload.question)
    lowered_question = payload.question.lower()

    if "opencravat" in lowered_question or "open cravat" in lowered_question:
        return AnalysisChatResponse(
            answer=(
                "OpenCRAVAT is not available in this ChatGenome build.\n\n"
                "- The OpenCRAVAT plugin and Studio card have been removed because the runtime was unstable.\n"
                "- So this request did not run OpenCRAVAT.\n"
                "- If you want additional deterministic annotation, please use the currently supported tools such as SnpEff, CADD lookup, or REVEL lookup."
            ),
            citations=[],
            used_fallback=False,
            used_tools=[],
        )

    if direct_tool and direct_tool.get("name") == "ldblockshow_execution_tool":
        try:
            return _handle_ldblockshow_request(payload)
        except Exception as exc:
            region = _extract_ldblockshow_region(payload.question) or "unknown"
            return AnalysisChatResponse(
                answer=f"LDBlockShow execution failed: {exc}",
                citations=[],
                used_fallback=True,
                used_tools=["ldblockshow_execution_tool"],
                ldblockshow_result=LDBlockShowResponse(
                    tool="ldblockshow",
                    input_path=payload.analysis.source_vcf_path or "",
                    region=region,
                    output_prefix="",
                    command_preview="LDBlockShow -InVCF <source.vcf.gz> -Region chr:start:end ...",
                    attempted_regions=[region] if region != "unknown" else [],
                    warnings=[str(exc)],
                ),
            )

    if direct_tool and direct_tool.get("name") == "gatk_liftover_vcf_tool":
        try:
            return _handle_liftover_request(payload)
        except Exception as exc:
            return AnalysisChatResponse(
                answer=f"GATK liftover execution failed: {exc}",
                citations=[],
                used_fallback=True,
                used_tools=["gatk_liftover_vcf_tool"],
            )

    if direct_tool and direct_tool.get("name") == "snpeff_execution_tool":
        try:
            return _handle_snpeff_request(payload)
        except Exception as exc:
            return AnalysisChatResponse(
                answer=f"SnpEff execution failed: {exc}",
                citations=[],
                used_fallback=True,
                used_tools=["snpeff_execution_tool"],
            )

    if direct_tool and direct_tool.get("name") == "plink_execution_tool":
        try:
            return _handle_plink_request(payload)
        except Exception as exc:
            return AnalysisChatResponse(
                answer=f"PLINK request handling failed: {exc}",
                citations=[],
                used_fallback=True,
                used_tools=["plink_execution_tool"],
                requested_view="plink",
            )
    try:
        return _call_openai(payload)
    except Exception:
        return _fallback_answer(payload)


def _fallback_raw_qc_answer(payload: RawQcChatRequest) -> RawQcChatResponse:
    answer = (
        "I could not complete the raw-QC chat response right now.\n\n"
        "- Please retry the question.\n"
        "- If you want the answer grounded in the current Studio state, add `$studio` to the message."
    )
    return RawQcChatResponse(answer=answer, citations=[], used_fallback=True)


def answer_raw_qc_chat(payload: RawQcChatRequest) -> RawQcChatResponse:
    if _needs_grounded_clarification(payload.question):
        return RawQcChatResponse(
            answer=_grounded_clarification_text(),
            citations=[],
            used_fallback=False,
        )

    lowered = payload.question.lower()
    if "samtools" in lowered:
        alignment_kind = (payload.analysis.facts.file_kind or "").upper()
        if alignment_kind not in {"BAM", "SAM", "CRAM", "ALIGNMENT"}:
            return RawQcChatResponse(
                answer=(
                    "samtools is intended for alignment files such as BAM, SAM, or CRAM. "
                    f"The current uploaded source is `{payload.analysis.facts.file_name}` ({payload.analysis.facts.file_kind})."
                ),
                citations=[],
                used_fallback=False,
            )
        raw_path = payload.analysis.source_raw_path
        if not raw_path:
            return RawQcChatResponse(
                answer=(
                    "samtools could not be run because this raw-QC session does not retain a durable source path yet. "
                    "Please re-upload the BAM/SAM/CRAM file and try again."
                ),
                citations=[],
                used_fallback=False,
            )
        try:
            result = run_samtools(
                SamtoolsRequest(
                    raw_path=raw_path,
                    original_name=payload.analysis.facts.file_name,
                )
            )
            idx_lines = [
                f"- {item.contig}: mapped {item.mapped}, unmapped {item.unmapped}, length {item.length_bp}"
                for item in result.idxstats_rows[:5]
            ]
            answer = (
                f"samtools reviewed `{result.display_name}` ({result.file_kind}).\n\n"
                f"- Quickcheck: {'PASS' if result.quickcheck_ok else 'issue detected'}\n"
                f"- Total reads: {result.total_reads if result.total_reads is not None else 'unknown'}\n"
                f"- Mapped reads: {result.mapped_reads if result.mapped_reads is not None else 'unknown'}"
                f"{f' ({result.mapped_rate:.2f}%)' if result.mapped_rate is not None else ''}\n"
                f"- Properly paired reads: {result.properly_paired_reads if result.properly_paired_reads is not None else 'unknown'}"
                f"{f' ({result.properly_paired_rate:.2f}%)' if result.properly_paired_rate is not None else ''}\n"
                f"- Singleton reads: {result.singleton_reads if result.singleton_reads is not None else 'unknown'}\n"
                f"- Index path: {result.index_path or 'none'}\n\n"
                "Top idxstats rows:\n"
                f"{chr(10).join(idx_lines) if idx_lines else '- idxstats rows are not available for this input.'}"
            )
            if result.warnings:
                answer += "\n\nWarnings:\n" + "\n".join(f"- {warning}" for warning in result.warnings[:5])
            return RawQcChatResponse(
                answer=answer,
                citations=[],
                used_fallback=False,
                samtools_result=result,
            )
        except Exception as exc:
            return RawQcChatResponse(
                answer=f"samtools 실행 중 오류가 발생했습니다: {exc}",
                citations=[],
                used_fallback=False,
            )

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    if not api_key:
        return _fallback_raw_qc_answer(payload)

    compact_context = {
        "analysis_id": payload.analysis.analysis_id,
        "draft_answer": payload.analysis.draft_answer,
        "facts": payload.analysis.facts.model_dump(),
        "modules": [item.model_dump() for item in payload.analysis.modules[:12]],
    }
    grounded = _has_studio_trigger(payload.question)
    if grounded:
        system_prompt = (
            "You are a sequencing QC copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer from the provided FastQC context and do not invent QC metrics. "
            "Be concise, grounded, and practical. "
            "If there are WARN or FAIL modules, explain why they matter for downstream genomics workflows."
        )
        user_content = (
            "Question:\n"
            f"{payload.question}\n\n"
            "FastQC context JSON:\n"
            f"{json.dumps(compact_context, ensure_ascii=False)}"
        )
    else:
        system_prompt = (
            "You are a helpful general assistant. "
            "The user did not request analysis grounding. "
            "Answer from general knowledge only and ignore any uploaded FastQC/raw-QC context unless the user explicitly asks with a grounding trigger such as $studio."
        )
        user_content = payload.question
    history_lines = [{"role": turn.role, "content": turn.content} for turn in payload.history[-6:]]
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            *history_lines,
            {"role": "user", "content": user_content},
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))) as response:
            result = json.loads(response.read().decode("utf-8"))
        output_text = result.get("output_text")
        if not output_text:
            output = result.get("output", [])
            texts: list[str] = []
            for item in output:
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        texts.append(content.get("text", ""))
            output_text = "\n".join(texts).strip()
        return RawQcChatResponse(answer=output_text or _fallback_raw_qc_answer(payload).answer, citations=[], used_fallback=False)
    except Exception:
        return _fallback_raw_qc_answer(payload)


def _fallback_summary_stats_answer(payload: SummaryStatsChatRequest) -> SummaryStatsChatResponse:
    answer = (
        "I could not complete the summary-statistics chat response right now.\n\n"
        "- Please retry the question.\n"
        "- If you want the answer grounded in the current Studio state, add `$studio` to the message."
    )
    return SummaryStatsChatResponse(answer=answer, citations=[], used_fallback=True)


def answer_summary_stats_chat(payload: SummaryStatsChatRequest) -> SummaryStatsChatResponse:
    if _needs_grounded_clarification(payload.question):
        return SummaryStatsChatResponse(
            answer=_grounded_clarification_text(),
            citations=[],
            used_fallback=False,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    if not api_key:
        return _fallback_summary_stats_answer(payload)

    compact_context = {
        "analysis_id": payload.analysis.analysis_id,
        "file_name": payload.analysis.file_name,
        "genome_build": payload.analysis.genome_build,
        "trait_type": payload.analysis.trait_type,
        "delimiter": payload.analysis.delimiter,
        "detected_columns": payload.analysis.detected_columns,
        "mapped_fields": payload.analysis.mapped_fields.model_dump(),
        "row_count": payload.analysis.row_count,
        "warnings": payload.analysis.warnings[:12],
        "preview_rows": payload.analysis.preview_rows[:8],
        "draft_answer": payload.analysis.draft_answer,
        "used_tools": payload.analysis.used_tools,
    }
    grounded = _has_studio_trigger(payload.question)
    if grounded:
        system_prompt = (
            "You are a post-GWAS and summary-statistics copilot. "
            "The user explicitly requested grounded reasoning via a trigger such as $studio or $current analysis. "
            "Answer from the provided summary statistics context. "
            "When the answer is grounded in the uploaded file, distinguish that clearly from general knowledge. "
            "Do not claim that a downstream tool has already been run unless it appears in used_tools."
        )
        user_content = (
            "Question:\n"
            f"{payload.question}\n\n"
            "Summary statistics context JSON:\n"
            f"{json.dumps(compact_context, ensure_ascii=False)}"
        )
    else:
        system_prompt = (
            "You are a helpful general assistant. "
            "The user did not request analysis grounding. "
            "Answer from general knowledge only and ignore any uploaded summary-statistics context unless the user explicitly asks with a grounding trigger such as $studio."
        )
        user_content = payload.question
    history_lines = [{"role": turn.role, "content": turn.content} for turn in payload.history[-6:]]
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            *history_lines,
            {"role": "user", "content": user_content},
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))) as response:
            result = json.loads(response.read().decode("utf-8"))
        output_text = result.get("output_text")
        if not output_text:
            output = result.get("output", [])
            texts: list[str] = []
            for item in output:
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        texts.append(content.get("text", ""))
            output_text = "\n".join(texts).strip()
        citations = sorted(set(re.findall(r"\bREF\d+\b", output_text or "")))
        return SummaryStatsChatResponse(
            answer=output_text or _fallback_summary_stats_answer(payload).answer,
            citations=citations,
            used_fallback=False,
        )
    except Exception:
        return _fallback_summary_stats_answer(payload)
