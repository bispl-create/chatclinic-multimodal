from __future__ import annotations

import json
import os
import re
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.models import (
    AnalysisChatRequest,
    AnalysisChatResponse,
    QqmanAssociationRequest,
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
from app.services.r_vcf_plots import run_qqman_association
from app.services.samtools import run_samtools
from app.services.snpeff import run_snpeff
from app.services.workflows import (
    analyze_prs_prep_workflow,
    analyze_raw_qc_workflow,
    analyze_summary_stats_workflow,
    analyze_vcf_workflow,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
PLUGINS_DIR = ROOT_DIR / "plugins"
WORKFLOWS_DIR = ROOT_DIR / "skills" / "chatgenome-orchestrator" / "workflows"
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))

TOOL_ALIAS_REGISTRY: dict[str, dict[str, Any]] = {
    "liftover": {
        "plugin_name": "gatk_liftover_vcf_tool",
        "aliases": ["liftover"],
        "source_types": ["vcf"],
        "result_kind": "liftover_result",
        "help_supported": True,
        "direct_preanalysis_supported": True,
    },
    "samtools": {
        "plugin_name": "samtools_execution_tool",
        "aliases": ["samtools"],
        "source_types": ["raw_qc"],
        "result_kind": "samtools_result",
        "help_supported": True,
        "direct_preanalysis_supported": True,
    },
    "plink": {
        "plugin_name": "plink_execution_tool",
        "aliases": ["plink"],
        "source_types": ["vcf"],
        "result_kind": "plink_result",
        "help_supported": True,
        "direct_preanalysis_supported": True,
    },
    "snpeff": {
        "plugin_name": "snpeff_execution_tool",
        "aliases": ["snpeff"],
        "source_types": ["vcf"],
        "result_kind": "snpeff_result",
        "help_supported": True,
        "direct_preanalysis_supported": True,
    },
    "ldblockshow": {
        "plugin_name": "ldblockshow_execution_tool",
        "aliases": ["ldblockshow"],
        "source_types": ["vcf"],
        "result_kind": "ldblockshow_result",
        "help_supported": True,
        "direct_preanalysis_supported": True,
    },
    "qqman": {
        "plugin_name": "qqman_execution_tool",
        "aliases": ["qqman"],
        "source_types": ["summary_stats"],
        "result_kind": "qqman_result",
        "help_supported": True,
        "direct_preanalysis_supported": True,
    },
}


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def _load_workflow_manifests() -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    for manifest in sorted(WORKFLOWS_DIR.glob("*.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            manifests.append(payload)
    return manifests


def _tool_registry_entry_for_manifest(manifest: dict[str, object]) -> dict[str, Any] | None:
    plugin_name = str(manifest.get("name") or "").strip()
    for entry in TOOL_ALIAS_REGISTRY.values():
        if str(entry.get("plugin_name") or "").strip() == plugin_name:
            return entry
    return None


def _tool_registry_entry_for_alias(alias: str) -> tuple[str | None, dict[str, Any] | None]:
    lowered = alias.strip().lower()
    for canonical_alias, entry in TOOL_ALIAS_REGISTRY.items():
        aliases = [str(item).strip().lower() for item in entry.get("aliases", [])]
        if lowered in aliases:
            return canonical_alias, entry
    return None, None


def _manifest_for_plugin_name(plugin_name: str | None) -> dict[str, object] | None:
    if not plugin_name:
        return None
    normalized = str(plugin_name).strip()
    for manifest in _load_tool_manifests():
        if str(manifest.get("name") or "").strip() == normalized:
            return manifest
    return None


def _tool_aliases(manifest: dict[str, object]) -> list[str]:
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
            if text and re.fullmatch(r"[a-z0-9_-]+", text):
                aliases.add(text)
    return sorted(aliases)


def _parse_at_tool_request(question: str) -> dict[str, object] | None:
    stripped = question.strip()
    match = re.match(r"^@([A-Za-z0-9_-]+)(?:\s+(.*))?$", stripped, flags=re.DOTALL)
    if not match:
        return None
    raw_alias = match.group(1).strip().lower()
    if raw_alias == "skill":
        return None
    remainder = (match.group(2) or "").strip()
    lowered = remainder.lower()
    canonical_alias, registry_entry = _tool_registry_entry_for_alias(raw_alias)
    if registry_entry is not None:
        manifest = _manifest_for_plugin_name(registry_entry.get("plugin_name"))
        return {
            "manifest": manifest,
            "alias": canonical_alias or raw_alias,
            "input_alias": raw_alias,
            "registry_entry": registry_entry,
            "remainder": remainder,
            "is_help": lowered in {"help", "--help", "-h"} or lowered.startswith("help "),
        }

    for manifest in _load_tool_manifests():
        if raw_alias in _tool_aliases(manifest):
            return {
                "manifest": manifest,
                "alias": raw_alias,
                "input_alias": raw_alias,
                "registry_entry": _tool_registry_entry_for_manifest(manifest),
                "remainder": remainder,
                "is_help": lowered in {"help", "--help", "-h"} or lowered.startswith("help "),
            }
    return {
        "manifest": None,
        "alias": raw_alias,
        "input_alias": raw_alias,
        "registry_entry": None,
        "remainder": remainder,
        "is_help": False,
    }


def _parse_skill_request(question: str) -> dict[str, object] | None:
    stripped = question.strip()
    match = re.match(r"^@skill(?:\s+(.*))?$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    remainder = (match.group(1) or "").strip()
    lowered = remainder.lower()
    if not remainder or lowered in {"help", "--help", "-h"}:
        return {"name": None, "remainder": remainder, "is_help": True, "manifest": None}
    workflow_name = remainder.split()[0].strip()
    is_help = remainder.lower().endswith(" help")
    target_name = workflow_name
    manifest = next(
        (
            item
            for item in _load_workflow_manifests()
            if str(item.get("name") or "").strip().lower() == workflow_name.lower()
        ),
        None,
    )
    return {
        "name": target_name,
        "remainder": remainder,
        "is_help": is_help,
        "manifest": manifest,
    }


def _render_tool_help(manifest: dict[str, object]) -> str:
    name = str(manifest.get("name") or "tool")
    registry_entry = _tool_registry_entry_for_manifest(manifest)
    help_block = manifest.get("help")
    if not isinstance(help_block, dict):
        aliases = ", ".join(f"@{item}" for item in _tool_aliases(manifest)[:4])
        return (
            f"`{name}` is registered, but no curated help metadata is available yet.\n\n"
            f"- Try one of these aliases: {aliases or '@tool'}"
        )
    input_hint = "active source"
    if registry_entry:
        source_types = [str(item).lower() for item in registry_entry.get("source_types", [])]
        if "vcf" in source_types:
            input_hint = "active VCF source"
        elif "raw_qc" in source_types:
            input_hint = "active BAM/SAM/CRAM source"
        elif "summary_stats" in source_types:
            input_hint = "active summary-statistics source"
    else:
        orchestration = manifest.get("orchestration") if isinstance(manifest.get("orchestration"), dict) else {}
        consumes = orchestration.get("consumes") if isinstance(orchestration, dict) else []
        if isinstance(consumes, list):
            lowered = [str(item).lower() for item in consumes]
            if "vcf_path" in lowered:
                input_hint = "active VCF source"
            elif "alignment_file" in lowered:
                input_hint = "active BAM/SAM/CRAM source"
            elif "summary_stats_path" in lowered:
                input_hint = "active summary-statistics source"

    alias_candidates = (
        [str(item).strip() for item in registry_entry.get("aliases", []) if str(item).strip()]
        if registry_entry
        else _tool_aliases(manifest)[:4]
    )
    alias_list = [f"@{item}" for item in alias_candidates[:4]]
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


def _resolve_tool_help_response(tool_request: dict[str, object] | None) -> str | None:
    if not tool_request or not bool(tool_request.get("is_help")):
        return None
    manifest = tool_request.get("manifest")
    alias = str(tool_request.get("alias") or tool_request.get("input_alias") or "tool").strip()
    if isinstance(manifest, dict):
        return _render_tool_help(manifest)
    return f"`@{alias}` is not a registered ChatGenome tool."


def _describe_source_type(source_type: str) -> str:
    normalized = source_type.strip().lower()
    if normalized == "vcf":
        return "VCF session"
    if normalized == "raw_qc":
        return "raw-QC/alignment session"
    if normalized == "summary_stats":
        return "summary-statistics session"
    return "current session"


def _tool_input_hint(source_type: str) -> str:
    normalized = source_type.strip().lower()
    if normalized == "vcf":
        return "active VCF source"
    if normalized == "raw_qc":
        return "active BAM/SAM/CRAM source"
    if normalized == "summary_stats":
        return "active summary-statistics source"
    return "active source"


def _resolve_tool_source_mismatch_response(
    tool_request: dict[str, object] | None, current_source_type: str
) -> str | None:
    if not tool_request:
        return None
    alias = str(tool_request.get("alias") or tool_request.get("input_alias") or "tool").strip()
    registry_entry = tool_request.get("registry_entry")
    if not isinstance(registry_entry, dict):
        manifest = tool_request.get("manifest")
        if isinstance(manifest, dict):
            registry_entry = _tool_registry_entry_for_manifest(manifest)
    allowed_source_types = [
        str(item).strip().lower()
        for item in (registry_entry.get("source_types", []) if isinstance(registry_entry, dict) else [])
        if str(item).strip()
    ]
    current = current_source_type.strip().lower()
    if not allowed_source_types or current in allowed_source_types:
        return None
    if len(allowed_source_types) == 1:
        target_source_type = allowed_source_types[0]
        return (
            f"`@{alias}` uses the {_tool_input_hint(target_source_type)}. "
            f"Run it from a {_describe_source_type(target_source_type)} rather than a {_describe_source_type(current)}."
        )
    expected = ", ".join(_describe_source_type(item) for item in allowed_source_types)
    return (
        f"`@{alias}` is not available for the current source type. "
        f"It expects one of: {expected}."
    )


def _render_skill_help(source_type: str | None = None, selected: dict[str, object] | None = None) -> str:
    manifests = _load_workflow_manifests()
    if source_type:
        manifests = [item for item in manifests if str(item.get("source_type") or "").strip().lower() == source_type.lower()]
    if selected is not None:
        manifests = [selected]
    if not manifests:
        return "No workflow registry entries are available for the current source."
    tool_lookup = {
        str(item.get("name") or "").strip(): item
        for item in _load_tool_manifests()
        if isinstance(item, dict)
    }
    lines: list[str] = ["**Workflow registry**", ""]
    for item in manifests:
        name = str(item.get("name") or "workflow")
        description = str(item.get("description") or "").strip()
        lines.append(f"- `@skill {name}`: {description}")
        if selected is not None:
            steps = item.get("steps") or []
            if isinstance(steps, list) and steps:
                lines.append("")
                lines.append("Steps")
                for step in steps:
                    step_name = str(step).strip()
                    manifest = tool_lookup.get(step_name)
                    if manifest is not None:
                        step_description = str(manifest.get("description") or "").strip()
                        lines.append(f"- `{step_name}`: {step_description}")
                    else:
                        lines.append(f"- `{step_name}`")
    lines.append("")
    lines.append("Examples")
    if source_type == "vcf":
        lines.append("- `@skill representative_vcf_review`")
    elif source_type == "raw_qc":
        lines.append("- `@skill raw_qc_review`")
    elif source_type == "summary_stats":
        lines.append("- `@skill summary_stats_review`")
        lines.append("- `@skill prs_prep`")
    else:
        lines.append("- `@skill help`")
        lines.append("- `@skill representative_vcf_review`")
    return "\n".join(lines)


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
    with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT_SECONDS) as response:
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


def _extract_key_value_options(text: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_-]*)=([A-Za-z0-9._:-]+)", text):
        options[key.lower()] = value
    return options


def _handle_liftover_request(payload: AnalysisChatRequest, *, remainder: str = "") -> AnalysisChatResponse:
    source_vcf_path = payload.analysis.source_vcf_path
    if not source_vcf_path:
        return AnalysisChatResponse(
            answer="The current analysis context does not include a source VCF path, so liftover cannot be run from this chat turn.",
            citations=[],
            used_fallback=True,
            used_tools=["gatk_liftover_vcf_tool"],
        )

    options = _extract_key_value_options(remainder)
    target_build, target_label = _extract_liftover_target_build(
        options.get("target") or payload.question,
        payload.analysis.facts.genome_build_guess,
    )
    source_build = options.get("source_build") or payload.analysis.facts.genome_build_guess
    output_prefix = options.get("output_prefix") or f"{payload.analysis.analysis_id}-liftover-{target_label}"
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
            source_build=source_build or None,
            target_build=target_build,
            output_prefix=output_prefix,
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


def _dispatch_analysis_liftover(payload: AnalysisChatRequest, tool_request: dict[str, object]) -> AnalysisChatResponse:
    return _handle_liftover_request(payload, remainder=str(tool_request.get("remainder") or ""))


def _dispatch_analysis_snpeff(payload: AnalysisChatRequest, tool_request: dict[str, object]) -> AnalysisChatResponse:
    return _handle_snpeff_request(payload)


def _dispatch_analysis_plink(payload: AnalysisChatRequest, tool_request: dict[str, object]) -> AnalysisChatResponse:
    return _handle_plink_request(payload)


def _dispatch_analysis_ldblockshow(payload: AnalysisChatRequest, tool_request: dict[str, object]) -> AnalysisChatResponse:
    return _handle_ldblockshow_request(payload)


ANALYSIS_TOOL_DISPATCH: dict[str, Any] = {
    "gatk_liftover_vcf_tool": _dispatch_analysis_liftover,
    "plink_execution_tool": _dispatch_analysis_plink,
    "snpeff_execution_tool": _dispatch_analysis_snpeff,
    "ldblockshow_execution_tool": _dispatch_analysis_ldblockshow,
}


def _handle_analysis_at_tool_request(payload: AnalysisChatRequest, tool_request: dict[str, object]) -> AnalysisChatResponse | None:
    manifest = tool_request.get("manifest")
    alias = str(tool_request.get("alias") or "")
    remainder = str(tool_request.get("remainder") or "")
    help_text = _resolve_tool_help_response(tool_request)
    if help_text is not None:
        return AnalysisChatResponse(answer=help_text, citations=[], used_fallback=False)
    mismatch_text = _resolve_tool_source_mismatch_response(tool_request, "vcf")
    if mismatch_text is not None:
        return AnalysisChatResponse(answer=mismatch_text, citations=[], used_fallback=False)
    if manifest is None:
        return AnalysisChatResponse(
            answer=f"`@{alias}` is not a registered ChatGenome tool.",
            citations=[],
            used_fallback=False,
        )
    name = str(manifest.get("name") or "")
    dispatch = ANALYSIS_TOOL_DISPATCH.get(name)
    if dispatch is not None:
        return dispatch(payload, tool_request)
    return None


def _handle_analysis_skill_request(payload: AnalysisChatRequest, skill_request: dict[str, object]) -> AnalysisChatResponse:
    manifest = skill_request.get("manifest")
    if skill_request.get("is_help") and manifest is None:
        return AnalysisChatResponse(answer=_render_skill_help("vcf"), citations=[], used_fallback=False)
    if manifest is None:
        name = str(skill_request.get("name") or "workflow")
        return AnalysisChatResponse(
            answer=f"`@skill {name}` is not a registered workflow for the current build.",
            citations=[],
            used_fallback=False,
        )
    if skill_request.get("is_help"):
        return AnalysisChatResponse(answer=_render_skill_help("vcf", selected=manifest), citations=[], used_fallback=False)
    workflow_name = str(manifest.get("name") or "")
    if workflow_name == "representative_vcf_review":
        source_vcf_path = payload.analysis.source_vcf_path
        if not source_vcf_path:
            return AnalysisChatResponse(
                answer="The active analysis does not expose a source VCF path, so this workflow cannot be rerun from chat.",
                citations=[],
                used_fallback=False,
            )
        refreshed = analyze_vcf_workflow(source_vcf_path, annotation_scope="representative", annotation_limit=None)
        return AnalysisChatResponse(
            answer=(
                "The representative VCF review workflow was rerun on the active source.\n\n"
                f"- Workflow: `{workflow_name}`\n"
                f"- Active file: `{refreshed.facts.file_name}`\n"
                f"- Logged tools: {', '.join(refreshed.used_tools or []) or 'none'}\n"
                f"- Candidate variants: {len(refreshed.candidate_variants or [])}\n\n"
                "The active VCF analysis state has been refreshed. Open Studio cards or ask follow-up questions. Use `$studio ...` if you want the answer grounded in the current VCF review state."
            ),
            citations=[],
            used_fallback=False,
            analysis=refreshed,
        )
    return AnalysisChatResponse(
        answer=f"`@skill {workflow_name}` is registered but not yet executable in analysis chat.",
        citations=[],
        used_fallback=False,
    )


def _dispatch_raw_qc_samtools(payload: RawQcChatRequest, tool_request: dict[str, object]) -> RawQcChatResponse:
    alignment_kind = (payload.analysis.facts.file_kind or "").upper()
    if alignment_kind not in {"BAM", "SAM", "CRAM", "ALIGNMENT"}:
        return RawQcChatResponse(
            answer=(
                "samtools is intended for alignment files such as BAM, SAM, or CRAM. "
                f"The current active source is `{payload.analysis.facts.file_name}` ({payload.analysis.facts.file_kind})."
            ),
            citations=[],
            used_fallback=False,
        )
    raw_path = payload.analysis.source_raw_path
    if not raw_path:
        return RawQcChatResponse(
            answer="The active raw-QC session does not include a durable alignment-file path, so `@samtools` cannot run yet.",
            citations=[],
            used_fallback=False,
        )
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
        f"samtools reviewed the active source `{result.display_name}` ({result.file_kind}).\n\n"
        f"- Quickcheck: {'PASS' if result.quickcheck_ok else 'issue detected'}\n"
        f"- Total reads: {result.total_reads if result.total_reads is not None else 'unknown'}\n"
        f"- Mapped reads: {result.mapped_reads if result.mapped_reads is not None else 'unknown'}"
        f"{f' ({result.mapped_rate:.2f}%)' if result.mapped_rate is not None else ''}\n"
        f"- Properly paired reads: {result.properly_paired_reads if result.properly_paired_reads is not None else 'unknown'}"
        f"{f' ({result.properly_paired_rate:.2f}%)' if result.properly_paired_rate is not None else ''}\n"
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


RAW_QC_TOOL_DISPATCH: dict[str, Any] = {
    "samtools_execution_tool": _dispatch_raw_qc_samtools,
}


def _handle_raw_qc_at_tool_request(payload: RawQcChatRequest, tool_request: dict[str, object]) -> RawQcChatResponse:
    manifest = tool_request.get("manifest")
    alias = str(tool_request.get("alias") or "")
    help_text = _resolve_tool_help_response(tool_request)
    if help_text is not None:
        return RawQcChatResponse(answer=help_text, citations=[], used_fallback=False)
    mismatch_text = _resolve_tool_source_mismatch_response(tool_request, "raw_qc")
    if mismatch_text is not None:
        return RawQcChatResponse(answer=mismatch_text, citations=[], used_fallback=False)
    if manifest is None:
        return RawQcChatResponse(
            answer=f"`@{alias}` is not a registered ChatGenome tool.",
            citations=[],
            used_fallback=False,
        )
    name = str(manifest.get("name") or "")
    dispatch = RAW_QC_TOOL_DISPATCH.get(name)
    if dispatch is not None:
        return dispatch(payload, tool_request)
    return RawQcChatResponse(answer=f"`@{alias}` is not a registered ChatGenome tool.", citations=[], used_fallback=False)


def _handle_raw_qc_skill_request(payload: RawQcChatRequest, skill_request: dict[str, object]) -> RawQcChatResponse:
    manifest = skill_request.get("manifest")
    if skill_request.get("is_help") and manifest is None:
        return RawQcChatResponse(answer=_render_skill_help("raw_qc"), citations=[], used_fallback=False)
    if manifest is None:
        name = str(skill_request.get("name") or "workflow")
        return RawQcChatResponse(
            answer=f"`@skill {name}` is not a registered workflow for the current build.",
            citations=[],
            used_fallback=False,
        )
    if skill_request.get("is_help"):
        return RawQcChatResponse(answer=_render_skill_help("raw_qc", selected=manifest), citations=[], used_fallback=False)
    workflow_name = str(manifest.get("name") or "")
    if workflow_name == "raw_qc_review":
        source_raw_path = payload.analysis.source_raw_path
        if not source_raw_path:
            return RawQcChatResponse(
                answer="The active raw-QC session does not expose a durable source file path, so this workflow cannot be rerun from chat.",
                citations=[],
                used_fallback=False,
            )
        refreshed = analyze_raw_qc_workflow(source_raw_path, payload.analysis.facts.file_name)
        return RawQcChatResponse(
            answer=(
                "The raw_qc_review workflow was rerun on the active source.\n\n"
                f"- Workflow: `{workflow_name}`\n"
                f"- Active file: `{refreshed.facts.file_name}`\n"
                f"- Logged tools: {', '.join(refreshed.used_tools or []) or 'none'}\n"
                f"- Modules: {len(refreshed.modules)}\n\n"
                "The raw-QC state has been refreshed. Use `@samtools` for additional alignment review on compatible sources, or `$studio ...` for grounded explanation of the current Studio state."
            ),
            citations=[],
            used_fallback=False,
            requested_view="rawqc",
            analysis=refreshed,
        )
    return RawQcChatResponse(
        answer=f"`@skill {workflow_name}` is registered but not yet executable in raw-QC chat.",
        citations=[],
        used_fallback=False,
    )


def _handle_summary_stats_at_tool_request(
    payload: SummaryStatsChatRequest, tool_request: dict[str, object]
) -> SummaryStatsChatResponse:
    manifest = tool_request.get("manifest")
    alias = str(tool_request.get("alias") or "")
    help_text = _resolve_tool_help_response(tool_request)
    if help_text is not None:
        return SummaryStatsChatResponse(answer=help_text, citations=[], used_fallback=False)
    mismatch_text = _resolve_tool_source_mismatch_response(tool_request, "summary_stats")
    if mismatch_text is not None:
        return SummaryStatsChatResponse(answer=mismatch_text, citations=[], used_fallback=False)
    if manifest is None:
        return SummaryStatsChatResponse(
            answer=f"`@{alias}` is not a registered ChatGenome tool.",
            citations=[],
            used_fallback=False,
        )

    name = str(manifest.get("name") or "")
    dispatch = SUMMARY_STATS_TOOL_DISPATCH.get(name)
    if dispatch is not None:
        return dispatch(payload, tool_request)
    return SummaryStatsChatResponse(answer=f"`@{alias}` is not a registered ChatGenome tool.", citations=[], used_fallback=False)


def _dispatch_summary_stats_qqman(
    payload: SummaryStatsChatRequest, tool_request: dict[str, object]
) -> SummaryStatsChatResponse:
    remainder = str(tool_request.get("remainder") or "")
    source_stats_path = payload.analysis.source_stats_path
    if not source_stats_path:
        return SummaryStatsChatResponse(
            answer="The active summary-statistics session does not expose a durable source file path, so `@qqman` cannot run yet.",
            citations=[],
            used_fallback=False,
        )
    options = _extract_key_value_options(remainder)
    result = run_qqman_association(
        QqmanAssociationRequest(
            association_path=source_stats_path,
            output_prefix=options.get("output_prefix") or f"{payload.analysis.analysis_id}-qqman",
        )
    )
    return SummaryStatsChatResponse(
        answer=(
            "qqman plots were generated for the active summary-statistics source.\n\n"
            f"- Output directory: `{result.output_dir}`\n"
            f"- Plot artifacts: {len(result.artifacts)}\n"
            f"- Warnings: {len(result.warnings)}\n\n"
            "The Studio card has been updated with the latest qqman result."
        ),
        citations=[],
        used_fallback=False,
        requested_view="qqman",
        qqman_result=result,
    )


SUMMARY_STATS_TOOL_DISPATCH: dict[str, Any] = {
    "qqman_execution_tool": _dispatch_summary_stats_qqman,
}


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

    skill_request = _parse_skill_request(payload.question)
    if skill_request:
        try:
            return _handle_analysis_skill_request(payload, skill_request)
        except Exception as exc:
            name = str(skill_request.get("name") or "workflow")
            return AnalysisChatResponse(
                answer=f"`@skill {name}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    at_tool_request = _parse_at_tool_request(payload.question)
    if at_tool_request:
        try:
            handled = _handle_analysis_at_tool_request(payload, at_tool_request)
            if handled is not None:
                return handled
        except Exception as exc:
            alias = str(at_tool_request.get("alias") or "tool")
            return AnalysisChatResponse(
                answer=f"`@{alias}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

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

    skill_request = _parse_skill_request(payload.question)
    if skill_request:
        try:
            return _handle_raw_qc_skill_request(payload, skill_request)
        except Exception as exc:
            name = str(skill_request.get("name") or "workflow")
            return RawQcChatResponse(
                answer=f"`@skill {name}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    at_tool_request = _parse_at_tool_request(payload.question)
    if at_tool_request:
        try:
            return _handle_raw_qc_at_tool_request(payload, at_tool_request)
        except Exception as exc:
            alias = str(at_tool_request.get("alias") or "tool")
            return RawQcChatResponse(
                answer=f"`@{alias}` execution failed: {exc}",
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
        with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT_SECONDS) as response:
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

    at_tool_request = _parse_at_tool_request(payload.question)
    if at_tool_request:
        try:
            return _handle_summary_stats_at_tool_request(payload, at_tool_request)
        except Exception as exc:
            alias = str(at_tool_request.get("alias") or "tool")
            return SummaryStatsChatResponse(
                answer=f"`@{alias}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
            )

    skill_request = _parse_skill_request(payload.question)
    if skill_request:
        try:
            manifest = skill_request.get("manifest")
            if skill_request.get("is_help") and manifest is None:
                return SummaryStatsChatResponse(answer=_render_skill_help("summary_stats"), citations=[], used_fallback=False)
            if manifest is None:
                name = str(skill_request.get("name") or "workflow")
                return SummaryStatsChatResponse(
                    answer=f"`@skill {name}` is not a registered workflow for the current build.",
                    citations=[],
                    used_fallback=False,
                )
            if skill_request.get("is_help"):
                return SummaryStatsChatResponse(
                    answer=_render_skill_help("summary_stats", selected=manifest),
                    citations=[],
                    used_fallback=False,
                )
            workflow_name = str(manifest.get("name") or "")
            if workflow_name == "summary_stats_review":
                source_stats_path = payload.analysis.source_stats_path
                if not source_stats_path:
                    return SummaryStatsChatResponse(
                        answer="The active summary-statistics session does not expose a durable source file path, so this workflow cannot be rerun from chat.",
                        citations=[],
                        used_fallback=False,
                    )
                refreshed = analyze_summary_stats_workflow(
                    source_stats_path,
                    payload.analysis.file_name,
                    genome_build=payload.analysis.genome_build,
                    trait_type=payload.analysis.trait_type,
                )
                return SummaryStatsChatResponse(
                    answer=(
                        "The summary_stats_review workflow was rerun on the active source.\n\n"
                        f"- Workflow: `{workflow_name}`\n"
                        f"- Active file: `{refreshed.file_name}`\n"
                        f"- Rows detected: {refreshed.row_count}\n"
                        f"- Auto-mapped fields: {sum(1 for value in refreshed.mapped_fields.model_dump().values() if value)}\n\n"
                        "The Summary Statistics Review state has been refreshed. Use `$studio ...` for grounded explanation of the current review state, or ask for a downstream workflow such as PRS preparation."
                    ),
                    citations=[],
                    used_fallback=False,
                    requested_view="sumstats",
                    analysis=refreshed,
                )
            if workflow_name == "prs_prep":
                source_stats_path = payload.analysis.source_stats_path
                if not source_stats_path:
                    return SummaryStatsChatResponse(
                        answer="The active summary-statistics session does not expose a durable source file path, so PRS preparation cannot be run from chat.",
                        citations=[],
                        used_fallback=False,
                    )
                prs_prep_result = analyze_prs_prep_workflow(
                    source_stats_path,
                    payload.analysis.file_name,
                    genome_build=payload.analysis.genome_build,
                )
                refreshed = payload.analysis.model_copy(update={"prs_prep_result": prs_prep_result})
                return SummaryStatsChatResponse(
                    answer=(
                        "The prs_prep workflow was run on the active summary-statistics source.\n\n"
                        f"- Workflow: `{workflow_name}`\n"
                        f"- Active file: `{prs_prep_result.file_name}`\n"
                        f"- Build check: {prs_prep_result.build_check.inferred_build} ({prs_prep_result.build_check.build_confidence})\n"
                        f"- Score-file rows kept: {prs_prep_result.kept_rows}\n"
                        f"- Score-file rows dropped: {prs_prep_result.dropped_rows}\n"
                        f"- Score file ready: {'yes' if prs_prep_result.score_file_ready else 'no'}\n\n"
                        "The PRS Prep Review state has been added to Studio. Use `$studio ...` to ask grounded questions about build check, harmonization, or score-file readiness."
                    ),
                    citations=[],
                    used_fallback=False,
                    requested_view="prs_prep",
                    analysis=refreshed,
                    prs_prep_result=prs_prep_result,
                )
            return SummaryStatsChatResponse(
                answer=f"`@skill {workflow_name}` is registered but not yet executable in summary-statistics chat.",
                citations=[],
                used_fallback=False,
            )
        except Exception as exc:
            name = str(skill_request.get("name") or "workflow")
            return SummaryStatsChatResponse(
                answer=f"`@skill {name}` execution failed: {exc}",
                citations=[],
                used_fallback=True,
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
        with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT_SECONDS) as response:
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
