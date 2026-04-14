---
name: chatgenome-orchestrator
description: Use when ChatGenome should decide which genomics analysis tools to run, in which order, and how to explain the resulting Studio state.
---

# ChatGenome Orchestrator

This skill defines the intended orchestration layer for `ChatGenome`.

## Welcome message

Upload a source file to get started. Supported formats: DICOM images, PNG/JPG/TIFF images, NIfTI volumes (.nii, .nii.gz), FHIR clinical bundles (.fhir.json, .fhir.xml, .ndjson), VCF (variant interpretation), Excel workbooks, text/markdown notes, FASTQ/BAM/SAM (raw sequencing QC), and summary statistics. The appropriate tools will run automatically after upload. Type `@help` for detailed tool options and usage tips.

## Help message

### Available tools by source type

**DICOM**
- Auto: DICOM Review (metadata, series summary, preview)

**PNG / JPG / TIFF Image**
- Auto: Image Review (metadata, EXIF, thumbnail)
- `@unsb [num_timesteps=5 tau=0.01]` — Run local UNSB enhancement on the active image source

**NIfTI Volume (.nii, .nii.gz)**
- Auto: NIfTI Review (shape, voxel dimensions, orientation, 3D viewer via Niivue)

**FHIR Bundle**
- Auto: FHIR Browser (patient, medications, labs, care team)

**Excel Workbook**
- Auto: Cohort Browser (sheets, schema, missingness)

**Text / Markdown**
- Auto: Text Review (preview, grounded summary)

**VCF (Variant Interpretation)**
- `@liftover [target=hg38]` — Convert genome build (hg19 ↔ hg38)
- `@snpeff [genome=GRCh38.mane.1.2.105]` — Run local SnpEff variant annotation
- `@plink [mode=qc|score]` — PLINK 2 QC or PRS scoring
- `@ldblockshow chr:start:end` — LD heatmap for a genomic region
- Auto: QC Summary, Annotation, ClinVar Review, VEP Consequence, Candidate Ranking, ROH Analysis, CADD/REVEL Enrichment, Grounded Summary

**FASTQ / BAM / SAM (Raw Sequencing QC)**
- `@samtools [mode=qc]` — samtools flagstat / stats / idxstats alignment QC
- Auto: FastQC Review

**Summary Statistics (GWAS)**
- `@qqman` — Manhattan plot + QQ plot
- `@prs_prep` — Build check, harmonization, score-file preparation
- Auto: Summary Stats Review (column detection, schema mapping)

### Tips

- `@toolname help` — Show detailed options for any tool
- `@help` — Show this guide

### Studio grounding

Use these prefixes to get answers grounded in the current analysis state:

- `$studio` — Interpret the active Studio cards
- `$current analysis` — Summarize the current analysis artifacts
- `$current card` — Explain the currently open card
- `$grounded` — Answer from tool-derived state only

Without a grounding prefix, questions are answered as general knowledge.



## Purpose

Use this skill when deciding which registered genomics tool should be used for a user request or an upload workflow.

## Workflow prompts

### Initial scope prompt

{file_name} 파일을 받았습니다. VCF analysis scope와 range(limit)를 입력해 주세요. 별도 지시가 없으면 representative로 시작합니다. 예: all로 200개, representative로 진행.

The architecture target is:

1. ChatGenome receives a VCF and user request.
2. The orchestrator chooses one or more tools from the registry.
3. The shared runner executes those tools.
4. Studio cards render the tool outputs.
5. Chat explains the grounded outputs, not raw VCF rows.

## Initial tool ordering

For the current migration stage, the preferred initial order is:

1. `vcf_qc_tool`
2. `annotation_tool`
3. `roh_analysis_tool`
4. `snpeff_execution_tool` when the user explicitly requests local SnpEff annotation
5. `cadd_lookup_tool`
6. `revel_lookup_tool`
7. `candidate_ranking_tool`
8. `grounded_summary_tool`
9. `clinvar_review_tool`
10. `vep_consequence_tool`
11. `clinical_coverage_tool`
12. `filtering_view_tool`
13. `symbolic_alt_tool`
14. `ldblockshow_execution_tool` when the user explicitly requests locus-level LD heatmap visualization for a VCF region
15. `samtools_execution_tool` when the user explicitly requests BAM/SAM/CRAM alignment QC, flagstat, idxstats, or post-alignment summary review

Later tools should include:

11. `igv_snapshot_tool`

## Rules

- Always prefer deterministic genomics tools over free-form model interpretation.
- Use `vcf_qc_tool` to establish base facts for any VCF workflow.
- Use `annotation_tool` to generate transcript-aware annotation state before downstream ranking.
- Use `snpeff_execution_tool` when the user explicitly asks to run SnpEff on a local VCF and the required local Java runtime, jar, and genome database are available.
- Use `ldblockshow_execution_tool` when the user explicitly asks for LD heatmap or block visualization over a region and provides or implies a concrete locus in `chr:start:end` format.
- Use `samtools_execution_tool` when the user explicitly asks for post-alignment QC or BAM/SAM/CRAM inspection such as `flagstat`, `idxstats`, or `samtools stats`.
- Use `cadd_lookup_tool` to enrich shortlisted annotated variants with local CADD scores when a build-matched local table is available.
- Use `revel_lookup_tool` to enrich shortlisted missense variants with local REVEL scores when a matching local segment file is available.
- Use `roh_analysis_tool` when ROH/recessive review is shown or requested.
- Use `candidate_ranking_tool` to populate the ranked shortlist shown in Studio after optional score enrichment such as local CADD and REVEL lookup.
- Use `grounded_summary_tool` to compose the narrative answer from trusted tool-derived state.
- Use `clinvar_review_tool` to populate the clinical significance distribution.
- Use `vep_consequence_tool` to populate the consequence distribution shown in Studio.
- Use `clinical_coverage_tool` to summarize annotation completeness.
- Use `filtering_view_tool` to populate filtering/triage overview metrics for the variant table.
- Use `symbolic_alt_tool` to split symbolic ALT records into a dedicated review path.
- Chat should refer to tool outputs and Studio summaries as the trusted state.
- If a tool fails, preserve the prior direct implementation as fallback until migration is complete.

## Chat policy

The chat layer should separate general conversation from grounded Studio interpretation.

### Default chat mode

- If the user does **not** include an explicit grounding trigger, answer as a normal GPT assistant.
- In default mode, general knowledge questions should be answered from general knowledge even when a genomics dataset is loaded.
- Do not automatically reinterpret general questions as requests for Studio card summaries just because a keyword overlaps with a card name or tool name.
- Examples:
  - `ROH가 뭐야?` -> answer as a general definition.
  - `BTS가 누구야?` -> answer as a general knowledge question.
  - `candidate variant가 뭔지 설명해줘` -> answer as a general genomics concept unless grounding is explicitly requested.

### Grounded Studio mode

- If the user includes one of the following triggers, answer from the active Studio state and current analysis artifacts:
  - `$studio`
  - `$current analysis`
  - `$current card`
  - `$grounded`
- In grounded mode, prefer current card data, tool outputs, and current analysis artifacts over generic explanation.
- In grounded mode, make it explicit that the answer is based on the currently loaded Studio state.
- Examples:
  - `$studio ROH 결과 설명해줘`
  - `$current analysis candidate card를 해석해줘`

### Tool execution policy

- A tool should run only when the user explicitly requests execution or when the workflow stage deterministically requires it.
- Prefer `@toolname` as the explicit execution trigger for tool calls.
- Prefer `@toolname help` when the user wants to understand tool options before execution.
- `@toolname` should use the current active source by default.
- `@unsb` should use the active image source and attach the enhanced preview to the Image Review card.
- Tool recommendation policy belongs in skill; actual tool invocation belongs in backend/runtime.
- Do not execute a tool just because a user asked a conceptual question about the tool domain.
- For the current migration stage:
  - `@liftover` should use the active VCF source.
  - `@samtools` should use the active BAM/SAM/CRAM raw-QC source.
  - `@plink` should use the active VCF source and open the curated PLINK Studio flow.
  - `@liftover help` and `@samtools help` should render curated option help from tool metadata instead of a hard-coded backend explanation.
  - `@plink help` should render curated option help from tool metadata instead of a hard-coded backend explanation.

## Source-specific follow-up policy

### VCF workflows

- After VCF analysis, prefer follow-up suggestions such as candidate review, ClinVar review, VEP consequence review, ROH interpretation, local annotation enrichment, and liftover when relevant.
- Use grounded interpretation only when the user explicitly invokes Studio grounding.

### Raw sequencing workflows

- After raw sequencing intake, prefer follow-up suggestions such as FastQC review, samtools review, alignment QC, and file integrity checks.
- General sequencing questions without grounding triggers should still be answered as normal GPT responses.

### Summary statistics workflows

- After summary statistics review, prefer follow-up suggestions such as Manhattan plot, QQ plot, PRS preparation, harmonization, and post-GWAS next steps.
- Do not automatically generate follow-up plots unless the workflow explicitly requires it or the user explicitly asks for them.
- `@skill prs_prep` should prepare build check and harmonization readiness before any PLINK score-file generation step.
- General statistical or genetics questions without grounding triggers should be answered as normal GPT responses.

## Interpretation policy

- Skill should define *how to interpret* candidate variants, ROH regions, ClinVar summaries, consequence summaries, and summary-statistics review cards.
- Backend should only provide the relevant current artifacts and source type to the model/runtime.
- Keep the backend responsible for trigger detection, source-type detection, state assembly, tool execution, and result persistence.

## Output expectations

The orchestrator should ensure that:

- used tool names are recorded
- registry tools are discoverable
- Studio cards can be traced to tool outputs
- grounded chat cites the current tool-derived state
