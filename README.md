# ChatGenome

Interactive genomics workspace for mode-driven review sessions, explicit workflow execution, direct tool runs, Studio cards, and grounded chat.

![ChatGenome workspace preview](docs/chatgenome-ui-preview.svg)

## Overview

ChatGenome is organized around three explicit trigger families:

- `@mode ...`
  Selects the session purpose and configures the source-input UI.
- `@skill ...`
  Runs a named multi-step workflow.
- `@toolname`
  Runs a single deterministic tool.
- `$studio ...`
  Grounds a chat response in the current Studio state.

The current architecture is intentionally explicit:

1. Choose a session mode.
2. Attach the source files required by that mode.
3. Run workflows with `@skill ...` or tools with `@toolname`.
4. Review outputs in `Studio`.
5. Use `$studio` only when you want the chat answer grounded in the current Studio artifacts.

This keeps general GPT conversation separate from deterministic genomics execution.

## Current Session Modes

### `@mode prs`

Purpose:
- post-GWAS PRS preparation and scoring

Expected sources:
- summary statistics
- target genotype VCF

Typical sequence:
- `@mode prs`
- upload summary statistics
- upload target genotype VCF
- `@skill prs_prep`
- `@plink score`

### `@mode vcf_analysis`

Purpose:
- single-input variant interpretation on a VCF

Expected source:
- VCF or VCF.gz

Typical sequence:
- `@mode vcf_analysis`
- upload VCF
- `@skill representative_vcf_review`
- optional follow-ups such as `@liftover`, `@snpeff`, `@ldblockshow`

### `@mode raw_sequence`

Purpose:
- raw sequencing or alignment QC

Expected source:
- FASTQ, BAM, SAM, or CRAM

Typical sequence:
- `@mode raw_sequence`
- upload source
- `@skill raw_qc_review`
- optional follow-up `@samtools`

Use `@mode help` at any time to reprint the mode list.

## Current Skills

The current orchestrator workflows live in:
- [skills/chatgenome-orchestrator/workflows](skills/chatgenome-orchestrator/workflows)

Available workflow triggers:

| Trigger | Purpose | Input expectation |
|---|---|---|
| `@skill representative_vcf_review` | Default VCF review workflow | Active VCF source |
| `@skill raw_qc_review` | Default raw-sequencing QC workflow | Active raw source |
| `@skill summary_stats_review` | Summary-statistics intake and review | Active summary-statistics source |
| `@skill prs_prep` | Build check, harmonization prep, and PLINK score-file preparation | Summary-statistics source in PRS mode or active summary-statistics source |

Use:
- `@skill help`
- `@skill <workflow_name> help`

to inspect the available workflows and their steps.

## Current Tools

### Direct user-facing `@tool` aliases

These are the main explicit tool triggers exposed in the UI.

| Trigger | Purpose | Typical input |
|---|---|---|
| `@liftover` | Run GATK LiftoverVcf on the current VCF | VCF |
| `@samtools` | Alignment/QC review for BAM/SAM/CRAM | raw sequencing or alignment source |
| `@plink` | PLINK QC or scoring | VCF target genotype |
| `@snpeff` | Local SnpEff consequence annotation | VCF |
| `@ldblockshow` | LD block visualization for a locus | VCF plus region |
| `@qqman` | Manhattan and QQ plot generation | summary statistics |

Each direct tool also supports:
- `@toolname help`

Example:
- `@liftover help`
- `@plink help`
- `@qqman help`

### Internal plugin inventory

These plugins currently exist under `/plugins`:

| Plugin | Role |
|---|---|
| `vcf_qc_tool` | VCF-level QC facts |
| `annotation_tool` | Core transcript-aware annotation |
| `roh_analysis_tool` | ROH-oriented review artifacts |
| `candidate_ranking_tool` | Candidate-variant prioritization |
| `clinvar_review_tool` | Clinical-significance summaries |
| `vep_consequence_tool` | Consequence distribution summaries |
| `clinical_coverage_tool` | Annotation completeness coverage |
| `filtering_view_tool` | Filtering and triage summaries |
| `symbolic_alt_tool` | Symbolic ALT review path |
| `grounded_summary_tool` | Draft grounded narrative summary |
| `cadd_lookup_tool` | Local CADD enrichment |
| `revel_lookup_tool` | Local REVEL enrichment |
| `fastqc_execution_tool` | FastQC execution |
| `samtools_execution_tool` | Samtools QC/review |
| `gatk_liftover_vcf_tool` | GATK LiftoverVcf execution |
| `plink_execution_tool` | PLINK QC and score modes |
| `snpeff_execution_tool` | Local SnpEff execution |
| `ldblockshow_execution_tool` | LD block plotting |
| `qqman_execution_tool` | Manhattan/QQ plotting |

Some of these plugins are used indirectly through workflows; others are exposed directly via `@tool` aliases.

## Trigger Semantics

### `@mode`

`@mode` chooses the session purpose and changes the input UI.

Examples:

```text
@mode help
@mode prs
@mode vcf_analysis
@mode raw_sequence
```

Effects:
- chooses the current session mode
- changes which source slots are shown
- changes which workflows are recommended
- changes how uploads are interpreted

### `@skill`

`@skill` runs a named workflow.

Examples:

```text
@skill help
@skill representative_vcf_review
@skill representative_vcf_review help
@skill prs_prep
```

Effects:
- runs an ordered multi-step workflow
- updates `Studio`
- may add grounded workflow summaries to chat

### `@toolname`

`@toolname` runs a single deterministic tool against the current source context.

Examples:

```text
@liftover target=hg38
@samtools
@plink score
@qqman
```

Effects:
- uses the active source or current mode-specific source slot
- returns a direct result
- creates or updates the corresponding Studio card

### `$studio`

`$studio` switches chat into grounded mode.

Examples:

```text
$studio summarize the current candidate card
$studio explain this PRS Prep Review card
```

Effects:
- the answer is generated from current Studio state, not general knowledge
- use this when you want explanations of current results

Without `$studio`, ChatGenome should answer as a normal GPT assistant.

## PRS Flow

The current PRS MVP is:

1. `@mode prs`
2. Upload summary statistics
3. Upload target genotype VCF
4. `@skill prs_prep`
   - build check
   - harmonization prep
   - PLINK score-file generation
5. `@plink score`
6. Review:
   - `PRS Prep Review`
   - `PLINK` score review card

For a known-good local smoke test pair:

- summary statistics:
  - [examples/prs_overlap_sumstats.tsv](examples/prs_overlap_sumstats.tsv)
- target genotype:
  - [examples/roh.1.vcf.gz](examples/roh.1.vcf.gz)

## Quick Start

### Environment

```bash
git clone https://github.com/bispl-create/chatgenome.git
cd chatgenome
cp .env.example .env
```

Set your API key in `.env` if you want GPT-backed behavior:

```bash
OPENAI_API_KEY=sk-...
OPENAI_WORKFLOW_MODEL=gpt-5-nano
OPENAI_MODEL=gpt-5-mini
```

### Install

Python dependencies:

```bash
python3 -m pip install --target /Users/jongcye/Documents/Codex/.vendor -r requirements.txt
```

Frontend dependencies:

```bash
PATH=/Users/jongcye/Documents/Codex/.local/node/node-v22.22.1-darwin-arm64/bin:$PATH npm install
```

### Run

Backend:

```bash
cd bioinformatics_vcf_evidence_mvp
PYTHONPATH=/Users/jongcye/Documents/Codex/.vendor uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Frontend build:

```bash
cd webapp
PATH=/Users/jongcye/Documents/Codex/.local/node/node-v22.22.1-darwin-arm64/bin:$PATH npm run build:local
```

Frontend start:

```bash
cd webapp
HOST=127.0.0.1 PORT=3003 PATH=/Users/jongcye/Documents/Codex/.local/node/node-v22.22.1-darwin-arm64/bin:$PATH npm run start:local
```

Then open:
- [http://127.0.0.1:3003](http://127.0.0.1:3003)

## Documentation

- Revision history:
  - [docs/REVISION_HISTORY.md](docs/REVISION_HISTORY.md)
- Developer manual:
  - [docs/DEVELOPER_MANUAL.md](docs/DEVELOPER_MANUAL.md)
- Tool plugin guide:
  - [docs/TOOL_PLUGIN_GUIDE.md](docs/TOOL_PLUGIN_GUIDE.md)
- Contributing:
  - [CONTRIBUTING.md](CONTRIBUTING.md)

## Current Limits

- `@tool` and `@skill` dispatch are much thinner than before, but not yet fully generic
- some evidence sources such as standalone `@clinvar`, `@gnomad`, and `@vep` are not yet exposed as direct tools
- PRS scoring currently uses a lightweight PLINK-based MVP, not a full PRSice/PRS-CS stack
- `ANNOVAR`, `dbNSFP`, and `SnpSift` are not yet integrated

## Design Principle

For genomics work, the intended pattern is:

1. use deterministic tools to establish facts
2. render those outputs in Studio
3. use `$studio` only when you want the model to explain the grounded result

This keeps execution, evidence, and explanation clearly separated.
