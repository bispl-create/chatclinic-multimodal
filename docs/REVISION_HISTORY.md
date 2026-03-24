# Revision History

This file summarizes the major architecture revisions that led to the current `@mode / @skill / @tool / $studio` design.

## Current Architectural Direction

ChatGenome now prefers:

1. explicit session selection with `@mode`
2. explicit workflow execution with `@skill`
3. explicit deterministic tool execution with `@toolname`
4. explicit grounded explanation with `$studio`

This replaced earlier patterns where uploads triggered large backend pipelines automatically and where chat intent could be inferred from loose keywords.

## Key Revision Groups

### 1. Studio-grounded chat became explicit

Relevant commits:
- `852c02e` Fix studio trigger flow and frontend upload routing
- `4c95e58` Remove automatic grounded summary after upload
- `2f15e0f` Fix workflow summary message ordering

What changed:
- general GPT chat was separated from Studio-grounded chat
- `$studio` became the explicit grounding trigger
- automatic grounded summary on upload was removed
- workflow summaries were moved into normal assistant-message ordering

Why it mattered:
- normal questions no longer get incorrectly interpreted as Studio questions
- grounded responses are opt-in and more predictable

### 2. Tool execution moved to explicit `@tool` triggers

Relevant commits:
- `9a570a3` Add @toolname chat routing and help metadata
- `6e29d12` Add qqman summary statistics tool
- `11dbdb4` Improve @tool help formatting
- `46619f3` Support direct tool runs from active sources

What changed:
- `@liftover`, `@samtools`, `@plink`, `@snpeff`, `@ldblockshow`, `@qqman`
  became the primary tool entrypoints
- `@toolname help` now renders metadata-backed help text
- direct pre-analysis tool runs can create Studio cards without first running a full workflow

Why it mattered:
- deterministic tool execution is easier to reason about
- users can inspect help and run tools on demand
- backend keyword-based branching was reduced

### 3. Workflows moved toward explicit `@skill` execution

Relevant commits:
- `acfc0c4` Add explicit skill and tool trigger workflows
- `9e294d9` Add workflow-driven skill chat flow

What changed:
- workflow registries were introduced for:
  - `representative_vcf_review`
  - `raw_qc_review`
  - `summary_stats_review`
  - `prs_prep`
- `@skill help` and `@skill <workflow> help` became available
- uploads stopped auto-running the full review workflow by default

Why it mattered:
- workflows became visible and inspectable
- execution intent became explicit instead of implicit

### 4. PRS workflow became a first-class path

Relevant commits:
- `fcbbd24` Add standalone PRS prep workflow
- `e1d4703` Add PLINK score workflow and PRS prep wiring
- `edee59b` Fix PLINK score parsing and PRS mode flow

What changed:
- `@skill prs_prep` was added
- PRS prep now performs:
  - build check
  - harmonization prep
  - PLINK score-file generation
- `@plink score` was added as a direct scoring path
- `PRS Prep Review` and `PLINK` score review UI were added
- synthetic overlap test data was added for local smoke tests

Why it mattered:
- post-GWAS analysis is no longer limited to plotting
- ChatGenome now has an explicit MVP PRS workflow

### 5. Session modes were separated from workflows

Relevant commits:
- `f8b6d09` Update startup messaging for @mode workflow

What changed:
- `@mode` became a separate concept from `@skill`
- current modes:
  - `prs`
  - `vcf_analysis`
  - `raw_sequence`
- PRS mode now presents two source roles:
  - summary statistics
  - target genotype

Why it mattered:
- workflows and UI layout are now mode-driven
- PRS no longer has to overload a single active source

### 6. `@tool` routing was refactored into a generic dispatch path

Relevant commits:
- `cdc74d0` Refactor @tool parsing and compatibility checks
- `7345a9f` Refactor direct tool dispatch routing
- `2dbbdd5` Normalize direct tool chat responses
- `b61b52c` Clean up generic @tool chat helpers

What changed:
- a shared alias registry was introduced for direct tools
- `@tool` parsing was normalized into a common request shape
- `@tool help` rendering and source-compatibility checks were centralized
- tool routing was moved from scattered `if/elif` branches toward dispatch tables
- direct tool responses started using shared response helpers

Why it mattered:
- adding new tools now requires fewer bespoke chat branches
- tool behavior is more consistent across VCF, raw-QC, and summary-statistics sessions
- the backend is closer to a thin runtime layer rather than a keyword router

### 7. Direct tool execution status became visible in the UI

Relevant commits:
- `d6aee4c` Show running status for direct tool execution

What changed:
- long-running direct tools now update the top status badge while running
- dedicated running/ready/failed states were added for:
  - `@liftover`
  - `@qqman`
  - `@samtools`
  - `@snpeff`
  - `@ldblockshow`
  - `@plink`
  - `@plink score`
- the status-detail text now explains what each direct tool is currently doing

Why it mattered:
- users can see that long-running deterministic tools are still working
- stale “ready” states no longer make active tool runs look frozen

## Current State Summary

ChatGenome now supports:

- explicit session-mode selection
- explicit workflow execution
- explicit tool execution
- explicit grounded Studio chat
- direct Studio cards for pre-analysis tool runs
- summary-statistics plotting with `qqman`
- PRS prep plus PLINK scoring MVP
- genericized `@tool` parsing, help, compatibility checks, and dispatch routing
- direct-tool running status feedback in the chat header

## Known Follow-up Areas

Still desirable:

- more generic backend runners for `@skill`
- standalone evidence tools such as `@clinvar`, `@gnomad`, and `@vep`
- improved contributor-facing tool metadata and registration patterns
- more polished onboarding and mode-specific guidance
