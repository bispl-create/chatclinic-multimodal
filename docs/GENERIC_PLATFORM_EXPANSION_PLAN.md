# Generic Platform Expansion Plan

## Goal

This project is moving from a genomics-specific application toward a generic multimodal platform.

The long-term target is:

- Adding a new tool should usually require only:
  - `plugins/<tool>/tool.json`
  - `plugins/<tool>/run.py`
- Adding a new workflow should usually require only:
  - `skills/.../workflows/<workflow>.json`
- Core files such as:
  - `app/main.py`
  - `app/services/chat.py`
  - `app/services/workflows.py`
  - `webapp/app/page.tsx`
  should not need routine changes for normal extensions.

The future input formats in scope are:

- Genomics:
  - VCF
  - BAM/SAM/CRAM
  - summary statistics
- Imaging:
  - DICOM
  - PNG/JPG/TIFF
- Clinical structured data:
  - FHIR
  - HL7
  - CSV/TSV
  - multi-sheet Excel
- Clinical unstructured data:
  - text notes
  - reports

## Current Direction

The recommended direction is:

1. Finish genericization of the current genomics platform.
2. Generalize source types, bootstrap workflows, and Studio rendering.
3. Integrate new formats in this order:
   - spreadsheet/text
   - DICOM/image
   - FHIR/HL7

This order keeps the application runnable throughout the migration and allows browser validation after each stage.

## Stage 1. Finish Direct `@tool` Genericization

### Objective

Make direct tool execution metadata-driven so that new direct tools are added through plugin manifests rather than editing chat routing.

### Work

- Finalize `tool.json.direct_chat` usage.
- Keep direct tool parsing generic in `chat.py`.
- Remove remaining legacy direct helper code where possible.
- Standardize response formatting for direct tools.

### Current Status

Completed on branch `codex/generic_tool` for the current direct tools:

- `liftover`
- `samtools`
- `qqman`
- `snpeff`
- `ldblockshow`
- `plink`

What changed:

- direct aliases are resolved from `tool.json`
- direct execution metadata is read from `tool.json.direct_chat`
- direct argument parsing is driven by `argument_mode`
- legacy direct wrappers were removed for the migrated tools
- direct dispatch now uses generic endpoint maps instead of alias-specific routing tables

### Validation

In the browser:

- VCF session:
  - `@liftover help`
  - `@liftover target=hg38`
  - `@snpeff`
  - `@ldblockshow chr11:24100000:24200000`
  - `@plink score`
- Raw QC session:
  - `@samtools`
- Summary stats session:
  - `@qqman`

### Completion Criteria

- Direct tool aliases come from `tool.json`.
- Direct tool argument parsing comes from `tool.json`.
- Adding a normal direct tool does not require editing `chat.py`.

## Stage 2. Make Workflow Dispatch Manifest-Driven

### Objective

Remove workflow-name dispatch tables from `chat.py`.

### Work

Add execution metadata to each workflow manifest:

- `source_type`
- `requested_view`
- `response_kind`
- `answer_template`

Then make `chat.py` do only:

1. parse `@skill`
2. load workflow manifest
3. validate source compatibility
4. call source-type generic workflow runner
5. assemble response from manifest metadata

### Validation

- `@skill representative_vcf_review`
- `@skill raw_qc_review`
- `@skill summary_stats_review`
- `@skill prs_prep`

### Completion Criteria

- No workflow-name dispatch table required for normal workflow execution.

## Stage 3. Normalize Workflow Manifests as Structured Batch Definitions

### Objective

Treat workflows as structured batch files that define tool execution order.

### Standard Step Shape

```json
{
  "tool": "annotation_tool",
  "bind": "annotations",
  "needs": ["facts"],
  "on_fail": "continue"
}
```

### Required Workflow Fields

- `name`
- `source_type`
- `requested_view`
- `response_kind`
- `answer_template`
- `steps`

### Validation

- Workflow help output remains correct.
- Existing workflows still execute in the browser.

### Completion Criteria

- Workflow manifests are the authoritative execution definition.

### Current Status

Partially completed on branch `codex/generic_tool`:

- all current workflow manifests use structured object steps
- workflow manifests now carry:
  - `source_type`
  - `requested_view`
  - `response_kind`
  - `answer_template`
- workflow loading normalizes and validates:
  - `tool`
  - `bind`
  - `needs`
  - optional `on_fail`

Remaining work for later stages:

- eliminate remaining workflow-name special handling in `workflows.py`
- move more execution behavior from runners into manifest + tool metadata

## Stage 4. Make Workflow Step Execution Metadata-First

### Objective

Reduce tool-specific step execution logic in `workflows.py`.

### Work

Expand `tool.json.workflow_binding` to support:

- `source_type`
- `input_map`
- `result_path`
- `transform`
- `fallback_transform`
- optional `preprocess`
- optional `postprocess`

The generic step runner should:

1. resolve inputs from context
2. call the tool
3. transform results
4. bind outputs back into context

### Validation

- `@skill representative_vcf_review`
- `@skill raw_qc_review`
- `@skill summary_stats_review`
- `@skill prs_prep`

### Completion Criteria

- Adding an already-supported tool to a workflow does not require editing `workflows.py`.

### Current Status

Partially completed on branch `codex/generic_tool`:

- VCF step execution now prefers `tool.json.workflow_binding`
- `workflow_binding` supports optional:
  - `preprocess`
  - `postprocess`
- the remaining representative-VCF custom steps were moved closer to metadata-first execution:
  - `snpeff_execution_tool`
  - `cadd_lookup_tool`
  - `revel_lookup_tool`
  - `grounded_summary_tool`

Remaining work for later stages:

- reduce or remove summary-stats and raw-QC pseudo-tool step registries
- split transform, fallback, and hook code out of `workflows.py`
- remove the remaining legacy VCF workflow path

## Stage 5. Split Transform, Fallback, and Hook Registries

### Objective

Move domain-specific execution details out of `workflows.py`.

### Proposed Modules

- `app/services/workflow_transforms.py`
- `app/services/workflow_fallbacks.py`
- `app/services/workflow_hooks.py`

### Validation

- Existing workflows still produce the same outputs.
- Fallback paths still work when a tool does not produce its expected output.

### Completion Criteria

- `workflows.py` becomes orchestration-only.

### Current Status

Partially completed on branch `codex/generic_tool`:

- VCF workflow transforms were moved to:
  - `app/services/workflow_transforms.py`
- VCF fallback logic was moved to:
  - `app/services/workflow_fallbacks.py`
- VCF preprocess/postprocess hooks were moved to:
  - `app/services/workflow_hooks.py`
- `workflows.py` now calls these registries instead of carrying the VCF hook and transform bodies inline

Remaining work for later stages:

- move or reduce remaining non-VCF step registries
- shrink the legacy VCF path
- continue moving response assembly and bootstrap logic out of `workflows.py`

Additional progress on branch `codex/generic_tool`:

- summary-statistics and raw-QC internal pseudo-step execution was moved out of `workflows.py` into:
  - `app/services/workflow_internal_steps.py`

This does not make those steps fully pluginized yet, but it removes another source-specific executor registry from the main orchestration file.

## Stage 6. Convert Upload Bootstrap Into Workflow-Driven Source Initialization

### Objective

Make `main.py` generic by replacing source-type bootstrap functions with source bootstrap workflows.

### Proposed Bootstrap Manifests

- `vcf_bootstrap.json`
- `raw_qc_bootstrap.json`
- `summary_stats_bootstrap.json`
- later:
  - `spreadsheet_bootstrap.json`
  - `text_bootstrap.json`
  - `dicom_bootstrap.json`
  - `image_bootstrap.json`
  - `fhir_bootstrap.json`
  - `hl7_bootstrap.json`

### Validation

Upload and confirm initial cards/state for:

- VCF
- BAM/SAM/CRAM
- summary statistics

### Completion Criteria

- `main.py` becomes an API router plus generic bootstrap launcher.

## Stage 7. Introduce a Generic Source Registry

### Objective

Generalize source handling beyond genomics.

### Proposed Source Types

- `genomic_vcf`
- `genomic_alignment`
- `genomic_sumstats`
- `dicom`
- `medical_image`
- `spreadsheet`
- `tabular`
- `text_note`
- `fhir`
- `hl7`

### Validation

Upload sample files and confirm source classification is correct.

### Completion Criteria

- New source types can be added without special-case routing in core files.

## Stage 8. Create a Generic Studio Renderer Layer

### Objective

Reduce card-specific rendering logic in `page.tsx`.

### Generic Card Types

- metadata table
- preview table
- warning list
- text section viewer
- artifact link list

### Specialized Cards Kept as Exceptions

- IGV or genomics visualization
- DICOM viewer
- PRS review
- FHIR resource browser

### Validation

- Existing genomics Studio cards still render.
- Generic fallback card renders for unrecognized result types.

### Completion Criteria

- Most new result types do not require custom frontend code.

## Stage 9. Integrate Spreadsheet and Text Inputs

### Objective

Add the easiest non-genomics formats first.

### Scope

- multi-sheet Excel
- CSV/TSV
- plain text notes

### Required Work

- source classification
- bootstrap workflow
- basic preview cards
- grounded `$studio` support

### Validation

- Upload workbook
- inspect sheet inventory
- inspect per-sheet preview
- ask `$studio` questions on current sheet/card

### Completion Criteria

- Spreadsheet and text become first-class source types.

## Stage 10. Integrate DICOM and Raster Image Inputs

### Objective

Add imaging support after the generic source/bootstrap system exists.

### Scope

- DICOM
- PNG/JPG/TIFF

### Required Work

- source classification
- bootstrap workflow
- image preview generation
- DICOM metadata extraction
- custom viewer card
- grounded `$studio` support

### Validation

- Upload DICOM
- Upload PNG/TIFF
- View images in Studio
- Ask grounded imaging questions

### Completion Criteria

- Imaging becomes a standard source category using the same platform shell.

## Stage 11. Integrate FHIR and HL7 Inputs

### Objective

Add complex clinical structured data last, after the generic platform is stable.

### Required Work

- source classification
- bootstrap workflow
- resource/segment parsing
- patient/chart summary cards
- grounded `$studio` support

### Validation

- Upload FHIR bundle or resource file
- Upload HL7 message
- Inspect chart cards
- Ask grounded chart questions

### Completion Criteria

- FHIR and HL7 become standard source types on the same platform.

## Validation Principle for Every Stage

For each stage:

1. restart backend
2. refresh frontend
3. run the smallest possible smoke test
4. confirm Studio cards and `$studio` behavior
5. commit only the focused stage changes

This keeps the application working continuously throughout the migration.

## Practical End State

### New Tool

Normally add only:

- `plugins/<tool>/tool.json`
- `plugins/<tool>/run.py`

Rare exceptions:

- a new transform
- a new fallback hook
- a new specialized Studio renderer

### New Workflow

Normally add only:

- `skills/.../workflows/<workflow>.json`

Rare exceptions:

- a new response kind
- a new bootstrap type

## Summary

The recommended path is:

1. finish genericization of direct tools
2. genericize workflow dispatch
3. standardize workflow manifests as structured batch definitions
4. genericize workflow step execution
5. genericize bootstrap analysis
6. generalize source types
7. genericize Studio rendering
8. integrate new formats incrementally

This is the safest path to a platform where most new tools require only plugin files and most new workflows require only workflow JSON.
