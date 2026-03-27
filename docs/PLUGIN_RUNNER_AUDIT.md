# Plugin Runner Audit

## Goal

Track which plugin `run.py` files can now be treated as optional compatibility wrappers, and define a safe batch plan for removing them.

Audit date:
- 2026-03-27

## Ready For `run.py` Removal

These plugins currently have:
- `logic.py`
- `def execute(payload)`
- `tool.json.entrypoint`

So backend execution can use direct import/execute without subprocess `run.py`.

- `annotation_tool`
- `cadd_lookup_tool`
- `candidate_ranking_tool`
- `clinical_coverage_tool`
- `clinvar_review_tool`
- `fastqc_execution_tool`
- `filtering_view_tool`
- `gatk_liftover_vcf_tool`
- `grounded_summary_tool`
- `ldblockshow_execution_tool`
- `plink_execution_tool`
- `prs_prep_tool`
- `qqman_execution_tool`
- `raw_qc_review_tool`
- `revel_lookup_tool`
- `roh_analysis_tool`
- `samtools_execution_tool`
- `snpeff_execution_tool`
- `summary_stats_review_tool`
- `symbolic_alt_tool`
- `vcf_qc_tool`
- `vep_consequence_tool`

Notes:
- `run.py` may still be kept as a compatibility wrapper for manual CLI use.
- From a backend/plugin-runtime standpoint, these are already entrypoint-first.

## Not Yet Ready For `run.py` Removal

No remaining plugin is blocked on missing `logic.py`, missing `execute(payload)`, or missing `tool.json.entrypoint`.

At this point, removal is purely a compatibility decision, not a runtime-engineering blocker.

## Core Status

Current backend state:
- `tool_runner.run_tool()` prefers `tool.json.entrypoint`
- it falls back to `run.py` only when `entrypoint` is absent
- `main.py` tool API endpoints already route through the generic tool runner

This means the platform is now:
- entrypoint-first
- `run.py`-compatible
- ready for staged `run.py` removal

## Deletion Batch Plan

### Batch 1. Direct Tool Endpoints

Lowest-risk batch because these are already exercised through generic API/tool routing.

- `annotation_tool`
- `gatk_liftover_vcf_tool`
- `samtools_execution_tool`
- `snpeff_execution_tool`
- `ldblockshow_execution_tool`
- `plink_execution_tool`
- `qqman_execution_tool`
- `filtering_view_tool`

Check after removal:
- `@liftover`
- `@samtools`
- `@snpeff`
- `@ldblockshow`
- `@plink`
- `@qqman`

### Batch 2. Workflow-Core VCF Plugins

These are critical to the representative VCF workflow, so remove only after Batch 1 is stable.

- `vcf_qc_tool`
- `candidate_ranking_tool`
- `roh_analysis_tool`
- `cadd_lookup_tool`
- `revel_lookup_tool`
- `clinical_coverage_tool`
- `clinvar_review_tool`
- `vep_consequence_tool`
- `symbolic_alt_tool`
- `grounded_summary_tool`

Check after removal:
- `@skill representative_vcf_review`
- grounded `$studio` questions
- candidate variants / summary cards

### Batch 3. Review / Bootstrap Plugins

These affect source bootstrap and review workflows.

- `fastqc_execution_tool`
- `raw_qc_review_tool`
- `summary_stats_review_tool`
- `prs_prep_tool`

Check after removal:
- `@skill raw_qc_review`
- `@skill summary_stats_review`
- `@skill prs_prep`
- source upload bootstrap for raw-QC and summary stats

## Recommended Next Step

1. decide whether manual `python run.py --input ... --output ...` compatibility is still needed
2. if not needed, remove Batch 1 `run.py` files first
3. verify the checks listed above
4. continue with Batch 2 and Batch 3 only after Batch 1 is stable
