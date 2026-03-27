# Plugin Runner Audit

## Goal

Track which plugin `run.py` files can now be treated as optional compatibility wrappers, and which plugins still need `run.py` because they do not yet expose a direct `entrypoint`.

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
- `fastqc_execution_tool`
- `filtering_view_tool`
- `gatk_liftover_vcf_tool`
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
- `vcf_qc_tool`

Notes:
- `run.py` may still be kept as a compatibility wrapper for manual CLI use.
- From a backend/plugin-runtime standpoint, these are already entrypoint-first.

## Not Yet Ready For `run.py` Removal

These plugins still have `run.py`, but do not yet have both:
- `logic.py`
- `tool.json.entrypoint`

They still rely on the older subprocess runner contract.

- `clinical_coverage_tool`
- `clinvar_review_tool`
- `grounded_summary_tool`
- `symbolic_alt_tool`
- `vep_consequence_tool`

Recommended next step for each:
1. add `logic.py`
2. add `def execute(payload)`
3. add `tool.json.entrypoint`
4. keep `run.py` only as optional compatibility layer

## Core Status

Current backend state:
- `tool_runner.run_tool()` prefers `tool.json.entrypoint`
- it falls back to `run.py` only when `entrypoint` is absent
- `main.py` tool API endpoints already route through the generic tool runner

This means the platform is now:
- entrypoint-first
- `run.py`-compatible
- not yet fully `run.py`-free

## Recommended Next Step

To complete the entrypoint-first migration:

1. convert the 5 remaining plugins listed above to `logic.py + entrypoint`
2. decide whether `run.py` should remain for manual CLI compatibility
3. if CLI compatibility is not needed, delete `run.py` from entrypoint-ready plugins in batches
