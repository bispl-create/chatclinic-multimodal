# Services Classification

This file captures the intended steady-state role of modules under `app/services`.

## Platform Core

These files should remain in `app/services` because they implement routing, orchestration, registry, or runtime infrastructure.

- `chat.py`
- `jobs.py`
- `plugin_runtime.py`
- `source_bootstrap.py`
- `source_registry.py`
- `tool_runner.py`
- `workflow_agent.py`
- `workflow_fallbacks.py`
- `workflow_hooks.py`
- `workflow_internal_steps.py`
- `workflow_responses.py`
- `workflow_transforms.py`
- `workflows.py`

## Shared Helpers

These files are domain-aware, but they assemble or enrich workflow state rather than representing a single standalone tool.

- `annotation.py`
- `cache_store.py`
- `recommendation.py`
- `references.py`

## Shim Layer

These files now exist primarily as compatibility imports while tool logic lives in plugin packages.

- `cadd_lookup.py`
- `candidate_ranking.py`
- `fastqc.py`
- `filtering.py`
- `gatk_liftover.py`
- `ldblockshow.py`
- `plink.py`
- `prs_prep.py`
- `r_vcf_plots.py`
- `revel_lookup.py`
- `roh_analysis.py`
- `samtools.py`
- `snpeff.py`
- `summary_stats.py`
- `variant_annotation.py`
- `vcf_summary.py`

## Guiding Rule

- New standalone execution logic belongs under `plugins/<tool>/`.
- `app/services` should only gain new files when the behavior is cross-tool infrastructure or a reusable shared helper.
- Shim files can be removed once all direct imports are migrated to plugin entrypoints or plugin logic modules.
