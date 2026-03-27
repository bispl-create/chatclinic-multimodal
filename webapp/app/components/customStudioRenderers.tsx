"use client";

import { useMemo } from "react";

import IgvBrowser from "./IgvBrowser";
import { type StudioRendererBuilderArgs, type StudioRendererRegistry } from "./studioRendererTypes";

function MetricBarList({
  items,
  emptyLabel,
}: {
  items: Array<{ label: string; detail: string; value: number }>;
  emptyLabel: string;
}) {
  const maxValue = items.reduce((acc, item) => Math.max(acc, item.value), 0);
  if (!items.length) {
    return <p className="emptyState">{emptyLabel}</p>;
  }

  return (
    <div className="distributionList">
      {items.map((item) => {
        const width = maxValue > 0 ? Math.max((item.value / maxValue) * 100, 6) : 0;
        return (
          <div key={`${item.label}-${item.detail}`} className="distributionRow">
            <div className="distributionMeta">
              <span>{item.label}</span>
              <strong>{item.detail}</strong>
            </div>
            <div className="distributionTrack">
              <div className="distributionFill" style={{ width: `${width}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function CohortBrowserCard({
  activeView,
  analysis,
  components,
  helpers,
}: {
  activeView: string | null;
  analysis: any;
  components: StudioRendererBuilderArgs["components"];
  helpers: StudioRendererBuilderArgs["helpers"];
}) {
  const { StudioMetricGrid, StudioPreviewTable, StudioSimpleList, WarningListCard } = components;
  const { formatNumber } = helpers;
  const sheetNames = Array.isArray(analysis?.sheet_names) ? analysis.sheet_names : [];
  const selectedSheetFromView =
    typeof activeView === "string" && activeView.startsWith("sheet::") && activeView.endsWith("::cohort_browser")
      ? activeView.slice("sheet::".length, -"::cohort_browser".length)
      : null;
  const selectedSheet =
    (selectedSheetFromView && sheetNames.includes(selectedSheetFromView) && selectedSheetFromView) ||
    (typeof analysis?.selected_sheet === "string" && analysis.selected_sheet.trim()) ||
    sheetNames[0] ||
    null;

  const artifact = useMemo(() => {
    if (!selectedSheet || !analysis?.artifacts) {
      return null;
    }
    return analysis.artifacts[`sheet::${selectedSheet}::cohort_browser`] ?? null;
  }, [analysis, selectedSheet]);

  const sheetDetails = useMemo(() => {
    const details = Array.isArray(analysis?.sheet_details) ? analysis.sheet_details : [];
    return details.find((item: any) => item?.sheet_name === selectedSheet) ?? null;
  }, [analysis, selectedSheet]);

  const previewRows = Array.isArray(artifact?.grid?.rows)
    ? artifact.grid.rows.map((row: Record<string, unknown>) =>
        Object.fromEntries(Object.entries(row).map(([key, value]) => [key, String(value ?? "")])),
      )
    : [];
  const previewColumns = Array.isArray(artifact?.grid?.columns) ? artifact.grid.columns : [];
  const schemaHighlights = Array.isArray(artifact?.schema_highlights) ? artifact.schema_highlights : [];
  const missingnessRows = Array.isArray(artifact?.missingness?.top_missing_columns)
    ? artifact.missingness.top_missing_columns
    : Array.isArray(artifact?.missingness)
      ? artifact.missingness
      : [];
  const schemaGraphItems = schemaHighlights.map((item: any) => ({
    label: String(item.name ?? item.column ?? "column"),
    detail: `${item.inferred_type ?? item.type ?? "unknown"} | ${formatNumber(item.unique_count)} unique`,
    value: Number(item.unique_count ?? 0),
  }));
  const missingnessGraphItems = missingnessRows.map((item: any) => {
    const rate = typeof item.missing_rate === "number" ? item.missing_rate * 100 : Number(String(item.percent ?? "0").replace("%", "")) || 0;
    return {
      label: String(item.column ?? "column"),
      detail: `${formatNumber(item.missing_count ?? item.missing)} missing | ${rate.toFixed(1)}%`,
      value: rate,
    };
  });
  const overviewItems = [
    { label: "Workbook", value: analysis?.file_name ?? "n/a", tone: "neutral" as const },
    { label: "Sheets", value: formatNumber(analysis?.sheet_count), tone: "good" as const },
    { label: "Selected", value: selectedSheet ?? "n/a", tone: "neutral" as const },
    { label: "Rows", value: formatNumber(artifact?.overview?.row_count ?? sheetDetails?.row_count), tone: "good" as const },
    { label: "Columns", value: formatNumber(artifact?.overview?.column_count ?? sheetDetails?.column_count), tone: "neutral" as const },
    {
      label: "Shape",
      value:
        artifact?.overview?.row_count != null && artifact?.overview?.column_count != null
          ? `${artifact.overview.row_count} x ${artifact.overview.column_count}`
          : "n/a",
      tone: "neutral" as const,
    },
  ];

  return (
    <section className="notebookPanel studioCanvasPanel">
      <div className="notebookHeader">
        <h2>Cohort Browser</h2>
        <span className="pill">{analysis?.file_name ?? "spreadsheet"}</span>
      </div>
      <div className="studioCanvasBody">
        <StudioMetricGrid items={overviewItems} />
        {artifact ? (
          <>
            <div className="resultSectionSplit">
              <article className="miniCard">
                <h3>Schema highlights</h3>
                <MetricBarList
                  items={schemaGraphItems}
                  emptyLabel="No schema highlights available."
                />
              </article>
              <article className="miniCard">
                <h3>Missingness</h3>
                <MetricBarList
                  items={missingnessGraphItems}
                  emptyLabel="No missingness summary available."
                />
              </article>
            </div>
            <div className="resultSectionSplit">
              <article className="miniCard">
                <h3>Composition</h3>
                <StudioSimpleList
                  items={[
                    { label: "Records", detail: formatNumber(artifact.composition?.record_count) },
                    { label: "Fields", detail: formatNumber(artifact.composition?.field_count) },
                    { label: "Categorical columns", detail: formatNumber((artifact.composition?.categorical_breakdowns ?? []).length) },
                    { label: "Numeric columns", detail: formatNumber((artifact.composition?.numeric_breakdowns ?? []).length) },
                  ]}
                  emptyLabel="No composition summary available."
                />
              </article>
              <article className="miniCard">
                <h3>Top categorical values</h3>
                <StudioSimpleList
                  items={(artifact.composition?.categorical_breakdowns ?? []).slice(0, 6).map((item: any) => ({
                    label: item.column ?? "column",
                    detail: Array.isArray(item.top_values)
                      ? item.top_values.map((entry: any) => `${entry.label}: ${formatNumber(entry.count)}`).join(" | ")
                      : "No values",
                  }))}
                  emptyLabel="No categorical breakdown is available."
                />
              </article>
            </div>
            <article className="miniCard">
              <h3>Grid preview</h3>
              {previewColumns.length && previewRows.length ? (
                <StudioPreviewTable
                  columns={previewColumns}
                  rows={previewRows}
                  rowHeaderLabel="Row"
                  footer={
                    <p className="summaryStatsGridMeta">
                      Showing {previewRows.length} preview rows from the selected sheet.
                    </p>
                  }
                />
              ) : (
                <p className="emptyState">No grid preview is available for this sheet.</p>
              )}
            </article>
            <WarningListCard
              warnings={Array.isArray(analysis?.warnings) ? analysis.warnings : []}
              emptyLabel="No workbook warnings."
            />
          </>
        ) : (
          <p className="emptyState">No cohort browser artifact is available for the selected sheet.</p>
        )}
      </div>
    </section>
  );
}

export function buildCustomStudioRendererRegistry({
  activeStudioView,
  apiBase,
  analysis,
  prsPrepResultForStudio,
  plinkResultForStudio,
  plinkConfig,
  setPlinkConfig,
  plinkCommandPreview,
  handleRunPlink,
  plinkRunning,
  activeSource,
  attachedFile,
  spreadsheetAnalysis,
  candidateVariants,
  searchedAnnotations,
  setSelectedAnnotationIndex,
  setActiveStudioView,
  buildAcmgHints,
  filteringSummary,
  annotationSearch,
  setAnnotationSearch,
  symbolicAnnotations,
  rohCandidates,
  recessiveShortlist,
  clinvarCounts,
  consequenceCounts,
  geneCounts,
  safeSelectedIndex,
  selectedAnnotation,
  components,
  helpers,
}: StudioRendererBuilderArgs): StudioRendererRegistry {
  const {
    ArtifactLinksRow,
    StudioSimpleList,
    DistributionList,
    VariantTable,
    MetricTile,
    AnnotationDetailCard,
    WarningListCard,
  } = components;
  const { summarizeLabel } = helpers;

  return {
    cohort_browser: () =>
      spreadsheetAnalysis ? (
        <CohortBrowserCard activeView={activeStudioView} analysis={spreadsheetAnalysis} components={components} helpers={helpers} />
      ) : null,
    prs_prep: () =>
      prsPrepResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>PRS Prep Review</h2></div><div className="studioCanvasBody"><div className="resultMetricGrid"><MetricTile label="Inferred build" value={prsPrepResultForStudio.build_check.inferred_build} tone="good" /><MetricTile label="Build confidence" value={prsPrepResultForStudio.build_check.build_confidence} tone="neutral" /><MetricTile label="Effect size" value={prsPrepResultForStudio.harmonization.effect_size_kind} tone="neutral" /><MetricTile label="Ready rows" value={String(prsPrepResultForStudio.kept_rows)} tone={prsPrepResultForStudio.kept_rows > 0 ? "good" : "warn"} /><MetricTile label="Dropped rows" value={String(prsPrepResultForStudio.dropped_rows)} tone="neutral" /><MetricTile label="Score file" value={prsPrepResultForStudio.score_file_ready ? "ready" : "not ready"} tone={prsPrepResultForStudio.score_file_ready ? "good" : "warn"} /></div><div className="resultSectionSplit"><article className="miniCard"><h3>Build check</h3><ul className="hintList"><li><strong>Source build</strong>: {prsPrepResultForStudio.build_check.source_build}</li><li><strong>Target build</strong>: {prsPrepResultForStudio.build_check.target_build}</li><li><strong>Build match</strong>: {prsPrepResultForStudio.build_check.build_match == null ? "undetermined" : prsPrepResultForStudio.build_check.build_match ? "yes" : "no"}</li></ul></article><article className="miniCard"><h3>Harmonization</h3><ul className="hintList"><li><strong>Required fields present</strong>: {prsPrepResultForStudio.harmonization.required_fields_present ? "yes" : "no"}</li><li><strong>Preview rows harmonizable</strong>: {prsPrepResultForStudio.harmonization.harmonizable_preview_rows}</li><li><strong>Ambiguous SNPs</strong>: {prsPrepResultForStudio.harmonization.ambiguous_snp_count}</li>{prsPrepResultForStudio.harmonization.missing_fields.length ? <li><strong>Missing fields</strong>: {prsPrepResultForStudio.harmonization.missing_fields.join(", ")}</li> : null}</ul></article></div><article className="miniCard"><h3>PLINK score-file preview</h3><p className="summaryStatsGridMeta">Columns: {prsPrepResultForStudio.score_file_columns.join(", ") || "ID, A1, BETA"}</p><div className="variantTableWrap summaryStatsTableWrap"><table className="variantTable summaryStatsTable"><thead><tr>{prsPrepResultForStudio.score_file_columns.map((column: string) => <th key={column}>{column}</th>)}</tr></thead><tbody>{prsPrepResultForStudio.score_file_preview_rows.map((row: any, index: number) => <tr key={`prs-prep-preview-${index}`}>{prsPrepResultForStudio.score_file_columns.map((column: string) => <td key={`${index}-${column}`}>{row[column] || ""}</td>)}</tr>)}</tbody></table></div><ArtifactLinksRow items={prsPrepResultForStudio.score_file_path ? [{ label: "Open output", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(prsPrepResultForStudio.score_file_path)}` }] : []} /></article><WarningListCard warnings={[...prsPrepResultForStudio.build_check.warnings, ...prsPrepResultForStudio.harmonization.warnings]} /></div></section>
      ) : null,
    plink: () =>
      analysis || plinkResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>PLINK</h2></div><div className="studioCanvasBody"><div className="resultMetricGrid"><MetricTile label="Mode" value={plinkConfig.mode === "score" ? "Score" : "QC"} tone="good" /><MetricTile label="Source" value={analysis?.facts.file_name ?? activeSource?.file_name ?? attachedFile?.name ?? "n/a"} tone="neutral" /><MetricTile label="Existing result" value={plinkResultForStudio ? "Available" : "Not run yet"} tone={plinkResultForStudio ? "good" : "neutral"} /><MetricTile label="Runner" value={plinkRunning ? "Running" : "Ready"} tone={plinkRunning ? "warn" : "good"} /></div><div className="resultList"><article className="resultListItem resultListStatic"><strong>Run configuration</strong><div className="annotationMetaGrid"><label className="field compactField"><span>Mode</span><select value={plinkConfig.mode} onChange={(event) => setPlinkConfig((current: any) => ({ ...current, mode: event.target.value }))}><option value="qc">qc</option><option value="score">score</option></select></label><label className="field compactField"><span>Output prefix</span><input type="text" value={plinkConfig.outputPrefix} onChange={(event) => setPlinkConfig((current: any) => ({ ...current, outputPrefix: event.target.value }))} /></label><label className="field compactField"><span>Frequency summary</span><input type="checkbox" checked={plinkConfig.runFreq} disabled={plinkConfig.mode === "score"} onChange={(event) => setPlinkConfig((current: any) => ({ ...current, runFreq: event.target.checked }))} /></label><label className="field compactField"><span>Missingness summary</span><input type="checkbox" checked={plinkConfig.runMissing} disabled={plinkConfig.mode === "score"} onChange={(event) => setPlinkConfig((current: any) => ({ ...current, runMissing: event.target.checked }))} /></label><label className="field compactField"><span>Hardy-Weinberg summary</span><input type="checkbox" checked={plinkConfig.runHardy} disabled={plinkConfig.mode === "score"} onChange={(event) => setPlinkConfig((current: any) => ({ ...current, runHardy: event.target.checked }))} /></label><label className="field compactField"><span>Allow extra chr labels</span><input type="checkbox" checked={plinkConfig.allowExtraChr} onChange={(event) => setPlinkConfig((current: any) => ({ ...current, allowExtraChr: event.target.checked }))} /></label></div>{plinkConfig.mode === "score" ? <p className="resultNote">Score mode uses the latest PRS prep score file. Run <code>@skill prs_prep</code> on a summary-statistics source first, then upload a target genotype VCF and run <code>@plink score</code>.</p> : null}</article><article className="resultListItem resultListStatic"><strong>Command preview</strong><pre className="codeBlock">{plinkCommandPreview}</pre></article></div><div className="resultActionRow"><button className="sourceAddButton" type="button" onClick={() => void handleRunPlink()} disabled={plinkRunning || !(analysis?.source_vcf_path || (activeSource?.source_type === "vcf" ? activeSource.source_path : null))}>{plinkRunning ? "Running PLINK..." : "Run PLINK"}</button></div>{plinkResultForStudio ? plinkResultForStudio.mode === "score" ? <><div className="resultMetricGrid"><MetricTile label="Samples scored" value={plinkResultForStudio.sample_count != null ? String(plinkResultForStudio.sample_count) : String(plinkResultForStudio.score_rows.length)} tone="good" /><MetricTile label="Mean score" value={plinkResultForStudio.score_mean != null ? plinkResultForStudio.score_mean.toFixed(4) : "n/a"} tone="neutral" /><MetricTile label="Min score" value={plinkResultForStudio.score_min != null ? plinkResultForStudio.score_min.toFixed(4) : "n/a"} tone="neutral" /><MetricTile label="Max score" value={plinkResultForStudio.score_max != null ? plinkResultForStudio.score_max.toFixed(4) : "n/a"} tone="neutral" /><MetricTile label="Preview rows" value={String(plinkResultForStudio.score_rows.length)} tone="good" /><MetricTile label="Warnings" value={String(plinkResultForStudio.warnings.length)} tone="neutral" /></div><div className="resultList"><article className="resultListItem resultListStatic"><strong>PRS score inputs</strong><span>Target genotype: {plinkResultForStudio.input_path}</span><span>Score file: {plinkResultForStudio.score_file_path || "n/a"}</span></article>{plinkResultForStudio.score_rows.slice(0, 10).map((row: any, index: number) => <article key={`plink-score-${row.sample_id}-${index}`} className="resultListItem resultListStatic"><strong>{row.sample_id}</strong><span>allele ct {row.allele_ct != null ? row.allele_ct.toFixed(2) : "n/a"} | dosage sum {row.named_allele_dosage_sum != null ? row.named_allele_dosage_sum.toFixed(2) : "n/a"} | score {row.score_sum != null ? row.score_sum.toFixed(4) : "n/a"}</span></article>)}{plinkResultForStudio.warnings.map((warning: string, index: number) => <article key={`plink-warning-${index}`} className="resultListItem resultListStatic"><strong>Warning {index + 1}</strong><span>{warning}</span></article>)}</div><ArtifactLinksRow items={[...(plinkResultForStudio.score_output_path ? [{ label: "Open output", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.score_output_path)}` }] : []), ...(plinkResultForStudio.log_path ? [{ label: "Open log", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.log_path)}` }] : [])]} /></> : <><div className="resultMetricGrid"><MetricTile label="Samples" value={plinkResultForStudio.sample_count != null ? String(plinkResultForStudio.sample_count) : "n/a"} tone="neutral" /><MetricTile label="Variants" value={plinkResultForStudio.variant_count != null ? String(plinkResultForStudio.variant_count) : "n/a"} tone="neutral" /><MetricTile label="Freq rows" value={String(plinkResultForStudio.freq_rows.length)} tone="good" /><MetricTile label="Missing rows" value={String(plinkResultForStudio.missing_rows.length)} tone="good" /><MetricTile label="Hardy rows" value={String(plinkResultForStudio.hardy_rows.length)} tone="good" /><MetricTile label="Warnings" value={String(plinkResultForStudio.warnings.length)} tone="neutral" /></div><div className="resultList">{plinkResultForStudio.freq_rows.slice(0, 5).map((row: any, index: number) => <article key={`plink-freq-${row.variant_id}-${index}`} className="resultListItem resultListStatic"><strong>{row.chrom}:{row.variant_id}</strong><span>{row.ref_allele}&gt;{row.alt_allele} | AF {row.alt_freq != null ? row.alt_freq.toFixed(4) : "n/a"} | OBS {row.observation_count ?? "n/a"}</span></article>)}{plinkResultForStudio.missing_rows.slice(0, 3).map((row: any, index: number) => <article key={`plink-missing-${row.sample_id}-${index}`} className="resultListItem resultListStatic"><strong>Missingness {row.sample_id}</strong><span>{row.missing_genotype_count} missing / {row.observation_count} obs | rate {(row.missing_rate * 100).toFixed(2)}%</span></article>)}{plinkResultForStudio.hardy_rows.slice(0, 5).map((row: any, index: number) => <article key={`plink-hardy-${row.variant_id}-${index}`} className="resultListItem resultListStatic"><strong>Hardy {row.chrom}:{row.variant_id}</strong><span>obs het {row.observed_hets ?? "n/a"} | exp het {row.expected_hets != null ? row.expected_hets.toFixed(2) : "n/a"} | p {row.p_value != null ? row.p_value.toExponential(3) : "n/a"}</span></article>)}{plinkResultForStudio.warnings.map((warning: string, index: number) => <article key={`plink-warning-${index}`} className="resultListItem resultListStatic"><strong>Warning {index + 1}</strong><span>{warning}</span></article>)}</div><ArtifactLinksRow items={[...(plinkResultForStudio.freq_path ? [{ label: "Open afreq", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.freq_path)}` }] : []), ...(plinkResultForStudio.missing_path ? [{ label: "Open smiss", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.missing_path)}` }] : []), ...(plinkResultForStudio.hardy_path ? [{ label: "Open hardy", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.hardy_path)}` }] : []), ...(plinkResultForStudio.log_path ? [{ label: "Open log", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.log_path)}` }] : [])]} /></> : <p className="emptyState">No PLINK result is available yet. Configure the run and execute it from this card.</p>}</div></section>
      ) : null,
    candidates: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>Candidate Variants</h2></div><div className="studioCanvasBody"><div className="resultList">{candidateVariants.map(({ item, score, inRoh }: any) => <button type="button" key={`${item.contig}-${item.pos_1based}-${item.rsid}-candidate`} className="resultListItem" onClick={() => { const nextIndex = searchedAnnotations.findIndex((candidate: any) => candidate.contig === item.contig && candidate.pos_1based === item.pos_1based && candidate.rsid === item.rsid); setSelectedAnnotationIndex(nextIndex >= 0 ? nextIndex : 0); setActiveStudioView("annotations"); }}><strong>{item.gene || "Unknown"} | {item.contig}:{item.pos_1based}</strong><span>Score {score} | {summarizeLabel(item.consequence, "Unclassified")} | {summarizeLabel(item.clinical_significance, "Unreviewed")}{inRoh ? " | inside ROH" : ""}{item.cadd_phred != null ? ` | CADD ${item.cadd_phred.toFixed(1)}` : ""}{item.revel_score != null ? ` | REVEL ${item.revel_score.toFixed(3)}` : ""}{" | "}gnomAD {item.gnomad_af || "n/a"}</span></button>)}</div></div></section>
      ) : null,
    acmg: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>ACMG Review</h2></div><div className="studioCanvasBody"><p className="emptyState acmgNote">This is a triage view with ACMG-style evidence hints. It is not a final clinical classification.</p><div className="resultList">{candidateVariants.slice(0, 6).map(({ item }: any) => <article key={`${item.contig}-${item.pos_1based}-${item.rsid}-acmg`} className="resultListItem resultListStatic"><strong>{item.gene || "Unknown"} | {item.rsid || `${item.contig}:${item.pos_1based}`}</strong><span>{summarizeLabel(item.consequence, "Unclassified")} | {summarizeLabel(item.clinical_significance, "Unreviewed")}</span><ul className="hintList">{buildAcmgHints(item).map((hint) => <li key={hint}>{hint}</li>)}</ul></article>)}</div></div></section>
      ) : null,
    table: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>Variant Table</h2><span className="pill">{searchedAnnotations.length} rows</span></div><div className="studioCanvasBody"><StudioSimpleList items={filteringSummary.map((item) => ({ label: item.label, detail: item.detail }))} /><div className="oeAnnotationControls"><label className="field"><span>Search gene / consequence / ClinVar</span><input value={annotationSearch} onChange={(event) => { setAnnotationSearch(event.target.value); setSelectedAnnotationIndex(0); }} placeholder="e.g. PALMD, missense_variant, benign" /></label></div><VariantTable items={searchedAnnotations} onSelect={(item: any) => { const nextIndex = searchedAnnotations.findIndex((candidate: any) => candidate.contig === item.contig && candidate.pos_1based === item.pos_1based && candidate.rsid === item.rsid); setSelectedAnnotationIndex(nextIndex >= 0 ? nextIndex : 0); setActiveStudioView("annotations"); }} /></div></section>
      ) : null,
    symbolic: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>Symbolic ALT Review</h2></div><div className="studioCanvasBody"><p className="emptyState acmgNote">Symbolic ALT records are separated here so they are not over-interpreted as ordinary SNV/indel calls.</p>{symbolicAnnotations.length ? <div className="resultList">{symbolicAnnotations.map((item: any) => <button type="button" key={`${item.contig}-${item.pos_1based}-${item.rsid}-symbolic`} className="resultListItem" onClick={() => { const nextIndex = searchedAnnotations.findIndex((candidate: any) => candidate.contig === item.contig && candidate.pos_1based === item.pos_1based && candidate.rsid === item.rsid); setSelectedAnnotationIndex(nextIndex >= 0 ? nextIndex : 0); setActiveStudioView("annotations"); }}><strong>{item.contig}:{item.pos_1based} | {item.gene || "Unknown"} | {item.alts.join(",")}</strong><span>{summarizeLabel(item.consequence, "Unclassified")} | genotype {item.genotype}</span></button>)}</div> : <p className="emptyState">No symbolic ALT records are present in the current annotated subset.</p>}</div></section>
      ) : null,
    roh: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>ROH / Recessive Review</h2></div><div className="studioCanvasBody"><p className="emptyState acmgNote">This is a homozygous-alt review and ROH-style heuristic from the current annotated subset. A full production workflow should add `bcftools roh` on the complete callset.</p><div className="resultSectionSplit"><article className="miniCard"><h3>ROH-style segments</h3>{rohCandidates.segments.length ? <div className="resultList">{rohCandidates.segments.map((segment: any) => <article key={segment.label} className="resultListItem resultListStatic"><strong>{segment.label}</strong><span>{segment.count} markers | span {segment.spanMb}{segment.quality != null ? ` | quality ${segment.quality.toFixed(1)}` : ""}{segment.sample ? ` | ${segment.sample}` : ""}</span></article>)}</div> : <p className="emptyState">No multi-site homozygous stretches were detected in the current subset.</p>}</article><article className="miniCard"><h3>Recessive-model candidates</h3>{recessiveShortlist.length ? <div className="resultList">{recessiveShortlist.map(({ item, score, inRoh }: any) => <button type="button" key={`${item.contig}-${item.pos_1based}-${item.rsid}-roh`} className="resultListItem" onClick={() => { const nextIndex = searchedAnnotations.findIndex((candidate: any) => candidate.contig === item.contig && candidate.pos_1based === item.pos_1based && candidate.rsid === item.rsid); setSelectedAnnotationIndex(nextIndex >= 0 ? nextIndex : 0); setActiveStudioView("annotations"); }}><strong>{item.gene || "Unknown"} | {item.contig}:{item.pos_1based}</strong><span>score {score} | genotype {item.genotype}{inRoh ? " | inside ROH" : ""}{" | "}{summarizeLabel(item.consequence, "Unclassified")} | gnomAD {item.gnomad_af || "n/a"}</span></button>)}</div> : <p className="emptyState">No homozygous alternate candidates are present in the current annotated subset.</p>}</article></div></div></section>
      ) : null,
    clinvar: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>ClinVar Review</h2></div><div className="studioCanvasBody"><div className="resultSectionSplit"><article className="miniCard"><h3>Clinical significance mix</h3><DistributionList items={clinvarCounts} emptyLabel="No ClinVar-style labels were found." /></article><article className="miniCard"><h3>Representative records</h3><div className="resultList">{analysis.annotations.slice(0, 8).map((item: any) => <button type="button" key={`${item.contig}-${item.pos_1based}-${item.rsid}-clinvar`} className="resultListItem" onClick={() => { const nextIndex = searchedAnnotations.findIndex((candidate: any) => candidate.contig === item.contig && candidate.pos_1based === item.pos_1based && candidate.rsid === item.rsid); setSelectedAnnotationIndex(nextIndex >= 0 ? nextIndex : 0); setActiveStudioView("annotations"); }}><strong>{item.gene || "Unknown"} | {item.rsid || `${item.contig}:${item.pos_1based}`}</strong><span>{summarizeLabel(item.clinical_significance, "Unreviewed")} | {summarizeLabel(item.clinvar_conditions, "No condition")}</span></button>)}</div></article></div></div></section>
      ) : null,
    vep: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>VEP Consequence</h2></div><div className="studioCanvasBody"><div className="resultSectionSplit"><article className="miniCard"><h3>Top consequences</h3><DistributionList items={consequenceCounts} emptyLabel="No consequence labels were found." /></article><article className="miniCard"><h3>Gene burden</h3><DistributionList items={geneCounts} emptyLabel="No gene burden summary is available." /></article></div></div></section>
      ) : null,
    igv: () =>
      analysis ? (
        <section className="studioCanvasPanel"><IgvBrowser buildGuess={analysis.facts.genome_build_guess ?? null} annotations={searchedAnnotations} selectedIndex={safeSelectedIndex} /></section>
      ) : null,
    annotations: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>Annotations</h2></div><div className="studioCanvasBody"><div className="oeAnnotationControls"><label className="field"><span>Search gene / consequence / ClinVar</span><input value={annotationSearch} onChange={(event) => { setAnnotationSearch(event.target.value); setSelectedAnnotationIndex(0); }} placeholder="e.g. PALMD, missense_variant, benign" /></label><label className="field"><span>Annotation dropdown</span><select value={safeSelectedIndex} onChange={(event) => setSelectedAnnotationIndex(Number(event.target.value))} disabled={!searchedAnnotations.length}>{searchedAnnotations.length ? searchedAnnotations.map((item: any, index: number) => <option key={`${item.contig}-${item.pos_1based}-${item.rsid}-${index}`} value={index}>{item.gene || "Unknown"} | {item.contig}:{item.pos_1based} | {item.rsid || "no-rsID"} | {item.consequence}</option>) : <option value={0}>No annotations matched the search</option>}</select></label></div>{selectedAnnotation ? <AnnotationDetailCard item={selectedAnnotation} /> : <p className="emptyState">No annotation is available for the current selection.</p>}</div></section>
      ) : null,
  };
}
