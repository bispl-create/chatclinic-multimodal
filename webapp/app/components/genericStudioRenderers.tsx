"use client";

import { type StudioRendererBuilderArgs, type StudioRendererRegistry } from "./studioRendererTypes";

export function buildGenericStudioRendererRegistry({
  apiBase,
  analysis,
  rawQcAnalysis,
  summaryStatsAnalysis,
  textAnalysis,
  qqmanResultForStudio,
  samtoolsResultForStudio,
  snpeffResultForStudio,
  liftoverResultForStudio,
  ldblockshowResultForStudio,
  summaryStatsGridRows,
  summaryStatsRowsLoading,
  summaryStatsHasMore,
  summaryStatsGridRef,
  handleSummaryStatsGridScroll,
  loadMoreSummaryStatsRows,
  annotationScope,
  annotationLimit,
  qcMetrics,
  clinicalCoverage,
  components,
  helpers,
}: StudioRendererBuilderArgs): StudioRendererRegistry {
  const {
    StudioMetricGrid,
    StudioPreviewTable,
    WarningListCard,
    ArtifactLinksRow,
    StudioSimpleList,
    DistributionList,
    ReferenceListCard,
  } = components;
  const { formatPercent, formatNumber } = helpers;

  return {
    rawqc: () =>
      rawQcAnalysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader"><h2>FastQC Review</h2></div>
          <div className="studioCanvasBody">
            <StudioMetricGrid
              items={[
                { label: "Total sequences", value: rawQcAnalysis.facts.total_sequences != null ? String(rawQcAnalysis.facts.total_sequences) : "n/a", tone: "good" },
                { label: "Sequence length", value: rawQcAnalysis.facts.sequence_length ?? "n/a", tone: "neutral" },
                { label: "%GC", value: rawQcAnalysis.facts.gc_content != null ? `${rawQcAnalysis.facts.gc_content.toFixed(1)}%` : "n/a", tone: "neutral" },
                { label: "Encoding", value: rawQcAnalysis.facts.encoding ?? "n/a", tone: "neutral" },
              ]}
            />
            <div className="resultList">
              {rawQcAnalysis.modules.map((module: any) => (
                <article key={module.name} className="miniCard">
                  <h3>{module.name}</h3>
                  <p>Status: {module.status}</p>
                  {module.detail ? <p>{module.detail}</p> : null}
                </article>
              ))}
            </div>
            <ArtifactLinksRow
              items={[
                ...(rawQcAnalysis.report_html_path ? [{ label: "Open HTML report", href: `${apiBase.replace(/\/$/, "")}/api/v1/raw-qc/report?path=${encodeURIComponent(rawQcAnalysis.report_html_path)}` }] : []),
                ...(rawQcAnalysis.report_zip_path ? [{ label: "Download ZIP", href: `${apiBase.replace(/\/$/, "")}/api/v1/raw-qc/report?path=${encodeURIComponent(rawQcAnalysis.report_zip_path)}` }] : []),
              ]}
            />
          </div>
        </section>
      ) : null,
    sumstats: () =>
      summaryStatsAnalysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader"><h2>Summary Stats Review</h2></div>
          <div className="studioCanvasBody">
            <StudioMetricGrid
              items={[
                { label: "Rows", value: String(summaryStatsAnalysis.row_count), tone: "good" },
                { label: "Columns", value: String(summaryStatsAnalysis.detected_columns.length) },
                { label: "Build", value: summaryStatsAnalysis.genome_build },
                { label: "Trait", value: summaryStatsAnalysis.trait_type },
                { label: "Delimiter", value: summaryStatsAnalysis.delimiter },
                { label: "Warnings", value: String(summaryStatsAnalysis.warnings.length) },
              ]}
            />
            <div className="resultSectionSplit">
              <article className="miniCard">
                <h3>Detected columns</h3>
                <ul className="hintList">
                  {summaryStatsAnalysis.detected_columns.map((column: string) => <li key={column}>{column}</li>)}
                </ul>
              </article>
              <article className="miniCard">
                <h3>Auto-mapped fields</h3>
                <ul className="hintList">
                  {Object.entries(summaryStatsAnalysis.mapped_fields).map(([field, value]) => (
                    <li key={field}><strong>{field}</strong>: {(value as string) || "not detected"}</li>
                  ))}
                </ul>
              </article>
            </div>
            <article className="miniCard">
              <h3>Preview grid</h3>
              <p className="summaryStatsGridMeta">Showing {summaryStatsGridRows.length} of {summaryStatsAnalysis.row_count} rows</p>
              <div ref={summaryStatsGridRef} onScroll={handleSummaryStatsGridScroll}>
                <StudioPreviewTable
                  columns={summaryStatsAnalysis.detected_columns}
                  rows={summaryStatsGridRows}
                  rowHeaderLabel="#"
                  footer={
                    <>
                      {summaryStatsRowsLoading ? <div className="summaryStatsGridFooter">Loading more rows...</div> : null}
                      {!summaryStatsRowsLoading && summaryStatsHasMore ? (
                        <div className="summaryStatsGridFooter">
                          <button type="button" className="sourceAddButton summaryStatsLoadMoreButton" onClick={() => void loadMoreSummaryStatsRows()}>
                            Load more rows
                          </button>
                        </div>
                      ) : null}
                      {!summaryStatsHasMore && summaryStatsGridRows.length ? <div className="summaryStatsGridFooter">All loaded rows are shown.</div> : null}
                    </>
                  }
                />
              </div>
            </article>
            <WarningListCard warnings={summaryStatsAnalysis.warnings} />
          </div>
        </section>
      ) : null,
    samtools: () =>
      rawQcAnalysis || samtoolsResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader"><h2>Samtools Review</h2></div>
          <div className="studioCanvasBody">
            {samtoolsResultForStudio ? (
              <>
                <StudioMetricGrid
                  items={[
                    { label: "File kind", value: samtoolsResultForStudio.file_kind, tone: "good" },
                    { label: "Total reads", value: samtoolsResultForStudio.total_reads != null ? String(samtoolsResultForStudio.total_reads) : "n/a" },
                    { label: "Mapped", value: samtoolsResultForStudio.mapped_reads != null ? `${samtoolsResultForStudio.mapped_reads}${samtoolsResultForStudio.mapped_rate != null ? ` (${samtoolsResultForStudio.mapped_rate.toFixed(2)}%)` : ""}` : "n/a", tone: "good" },
                    { label: "Properly paired", value: samtoolsResultForStudio.properly_paired_reads != null ? `${samtoolsResultForStudio.properly_paired_reads}${samtoolsResultForStudio.properly_paired_rate != null ? ` (${samtoolsResultForStudio.properly_paired_rate.toFixed(2)}%)` : ""}` : "n/a" },
                    { label: "Quickcheck", value: samtoolsResultForStudio.quickcheck_ok ? "PASS" : "Issue detected", tone: samtoolsResultForStudio.quickcheck_ok ? "good" : "warn" },
                    { label: "Index", value: samtoolsResultForStudio.index_path ? "Created / available" : "n/a" },
                  ]}
                />
                <div className="resultSectionSplit">
                  <article className="miniCard">
                    <h3>samtools stats highlights</h3>
                    <StudioSimpleList items={samtoolsResultForStudio.stats_highlights.map((item: any) => ({ label: item.label, detail: item.value }))} emptyLabel="No samtools stats highlights are available." />
                  </article>
                  <article className="miniCard">
                    <h3>idxstats preview</h3>
                    <StudioSimpleList items={samtoolsResultForStudio.idxstats_rows.map((row: any) => ({ label: row.contig, detail: `mapped ${row.mapped} | unmapped ${row.unmapped} | length ${row.length_bp}` }))} emptyLabel="No idxstats preview rows are available." />
                  </article>
                </div>
                <WarningListCard warnings={samtoolsResultForStudio.warnings} />
              </>
            ) : <p className="emptyState">No samtools result is available for the current raw-QC session.</p>}
          </div>
        </section>
      ) : null,
    qqman: () =>
      qqmanResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>qqman Plots</h2></div><div className="studioCanvasBody"><StudioMetricGrid items={[{ label: "Tool", value: qqmanResultForStudio.tool, tone: "good" }, { label: "Artifacts", value: String(qqmanResultForStudio.artifacts.length) }, { label: "Warnings", value: String(qqmanResultForStudio.warnings.length) }]} /><div className="resultList"><article className="resultListItem resultListStatic"><strong>Command preview</strong><pre className="codeBlock">{qqmanResultForStudio.command_preview}</pre></article></div><div className="resultSectionSplit">{qqmanResultForStudio.artifacts.map((artifact: any) => <article key={artifact.api_path} className="miniCard"><h3>{artifact.title}</h3><img src={`${apiBase.replace(/\/$/, "")}${artifact.api_path}`} alt={artifact.title} className="plotPreviewImage" /><p className="resultNote">{artifact.note}</p><div className="resultActionRow"><a className="sourceAddButton" href={`${apiBase.replace(/\/$/, "")}${artifact.api_path}`} target="_blank" rel="noreferrer">Open image</a></div></article>)}</div><WarningListCard warnings={qqmanResultForStudio.warnings} /></div></section>
      ) : null,
    provenance: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>Analysis Provenance</h2></div><div className="studioCanvasBody"><StudioMetricGrid items={[{ label: "Annotation scope", value: annotationScope }, { label: "Annotation limit", value: annotationScope === "all" ? annotationLimit || "n/a" : "representative" }, { label: "References", value: String(analysis.references.length) }, { label: "Annotations", value: String(analysis.annotations.length) }]} /><div className="resultSectionSplit"><article className="miniCard"><h3>Tool chain</h3><ul className="hintList"><li>`pysam` for VCF parsing, file summary, and QC metrics</li><li>`Ensembl VEP REST` for consequence, transcript, HGVS, and protein fields</li><li>`ClinVar / NCBI refsnp` for clinical significance and condition labels</li><li>`gnomAD` frequency joins for population rarity context</li><li>`OpenAI` models for workflow intake and grounded narrative explanation</li></ul></article><article className="miniCard"><h3>Current run policy</h3><ul className="hintList"><li>Filtering tools such as `bcftools` and `GATK` are available but were not automatically applied in this summary-first run.</li><li>Representative annotation is the default unless the user explicitly requests a wider range.</li><li>Studio cards are derived from the current annotated subset, not from a separate hidden analysis branch.</li></ul></article></div></div></section>
      ) : null,
    qc: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>QC Summary</h2></div><div className="studioCanvasBody"><StudioMetricGrid items={[{ label: "PASS rate", value: formatPercent(qcMetrics?.pass_rate), tone: "good" }, { label: "Ti/Tv", value: formatNumber(qcMetrics?.transition_transversion_ratio) }, { label: "Missing GT", value: formatPercent(qcMetrics?.missing_gt_rate), tone: "warn" }, { label: "Het/HomAlt", value: formatNumber(qcMetrics?.het_hom_alt_ratio) }, { label: "Multi-allelic", value: formatPercent(qcMetrics?.multi_allelic_rate) }, { label: "Symbolic ALT", value: formatPercent(qcMetrics?.symbolic_alt_rate) }, { label: "SNV fraction", value: formatPercent(qcMetrics?.snv_fraction), tone: "good" }, { label: "Indel fraction", value: formatPercent(qcMetrics?.indel_fraction) }]} /><div className="resultSectionSplit"><article className="miniCard"><h3>Genotype composition</h3><DistributionList items={Object.entries(analysis.facts.genotype_counts).map(([label, count]) => ({ label, count: count as number })).sort((left: any, right: any) => right.count - left.count)} emptyLabel="No genotype counts are available." /></article><article className="miniCard"><h3>Variant classes</h3><DistributionList items={Object.entries(analysis.facts.variant_types).map(([label, count]) => ({ label, count: count as number })).sort((left: any, right: any) => right.count - left.count)} emptyLabel="No variant class counts are available." /></article></div></div></section>
      ) : null,
    coverage: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>Clinical Annotation Coverage</h2></div><div className="studioCanvasBody"><StudioSimpleList items={clinicalCoverage.map((item) => ({ label: item.label, detail: item.detail }))} emptyLabel="No clinical coverage summary is available." /></div></section>
      ) : null,
    snpeff: () =>
      analysis || snpeffResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>SnpEff Review</h2></div><div className="studioCanvasBody">{snpeffResultForStudio ? <><StudioMetricGrid items={[{ label: "Genome DB", value: snpeffResultForStudio.genome, tone: "good" }, { label: "Preview rows", value: String(snpeffResultForStudio.parsed_records.length) }, { label: "Tool", value: snpeffResultForStudio.tool }]} /><div className="resultList">{snpeffResultForStudio.parsed_records.map((record: any, index: number) => <article key={`${record.contig}-${record.pos_1based}-${record.alt}-${index}`} className="resultListItem resultListStatic"><strong>{record.contig}:{record.pos_1based} {record.ref}&gt;{record.alt}</strong><span>{record.ann.length ? record.ann.slice(0, 2).map((ann: any) => `${ann.gene_name || "Unknown"} | ${ann.annotation} | ${ann.impact} | ${ann.hgvs_c || "."} | ${ann.hgvs_p || "."}`).join(" || ") : "No parsed ANN entries"}</span></article>)}</div><ArtifactLinksRow items={[{ label: "Open annotated VCF", href: `file://${snpeffResultForStudio.output_path}` }]} /></> : <p className="emptyState">No auxiliary SnpEff result is available for the current analysis.</p>}</div></section>
      ) : null,
    liftover: () =>
      analysis || liftoverResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>LiftOver Review</h2></div><div className="studioCanvasBody">{liftoverResultForStudio ? <><StudioMetricGrid items={[{ label: "Source build", value: liftoverResultForStudio.source_build ?? "unknown", tone: "neutral" }, { label: "Target build", value: liftoverResultForStudio.target_build ?? "unknown", tone: "good" }, { label: "Lifted records", value: String(liftoverResultForStudio.lifted_record_count ?? 0), tone: "good" }, { label: "Rejected records", value: String(liftoverResultForStudio.rejected_record_count ?? 0), tone: "neutral" }, { label: "Warnings", value: String(liftoverResultForStudio.warnings.length), tone: "neutral" }, { label: "Tool", value: liftoverResultForStudio.tool, tone: "neutral" }]} /><div className="resultList">{liftoverResultForStudio.parsed_records.length ? liftoverResultForStudio.parsed_records.map((record: any, index: number) => <article key={`${record.contig}-${record.pos_1based}-${index}`} className="resultListItem resultListStatic"><strong>{record.contig}:{record.pos_1based} {record.ref}&gt;{record.alts.join(",")}</strong><span>Lifted preview record</span></article>) : <p className="emptyState">No lifted preview records are available for this result.</p>}{liftoverResultForStudio.warnings.length ? liftoverResultForStudio.warnings.map((warning: string, index: number) => <article key={`liftover-warning-${index}`} className="resultListItem resultListStatic"><strong>Warning {index + 1}</strong><span>{warning}</span></article>) : null}</div><ArtifactLinksRow items={[{ label: "Open lifted VCF", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(liftoverResultForStudio.output_path)}` }, { label: "Open reject VCF", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(liftoverResultForStudio.reject_path)}` }]} /></> : <p className="emptyState">No liftover result is available for the current analysis.</p>}</div></section>
      ) : null,
    ldblockshow: () =>
      analysis || ldblockshowResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>LD Block Review</h2></div><div className="studioCanvasBody">{ldblockshowResultForStudio ? <><StudioMetricGrid items={[{ label: "Region", value: ldblockshowResultForStudio.region, tone: "good" }, { label: "Tried regions", value: String(ldblockshowResultForStudio.attempted_regions?.length ?? 0), tone: "neutral" }, { label: "Site rows", value: String(ldblockshowResultForStudio.site_row_count ?? 0), tone: "neutral" }, { label: "Triangle pairs", value: String(ldblockshowResultForStudio.triangle_pair_count ?? 0), tone: "neutral" }, { label: "Warnings", value: String(ldblockshowResultForStudio.warnings.length), tone: "neutral" }, { label: "Tool", value: ldblockshowResultForStudio.tool, tone: "neutral" }]} /><div className="resultList">{ldblockshowResultForStudio.attempted_regions?.length ? <article className="resultListItem resultListStatic"><strong>Attempted regions</strong><span>{ldblockshowResultForStudio.attempted_regions.join(" -> ")}</span></article> : null}</div><WarningListCard warnings={ldblockshowResultForStudio.warnings} emptyLabel="No LDBlockShow warnings were reported." emptyAsParagraph /><ArtifactLinksRow items={[...(ldblockshowResultForStudio.svg_path ? [{ label: "Open LD SVG", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(ldblockshowResultForStudio.svg_path)}` }] : []), ...(ldblockshowResultForStudio.block_path ? [{ label: "Open block table", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(ldblockshowResultForStudio.block_path)}` }] : []), ...(ldblockshowResultForStudio.site_path ? [{ label: "Open site table", href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(ldblockshowResultForStudio.site_path)}` }] : [])]} /></> : <p className="emptyState">No LDBlockShow result is available for the current analysis.</p>}</div></section>
      ) : null,
    references: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel"><div className="notebookHeader"><h2>References</h2></div><div className="studioCanvasBody"><ReferenceListCard items={analysis.references} /></div></section>
      ) : null,
    text: () =>
      textAnalysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader"><h2>Text Review</h2></div>
          <div className="studioCanvasBody">
            <StudioMetricGrid
              items={[
                { label: "Media type", value: textAnalysis.media_type, tone: "good" },
                { label: "Characters", value: String(textAnalysis.char_count) },
                { label: "Words", value: String(textAnalysis.word_count) },
                { label: "Lines", value: String(textAnalysis.line_count) },
              ]}
            />
            <article className="miniCard">
              <h3>Preview</h3>
              <StudioSimpleList
                items={(textAnalysis.preview_lines || []).map((line: string, index: number) => ({
                  label: `Line ${index + 1}`,
                  detail: line,
                }))}
                emptyLabel="No preview lines are available for this text source."
              />
            </article>
            <WarningListCard warnings={textAnalysis.warnings} />
          </div>
        </section>
      ) : null,
  };
}
