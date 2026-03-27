"use client";

import { useEffect, useMemo, useRef, useState, type UIEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import IgvBrowser from "./components/IgvBrowser";

type TranscriptAnnotation = {
  transcript_id: string;
  transcript_biotype: string;
  canonical: string;
  exon: string;
  intron: string;
  hgvsc: string;
  hgvsp: string;
  protein_id: string;
  amino_acids: string;
  codons: string;
};

type VariantAnnotation = {
  contig: string;
  pos_1based: number;
  ref: string;
  alts: string[];
  genotype: string;
  gene: string;
  consequence: string;
  rsid: string;
  transcript_id: string;
  transcript_biotype: string;
  canonical: string;
  exon: string;
  intron: string;
  hgvsc: string;
  hgvsp: string;
  protein_id: string;
  amino_acids: string;
  codons: string;
  transcript_options: TranscriptAnnotation[];
  clinical_significance: string;
  clinvar_conditions: string;
  gnomad_af: string;
  source_url: string;
  cadd_raw_score?: number | null;
  cadd_phred?: number | null;
  cadd_lookup_status?: string | null;
  revel_score?: number | null;
  revel_lookup_status?: string | null;
};

type AnalysisResponse = {
  analysis_id: string;
  source_vcf_path?: string | null;
  draft_answer: string;
  facts: {
    file_name: string;
    record_count: number;
    samples: string[];
    genome_build_guess: string | null;
    variant_types: Record<string, number>;
    genotype_counts: Record<string, number>;
    qc: {
      pass_rate: number | null;
      missing_gt_rate: number | null;
      multi_allelic_rate: number | null;
      symbolic_alt_rate: number | null;
      snv_fraction: number | null;
      indel_fraction: number | null;
      transition_transversion_ratio: number | null;
      het_hom_alt_ratio: number | null;
    };
  };
  annotations: VariantAnnotation[];
  snpeff_result?: {
    tool: string;
    genome: string;
    input_path: string;
    output_path: string;
    index_path?: string | null;
    command_preview: string;
    parsed_records: Array<{
      contig: string;
      pos_1based: number;
      ref: string;
      alt: string;
      ann: Array<{
        allele: string;
        annotation: string;
        impact: string;
        gene_name: string;
        gene_id: string;
        feature_type: string;
        feature_id: string;
        transcript_biotype: string;
        rank: string;
        hgvs_c: string;
        hgvs_p: string;
      }>;
    }>;
  } | null;
  plink_result?: {
    tool: string;
    mode?: string;
    input_path: string;
    command_preview: string;
    output_prefix: string;
    log_path?: string | null;
    freq_path?: string | null;
    missing_path?: string | null;
    hardy_path?: string | null;
    score_file_path?: string | null;
    score_output_path?: string | null;
    variant_count?: number | null;
    sample_count?: number | null;
    freq_rows: Array<{
      chrom: string;
      variant_id: string;
      ref_allele: string;
      alt_allele: string;
      alt_freq?: number | null;
      observation_count?: number | null;
    }>;
    missing_rows: Array<{
      sample_id: string;
      missing_genotype_count: number;
      observation_count: number;
      missing_rate: number;
    }>;
    hardy_rows: Array<{
      chrom: string;
      variant_id: string;
      observed_hets?: number | null;
      expected_hets?: number | null;
      p_value?: number | null;
    }>;
    score_rows: Array<{
      sample_id: string;
      allele_ct?: number | null;
      named_allele_dosage_sum?: number | null;
      score_sum?: number | null;
    }>;
    score_mean?: number | null;
    score_min?: number | null;
    score_max?: number | null;
    warnings: string[];
  } | null;
  liftover_result?: {
    tool: string;
    input_path: string;
    source_build?: string | null;
    target_build?: string | null;
    target_reference_fasta: string;
    chain_file: string;
    output_path: string;
    output_index_path?: string | null;
    reject_path: string;
    reject_index_path?: string | null;
    command_preview: string;
    lifted_record_count?: number | null;
    rejected_record_count?: number | null;
    parsed_records: Array<{
      contig: string;
      pos_1based: number;
      ref: string;
      alts: string[];
    }>;
    warnings: string[];
  } | null;
  ldblockshow_result?: {
    tool: string;
    input_path: string;
    region: string;
    output_prefix: string;
    command_preview: string;
    svg_path?: string | null;
    png_path?: string | null;
    pdf_path?: string | null;
    block_path?: string | null;
    site_path?: string | null;
    triangle_path?: string | null;
    attempted_regions?: string[];
    site_row_count?: number;
    block_row_count?: number;
    triangle_pair_count?: number;
    warnings: string[];
  } | null;
  candidate_variants?: Array<{
    item: VariantAnnotation;
    score: number;
    in_roh: boolean;
  }>;
  clinvar_summary?: Array<{
    label: string;
    count: number;
  }>;
  consequence_summary?: Array<{
    label: string;
    count: number;
  }>;
  clinical_coverage_summary?: Array<{
    label: string;
    count: number;
    detail: string;
  }>;
  filtering_summary?: Array<{
    label: string;
    count: number;
    detail: string;
  }>;
  symbolic_alt_summary?: {
    count: number;
    examples: Array<{
      locus: string;
      gene: string;
      alts: string[];
      consequence: string;
      genotype: string;
    }>;
  };
  roh_segments?: Array<{
    sample: string;
    contig: string;
    start_1based: number;
    end_1based: number;
    length_bp: number;
    marker_count: number;
    quality: number;
  }>;
  references: Array<{
    id: string;
    title: string;
    source: string;
    url: string;
    note: string;
  }>;
  used_tools?: string[];
  tool_registry?: Array<{
    name: string;
    description: string;
    task: string;
    modality: string;
    approval_required: boolean;
    source: string;
  }>;
};

type RawQcResponse = {
  analysis_id: string;
  source_raw_path?: string | null;
  samtools_result?: {
    tool: string;
    input_path: string;
    display_name: string;
    file_kind: string;
    command_preview: string;
    quickcheck_ok?: boolean | null;
    total_reads?: number | null;
    mapped_reads?: number | null;
    mapped_rate?: number | null;
    paired_reads?: number | null;
    properly_paired_reads?: number | null;
    properly_paired_rate?: number | null;
    singleton_reads?: number | null;
    index_path?: string | null;
    stats_highlights: Array<{ label: string; value: string }>;
    idxstats_rows: Array<{ contig: string; length_bp: number; mapped: number; unmapped: number }>;
    warnings: string[];
  } | null;
  draft_answer: string;
  facts: {
    file_name: string;
    file_kind: string;
    total_sequences: number | null;
    filtered_sequences: number | null;
    poor_quality_sequences: number | null;
    sequence_length: string | null;
    gc_content: number | null;
    encoding: string | null;
  };
  modules: Array<{
    name: string;
    status: string;
    detail: string;
  }>;
  report_html_path?: string | null;
  report_zip_path?: string | null;
  used_tools?: string[];
  tool_registry?: Array<{
    name: string;
    description: string;
    task: string;
    modality: string;
    approval_required: boolean;
    source: string;
  }>;
};

type SummaryStatsResponse = {
  analysis_id: string;
  source_stats_path?: string | null;
  file_name: string;
  genome_build: string;
  trait_type: string;
  delimiter: string;
  detected_columns: string[];
  mapped_fields: {
    chrom?: string | null;
    pos?: string | null;
    rsid?: string | null;
    effect_allele?: string | null;
    other_allele?: string | null;
    beta_or?: string | null;
    standard_error?: string | null;
    p_value?: string | null;
    n?: string | null;
    eaf?: string | null;
  };
  row_count: number;
  preview_rows: Array<Record<string, string>>;
  warnings: string[];
  qqman_result?: RPlotResponse | null;
  prs_prep_result?: PrsPrepResponse | null;
  draft_answer: string;
  used_tools?: string[];
  tool_registry?: Array<{
    name: string;
    description: string;
    task: string;
    modality: string;
    approval_required: boolean;
    source: string;
  }>;
};

type SummaryStatsRowsResponse = {
  rows: Array<Record<string, string>>;
  offset: number;
  limit: number;
  returned: number;
  has_more: boolean;
};

type SummaryStatsChatResponse = {
  answer: string;
  citations: string[];
  used_fallback: boolean;
  requested_view?: StudioView | null;
  analysis?: SummaryStatsResponse | null;
  qqman_result?: RPlotResponse | null;
  prs_prep_result?: PrsPrepResponse | null;
};

type RPlotResponse = {
  tool: string;
  input_path: string;
  output_dir: string;
  command_preview: string;
  artifacts: Array<{
    plot_type: string;
    title: string;
    image_path: string;
    api_path: string;
    note: string;
  }>;
  warnings: string[];
};

type AnalysisChatResponse = {
  answer: string;
  citations: string[];
  used_fallback: boolean;
  used_tools?: string[];
  requested_view?: StudioView | null;
  analysis?: AnalysisResponse | null;
  plink_result?: AnalysisResponse["plink_result"];
  liftover_result?: AnalysisResponse["liftover_result"];
  ldblockshow_result?: AnalysisResponse["ldblockshow_result"];
};

type RawQcChatResponse = {
  answer: string;
  citations: string[];
  used_fallback: boolean;
  requested_view?: StudioView | null;
  analysis?: RawQcResponse | null;
  samtools_result?: RawQcResponse["samtools_result"];
};

type WorkflowStep =
  | string
  | {
      tool?: string;
      bind?: string;
      needs?: string[];
      on_fail?: string;
    };

type WorkflowManifest = {
  name: string;
  description: string;
  source_type: "vcf" | "raw_qc" | "summary_stats" | string;
  steps?: WorkflowStep[];
  default_view?: string | null;
};

type SourceReadyResponse = {
  source_type: "vcf" | "raw_qc" | "summary_stats";
  file_name: string;
  source_path: string;
  file_kind?: string | null;
};

type SessionMode = "prs" | "vcf_analysis" | "raw_sequence";

type PrsPrepResponse = {
  analysis_id: string;
  source_stats_path: string;
  file_name: string;
  build_check: {
    inferred_build: string;
    build_confidence: string;
    source_build: string;
    target_build: string;
    build_match?: boolean | null;
    warnings: string[];
  };
  harmonization: {
    required_fields_present: boolean;
    effect_size_kind: string;
    ambiguous_snp_count: number;
    harmonizable_preview_rows: number;
    missing_fields: string[];
    warnings: string[];
  };
  score_file_path?: string | null;
  score_file_columns: string[];
  score_file_preview_rows: Array<Record<string, string>>;
  kept_rows: number;
  dropped_rows: number;
  score_file_ready: boolean;
  draft_answer: string;
};

const DEFAULT_WORKFLOW_REGISTRY: WorkflowManifest[] = [
  {
    name: "representative_vcf_review",
    description: "Run the default representative VCF interpretation workflow on the active VCF source.",
    source_type: "vcf",
    default_view: "candidates",
  },
  {
    name: "raw_qc_review",
    description: "Run the default raw sequencing QC workflow on the active FASTQ/BAM/SAM source.",
    source_type: "raw_qc",
    default_view: "rawqc",
  },
  {
    name: "summary_stats_review",
    description: "Run the default summary statistics intake and review workflow on the active summary-stats source.",
    source_type: "summary_stats",
    default_view: "sumstats",
  },
  {
    name: "prs_prep",
    description: "Run build check, harmonization prep, and score-file preparation on the active summary-statistics source.",
    source_type: "summary_stats",
    default_view: "prs_prep",
  },
];

const DEFAULT_LIFTOVER_CHAIN =
  "/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/references/liftover/chains/hg19ToHg38.over.chain.gz";
const DEFAULT_LIFTOVER_TARGET_FASTA =
  "/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/references/liftover/GRCh38/hg38.fa";

type ChatMessage = {
  role: "assistant" | "user";
  content: string;
  kind?: "status" | "summary";
};

type AnalysisQuestionTurn = {
  role: "user" | "assistant";
  content: string;
};

type StudioView =
  | "candidates"
  | "acmg"
  | "provenance"
  | "coverage"
  | "rawqc"
  | "sumstats"
  | "prs_prep"
  | "qqman"
  | "samtools"
  | "snpeff"
  | "plink"
  | "liftover"
  | "ldblockshow"
  | "symbolic"
  | "roh"
  | "qc"
  | "table"
  | "clinvar"
  | "vep"
  | "references"
  | "igv"
  | "annotations";

type RohStudioSegment = {
  label: string;
  count: number;
  spanMb: string;
  quality?: number;
  sample?: string;
};

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function describeWorkflowStep(step: WorkflowStep, availableTools: Record<string, string>) {
  if (typeof step === "string") {
    return `- \`${step}\`${availableTools[step] ? `: ${availableTools[step]}` : ""}`;
  }

  const toolName = String(step?.tool ?? "").trim();
  const bindName = String(step?.bind ?? "").trim();
  const needs = Array.isArray(step?.needs)
    ? step.needs.map((item) => String(item).trim()).filter(Boolean)
    : [];
  const onFail = String(step?.on_fail ?? "").trim().toLowerCase();
  const detailParts: string[] = [];
  if (bindName) {
    detailParts.push(`binds \`${bindName}\``);
  }
  if (needs.length) {
    detailParts.push(`needs \`${needs.join(", ")}\``);
  }
  if (onFail === "continue") {
    detailParts.push("continues on failure");
  }
  const details = detailParts.length ? ` (${detailParts.join("; ")})` : "";
  return `- \`${toolName}\`${availableTools[toolName] ? `: ${availableTools[toolName]}` : ""}${details}`;
}

function isRawQcFileName(fileName: string) {
  const lowered = fileName.toLowerCase();
  return (
    lowered.endsWith(".fastq") ||
    lowered.endsWith(".fastq.gz") ||
    lowered.endsWith(".fq") ||
    lowered.endsWith(".fq.gz") ||
    lowered.endsWith(".bam") ||
    lowered.endsWith(".sam")
  );
}

function isVcfFileName(fileName: string) {
  const lowered = fileName.toLowerCase();
  return lowered.endsWith(".vcf") || lowered.endsWith(".vcf.gz");
}

function isSummaryStatsFileName(fileName: string) {
  const lowered = fileName.toLowerCase();
  return (
    lowered.endsWith(".tsv") ||
    lowered.endsWith(".tsv.gz") ||
    lowered.endsWith(".txt") ||
    lowered.endsWith(".txt.gz") ||
    lowered.endsWith(".csv") ||
    lowered.endsWith(".csv.gz") ||
    lowered.endsWith(".sumstats") ||
    lowered.endsWith(".sumstats.gz")
  );
}

function formatPercent(value: number | null | undefined) {
  return value == null ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | null | undefined, digits = 2) {
  return value == null ? "n/a" : value.toFixed(digits);
}

function AnnotationDetailCard({ item }: { item: VariantAnnotation }) {
  const [selectedTranscriptIndex, setSelectedTranscriptIndex] = useState(0);
  const transcript = item.transcript_options[selectedTranscriptIndex] ?? {
    transcript_id: item.transcript_id,
    transcript_biotype: item.transcript_biotype,
    canonical: item.canonical,
    exon: item.exon,
    intron: item.intron,
    hgvsc: item.hgvsc,
    hgvsp: item.hgvsp,
    protein_id: item.protein_id,
    amino_acids: item.amino_acids,
    codons: item.codons,
  };

  return (
    <article className="miniCard annotationDetailCard">
      <h3>
        {item.gene || "Unknown gene"} | {item.contig}:{item.pos_1based}
      </h3>
      <p>
        {item.ref}&gt;{item.alts.join(",")} | {item.consequence} | {item.rsid || "no-rsID"}
      </p>
      <p>
        Genotype: {item.genotype} | ClinVar: {item.clinical_significance} | gnomAD AF: {item.gnomad_af}
      </p>
      {item.cadd_phred != null || item.cadd_raw_score != null ? (
        <p>
          CADD: PHRED {item.cadd_phred != null ? item.cadd_phred.toFixed(1) : "n/a"} | Raw{" "}
          {item.cadd_raw_score != null ? item.cadd_raw_score.toFixed(3) : "n/a"}
        </p>
      ) : null}
      {item.revel_score != null ? <p>REVEL: {item.revel_score.toFixed(3)}</p> : null}
      <p>Condition: {item.clinvar_conditions}</p>
      {item.transcript_options.length ? (
        <label className="field compactField">
          <span>Transcript</span>
          <select
            value={selectedTranscriptIndex}
            onChange={(event) => setSelectedTranscriptIndex(Number(event.target.value))}
          >
            {item.transcript_options.map((option, index) => (
              <option key={`${item.contig}-${item.pos_1based}-${option.transcript_id}-${index}`} value={index}>
                {option.transcript_id} | {option.transcript_biotype} | canonical {option.canonical}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <div className="annotationMetaGrid">
        <div className="factBox">
          <span>Transcript</span>
          <strong>{transcript.transcript_id}</strong>
        </div>
        <div className="factBox">
          <span>Biotype</span>
          <strong>{transcript.transcript_biotype}</strong>
        </div>
        <div className="factBox">
          <span>Exon / Intron</span>
          <strong>
            {transcript.exon} / {transcript.intron}
          </strong>
        </div>
        <div className="factBox">
          <span>Canonical</span>
          <strong>{transcript.canonical}</strong>
        </div>
      </div>
      <div className="annotationTextStack">
        <p>HGVSc: {transcript.hgvsc}</p>
        <p>HGVSp: {transcript.hgvsp}</p>
        <p>
          Protein: {transcript.protein_id} | AA: {transcript.amino_acids} | Codons: {transcript.codons}
        </p>
        <p>
          Source:{" "}
          <a href={item.source_url} target="_blank" rel="noreferrer">
            Open annotation reference
          </a>
        </p>
      </div>
    </article>
  );
}

function MarkdownAnswer({ content }: { content: string }) {
  return (
    <div className="markdownAnswer">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

function summarizeLabel(value: string, fallback: string) {
  const trimmed = value.trim();
  if (!trimmed || trimmed === "." || trimmed === "not available") {
    return fallback;
  }
  return trimmed
    .split(/[;,]/)[0]
    .trim()
    .replace(/_/g, " ");
}

function MetricTile({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "warn";
}) {
  return (
    <article className={`resultMetricTile resultMetricTile-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

type MetricItem = {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "warn";
};

function StudioMetricGrid({ items }: { items: MetricItem[] }) {
  return (
    <div className="resultMetricGrid">
      {items.map((item) => (
        <MetricTile
          key={`${item.label}-${item.value}`}
          label={item.label}
          value={item.value}
          tone={item.tone ?? "neutral"}
        />
      ))}
    </div>
  );
}

function DistributionList({
  items,
  emptyLabel,
}: {
  items: Array<{ label: string; count: number }>;
  emptyLabel: string;
}) {
  const maxValue = items[0]?.count ?? 0;
  if (!items.length) {
    return <p className="emptyState">{emptyLabel}</p>;
  }

  return (
    <div className="distributionList">
      {items.map((item) => {
        const width = maxValue > 0 ? Math.max((item.count / maxValue) * 100, 6) : 0;
        return (
          <div key={item.label} className="distributionRow">
            <div className="distributionMeta">
              <span>{item.label}</span>
              <strong>{item.count}</strong>
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

type WarningListCardProps = {
  warnings: string[];
  emptyLabel?: string;
  emptyAsParagraph?: boolean;
  titleBuilder?: (index: number) => string;
};

function WarningListCard({
  warnings,
  emptyLabel,
  emptyAsParagraph = false,
  titleBuilder = (index) => `Warning ${index + 1}`,
}: WarningListCardProps) {
  if (!warnings.length) {
    if (!emptyLabel) {
      return null;
    }
    return emptyAsParagraph ? <p className="emptyState">{emptyLabel}</p> : <div className="resultList"><article className="resultListItem resultListStatic"><strong>Status</strong><span>{emptyLabel}</span></article></div>;
  }

  return (
    <div className="resultList">
      {warnings.map((warning, index) => (
        <article key={`${warning}-${index}`} className="resultListItem resultListStatic">
          <strong>{titleBuilder(index)}</strong>
          <span>{warning}</span>
        </article>
      ))}
    </div>
  );
}

type ArtifactLinkItem = {
  label: string;
  href: string;
};

function ArtifactLinksRow({ items }: { items: ArtifactLinkItem[] }) {
  if (!items.length) {
    return null;
  }
  return (
    <div className="resultActionRow">
      {items.map((item) => (
        <a
          key={`${item.label}-${item.href}`}
          className="sourceAddButton"
          href={item.href}
          target="_blank"
          rel="noreferrer"
        >
          {item.label}
        </a>
      ))}
    </div>
  );
}

type SimpleListItem = {
  label: string;
  detail: string;
};

function StudioSimpleList({
  items,
  emptyLabel,
}: {
  items: SimpleListItem[];
  emptyLabel?: string;
}) {
  if (!items.length) {
    return emptyLabel ? <p className="emptyState">{emptyLabel}</p> : null;
  }

  return (
    <div className="resultList">
      {items.map((item) => (
        <article key={`${item.label}-${item.detail}`} className="resultListItem resultListStatic">
          <strong>{item.label}</strong>
          <span>{item.detail}</span>
        </article>
      ))}
    </div>
  );
}

function StudioPreviewTable({
  columns,
  rows,
  rowHeaderLabel,
  footer,
}: {
  columns: string[];
  rows: Array<Record<string, string>>;
  rowHeaderLabel?: string;
  footer?: React.ReactNode;
}) {
  return (
    <div className="variantTableWrap summaryStatsTableWrap">
      <table className="variantTable summaryStatsTable">
        <thead>
          <tr>
            {rowHeaderLabel ? <th className="summaryStatsRowHeader">{rowHeaderLabel}</th> : null}
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`preview-row-${index}`}>
              {rowHeaderLabel ? <td className="summaryStatsRowHeader">{index + 1}</td> : null}
              {columns.map((column) => (
                <td key={`${index}-${column}`}>{row[column] || ""}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {footer}
    </div>
  );
}

function ReferenceListCard({
  items,
}: {
  items: Array<{ id: string; title: string; url: string }>;
}) {
  if (!items.length) {
    return <p className="emptyState">No references are available for the current analysis.</p>;
  }

  return (
    <ol className="referenceList">
      {items.map((item, index) => (
        <li key={item.id}>
          <span className="referenceIndex">[{index + 1}]</span>{" "}
          <a href={item.url} target="_blank" rel="noreferrer">
            {item.title}
          </a>
        </li>
      ))}
    </ol>
  );
}

function VariantTable({
  items,
  onSelect,
}: {
  items: VariantAnnotation[];
  onSelect: (item: VariantAnnotation) => void;
}) {
  if (!items.length) {
    return <p className="emptyState">No annotation is available for the current selection.</p>;
  }

  return (
    <div className="variantTableWrap">
      <table className="variantTable">
        <thead>
          <tr>
            <th>Locus</th>
            <th>Gene</th>
            <th>Consequence</th>
            <th>ClinVar</th>
            <th>gnomAD</th>
            <th>HGVS</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr
              key={`${item.contig}-${item.pos_1based}-${item.rsid}-${item.hgvsc}`}
              onClick={() => onSelect(item)}
              className="variantTableRow"
            >
              <td>
                {item.contig}:{item.pos_1based}
              </td>
              <td>{item.gene || "Unknown"}</td>
              <td>{summarizeLabel(item.consequence, "Unclassified")}</td>
              <td>{summarizeLabel(item.clinical_significance, "Unreviewed")}</td>
              <td>{item.gnomad_af || "n/a"}</td>
              <td>{item.hgvsp !== "." ? item.hgvsp : item.hgvsc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function rankCandidateScore(item: VariantAnnotation) {
  let score = 0;
  const significance = item.clinical_significance.toLowerCase();
  const consequence = item.consequence.toLowerCase();
  const afText = (item.gnomad_af || "").trim();
  const af = Number(afText);

  if (significance.includes("pathogenic")) {
    score += 5;
  } else if (significance.includes("vus")) {
    score += 2;
  } else if (significance.includes("benign")) {
    score -= 2;
  }

  if (consequence.includes("splice")) {
    score += 4;
  } else if (consequence.includes("missense")) {
    score += 3;
  } else if (consequence.includes("stop") || consequence.includes("frameshift")) {
    score += 5;
  } else if (consequence.includes("synonymous")) {
    score -= 1;
  }

  if (!Number.isNaN(af)) {
    if (af < 0.001) {
      score += 3;
    } else if (af < 0.01) {
      score += 2;
    } else if (af > 0.05) {
      score -= 2;
    }
  }

  if (item.genotype === "1/1") {
    score += 1;
  }

  return score;
}

function isVariantInRoh(
  item: VariantAnnotation,
  rohSegments:
    | Array<{
        sample: string;
        contig: string;
        start_1based: number;
        end_1based: number;
      }>
    | undefined,
) {
  if (!rohSegments?.length) {
    return false;
  }
  return rohSegments.some(
    (segment) =>
      segment.contig === item.contig &&
      item.pos_1based >= segment.start_1based &&
      item.pos_1based <= segment.end_1based,
  );
}

function rankRecessiveScore(
  item: VariantAnnotation,
  rohSegments:
    | Array<{
        sample: string;
        contig: string;
        start_1based: number;
        end_1based: number;
      }>
    | undefined,
) {
  let score = 0;
  const consequence = item.consequence.toLowerCase();
  const significance = item.clinical_significance.toLowerCase();
  const af = Number((item.gnomad_af || "").trim());

  if (item.genotype === "1/1") {
    score += 4;
  }
  if (isVariantInRoh(item, rohSegments)) {
    score += 5;
  }
  if (consequence.includes("splice")) {
    score += 4;
  } else if (consequence.includes("missense")) {
    score += 3;
  } else if (consequence.includes("stop") || consequence.includes("frameshift")) {
    score += 5;
  } else if (consequence.includes("synonymous")) {
    score -= 2;
  }
  if (!Number.isNaN(af)) {
    if (af < 0.001) {
      score += 4;
    } else if (af < 0.01) {
      score += 2;
    } else if (af > 0.05) {
      score -= 3;
    }
  }
  if (significance.includes("pathogenic")) {
    score += 3;
  } else if (significance.includes("benign")) {
    score -= 3;
  }
  return score;
}

function buildAcmgHints(item: VariantAnnotation) {
  const hints: string[] = [];
  const significance = item.clinical_significance.toLowerCase();
  const consequence = item.consequence.toLowerCase();
  const af = Number((item.gnomad_af || "").trim());

  if (consequence.includes("splice")) {
    hints.push("PVS1-supporting candidate: splice consequence is present and may affect transcript processing.");
  }
  if (consequence.includes("missense")) {
    hints.push("PP3-style review candidate: missense consequence may warrant in-silico evidence review.");
  }
  if (!Number.isNaN(af) && af < 0.001) {
    hints.push("PM2-style review candidate: allele frequency appears very low in gnomAD.");
  }
  if (significance.includes("pathogenic")) {
    hints.push("ClinVar support: existing pathogenic-style assertion is present and should be reviewed for evidence level.");
  }
  if (significance.includes("benign")) {
    hints.push("Benign evidence note: ClinVar currently trends benign, so pathogenic interpretation is less likely.");
  }
  if (!hints.length) {
    hints.push("No strong ACMG-style hint is available from the current fields alone. Additional transcript, phenotype, and segregation review is needed.");
  }
  return hints;
}

function hasMeaningfulText(value: string) {
  const trimmed = value.trim();
  return Boolean(trimmed && trimmed !== "." && trimmed.toLowerCase() !== "not available");
}

const groundingTokens = ["$studio", "$current analysis", "$current card", "$grounded"];

function detectToolTriggers(text: string): string[] {
  return Array.from(new Set(Array.from(text.matchAll(/(^|\s)(@[A-Za-z0-9_-]+)/g)).map((match) => match[2])));
}

function renderUserPromptInline(content: string) {
  const tokenPattern = /(\$studio|\$current analysis|\$current card|\$grounded|@[A-Za-z0-9_-]+)/gi;
  const parts = content.split(tokenPattern);
  return parts.map((part, index) => {
    const lowered = part.toLowerCase();
    if (groundingTokens.includes(lowered)) {
      return (
        <span key={`user-token-${index}`} className="inlineTriggerChip">
          {part}
        </span>
      );
    }
    if (/^@[A-Za-z0-9_-]+$/.test(part)) {
      return (
        <span key={`user-tool-${index}`} className="inlineToolChip">
          {part}
        </span>
      );
    }
    return <span key={`user-text-${index}`}>{part}</span>;
  });
}

export default function Page() {
  const [apiBase, setApiBase] = useState("http://127.0.0.1:8001");
  const [toolRegistry, setToolRegistry] = useState<AnalysisResponse["tool_registry"]>([]);
  const [toolRegistryLoading, setToolRegistryLoading] = useState(true);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Select a session mode first. `@mode prs` starts the PRS workflow and expects two inputs: summary statistics and a target genotype. `@mode vcf_analysis` starts a single-input VCF variant interpretation session. `@mode raw_sequence` starts a FASTQ/BAM/SAM raw sequencing QC session. To see the available modes again, enter `@mode help`.",
    },
  ]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [rawQcAnalysis, setRawQcAnalysis] = useState<RawQcResponse | null>(null);
  const [summaryStatsAnalysis, setSummaryStatsAnalysis] = useState<SummaryStatsResponse | null>(null);
  const [summaryStatsGridRows, setSummaryStatsGridRows] = useState<Array<Record<string, string>>>([]);
  const [summaryStatsHasMore, setSummaryStatsHasMore] = useState(false);
  const [summaryStatsRowsLoading, setSummaryStatsRowsLoading] = useState(false);
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [attachedSourceType, setAttachedSourceType] = useState<"vcf" | "raw_qc" | "summary_stats" | null>(null);
  const [activeSource, setActiveSource] = useState<SourceReadyResponse | null>(null);
  const [sessionMode, setSessionMode] = useState<SessionMode | null>(null);
  const [pendingUploadRole, setPendingUploadRole] = useState<"default" | "prs_summary" | "prs_target">("default");
  const [prsSummaryFile, setPrsSummaryFile] = useState<File | null>(null);
  const [prsTargetFile, setPrsTargetFile] = useState<File | null>(null);
  const [prsSummarySource, setPrsSummarySource] = useState<SourceReadyResponse | null>(null);
  const [prsTargetSource, setPrsTargetSource] = useState<SourceReadyResponse | null>(null);
  const [directLiftoverResult, setDirectLiftoverResult] = useState<AnalysisResponse["liftover_result"] | null>(null);
  const [directSamtoolsResult, setDirectSamtoolsResult] = useState<RawQcResponse["samtools_result"] | null>(null);
  const [directPlinkResult, setDirectPlinkResult] = useState<AnalysisResponse["plink_result"] | null>(null);
  const [directSnpeffResult, setDirectSnpeffResult] = useState<AnalysisResponse["snpeff_result"] | null>(null);
  const [directLdblockshowResult, setDirectLdblockshowResult] = useState<AnalysisResponse["ldblockshow_result"] | null>(null);
  const [directQqmanResult, setDirectQqmanResult] = useState<RPlotResponse | null>(null);
  const [directPrsPrepResult, setDirectPrsPrepResult] = useState<PrsPrepResponse | null>(null);
  const [latestPrsPrepResult, setLatestPrsPrepResult] = useState<PrsPrepResponse | null>(null);
  const [annotationScope, setAnnotationScope] = useState<"representative" | "all">("representative");
  const [annotationLimit, setAnnotationLimit] = useState("200");
  const [status, setStatus] = useState("Waiting for a session mode");
  const [error, setError] = useState<string | null>(null);
  const [selectedAnnotationIndex, setSelectedAnnotationIndex] = useState(0);
  const [annotationSearch, setAnnotationSearch] = useState("");
  const [composerText, setComposerText] = useState("");
  const [isComposing, setIsComposing] = useState(false);
  const [analysisQa, setAnalysisQa] = useState<AnalysisQuestionTurn[]>([]);
  const [followUpAnswer, setFollowUpAnswer] = useState<string | null>(null);
  const [activeStudioView, setActiveStudioView] = useState<StudioView | null>(null);
  const [plinkRunning, setPlinkRunning] = useState(false);
  const [plinkConfig, setPlinkConfig] = useState({
    mode: "qc",
    runFreq: true,
    runMissing: true,
    runHardy: true,
    allowExtraChr: true,
    outputPrefix: "",
  });
  const [toolRegistryOpen, setToolRegistryOpen] = useState(false);
  const [skillRegistryOpen, setSkillRegistryOpen] = useState(false);
  const [workflowRegistry, setWorkflowRegistry] = useState<WorkflowManifest[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const studioCanvasRef = useRef<HTMLElement | null>(null);
  const chatStreamRef = useRef<HTMLDivElement | null>(null);
  const summaryStatsGridRef = useRef<HTMLDivElement | null>(null);
  const activeToolRegistry =
    analysis?.tool_registry?.length
      ? analysis.tool_registry
      : rawQcAnalysis?.tool_registry?.length
        ? rawQcAnalysis.tool_registry
        : summaryStatsAnalysis?.tool_registry?.length
          ? summaryStatsAnalysis.tool_registry
        : toolRegistry;
  const availableWorkflows = useMemo(() => {
    const registry = workflowRegistry.length ? workflowRegistry : DEFAULT_WORKFLOW_REGISTRY;
    const sourceType =
      sessionMode === "prs"
        ? "summary_stats"
        : analysis
          ? "vcf"
          : rawQcAnalysis
            ? "raw_qc"
            : summaryStatsAnalysis
              ? "summary_stats"
              : attachedSourceType;
    return sourceType ? registry.filter((item) => item.source_type === sourceType) : registry;
  }, [workflowRegistry, analysis, rawQcAnalysis, summaryStatsAnalysis, attachedSourceType, sessionMode]);

  const hasAttachedSource = Boolean(attachedFile || prsSummaryFile || prsTargetFile);

  function displayToolAlias(toolName: string) {
    const normalized = toolName.toLowerCase();
    if (normalized.includes("liftover")) {
      return "@liftover";
    }
    if (normalized.includes("samtools")) {
      return "@samtools";
    }
    if (normalized.includes("plink")) {
      return "@plink";
    }
    if (normalized.includes("snpeff")) {
      return "@snpeff";
    }
    if (normalized.includes("ldblockshow")) {
      return "@ldblockshow";
    }
    if (normalized.includes("qqman")) {
      return "@qqman";
    }
    return `@${normalized.replace(/_execution_tool$|_tool$|_vcf_tool$/g, "").replace(/^gatk_/, "").replace(/_/g, "")}`;
  }
  const plinkCommandPreview = useMemo(() => {
    const inputPath =
      analysis?.source_vcf_path ||
      (activeSource?.source_type === "vcf" ? activeSource.source_path : null) ||
      "<target-genotype.vcf.gz>";
    const outputPrefix = (plinkConfig.outputPrefix || `${analysis?.analysis_id ?? "analysis"}-plink`).trim();
    if (plinkConfig.mode === "score") {
      const scoreFilePath =
        latestPrsPrepResult?.score_file_path ||
        summaryStatsAnalysis?.prs_prep_result?.score_file_path ||
        directPrsPrepResult?.score_file_path ||
        "<prs_weights.tsv>";
      const flags = [plinkConfig.allowExtraChr ? "--allow-extra-chr" : ""].filter(Boolean);
      return `plink2 --vcf ${inputPath} dosage=DS ${flags.join(" ")} --score ${scoreFilePath} 1 2 3 header-read cols=scoresums --out ${outputPrefix}`.replace(
        /\s+/g,
        " ",
      );
    }
    const flags: string[] = [];
    if (plinkConfig.runFreq) {
      flags.push("--freq");
    }
    if (plinkConfig.runMissing) {
      flags.push("--missing");
    }
    if (plinkConfig.runHardy) {
      flags.push("--hardy");
    }
    if (plinkConfig.allowExtraChr) {
      flags.push("--allow-extra-chr");
    }
    return `plink2 --vcf ${inputPath} dosage=DS ${flags.join(" ")} --out ${outputPrefix}`.trim();
  }, [analysis, activeSource, plinkConfig, latestPrsPrepResult, summaryStatsAnalysis, directPrsPrepResult]);
  const normalizedComposerText = typeof composerText === "string" ? composerText.toLowerCase() : "";
  const detectedGroundingTriggers = groundingTokens.filter((token) => normalizedComposerText.includes(token));
  const detectedToolTriggers = detectToolTriggers(composerText);
  const composerInputClass = [
    "composerInput",
    detectedGroundingTriggers.length ? "composerInputTriggered" : "",
    detectedToolTriggers.length ? "composerInputToolTriggered" : "",
  ]
    .filter(Boolean)
    .join(" ");

  useEffect(() => {
    if (!analysis) {
      return;
    }
    setPlinkConfig((current) => ({ ...current, outputPrefix: `${analysis.analysis_id}-plink` }));
  }, [analysis]);

  useEffect(() => {
    if (!summaryStatsAnalysis) {
      setSummaryStatsGridRows([]);
      setSummaryStatsHasMore(false);
      return;
    }
    setSummaryStatsGridRows(summaryStatsAnalysis.preview_rows);
    setSummaryStatsHasMore(summaryStatsAnalysis.row_count > summaryStatsAnalysis.preview_rows.length);
  }, [summaryStatsAnalysis]);

  useEffect(() => {
    if (summaryStatsAnalysis?.prs_prep_result) {
      setDirectPrsPrepResult(summaryStatsAnalysis.prs_prep_result);
      setLatestPrsPrepResult(summaryStatsAnalysis.prs_prep_result);
    }
  }, [summaryStatsAnalysis?.prs_prep_result]);

  useEffect(() => {
    if (activeStudioView !== "sumstats" || !summaryStatsHasMore || summaryStatsRowsLoading) {
      return;
    }
    if (summaryStatsGridRows.length < 120) {
      void loadMoreSummaryStatsRows();
    }
  }, [activeStudioView, summaryStatsGridRows.length, summaryStatsHasMore, summaryStatsRowsLoading]);

  useEffect(() => {
    let cancelled = false;

    async function loadToolRegistry() {
      if (!cancelled) {
        setToolRegistryLoading(true);
      }
      try {
        const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/tools`);
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as AnalysisResponse["tool_registry"];
        if (!cancelled) {
          setToolRegistry(payload);
        }
      } catch {
        // Keep the shell usable even if the local backend is temporarily unavailable.
      } finally {
        if (!cancelled) {
          setToolRegistryLoading(false);
        }
      }
    }

    async function loadWorkflowRegistry() {
      try {
        const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/workflows`);
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as WorkflowManifest[];
        if (!cancelled) {
          setWorkflowRegistry(payload);
        }
      } catch {
        // best-effort refresh only
      }
    }

    void loadToolRegistry();
    void loadWorkflowRegistry();
    const retryTimer = window.setInterval(() => {
      if (!cancelled && (toolRegistry?.length ?? 0) === 0) {
        void loadToolRegistry();
      }
    }, 4000);

    return () => {
      cancelled = true;
      window.clearInterval(retryTimer);
    };
  }, [apiBase, toolRegistry?.length]);

  function addMessage(message: ChatMessage) {
    setMessages((current) => [...current, message]);
  }

  function buildCitationMap(references: AnalysisResponse["references"]) {
    return new Map(references.map((item, index) => [item.id, index + 1]));
  }

  function formatSummaryWithCitations(text: string, references: AnalysisResponse["references"]) {
    const referenceMap = new Map(
      references.map((item, index) => [
        item.id,
        {
          index: index + 1,
          url: item.url,
          title: item.title,
        },
      ]),
    );

    return text.replace(/\[?(REF\d+)\]?/g, (match, refId: string) => {
      const reference = referenceMap.get(refId);
      if (!reference) {
        return match;
      }
      return `[${reference.index}](${reference.url} "${reference.title}")`;
    });
  }

  function handleAttachClick(role: "default" | "prs_summary" | "prs_target" = "default") {
    setPendingUploadRole(role);
    fileInputRef.current?.click();
  }

  function workflowHelpText(sourceType: "vcf" | "raw_qc" | "summary_stats" | null) {
    const registry = workflowRegistry.length ? workflowRegistry : DEFAULT_WORKFLOW_REGISTRY;
    const filtered = registry.filter((item) => !sourceType || item.source_type === sourceType);
    if (filtered.length === 0) {
      return "No workflow registry entries are available for the current source.";
    }
    return [
      "**Workflow registry**",
      "",
      ...filtered.map((item) => `- \`@skill ${item.name}\`: ${item.description}`),
    ].join("\n");
  }

  function workflowDetailHelpText(workflowName: string) {
    const registry = workflowRegistry.length ? workflowRegistry : DEFAULT_WORKFLOW_REGISTRY;
    const workflow = registry.find((item) => item.name === workflowName);
    if (!workflow) {
      return `\`@skill ${workflowName}\` is not a registered workflow.`;
    }
    const availableTools = (toolRegistry ?? []).reduce<Record<string, string>>((acc, item) => {
      acc[item.name] = item.description;
      return acc;
    }, {});
    const lines = [
      `**${workflow.name}**`,
      "",
      workflow.description,
    ];
    if (workflow.steps?.length) {
      lines.push("", "Steps");
      workflow.steps.forEach((step) => {
        lines.push(describeWorkflowStep(step, availableTools));
      });
    }
    lines.push("", "Examples", `- \`@skill ${workflow.name}\``, `- \`@skill ${workflow.name} help\``);
    return lines.join("\n");
  }

  function modeHelpText() {
    return [
      "**Available modes**",
      "",
      "- `@mode prs`: polygenic risk score workflow with two inputs (`summary statistics` and `target genotype VCF`).",
      "- `@mode vcf_analysis`: variant interpretation workflow with one VCF source.",
      "- `@mode raw_sequence`: raw sequencing QC workflow with one FASTQ/BAM/SAM source.",
    ].join("\n");
  }

  function modeSelectionText(mode: SessionMode) {
    if (mode === "prs") {
      return [
        "**PRS mode**",
        "",
        "This session expects two sources:",
        "- `Summary statistics` for PRS preparation",
        "- `Target genotype VCF` for PLINK scoring",
        "",
        "Next steps:",
        "- Upload the summary-statistics file into the `Summary stats` slot",
        "- Upload the target genotype VCF into the `Target genotype` slot",
        "- Then run `@skill prs_prep` and `@plink score`",
      ].join("\n");
    }
    if (mode === "vcf_analysis") {
      return [
        "**VCF analysis mode**",
        "",
        "This session expects one VCF source for variant interpretation.",
        "",
        "Next steps:",
        "- Upload one VCF file",
        "- Run `@skill representative_vcf_review`",
      ].join("\n");
    }
    return [
      "**Raw sequencing mode**",
      "",
      "This session expects one FASTQ/BAM/SAM source for raw sequencing QC.",
      "",
      "Next steps:",
      "- Upload one raw sequencing file",
      "- Run `@skill raw_qc_review`",
    ].join("\n");
  }

  async function uploadActiveSource(file: File): Promise<SourceReadyResponse> {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/source/upload`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return (await response.json()) as SourceReadyResponse;
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (!file) {
      return;
    }
    const hadPreparedPrsScoreFile = Boolean(latestPrsPrepResult?.score_file_ready);
    const slotRole =
      sessionMode === "prs"
        ? pendingUploadRole !== "default"
          ? pendingUploadRole
          : isSummaryStatsFileName(file.name) && !isVcfFileName(file.name)
            ? "prs_summary"
            : "prs_target"
        : "default";
    const guessedSourceType =
      isRawQcFileName(file.name)
        ? "raw_qc"
        : isSummaryStatsFileName(file.name) && !isVcfFileName(file.name)
          ? "summary_stats"
          : "vcf";
    setAttachedFile(file);
    setAttachedSourceType(guessedSourceType);
    setActiveSource(null);
    if (sessionMode === "prs" && slotRole === "prs_summary") {
      setSummaryStatsAnalysis(null);
      setDirectQqmanResult(null);
      setDirectPrsPrepResult(null);
      setLatestPrsPrepResult(null);
    } else if (sessionMode === "prs" && slotRole === "prs_target") {
      setAnalysis(null);
      setDirectLiftoverResult(null);
      setDirectPlinkResult(null);
      setDirectSnpeffResult(null);
      setDirectLdblockshowResult(null);
    } else if (guessedSourceType === "vcf") {
      setAnalysis(null);
      setDirectLiftoverResult(null);
      setDirectPlinkResult(null);
      setDirectSnpeffResult(null);
      setDirectLdblockshowResult(null);
    } else if (guessedSourceType === "raw_qc") {
      setRawQcAnalysis(null);
      setDirectSamtoolsResult(null);
    } else if (guessedSourceType === "summary_stats") {
      setSummaryStatsAnalysis(null);
      setDirectQqmanResult(null);
      setDirectPrsPrepResult(null);
      setLatestPrsPrepResult(null);
    }
    setFollowUpAnswer(null);
    setAnalysisQa([]);
    setActiveStudioView(null);
    setSelectedAnnotationIndex(0);
    setAnnotationSearch("");
    setError(null);
    try {
      const source = await uploadActiveSource(file);
      setActiveSource(source);
      if (sessionMode === "prs") {
        if (slotRole === "prs_summary") {
          setPrsSummaryFile(file);
          setPrsSummarySource(source);
        } else {
          setPrsTargetFile(file);
          setPrsTargetSource(source);
        }
      } else {
        setPrsSummaryFile(null);
        setPrsSummarySource(null);
        setPrsTargetFile(null);
        setPrsTargetSource(null);
      }
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Source upload failed");
      addMessage({
        role: "assistant",
        content: `소스 업로드 중 오류가 발생했습니다: ${message}`,
      });
      event.target.value = "";
      setPendingUploadRole("default");
      return;
    }
    if (sessionMode === "prs") {
      setStatus(slotRole === "prs_summary" ? "PRS summary-statistics source ready" : "PRS target genotype source ready");
      addMessage({
        role: "assistant",
        content:
          slotRole === "prs_summary"
            ? `Summary statistics source \`${file.name}\` is loaded into the PRS session. Upload a target genotype VCF into the second slot, then run \`@skill prs_prep\`.`
            : hadPreparedPrsScoreFile
              ? `Target genotype VCF source \`${file.name}\` is loaded into the PRS session. You can run \`@plink score\` now.`
              : `Target genotype VCF source \`${file.name}\` is loaded into the PRS session. Prepare the summary-statistics source with \`@skill prs_prep\`, then run \`@plink score\`.`,
      });
      event.target.value = "";
      setPendingUploadRole("default");
      return;
    }
    if (guessedSourceType === "raw_qc") {
      setStatus("Raw sequencing source ready");
      addMessage({
        role: "assistant",
        content: `Raw sequencing source \`${file.name}\` is loaded. Run \`@skill raw_qc_review\` to start the default review workflow, or \`@skill help\` to see available workflows.`,
      });
    } else if (guessedSourceType === "summary_stats") {
      setStatus("Summary statistics source ready");
      addMessage({
        role: "assistant",
        content: `Summary statistics source \`${file.name}\` is loaded. Run \`@skill summary_stats_review\` to start the default review workflow, or \`@skill help\` to see available workflows.`,
      });
    } else {
      setStatus("VCF source ready");
      addMessage({
        role: "assistant",
        content: hadPreparedPrsScoreFile
          ? `VCF source \`${file.name}\` is loaded as a target genotype source. You can run \`@plink score\` now, or use \`@skill representative_vcf_review\` for the default review workflow.`
          : `VCF source \`${file.name}\` is loaded. Run \`@skill representative_vcf_review\` to start the default review workflow, or \`@skill help\` to see available workflows.`,
      });
    }
    event.target.value = "";
    setPendingUploadRole("default");
  }

  function parseInlineOptions(text: string) {
    const options: Record<string, string> = {};
    for (const token of text.split(/\s+/).filter(Boolean)) {
      const [key, ...rest] = token.split("=");
      if (!key || rest.length === 0) {
        continue;
      }
      options[key.trim().toLowerCase()] = rest.join("=").trim();
    }
    return options;
  }

  function toolRunningStatus(alias: string, remainder: string) {
    const normalized = alias.trim().toLowerCase();
    const wantsPlinkScore =
      normalized === "plink" &&
      (remainder.trim().toLowerCase() === "score" ||
        parseInlineOptions(remainder).mode?.toLowerCase() === "score");
    if (normalized === "liftover") {
      return "Running Liftover...";
    }
    if (normalized === "qqman") {
      return "Running qqman...";
    }
    if (normalized === "samtools") {
      return "Running samtools...";
    }
    if (normalized === "snpeff") {
      return "Running SnpEff...";
    }
    if (normalized === "ldblockshow") {
      return "Running LDBlockShow...";
    }
    if (normalized === "plink") {
      return wantsPlinkScore ? "Running PLINK score..." : "Running PLINK...";
    }
    return "Running tool...";
  }

  function toolReadyStatus(alias: string, remainder: string) {
    const normalized = alias.trim().toLowerCase();
    const wantsPlinkScore =
      normalized === "plink" &&
      (remainder.trim().toLowerCase() === "score" ||
        parseInlineOptions(remainder).mode?.toLowerCase() === "score");
    if (normalized === "liftover") {
      return "Liftover ready";
    }
    if (normalized === "qqman") {
      return "qqman ready";
    }
    if (normalized === "samtools") {
      return "samtools ready";
    }
    if (normalized === "snpeff") {
      return "SnpEff ready";
    }
    if (normalized === "ldblockshow") {
      return "LDBlockShow ready";
    }
    if (normalized === "plink") {
      return wantsPlinkScore ? "PLINK score ready" : "PLINK ready";
    }
    return "Tool ready";
  }

  function toolFailedStatus(alias: string, remainder: string) {
    const normalized = alias.trim().toLowerCase();
    const wantsPlinkScore =
      normalized === "plink" &&
      (remainder.trim().toLowerCase() === "score" ||
        parseInlineOptions(remainder).mode?.toLowerCase() === "score");
    if (normalized === "liftover") {
      return "Liftover failed";
    }
    if (normalized === "qqman") {
      return "qqman failed";
    }
    if (normalized === "samtools") {
      return "samtools failed";
    }
    if (normalized === "snpeff") {
      return "SnpEff failed";
    }
    if (normalized === "ldblockshow") {
      return "LDBlockShow failed";
    }
    if (normalized === "plink") {
      return wantsPlinkScore ? "PLINK score failed" : "PLINK failed";
    }
    return "Tool failed";
  }

  async function runPreAnalysisTool(alias: string, remainder: string) {
    const preAnalysisSource =
      sessionMode === "prs"
        ? alias === "plink" && (remainder.trim().toLowerCase() === "score" || parseInlineOptions(remainder).mode?.toLowerCase() === "score")
          ? prsTargetSource
          : alias === "qqman" || alias === "plink"
            ? prsSummarySource ?? activeSource
            : prsTargetSource ?? activeSource
        : activeSource;
    if (!preAnalysisSource) {
      addMessage({
        role: "assistant",
        content: "먼저 active source가 준비되어야 합니다. 파일을 다시 업로드해 주세요.",
      });
      return;
    }
    const options = parseInlineOptions(remainder);
    setStatus(toolRunningStatus(alias, remainder));

    if (alias === "liftover" && preAnalysisSource.source_type === "vcf") {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/liftover/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vcf_path: preAnalysisSource.source_path,
          chain_file: DEFAULT_LIFTOVER_CHAIN,
          target_reference_fasta: DEFAULT_LIFTOVER_TARGET_FASTA,
          target_build: options.target || "hg38",
          source_build: options.source_build || undefined,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as AnalysisResponse["liftover_result"];
      setDirectLiftoverResult(payload ?? null);
      setActiveStudioView("liftover");
      setStatus(toolReadyStatus(alias, remainder));
      addMessage({
        role: "assistant",
        content:
          `GATK LiftoverVcf was run for the current VCF.\n\n` +
          `- Source build: ${payload?.source_build || "unknown"}\n` +
          `- Target build: ${payload?.target_build || "unknown"}\n` +
          `- Lifted records: ${payload?.lifted_record_count ?? "unknown"}\n` +
          `- Rejected records: ${payload?.rejected_record_count ?? "unknown"}`,
      });
      return;
    }

    if (alias === "samtools" && preAnalysisSource.source_type === "raw_qc") {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/samtools/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          raw_path: preAnalysisSource.source_path,
          original_name: preAnalysisSource.file_name,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as RawQcResponse["samtools_result"];
      setDirectSamtoolsResult(payload ?? null);
      setActiveStudioView("samtools");
      setStatus(toolReadyStatus(alias, remainder));
      addMessage({
        role: "assistant",
        content:
          `samtools reviewed the active source \`${preAnalysisSource.file_name}\`.\n\n` +
          `- Quickcheck: ${payload?.quickcheck_ok ? "PASS" : "issue detected"}\n` +
          `- Total reads: ${payload?.total_reads ?? "unknown"}\n` +
          `- Mapped reads: ${payload?.mapped_reads ?? "unknown"}${
            payload?.mapped_rate != null ? ` (${payload.mapped_rate.toFixed(2)}%)` : ""
          }`,
      });
      return;
    }

    const wantsPlinkScore =
      alias === "plink" &&
      (remainder.trim().toLowerCase() === "score" ||
        options.mode?.toLowerCase() === "score");

    if (alias === "plink" && preAnalysisSource.source_type === "summary_stats" && wantsPlinkScore) {
      addMessage({
        role: "assistant",
        content:
          "PLINK score needs a target genotype source in addition to the prepared summary-statistics weights.\n\n" +
          "- Keep the current PRS prep result.\n" +
          "- Upload a target genotype VCF.\n" +
          "- Then run `@plink score` again.",
      });
      return;
    }

    if (alias === "plink" && preAnalysisSource.source_type === "vcf") {
      if (wantsPlinkScore && !latestPrsPrepResult?.score_file_ready) {
        addMessage({
          role: "assistant",
          content:
            "PLINK score needs a prepared score file.\n\n" +
            "- Upload a summary-statistics source.\n" +
            "- Run `@skill prs_prep`.\n" +
            "- Then upload the target genotype VCF and run `@plink score` again.",
        });
        return;
      }
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/plink/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vcf_path: preAnalysisSource.source_path,
          mode: wantsPlinkScore ? "score" : "qc",
          score_file_path: wantsPlinkScore ? latestPrsPrepResult?.score_file_path : undefined,
          output_prefix: `${preAnalysisSource.file_name}-plink`,
          allow_extra_chr: true,
          freq_limit: wantsPlinkScore ? 0 : 12,
          missing_limit: wantsPlinkScore ? 0 : 12,
          hardy_limit: wantsPlinkScore ? 0 : 12,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as AnalysisResponse["plink_result"];
      setDirectPlinkResult(payload ?? null);
      setActiveStudioView("plink");
      setStatus(toolReadyStatus(alias, remainder));
      addMessage({
        role: "assistant",
        content:
          wantsPlinkScore
            ? `PLINK score was run for the current target genotype VCF.\n\n` +
              `- Samples scored: ${payload?.score_rows?.length ?? "unknown"}\n` +
              `- Mean score: ${payload?.score_mean != null ? payload.score_mean.toFixed(4) : "n/a"}\n` +
              `- Output: ${payload?.score_output_path || "n/a"}`
            : `PLINK was run for the current VCF.\n\n` +
              `- Variants: ${payload?.variant_count ?? "unknown"}\n` +
              `- Samples: ${payload?.sample_count ?? "unknown"}\n` +
              `- Outputs: afreq ${payload?.freq_rows?.length ?? 0}, missing ${payload?.missing_rows?.length ?? 0}, hardy ${payload?.hardy_rows?.length ?? 0}`,
      });
      return;
    }

    if (alias === "snpeff" && preAnalysisSource.source_type === "vcf") {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/snpeff/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vcf_path: preAnalysisSource.source_path,
          genome: options.genome || "GRCh37.75",
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      setDirectSnpeffResult(payload ?? null);
      setActiveStudioView("snpeff");
      setStatus(toolReadyStatus(alias, remainder));
      addMessage({
        role: "assistant",
        content:
          `SnpEff was run for the current VCF.\n\n` +
          `- Genome: ${payload?.genome || "unknown"}\n` +
          `- Parsed preview records: ${payload?.parsed_records?.length ?? 0}\n` +
          `- Output: ${payload?.output_path || "n/a"}`,
      });
      return;
    }

    if (alias === "ldblockshow" && preAnalysisSource.source_type === "vcf") {
      if (!options.region) {
        addMessage({
          role: "assistant",
          content: "LDBlockShow requires a region. Try `@ldblockshow help` or `@ldblockshow region=chr1:1000-5000`.",
        });
        return;
      }
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/ldblockshow/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vcf_path: preAnalysisSource.source_path,
          region: options.region,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      setDirectLdblockshowResult(payload ?? null);
      setActiveStudioView("ldblockshow");
      setStatus(toolReadyStatus(alias, remainder));
      addMessage({
        role: "assistant",
        content:
          `LDBlockShow was run for the current VCF.\n\n` +
          `- Region: ${payload?.region || options.region}\n` +
          `- PNG: ${payload?.png_path || "n/a"}\n` +
          `- SVG: ${payload?.svg_path || "n/a"}`,
      });
      return;
    }

    if (alias === "qqman" && preAnalysisSource.source_type === "summary_stats") {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/qqman/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          association_path: preAnalysisSource.source_path,
          output_prefix: options.output_prefix || `${preAnalysisSource.file_name}-qqman`,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as RPlotResponse;
      setDirectQqmanResult(payload);
      setActiveStudioView("qqman");
      setStatus(toolReadyStatus(alias, remainder));
      addMessage({
        role: "assistant",
        content:
          `qqman plots were generated for the current summary-statistics source.\n\n` +
          `- Output directory: ${payload.output_dir}\n` +
          `- Plot artifacts: ${payload.artifacts.length}\n` +
          `- Warnings: ${payload.warnings.length}`,
      });
      return;
    }

    addMessage({
      role: "assistant",
      content:
        `\`@${alias}\` is not compatible with the current active source.` +
        (preAnalysisSource ? ` Current source type: \`${preAnalysisSource.source_type}\`.` : ""),
    });
  }

  async function handleStartRawQc(file: File, options?: { silent?: boolean }): Promise<RawQcResponse | null> {
    const silent = options?.silent ?? false;
    setError(null);
    if (!silent) {
      addMessage({
        role: "assistant",
        content: "FastQC로 raw sequencing QC를 실행하고 있습니다.",
        kind: "status",
      });
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/raw-qc/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload: RawQcResponse = await response.json();
      setRawQcAnalysis(payload);
      setAnalysis(null);
      setActiveStudioView("rawqc");
      setStatus("Raw QC ready");
      setComposerText("");
      return payload;
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Raw QC failed");
      addMessage({
        role: "assistant",
        content: `FastQC 실행 중 오류가 발생했습니다: ${message}`,
      });
      return null;
    }
  }

  async function handleStartSummaryStats(
    file: File,
    options?: { silent?: boolean },
  ): Promise<SummaryStatsResponse | null> {
    const silent = options?.silent ?? false;
    setError(null);
    if (!silent) {
      addMessage({
        role: "assistant",
        content: "Summary statistics 파일을 읽고 컬럼과 기본 QC를 확인하고 있습니다.",
        kind: "status",
      });
    }

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("genome_build", "unknown");
      formData.append("trait_type", "unknown");

      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/summary-stats/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload: SummaryStatsResponse = await response.json();
      setSummaryStatsAnalysis(payload);
      setAnalysis(null);
      setRawQcAnalysis(null);
      setActiveStudioView("sumstats");
      setStatus("Summary stats ready");
      setComposerText("");
      return payload;
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Summary stats failed");
      addMessage({
        role: "assistant",
        content: `Summary statistics intake 중 오류가 발생했습니다: ${message}`,
      });
      return null;
    }
  }

  async function handleStartPrsPrepFromSource(sourceOverride?: SourceReadyResponse | null): Promise<PrsPrepResponse | null> {
    const source = sourceOverride ?? prsSummarySource ?? activeSource;
    if (!source || source.source_type !== "summary_stats") {
      setError("PRS prep requires an active summary-statistics source.");
      return null;
    }

    setError(null);
    setStatus("Running PRS prep...");

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/prs-prep/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_stats_path: source.source_path,
          file_name: source.file_name,
          genome_build: summaryStatsAnalysis?.genome_build ?? "unknown",
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload: PrsPrepResponse = await response.json();
      setDirectPrsPrepResult(payload);
      setLatestPrsPrepResult(payload);
      setActiveStudioView("prs_prep");
      setStatus("PRS prep ready");
      setComposerText("");
      addMessage({
        role: "assistant",
        content:
          `PRS prep was run for \`${payload.file_name}\`.\n\n` +
          `- Build check: ${payload.build_check.inferred_build} (${payload.build_check.build_confidence})\n` +
          `- Score-file rows kept: ${payload.kept_rows}\n` +
          `- Score-file rows dropped: ${payload.dropped_rows}\n` +
          `- Score file ready: ${payload.score_file_ready ? "yes" : "no"}\n\n` +
          "Open the PRS Prep Review card in Studio to inspect harmonization warnings and the PLINK score-file preview."
      });
      return payload;
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("PRS prep failed");
      addMessage({
        role: "assistant",
        content: `PRS prep 중 오류가 발생했습니다: ${message}`,
      });
      return null;
    }
  }

  async function handleStartAnalysis(
    parsedScope?: "representative" | "all",
    parsedLimit?: string,
    fileOverride?: File | null,
    options?: { silent?: boolean },
  ): Promise<AnalysisResponse | null> {
    const inputFile = fileOverride ?? attachedFile;
    const silent = options?.silent ?? false;
    if (!inputFile) {
      setError("먼저 + 버튼으로 VCF 파일을 첨부해 주세요.");
      return null;
    }

    const effectiveScope = parsedScope ?? annotationScope;
    const effectiveLimit = parsedLimit ?? annotationLimit;
    setAnnotationScope(effectiveScope);
    setAnnotationLimit(effectiveLimit);
    setError(null);
    setStatus("Analyzing");
    if ((toolRegistry?.length ?? 0) === 0) {
      void (async () => {
        try {
          const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/tools`);
          if (!response.ok) {
            return;
          }
          const payload = (await response.json()) as AnalysisResponse["tool_registry"];
          setToolRegistry(payload);
        } catch {
          // best-effort refresh only
        }
      })();
    }
    if (!silent) {
      addMessage({
        role: "assistant",
        content: "pysam으로 VCF header, sample, record count, 기본 QC를 읽고 있습니다.",
        kind: "status",
      });
      await sleep(350);
      addMessage({
        role: "assistant",
        content:
          "현재 run은 요약과 주석 중심이라 bcftools/GATK hard filtering는 적용하지 않고, 원본 VCF를 그대로 해석합니다. 필터 조건이 필요해지면 그 단계에서 bcftools filter 또는 GATK VariantFiltration을 호출하겠습니다.",
        kind: "status",
      });
      await sleep(350);
      addMessage({
        role: "assistant",
        content:
          "변이 주석은 Ensembl VEP REST로 consequence, transcript, HGVS, protein 정보를 붙이고, ClinVar/refsnp와 gnomAD를 기준으로 clinical significance와 allele frequency를 확인하고 있습니다.",
        kind: "status",
      });
    }

    try {
      const formData = new FormData();
      formData.append("file", inputFile);
      formData.append("annotation_scope", effectiveScope);
      if (effectiveScope === "all" && effectiveLimit.trim()) {
        formData.append("annotation_limit", effectiveLimit);
      }

      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/analysis/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload: AnalysisResponse = await response.json();
      setAnalysis(payload);
      setFollowUpAnswer(null);
      setAnalysisQa([]);
      setActiveStudioView(null);
      setSelectedAnnotationIndex(0);
      setComposerText("");
      setStatus("Analysis ready");
      if (!silent && payload.draft_answer?.trim()) {
        addMessage({
          role: "assistant",
          content: formatSummaryWithCitations(payload.draft_answer, payload.references),
        });
      }
      return payload;
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Analysis failed");
      addMessage({
        role: "assistant",
        content: `분석 중 오류가 발생했습니다: ${message}`,
      });
      return null;
    }
  }

  async function fetchToolHelpText(alias: string) {
    const response = await fetch(
      `${apiBase.replace(/\/$/, "")}/api/v1/tools/help?alias=${encodeURIComponent(alias)}`,
    );
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = (await response.json()) as { help?: string };
    return payload.help ?? `\`@${alias}\` help is not available right now.`;
  }

  async function handleComposerSubmit() {
    const text = composerText.trim();
    if (!text) {
      return;
    }

    const modeMatch = text.match(/^@mode(?:\s+(.*))?$/i);
    if (modeMatch) {
      addMessage({ role: "user", content: text });
      setComposerText("");
      const remainder = (modeMatch[1] ?? "").trim().toLowerCase();
      if (!remainder || remainder === "help") {
        addMessage({ role: "assistant", content: modeHelpText() });
        return;
      }
      if (remainder === "prs") {
        setSessionMode("prs");
        setStatus("PRS mode selected");
        addMessage({ role: "assistant", content: modeSelectionText("prs") });
        return;
      }
      if (remainder === "vcf_analysis") {
        setSessionMode("vcf_analysis");
        setStatus("VCF analysis mode selected");
        addMessage({ role: "assistant", content: modeSelectionText("vcf_analysis") });
        return;
      }
      if (remainder === "raw_sequence") {
        setSessionMode("raw_sequence");
        setStatus("Raw sequencing mode selected");
        addMessage({ role: "assistant", content: modeSelectionText("raw_sequence") });
        return;
      }
      addMessage({ role: "assistant", content: `\`@mode ${remainder}\` is not available. Use \`@mode help\`.` });
      return;
    }

    if (!hasAttachedSource) {
      addMessage({ role: "user", content: text });
      addMessage({
        role: "assistant",
        content: sessionMode ? "먼저 현재 mode에 필요한 source 파일을 업로드해 주세요." : "먼저 `@mode prs`, `@mode vcf_analysis`, 또는 `@mode raw_sequence`를 선택한 뒤 source 파일을 업로드해 주세요.",
      });
      setComposerText("");
      return;
    }

    if (!analysis && !rawQcAnalysis && !summaryStatsAnalysis) {
      const skillMatch = text.match(/^@skill(?:\s+(.*))?$/i);
      if (skillMatch) {
        addMessage({ role: "user", content: text });
        setComposerText("");
        const remainder = (skillMatch[1] ?? "").trim();
        if (!remainder || /^help$/i.test(remainder)) {
          addMessage({
            role: "assistant",
            content: workflowHelpText(attachedSourceType),
          });
          return;
        }

        const workflowName = remainder.split(/\s+/)[0];
        const selectedVcfFile = sessionMode === "vcf_analysis" ? attachedFile : attachedFile;
        const selectedRawQcFile = attachedFile;
        const selectedSummaryStatsFile = sessionMode === "prs" ? prsSummaryFile : attachedFile;
        if (/\shelp$/i.test(remainder)) {
          addMessage({
            role: "assistant",
            content: workflowDetailHelpText(workflowName),
          });
          return;
        }
        if (workflowName === "representative_vcf_review" && (sessionMode === "vcf_analysis" || attachedSourceType === "vcf")) {
          if (!selectedVcfFile) {
            addMessage({ role: "assistant", content: "Upload a VCF source first." });
            return;
          }
          await handleStartAnalysis("representative", annotationLimit, selectedVcfFile);
          return;
        }
        if (workflowName === "raw_qc_review" && (sessionMode === "raw_sequence" || attachedSourceType === "raw_qc")) {
          if (!selectedRawQcFile) {
            addMessage({ role: "assistant", content: "Upload a raw sequencing source first." });
            return;
          }
          await handleStartRawQc(selectedRawQcFile);
          return;
        }
        if (workflowName === "summary_stats_review" && (sessionMode === null ? attachedSourceType === "summary_stats" : sessionMode === "prs")) {
          if (!selectedSummaryStatsFile) {
            addMessage({ role: "assistant", content: "Upload a summary-statistics source first." });
            return;
          }
          await handleStartSummaryStats(selectedSummaryStatsFile);
          return;
        }
        if (workflowName === "prs_prep" && (sessionMode === "prs" || attachedSourceType === "summary_stats")) {
          await handleStartPrsPrepFromSource(prsSummarySource ?? activeSource);
          return;
        }

        addMessage({
          role: "assistant",
          content:
            `\`@skill ${workflowName}\` is not compatible with the current active source.` +
            (attachedSourceType ? ` Current source type: \`${attachedSourceType}\`.` : ""),
        });
        return;
      }

      const atToolMatch = text.match(/^@([A-Za-z0-9_-]+)(?:\s+(.*))?$/);
      if (atToolMatch && atToolMatch[1].toLowerCase() !== "skill") {
        const alias = atToolMatch[1].trim();
        const remainder = (atToolMatch[2] ?? "").trim();
        const isHelp = /^(help|--help|-h)(\s+.*)?$/i.test(remainder);
        setComposerText("");

        if (isHelp) {
          addMessage({ role: "user", content: text });
          try {
            const helpText = await fetchToolHelpText(alias);
            addMessage({
              role: "assistant",
              content: helpText,
            });
          } catch (caught) {
            const message = caught instanceof Error ? caught.message : String(caught);
            addMessage({
              role: "assistant",
              content: `\`@${alias} help\` 조회 중 오류가 발생했습니다: ${message}`,
            });
          }
          return;
        }

        addMessage({ role: "user", content: text });
        try {
          await runPreAnalysisTool(alias.toLowerCase(), remainder);
        } catch (caught) {
          const message = caught instanceof Error ? caught.message : String(caught);
          setStatus(toolFailedStatus(alias.toLowerCase(), remainder));
          addMessage({
            role: "assistant",
            content: `\`@${alias}\` 실행 중 오류가 발생했습니다: ${message}`,
          });
        }
        return;
      }

      addMessage({ role: "user", content: text });
      setComposerText("");
      addMessage({
        role: "assistant",
        content:
          sessionMode === "prs"
            ? "PRS mode is active. Upload the summary-statistics and target-genotype sources as needed, then use `@skill prs_prep` and `@plink score`."
            : "A source is loaded, but no workflow is running yet. Use `@skill help` to see available workflows, then run one such as `@skill representative_vcf_review`.",
      });
      return;
    }

    if (analysis) {
      setComposerText("");
      await handleAskAnalysisQuestion(text);
      return;
    }

    if (rawQcAnalysis) {
      setComposerText("");
      await handleAskRawQcQuestion(text);
      return;
    }

    if (summaryStatsAnalysis) {
      setComposerText("");
      await handleAskSummaryStatsQuestion(text);
      return;
    }

    addMessage({ role: "user", content: text });
    addMessage({
      role: "assistant",
      content: "먼저 mode를 정하고 필요한 source를 업로드해 주세요.",
    });
    setComposerText("");
  }

  async function handleAskAnalysisQuestion(questionText?: string, analysisOverride?: AnalysisResponse | null) {
    const text = questionText?.trim() ?? "";
    const activeAnalysis = analysisOverride ?? analysis;
    if (!text || !activeAnalysis) {
      return;
    }

    setStatus("Generating answer...");
    setAnalysisQa((current) => [...current, { role: "user", content: text }]);

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          analysis: activeAnalysis,
          history: analysisQa.map((turn) => ({ role: turn.role, content: turn.content })),
          studio_context: studioContext,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload: AnalysisChatResponse = await response.json();
      if (payload.analysis) {
        setAnalysis(payload.analysis);
      }
      setAnalysisQa((current) => [...current, { role: "assistant", content: payload.answer }]);
      if (payload.requested_view === "plink") {
        setActiveStudioView("plink");
      } else if (payload.requested_view) {
        setActiveStudioView(payload.requested_view);
      }
      if (payload.plink_result) {
        setAnalysis((current) =>
          current
            ? {
                ...current,
                plink_result: payload.plink_result,
                used_tools: payload.used_tools ?? current.used_tools,
              }
            : current,
        );
        setActiveStudioView("plink");
      }
      if (payload.liftover_result) {
        setAnalysis((current) =>
          current
            ? {
                ...current,
                liftover_result: payload.liftover_result,
                used_tools: payload.used_tools ?? ["gatk_liftover_vcf_tool"],
              }
            : current,
        );
        setActiveStudioView("liftover");
      }
      if (payload.ldblockshow_result) {
        setAnalysis((current) =>
          current
            ? {
                ...current,
                ldblockshow_result: payload.ldblockshow_result,
                used_tools: payload.used_tools ?? ["ldblockshow_execution_tool"],
              }
            : current,
        );
        setActiveStudioView("ldblockshow");
      }
      setFollowUpAnswer(payload.answer);
      setStatus("Answer ready");
    } catch (caught) {
      const msg = caught instanceof Error ? caught.message : String(caught);
      setAnalysisQa((current) => [
        ...current,
        { role: "assistant", content: `설명 요청 중 오류가 발생했습니다: ${msg}` },
      ]);
      setStatus("Answer failed");
    }
  }

  async function handleRunPlink() {
    const sourceVcfPath =
      analysis?.source_vcf_path ??
      (sessionMode === "prs"
        ? prsTargetSource?.source_path ?? null
        : activeSource?.source_type === "vcf"
          ? activeSource.source_path
          : null);
    if (!sourceVcfPath) {
      setError("현재 분석에는 실행 가능한 source VCF path가 없습니다.");
      return;
    }

    const scoreMode = plinkConfig.mode === "score";
    setPlinkRunning(true);
    setStatus("Running PLINK...");
    setError(null);
    try {
      if (scoreMode && !latestPrsPrepResult?.score_file_ready) {
        throw new Error("Run @skill prs_prep first and provide a prepared score file before PLINK score mode.");
      }
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/plink/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vcf_path: sourceVcfPath,
          mode: scoreMode ? "score" : "qc",
          score_file_path: scoreMode ? latestPrsPrepResult?.score_file_path : undefined,
          output_prefix: (plinkConfig.outputPrefix || `${analysis?.analysis_id ?? "active-source"}-plink`).trim(),
          allow_extra_chr: plinkConfig.allowExtraChr,
          freq_limit: !scoreMode && plinkConfig.runFreq ? 12 : 0,
          missing_limit: !scoreMode && plinkConfig.runMissing ? 12 : 0,
          hardy_limit: !scoreMode && plinkConfig.runHardy ? 12 : 0,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      setAnalysis((current) => {
        if (!current) {
          return current;
        }
        return {
          ...current,
          plink_result: payload,
          used_tools: ["plink_execution_tool"],
        };
      });
      if (!analysis) {
        setDirectPlinkResult(payload ?? null);
      }
      setActiveStudioView("plink");
      setStatus("PLINK ready");
    } catch (caught) {
      const msg = caught instanceof Error ? caught.message : String(caught);
      setError(msg);
      setStatus("PLINK failed");
      setAnalysisQa((current) => [
        ...current,
        { role: "assistant", content: `PLINK 실행 중 오류가 발생했습니다: ${msg}` },
      ]);
    } finally {
      setPlinkRunning(false);
    }
  }

  async function loadMoreSummaryStatsRows() {
    if (!summaryStatsAnalysis?.source_stats_path || summaryStatsRowsLoading || !summaryStatsHasMore) {
      return;
    }

    setSummaryStatsRowsLoading(true);
    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/summary-stats/rows`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_stats_path: summaryStatsAnalysis.source_stats_path,
          offset: summaryStatsGridRows.length,
          limit: 200,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload: SummaryStatsRowsResponse = await response.json();
      setSummaryStatsGridRows((current) => [...current, ...payload.rows]);
      setSummaryStatsHasMore(payload.has_more);
    } catch (caught) {
      const msg = caught instanceof Error ? caught.message : String(caught);
      setError(msg);
    } finally {
      setSummaryStatsRowsLoading(false);
    }
  }

  function handleSummaryStatsGridScroll(event: UIEvent<HTMLDivElement>) {
    const target = event.currentTarget;
    const remaining = target.scrollHeight - target.scrollTop - target.clientHeight;
    if (remaining < 160) {
      void loadMoreSummaryStatsRows();
    }
  }

  async function handleAskRawQcQuestion(questionText?: string, analysisOverride?: RawQcResponse | null) {
    const text = questionText?.trim() ?? "";
    const activeAnalysis = analysisOverride ?? rawQcAnalysis;
    if (!text || !activeAnalysis) {
      return;
    }

    setStatus("Generating answer...");
    setAnalysisQa((current) => [...current, { role: "user", content: text }]);

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/raw-qc`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          analysis: activeAnalysis,
          history: analysisQa.map((turn) => ({ role: turn.role, content: turn.content })),
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload: RawQcChatResponse = await response.json();
      if (payload.analysis) {
        setRawQcAnalysis(payload.analysis);
      }
      if (payload.samtools_result) {
        setRawQcAnalysis((current) =>
          current
            ? {
                ...current,
                samtools_result: payload.samtools_result,
                used_tools: ["samtools_execution_tool"],
              }
            : current,
        );
        setActiveStudioView("samtools");
      } else if (payload.requested_view) {
        setActiveStudioView(payload.requested_view);
      }
      setAnalysisQa((current) => [...current, { role: "assistant", content: payload.answer }]);
      setFollowUpAnswer(payload.answer);
      setStatus("Answer ready");
    } catch (caught) {
      const msg = caught instanceof Error ? caught.message : String(caught);
      setAnalysisQa((current) => [
        ...current,
        { role: "assistant", content: `설명 요청 중 오류가 발생했습니다: ${msg}` },
      ]);
      setStatus("Answer failed");
    }
  }

  async function handleAskSummaryStatsQuestion(questionText?: string, analysisOverride?: SummaryStatsResponse | null) {
    const text = questionText?.trim() ?? "";
    const activeAnalysis = analysisOverride ?? summaryStatsAnalysis;
    if (!text || !activeAnalysis) {
      return;
    }

    setStatus("Generating answer...");
    setAnalysisQa((current) => [...current, { role: "user", content: text }]);

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/summary-stats`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          analysis: activeAnalysis,
          history: analysisQa.map((turn) => ({ role: turn.role, content: turn.content })),
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload: SummaryStatsChatResponse = await response.json();
      if (payload.analysis) {
        setSummaryStatsAnalysis(payload.analysis);
      }
      if (payload.qqman_result) {
        setDirectQqmanResult(payload.qqman_result);
      }
      if (payload.prs_prep_result && !payload.analysis) {
        setSummaryStatsAnalysis((current) =>
          current ? { ...current, prs_prep_result: payload.prs_prep_result ?? null } : current,
        );
      }
      if (payload.requested_view) {
        setActiveStudioView(payload.requested_view);
      }
      setAnalysisQa((current) => [...current, { role: "assistant", content: payload.answer }]);
      setFollowUpAnswer(payload.answer);
      setStatus("Answer ready");
    } catch (caught) {
      const msg = caught instanceof Error ? caught.message : String(caught);
      setAnalysisQa((current) => [
        ...current,
        { role: "assistant", content: `설명 요청 중 오류가 발생했습니다: ${msg}` },
      ]);
      setStatus("Answer failed");
    }
  }

  const searchedAnnotations = useMemo(() => {
    const query = annotationSearch.trim().toLowerCase();
    if (!analysis) {
      return [];
    }
    if (!query) {
      return analysis.annotations;
    }
    return analysis.annotations.filter((item) =>
      [item.gene, item.consequence, item.clinical_significance, item.clinvar_conditions, item.rsid]
        .join(" ")
        .toLowerCase()
        .includes(query),
    );
  }, [analysis, annotationSearch]);

  const safeSelectedIndex =
    searchedAnnotations.length === 0
      ? 0
      : Math.min(selectedAnnotationIndex, searchedAnnotations.length - 1);
  const selectedAnnotation = searchedAnnotations[safeSelectedIndex] ?? searchedAnnotations[0] ?? null;
  const summaryText = analysis
    ? formatSummaryWithCitations(analysis.draft_answer, analysis.references)
    : rawQcAnalysis
      ? rawQcAnalysis.draft_answer
      : summaryStatsAnalysis
        ? summaryStatsAnalysis.draft_answer
      : null;
  const displayedAnswer = followUpAnswer ?? summaryText;
  const hasInteractiveState = Boolean(attachedFile || analysis || rawQcAnalysis || summaryStatsAnalysis || messages.length > 1);
  const latestStatusMessage =
    [...messages].reverse().find((message) => message.kind === "status" || message.kind === "summary")?.content ?? "";
  const sourceStatusDetail = useMemo(() => {
    if (status === "Generating answer...") {
      return "ChatGenome is reading the current analysis and Studio results to prepare a grounded response.";
    }
    if (status === "Preparing analysis...") {
      return "The VCF is attached. ChatGenome is starting the default representative analysis run.";
    }
    if (status === "Analyzing") {
      return "Reading the VCF, attaching deterministic annotation, and preparing grounded outputs.";
    }
    if (status === "Running FastQC...") {
      return "Running local FastQC on the uploaded raw sequencing file and collecting module-level QC results.";
    }
    if (status === "Loading summary statistics...") {
      return "Reading the summary statistics file, detecting columns, and preparing a post-GWAS review surface.";
    }
    if (status === "Running Liftover...") {
      return "Running GATK LiftoverVcf on the active VCF and preparing lifted and rejected variant outputs.";
    }
    if (status === "Running qqman...") {
      return "Generating Manhattan and QQ plots from the active summary-statistics source.";
    }
    if (status === "Running samtools...") {
      return "Running samtools on the active alignment source and collecting quickcheck, stats, and idxstats summaries.";
    }
    if (status === "Running SnpEff...") {
      return "Running SnpEff annotation on the active VCF and preparing parsed preview records.";
    }
    if (status === "Running LDBlockShow...") {
      return "Running LDBlockShow for the requested region and preparing LD block visualization artifacts.";
    }
    if (status === "Running PLINK score...") {
      return "Scoring the active target genotype against the prepared PRS weight file.";
    }
    if (status === "Running PLINK...") {
      return "Running the PLINK QC workflow on the active VCF source.";
    }
    if (status === "Answer ready") {
      return "The latest answer is ready in Chat and grounded against the current analysis context.";
    }
    if (status === "Liftover ready") {
      return "Liftover finished and the LiftOver Review card has been updated in Studio.";
    }
    if (status === "qqman ready") {
      return "qqman plotting finished and the qqman Plots card is ready in Studio.";
    }
    if (status === "samtools ready") {
      return "samtools finished and the Samtools Review card is ready in Studio.";
    }
    if (status === "SnpEff ready") {
      return "SnpEff finished and the SnpEff Review card is ready in Studio.";
    }
    if (status === "LDBlockShow ready") {
      return "LDBlockShow finished and the LD block review card is ready in Studio.";
    }
    if (status === "PLINK score ready") {
      return "PLINK score finished and the PRS Review output is ready in Studio.";
    }
    if (status === "PLINK ready") {
      return "PLINK finished and the PLINK card has been updated in Studio.";
    }
    if (status === "Raw QC ready") {
      return "FastQC finished. You can inspect the module summary in Studio and ask follow-up questions in chat.";
    }
    if (status === "Summary stats ready") {
      return "Summary statistics intake finished. Review detected columns, mapping, and QC in Studio before post-GWAS analysis.";
    }
    if (status === "Raw QC failed") {
      return "FastQC failed for the uploaded raw sequencing file. Check the error and runtime prerequisites such as Java.";
    }
    if (status === "Summary stats failed") {
      return "Summary statistics intake failed. Check the file format, compression, and delimiter/header structure.";
    }
    if (status === "Answer failed") {
      return "The last chat response failed. Retry the question and ChatGenome will attempt the grounded explanation again.";
    }
    if (status === "Liftover failed") {
      return "Liftover failed for the active VCF. Check the error details and genome-build inputs.";
    }
    if (status === "qqman failed") {
      return "qqman failed for the active summary-statistics source. Check the file columns and plotting prerequisites.";
    }
    if (status === "samtools failed") {
      return "samtools failed for the active alignment source. Check the file type, index, and runtime prerequisites.";
    }
    if (status === "SnpEff failed") {
      return "SnpEff failed for the active VCF. Check the genome build and local runtime dependencies.";
    }
    if (status === "LDBlockShow failed") {
      return "LDBlockShow failed for the requested region. Check whether the locus contains enough variants for LD plotting.";
    }
    if (status === "PLINK score failed") {
      return "PLINK score failed. Check the target genotype source, score file, and variant overlap.";
    }
    if (status === "PLINK failed") {
      return "PLINK failed for the active VCF. Check the input file and selected QC settings.";
    }
    return latestStatusMessage;
  }, [latestStatusMessage, status]);
  const chatHeaderStatus =
    status === "Generating answer..." ||
    status === "Preparing analysis..." ||
    status === "Analyzing" ||
    status === "Running FastQC..." ||
    status === "Loading summary statistics..." ||
    status === "Running Liftover..." ||
    status === "Running qqman..." ||
    status === "Running samtools..." ||
    status === "Running SnpEff..." ||
    status === "Running LDBlockShow..." ||
    status === "Running PLINK..." ||
    status === "Running PLINK score..." ||
    status === "Raw QC failed" ||
    status === "Summary stats failed" ||
    status === "Answer failed" ||
    status === "Liftover failed" ||
    status === "qqman failed" ||
    status === "samtools failed" ||
    status === "SnpEff failed" ||
    status === "LDBlockShow failed" ||
    status === "PLINK failed" ||
    status === "PLINK score failed" ||
    status === "Liftover ready" ||
    status === "qqman ready" ||
    status === "samtools ready" ||
    status === "SnpEff ready" ||
    status === "LDBlockShow ready" ||
    status === "PLINK ready" ||
    status === "PLINK score ready"
      ? status
      : analysis || rawQcAnalysis || summaryStatsAnalysis
        ? analysis
          ? "Analysis ready"
          : rawQcAnalysis
            ? "Raw QC ready"
            : "Summary stats ready"
        : status;
  const summaryTurn =
    analysis || rawQcAnalysis || summaryStatsAnalysis
      ? [
          {
            role: "assistant" as const,
            content:
              summaryText ??
              "분석이 완료되면 grounded summary가 여기에 표시됩니다.",
          },
        ]
      : [];
  const messageTurns = messages
    .filter((message) => message.kind !== "status")
    .map((message) => ({ role: message.role, content: message.content }));
  const chatTurns =
    messageTurns.length === 0 && analysisQa.length === 0
      ? [...summaryTurn]
      : [...messageTurns, ...analysisQa];
  const qcMetrics = analysis?.facts.qc ?? null;
  const clinvarCounts = useMemo(() => {
    if (!analysis) {
      return [];
    }
    if (analysis.clinvar_summary?.length) {
      return analysis.clinvar_summary;
    }
    const counts = new Map<string, number>();
    analysis.annotations.forEach((item) => {
      const key = summarizeLabel(item.clinical_significance, "Unreviewed");
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
    return [...counts.entries()]
      .map(([label, count]) => ({ label, count }))
      .sort((left, right) => right.count - left.count);
  }, [analysis]);
  const consequenceCounts = useMemo(() => {
    if (!analysis) {
      return [];
    }
    if (analysis.consequence_summary?.length) {
      return analysis.consequence_summary;
    }
    const counts = new Map<string, number>();
    analysis.annotations.forEach((item) => {
      const key = summarizeLabel(item.consequence, "Unclassified");
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
    return [...counts.entries()]
      .map(([label, count]) => ({ label, count }))
      .sort((left, right) => right.count - left.count)
      .slice(0, 10);
  }, [analysis]);
  const geneCounts = useMemo(() => {
    if (!analysis) {
      return [];
    }
    const counts = new Map<string, number>();
    analysis.annotations.forEach((item) => {
      const key = item.gene?.trim() || "Unknown";
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
    return [...counts.entries()]
      .map(([label, count]) => ({ label, count }))
      .sort((left, right) => right.count - left.count)
      .slice(0, 8);
  }, [analysis]);
  const candidateVariants = useMemo(() => {
    if (!analysis) {
      return [];
    }
    if (analysis.candidate_variants?.length) {
      return analysis.candidate_variants.map((entry) => ({
        item: entry.item,
        score: entry.score,
        inRoh: entry.in_roh,
      }));
    }
    return [...analysis.annotations]
      .map((item) => {
        const rohBoost = isVariantInRoh(item, analysis.roh_segments) ? 3 : 0;
        const homAltBoost = item.genotype === "1/1" ? 1 : 0;
        return {
          item,
          score: rankCandidateScore(item) + rohBoost + homAltBoost,
          inRoh: isVariantInRoh(item, analysis.roh_segments),
        };
      })
      .sort((left, right) => right.score - left.score)
      .slice(0, 8);
  }, [analysis]);
  const clinicalCoverage = useMemo(() => {
    if (!analysis || analysis.annotations.length === 0) {
      return [];
    }
    if (analysis.clinical_coverage_summary?.length) {
      return analysis.clinical_coverage_summary;
    }
    const total = analysis.annotations.length;
    const ratio = (count: number) => `${Math.round((count / total) * 100)}%`;
    const clinvarCount = analysis.annotations.filter(
      (item) => hasMeaningfulText(item.clinical_significance) || hasMeaningfulText(item.clinvar_conditions),
    ).length;
    const gnomadCount = analysis.annotations.filter((item) => hasMeaningfulText(item.gnomad_af)).length;
    const geneCount = analysis.annotations.filter((item) => hasMeaningfulText(item.gene)).length;
    const hgvsCount = analysis.annotations.filter(
      (item) => hasMeaningfulText(item.hgvsc) || hasMeaningfulText(item.hgvsp),
    ).length;
    const proteinCount = analysis.annotations.filter((item) => hasMeaningfulText(item.hgvsp)).length;
    return [
      { label: "ClinVar coverage", count: clinvarCount, detail: `${clinvarCount}/${total} annotated (${ratio(clinvarCount)})` },
      { label: "gnomAD coverage", count: gnomadCount, detail: `${gnomadCount}/${total} annotated (${ratio(gnomadCount)})` },
      { label: "Gene mapping", count: geneCount, detail: `${geneCount}/${total} annotated (${ratio(geneCount)})` },
      { label: "HGVS coverage", count: hgvsCount, detail: `${hgvsCount}/${total} annotated (${ratio(hgvsCount)})` },
      { label: "Protein change", count: proteinCount, detail: `${proteinCount}/${total} annotated (${ratio(proteinCount)})` },
    ];
  }, [analysis]);
  const filteringSummary = useMemo(() => {
    if (!analysis) {
      return [];
    }
    if (analysis.filtering_summary?.length) {
      return analysis.filtering_summary;
    }
    const uniqueGenes = new Set(
      analysis.annotations.map((item) => item.gene?.trim() || "").filter((item) => item && item !== "."),
    );
    const clinvarLabeled = analysis.annotations.filter(
      (item) => item.clinical_significance && item.clinical_significance !== ".",
    ).length;
    const symbolicRows = analysis.annotations.filter((item) =>
      item.alts.some((alt) => alt.startsWith("<") && alt.endsWith(">")),
    ).length;
    return [
      { label: "Annotated rows", count: analysis.annotations.length, detail: `${analysis.annotations.length} rows currently available in the triage table` },
      { label: "Distinct genes", count: uniqueGenes.size, detail: `${uniqueGenes.size} genes represented in the annotated subset` },
      { label: "ClinVar-labeled rows", count: clinvarLabeled, detail: `${clinvarLabeled} rows contain a ClinVar-style significance label` },
      { label: "Symbolic ALT rows", count: symbolicRows, detail: `${symbolicRows} rows are symbolic ALT records that may need separate handling` },
    ];
  }, [analysis]);
  const symbolicAnnotations = useMemo(() => {
    if (!analysis) {
      return [];
    }
    if (analysis.symbolic_alt_summary?.examples?.length) {
      const lookup = new Map<string, VariantAnnotation>(
        analysis.annotations.map((item) => [`${item.contig}:${item.pos_1based}`, item] as const),
      );
      return analysis.symbolic_alt_summary.examples
        .map((item) => lookup.get(item.locus))
        .filter((item): item is VariantAnnotation => Boolean(item));
    }
    return analysis.annotations.filter((item) => item.alts.some((alt) => alt.startsWith("<") && alt.endsWith(">")));
  }, [analysis]);
  const rohCandidates = useMemo(() => {
    if (!analysis) {
      return { items: [] as VariantAnnotation[], segments: [] as RohStudioSegment[] };
    }
    const homAlt = [...analysis.annotations]
      .filter((item) => item.genotype === "1/1")
      .sort((left, right) =>
        left.contig === right.contig ? left.pos_1based - right.pos_1based : left.contig.localeCompare(right.contig),
      );

    const segments: RohStudioSegment[] = [];
    let start = homAlt[0];
    let previous = homAlt[0];
    let count = homAlt[0] ? 1 : 0;
    for (let index = 1; index < homAlt.length; index += 1) {
      const current = homAlt[index];
      const sameContig = current.contig === previous.contig;
      const closeEnough = current.pos_1based - previous.pos_1based <= 2_000_000;
      if (sameContig && closeEnough) {
        count += 1;
        previous = current;
        continue;
      }
      if (start && previous && count >= 2) {
        segments.push({
          label: `${start.contig}:${start.pos_1based}-${previous.pos_1based}`,
          count,
          spanMb: `${((previous.pos_1based - start.pos_1based) / 1_000_000).toFixed(2)} Mb`,
        });
      }
      start = current;
      previous = current;
      count = 1;
    }
    if (start && previous && count >= 2) {
      segments.push({
        label: `${start.contig}:${start.pos_1based}-${previous.pos_1based}`,
        count,
        spanMb: `${((previous.pos_1based - start.pos_1based) / 1_000_000).toFixed(2)} Mb`,
      });
    }
    const actualSegments: RohStudioSegment[] = (analysis.roh_segments ?? []).map((segment) => ({
      label: `${segment.contig}:${segment.start_1based}-${segment.end_1based}`,
      count: segment.marker_count,
      spanMb: `${(segment.length_bp / 1_000_000).toFixed(2)} Mb`,
      quality: segment.quality,
      sample: segment.sample,
    }));
    return { items: homAlt.slice(0, 8), segments: actualSegments.length ? actualSegments : segments.slice(0, 6) };
  }, [analysis]);
  const recessiveShortlist = useMemo(() => {
    if (!analysis) {
      return [];
    }
    return [...analysis.annotations]
      .filter((item) => item.genotype === "1/1" || isVariantInRoh(item, analysis.roh_segments))
      .map((item) => ({
        item,
        score: rankRecessiveScore(item, analysis.roh_segments),
        inRoh: isVariantInRoh(item, analysis.roh_segments),
      }))
      .sort((left, right) => right.score - left.score)
      .slice(0, 8);
  }, [analysis]);
  const snpeffResultForStudio = analysis?.snpeff_result ?? directSnpeffResult;
  const plinkResultForStudio = analysis?.plink_result ?? directPlinkResult;
  const liftoverResultForStudio = analysis?.liftover_result ?? directLiftoverResult;
  const ldblockshowResultForStudio = analysis?.ldblockshow_result ?? directLdblockshowResult;
  const samtoolsResultForStudio = rawQcAnalysis?.samtools_result ?? directSamtoolsResult;
  const qqmanResultForStudio = summaryStatsAnalysis?.qqman_result ?? directQqmanResult;
  const prsPrepResultForStudio = summaryStatsAnalysis?.prs_prep_result ?? directPrsPrepResult ?? null;
  const hasStudioState = Boolean(
    analysis ||
      rawQcAnalysis ||
      summaryStatsAnalysis ||
      prsPrepResultForStudio ||
      qqmanResultForStudio ||
      snpeffResultForStudio ||
      plinkResultForStudio ||
      liftoverResultForStudio ||
      ldblockshowResultForStudio ||
      samtoolsResultForStudio,
  );
  const studioCards: Array<{ id: StudioView; title: string; subtitle: string }> = rawQcAnalysis
    ? [
        { id: "rawqc", title: "FastQC Review", subtitle: "Raw sequencing module summary" },
        ...(samtoolsResultForStudio
          ? [{ id: "samtools" as StudioView, title: "Samtools Review", subtitle: "Alignment QC summary" }]
          : []),
      ]
    : summaryStatsAnalysis
      ? [
          { id: "sumstats" as StudioView, title: "Summary Stats Review", subtitle: "Post-GWAS intake and column mapping" },
          ...(prsPrepResultForStudio
            ? [{ id: "prs_prep" as StudioView, title: "PRS Prep Review", subtitle: "Build check, harmonization, and score-file readiness" }]
            : []),
          ...(qqmanResultForStudio
            ? [{ id: "qqman" as StudioView, title: "qqman Plots", subtitle: "Manhattan and QQ visualization" }]
            : []),
        ]
    : analysis
      ? [
          { id: "provenance", title: "Workflow Setup", subtitle: "Tools, scope, and run policy" },
          { id: "qc", title: "QC Summary", subtitle: "PASS, Ti/Tv, GT quality" },
          { id: "coverage", title: "Clinical Coverage", subtitle: "Annotation completeness view" },
          { id: "snpeff", title: "SnpEff Review", subtitle: "Local effect annotation preview" },
          { id: "plink", title: "PLINK", subtitle: "QC command runner and result review" },
          ...(liftoverResultForStudio
            ? [{ id: "liftover" as StudioView, title: "LiftOver Review", subtitle: "Genome build conversion result" }]
            : []),
          ...(ldblockshowResultForStudio
            ? [{ id: "ldblockshow" as StudioView, title: "LD Block Review", subtitle: "Locus-level LD heatmap" }]
            : []),
          { id: "table", title: "Filtering View", subtitle: "Searchable variant triage" },
          { id: "symbolic", title: "Symbolic ALT Review", subtitle: "Structural-style records split out" },
          { id: "roh", title: "ROH / Recessive", subtitle: "Hom-alt and ROH-style review" },
          { id: "candidates", title: "Candidate Variants", subtitle: "Ranked review shortlist" },
          { id: "vep", title: "VEP Consequence", subtitle: "Consequence and gene burden" },
          { id: "clinvar", title: "ClinVar Review", subtitle: "Clinical significance mix" },
          { id: "annotations", title: "Annotation Cards", subtitle: "Variant detail cards" },
          { id: "igv", title: "IGV Plot", subtitle: "Locus visualization" },
          { id: "acmg", title: "ACMG Review", subtitle: "Evidence hints, not final calls" },
          { id: "references", title: "References", subtitle: "Linked evidence" },
        ]
      : [
          ...(samtoolsResultForStudio
            ? [{ id: "samtools" as StudioView, title: "Samtools Review", subtitle: "Alignment QC summary" }]
            : []),
          ...(snpeffResultForStudio
            ? [{ id: "snpeff" as StudioView, title: "SnpEff Review", subtitle: "Local effect annotation preview" }]
            : []),
          ...(plinkResultForStudio
            ? [{ id: "plink" as StudioView, title: "PLINK", subtitle: "QC command runner and result review" }]
            : []),
          ...(liftoverResultForStudio
          ? [{ id: "liftover" as StudioView, title: "LiftOver Review", subtitle: "Genome build conversion result" }]
          : []),
          ...(ldblockshowResultForStudio
          ? [{ id: "ldblockshow" as StudioView, title: "LD Block Review", subtitle: "Locus-level LD heatmap" }]
          : []),
          ...(qqmanResultForStudio
          ? [{ id: "qqman" as StudioView, title: "qqman Plots", subtitle: "Manhattan and QQ visualization" }]
          : []),
          ...(prsPrepResultForStudio
          ? [{ id: "prs_prep" as StudioView, title: "PRS Prep Review", subtitle: "Build check, harmonization, and score-file readiness" }]
          : []),
        ];
  const studioRendererRegistry: Partial<Record<StudioView, () => React.ReactNode>> = {
    rawqc: () =>
      rawQcAnalysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>FastQC Review</h2>
          </div>
          <div className="studioCanvasBody">
            <StudioMetricGrid
              items={[
                {
                  label: "Total sequences",
                  value: rawQcAnalysis.facts.total_sequences != null ? String(rawQcAnalysis.facts.total_sequences) : "n/a",
                  tone: "good",
                },
                { label: "Sequence length", value: rawQcAnalysis.facts.sequence_length ?? "n/a", tone: "neutral" },
                {
                  label: "%GC",
                  value: rawQcAnalysis.facts.gc_content != null ? `${rawQcAnalysis.facts.gc_content.toFixed(1)}%` : "n/a",
                  tone: "neutral",
                },
                { label: "Encoding", value: rawQcAnalysis.facts.encoding ?? "n/a", tone: "neutral" },
              ]}
            />
            <div className="resultList">
              {rawQcAnalysis.modules.map((module) => (
                <article key={module.name} className="miniCard">
                  <h3>{module.name}</h3>
                  <p>Status: {module.status}</p>
                  {module.detail ? <p>{module.detail}</p> : null}
                </article>
              ))}
            </div>
            <ArtifactLinksRow
              items={[
                ...(rawQcAnalysis.report_html_path
                  ? [
                      {
                        label: "Open HTML report",
                        href: `${apiBase.replace(/\/$/, "")}/api/v1/raw-qc/report?path=${encodeURIComponent(rawQcAnalysis.report_html_path)}`,
                      },
                    ]
                  : []),
                ...(rawQcAnalysis.report_zip_path
                  ? [
                      {
                        label: "Download ZIP",
                        href: `${apiBase.replace(/\/$/, "")}/api/v1/raw-qc/report?path=${encodeURIComponent(rawQcAnalysis.report_zip_path)}`,
                      },
                    ]
                  : []),
              ]}
            />
          </div>
        </section>
      ) : null,
    sumstats: () =>
      summaryStatsAnalysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>Summary Stats Review</h2>
          </div>
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
                  {summaryStatsAnalysis.detected_columns.map((column) => (
                    <li key={column}>{column}</li>
                  ))}
                </ul>
              </article>
              <article className="miniCard">
                <h3>Auto-mapped fields</h3>
                <ul className="hintList">
                  {Object.entries(summaryStatsAnalysis.mapped_fields).map(([field, value]) => (
                    <li key={field}>
                      <strong>{field}</strong>: {value || "not detected"}
                    </li>
                  ))}
                </ul>
              </article>
            </div>
            <article className="miniCard">
              <h3>Preview grid</h3>
              <p className="summaryStatsGridMeta">
                Showing {summaryStatsGridRows.length} of {summaryStatsAnalysis.row_count} rows
              </p>
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
                      {!summaryStatsHasMore && summaryStatsGridRows.length ? (
                        <div className="summaryStatsGridFooter">All loaded rows are shown.</div>
                      ) : null}
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
          <div className="notebookHeader">
            <h2>Samtools Review</h2>
          </div>
          <div className="studioCanvasBody">
            {samtoolsResultForStudio ? (
              <>
                <StudioMetricGrid
                  items={[
                    { label: "File kind", value: samtoolsResultForStudio.file_kind, tone: "good" },
                    {
                      label: "Total reads",
                      value: samtoolsResultForStudio.total_reads != null ? String(samtoolsResultForStudio.total_reads) : "n/a",
                    },
                    {
                      label: "Mapped",
                      value:
                        samtoolsResultForStudio.mapped_reads != null
                          ? `${samtoolsResultForStudio.mapped_reads}${
                              samtoolsResultForStudio.mapped_rate != null ? ` (${samtoolsResultForStudio.mapped_rate.toFixed(2)}%)` : ""
                            }`
                          : "n/a",
                      tone: "good",
                    },
                    {
                      label: "Properly paired",
                      value:
                        samtoolsResultForStudio.properly_paired_reads != null
                          ? `${samtoolsResultForStudio.properly_paired_reads}${
                              samtoolsResultForStudio.properly_paired_rate != null
                                ? ` (${samtoolsResultForStudio.properly_paired_rate.toFixed(2)}%)`
                                : ""
                            }`
                          : "n/a",
                    },
                    {
                      label: "Quickcheck",
                      value: samtoolsResultForStudio.quickcheck_ok ? "PASS" : "Issue detected",
                      tone: samtoolsResultForStudio.quickcheck_ok ? "good" : "warn",
                    },
                    {
                      label: "Index",
                      value: samtoolsResultForStudio.index_path ? "Created / available" : "n/a",
                    },
                  ]}
                />
                <div className="resultSectionSplit">
                  <article className="miniCard">
                    <h3>samtools stats highlights</h3>
                    <StudioSimpleList
                      items={samtoolsResultForStudio.stats_highlights.map((item) => ({
                        label: item.label,
                        detail: item.value,
                      }))}
                      emptyLabel="No samtools stats highlights are available."
                    />
                  </article>
                  <article className="miniCard">
                    <h3>idxstats preview</h3>
                    <StudioSimpleList
                      items={samtoolsResultForStudio.idxstats_rows.map((row) => ({
                        label: row.contig,
                        detail: `mapped ${row.mapped} | unmapped ${row.unmapped} | length ${row.length_bp}`,
                      }))}
                      emptyLabel="No idxstats preview rows are available."
                    />
                  </article>
                </div>
                <WarningListCard warnings={samtoolsResultForStudio.warnings} />
              </>
            ) : (
              <p className="emptyState">No samtools result is available for the current raw-QC session.</p>
            )}
          </div>
        </section>
      ) : null,
    prs_prep: () =>
      prsPrepResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>PRS Prep Review</h2>
          </div>
          <div className="studioCanvasBody">
            <div className="resultMetricGrid">
              <MetricTile label="Inferred build" value={prsPrepResultForStudio.build_check.inferred_build} tone="good" />
              <MetricTile label="Build confidence" value={prsPrepResultForStudio.build_check.build_confidence} tone="neutral" />
              <MetricTile label="Effect size" value={prsPrepResultForStudio.harmonization.effect_size_kind} tone="neutral" />
              <MetricTile
                label="Ready rows"
                value={String(prsPrepResultForStudio.kept_rows)}
                tone={prsPrepResultForStudio.kept_rows > 0 ? "good" : "warn"}
              />
              <MetricTile label="Dropped rows" value={String(prsPrepResultForStudio.dropped_rows)} tone="neutral" />
              <MetricTile
                label="Score file"
                value={prsPrepResultForStudio.score_file_ready ? "ready" : "not ready"}
                tone={prsPrepResultForStudio.score_file_ready ? "good" : "warn"}
              />
            </div>
            <div className="resultSectionSplit">
              <article className="miniCard">
                <h3>Build check</h3>
                <ul className="hintList">
                  <li>
                    <strong>Source build</strong>: {prsPrepResultForStudio.build_check.source_build}
                  </li>
                  <li>
                    <strong>Target build</strong>: {prsPrepResultForStudio.build_check.target_build}
                  </li>
                  <li>
                    <strong>Build match</strong>:{" "}
                    {prsPrepResultForStudio.build_check.build_match == null
                      ? "undetermined"
                      : prsPrepResultForStudio.build_check.build_match
                        ? "yes"
                        : "no"}
                  </li>
                </ul>
              </article>
              <article className="miniCard">
                <h3>Harmonization</h3>
                <ul className="hintList">
                  <li>
                    <strong>Required fields present</strong>:{" "}
                    {prsPrepResultForStudio.harmonization.required_fields_present ? "yes" : "no"}
                  </li>
                  <li>
                    <strong>Preview rows harmonizable</strong>: {prsPrepResultForStudio.harmonization.harmonizable_preview_rows}
                  </li>
                  <li>
                    <strong>Ambiguous SNPs</strong>: {prsPrepResultForStudio.harmonization.ambiguous_snp_count}
                  </li>
                  {prsPrepResultForStudio.harmonization.missing_fields.length ? (
                    <li>
                      <strong>Missing fields</strong>: {prsPrepResultForStudio.harmonization.missing_fields.join(", ")}
                    </li>
                  ) : null}
                </ul>
              </article>
            </div>
            <article className="miniCard">
              <h3>PLINK score-file preview</h3>
              <p className="summaryStatsGridMeta">
                Columns: {prsPrepResultForStudio.score_file_columns.join(", ") || "ID, A1, BETA"}
              </p>
              <div className="variantTableWrap summaryStatsTableWrap">
                <table className="variantTable summaryStatsTable">
                  <thead>
                    <tr>
                      {prsPrepResultForStudio.score_file_columns.map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {prsPrepResultForStudio.score_file_preview_rows.map((row, index) => (
                      <tr key={`prs-prep-preview-${index}`}>
                        {prsPrepResultForStudio.score_file_columns.map((column) => (
                          <td key={`${index}-${column}`}>{row[column] || ""}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <ArtifactLinksRow
                items={
                  prsPrepResultForStudio.score_file_path
                    ? [
                        {
                          label: "Open output",
                          href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(prsPrepResultForStudio.score_file_path)}`,
                        },
                      ]
                    : []
                }
              />
            </article>
            <WarningListCard
              warnings={[...prsPrepResultForStudio.build_check.warnings, ...prsPrepResultForStudio.harmonization.warnings]}
            />
          </div>
        </section>
      ) : null,
    qqman: () =>
      qqmanResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>qqman Plots</h2>
          </div>
          <div className="studioCanvasBody">
            <StudioMetricGrid
              items={[
                { label: "Tool", value: qqmanResultForStudio.tool, tone: "good" },
                { label: "Artifacts", value: String(qqmanResultForStudio.artifacts.length) },
                { label: "Warnings", value: String(qqmanResultForStudio.warnings.length) },
              ]}
            />
            <div className="resultList">
              <article className="resultListItem resultListStatic">
                <strong>Command preview</strong>
                <pre className="codeBlock">{qqmanResultForStudio.command_preview}</pre>
              </article>
            </div>
            <div className="resultSectionSplit">
              {qqmanResultForStudio.artifacts.map((artifact) => (
                <article key={artifact.api_path} className="miniCard">
                  <h3>{artifact.title}</h3>
                  <img
                    src={`${apiBase.replace(/\/$/, "")}${artifact.api_path}`}
                    alt={artifact.title}
                    className="plotPreviewImage"
                  />
                  <p className="resultNote">{artifact.note}</p>
                  <div className="resultActionRow">
                    <a className="sourceAddButton" href={`${apiBase.replace(/\/$/, "")}${artifact.api_path}`} target="_blank" rel="noreferrer">
                      Open image
                    </a>
                  </div>
                </article>
              ))}
            </div>
            <WarningListCard warnings={qqmanResultForStudio.warnings} />
          </div>
        </section>
      ) : null,
    provenance: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>Analysis Provenance</h2>
          </div>
          <div className="studioCanvasBody">
            <StudioMetricGrid
              items={[
                { label: "Annotation scope", value: annotationScope },
                { label: "Annotation limit", value: annotationScope === "all" ? annotationLimit || "n/a" : "representative" },
                { label: "References", value: String(analysis.references.length) },
                { label: "Annotations", value: String(analysis.annotations.length) },
              ]}
            />
            <div className="resultSectionSplit">
              <article className="miniCard">
                <h3>Tool chain</h3>
                <ul className="hintList">
                  <li>`pysam` for VCF parsing, file summary, and QC metrics</li>
                  <li>`Ensembl VEP REST` for consequence, transcript, HGVS, and protein fields</li>
                  <li>`ClinVar / NCBI refsnp` for clinical significance and condition labels</li>
                  <li>`gnomAD` frequency joins for population rarity context</li>
                  <li>`OpenAI` models for workflow intake and grounded narrative explanation</li>
                </ul>
              </article>
              <article className="miniCard">
                <h3>Current run policy</h3>
                <ul className="hintList">
                  <li>Filtering tools such as `bcftools` and `GATK` are available but were not automatically applied in this summary-first run.</li>
                  <li>Representative annotation is the default unless the user explicitly requests a wider range.</li>
                  <li>Studio cards are derived from the current annotated subset, not from a separate hidden analysis branch.</li>
                </ul>
              </article>
            </div>
          </div>
        </section>
      ) : null,
    qc: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>QC Summary</h2>
          </div>
          <div className="studioCanvasBody">
            <StudioMetricGrid
              items={[
                { label: "PASS rate", value: formatPercent(qcMetrics?.pass_rate), tone: "good" },
                { label: "Ti/Tv", value: formatNumber(qcMetrics?.transition_transversion_ratio) },
                { label: "Missing GT", value: formatPercent(qcMetrics?.missing_gt_rate), tone: "warn" },
                { label: "Het/HomAlt", value: formatNumber(qcMetrics?.het_hom_alt_ratio) },
                { label: "Multi-allelic", value: formatPercent(qcMetrics?.multi_allelic_rate) },
                { label: "Symbolic ALT", value: formatPercent(qcMetrics?.symbolic_alt_rate) },
                { label: "SNV fraction", value: formatPercent(qcMetrics?.snv_fraction), tone: "good" },
                { label: "Indel fraction", value: formatPercent(qcMetrics?.indel_fraction) },
              ]}
            />
            <div className="resultSectionSplit">
              <article className="miniCard">
                <h3>Genotype composition</h3>
                <DistributionList
                  items={Object.entries(analysis.facts.genotype_counts)
                    .map(([label, count]) => ({ label, count }))
                    .sort((left, right) => right.count - left.count)}
                  emptyLabel="No genotype counts are available."
                />
              </article>
              <article className="miniCard">
                <h3>Variant classes</h3>
                <DistributionList
                  items={Object.entries(analysis.facts.variant_types)
                    .map(([label, count]) => ({ label, count }))
                    .sort((left, right) => right.count - left.count)}
                  emptyLabel="No variant class counts are available."
                />
              </article>
            </div>
          </div>
        </section>
      ) : null,
    coverage: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>Clinical Annotation Coverage</h2>
          </div>
          <div className="studioCanvasBody">
            <StudioSimpleList
              items={clinicalCoverage.map((item) => ({
                label: item.label,
                detail: item.detail,
              }))}
              emptyLabel="No clinical coverage summary is available."
            />
          </div>
        </section>
      ) : null,
    snpeff: () =>
      analysis || snpeffResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>SnpEff Review</h2>
          </div>
          <div className="studioCanvasBody">
            {snpeffResultForStudio ? (
              <>
                <StudioMetricGrid
                  items={[
                    { label: "Genome DB", value: snpeffResultForStudio.genome, tone: "good" },
                    { label: "Preview rows", value: String(snpeffResultForStudio.parsed_records.length) },
                    { label: "Tool", value: snpeffResultForStudio.tool },
                  ]}
                />
                <div className="resultList">
                  {snpeffResultForStudio.parsed_records.map((record, index) => (
                    <article key={`${record.contig}-${record.pos_1based}-${record.alt}-${index}`} className="resultListItem resultListStatic">
                      <strong>
                        {record.contig}:{record.pos_1based} {record.ref}&gt;{record.alt}
                      </strong>
                      <span>
                        {record.ann.length
                          ? record.ann
                              .slice(0, 2)
                              .map((ann) => `${ann.gene_name || "Unknown"} | ${ann.annotation} | ${ann.impact} | ${ann.hgvs_c || "."} | ${ann.hgvs_p || "."}`)
                              .join(" || ")
                          : "No parsed ANN entries"}
                      </span>
                    </article>
                  ))}
                </div>
                <ArtifactLinksRow
                  items={[
                    {
                      label: "Open annotated VCF",
                      href: `file://${snpeffResultForStudio.output_path}`,
                    },
                  ]}
                />
              </>
            ) : (
              <p className="emptyState">No auxiliary SnpEff result is available for the current analysis.</p>
            )}
          </div>
        </section>
      ) : null,
    liftover: () =>
      analysis || liftoverResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>LiftOver Review</h2>
          </div>
          <div className="studioCanvasBody">
            {liftoverResultForStudio ? (
              <>
                <StudioMetricGrid
                  items={[
                    { label: "Source build", value: liftoverResultForStudio.source_build ?? "unknown", tone: "neutral" },
                    { label: "Target build", value: liftoverResultForStudio.target_build ?? "unknown", tone: "good" },
                    { label: "Lifted records", value: String(liftoverResultForStudio.lifted_record_count ?? 0), tone: "good" },
                    { label: "Rejected records", value: String(liftoverResultForStudio.rejected_record_count ?? 0), tone: "neutral" },
                    { label: "Warnings", value: String(liftoverResultForStudio.warnings.length), tone: "neutral" },
                    { label: "Tool", value: liftoverResultForStudio.tool, tone: "neutral" },
                  ]}
                />
                <div className="resultList">
                  {liftoverResultForStudio.parsed_records.length ? (
                    liftoverResultForStudio.parsed_records.map((record, index) => (
                      <article key={`${record.contig}-${record.pos_1based}-${index}`} className="resultListItem resultListStatic">
                        <strong>
                          {record.contig}:{record.pos_1based} {record.ref}&gt;{record.alts.join(",")}
                        </strong>
                        <span>Lifted preview record</span>
                      </article>
                    ))
                  ) : (
                    <p className="emptyState">No lifted preview records are available for this result.</p>
                  )}
                  {liftoverResultForStudio.warnings.length
                    ? liftoverResultForStudio.warnings.map((warning, index) => (
                        <article key={`liftover-warning-${index}`} className="resultListItem resultListStatic">
                          <strong>Warning {index + 1}</strong>
                          <span>{warning}</span>
                        </article>
                      ))
                    : null}
                </div>
                <ArtifactLinksRow
                  items={[
                    {
                      label: "Open lifted VCF",
                      href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(liftoverResultForStudio.output_path)}`,
                    },
                    {
                      label: "Open reject VCF",
                      href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(liftoverResultForStudio.reject_path)}`,
                    },
                  ]}
                />
              </>
            ) : (
              <p className="emptyState">No liftover result is available for the current analysis.</p>
            )}
          </div>
        </section>
      ) : null,
    ldblockshow: () =>
      analysis || ldblockshowResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>LD Block Review</h2>
          </div>
          <div className="studioCanvasBody">
            {ldblockshowResultForStudio ? (
              <>
                <StudioMetricGrid
                  items={[
                    { label: "Region", value: ldblockshowResultForStudio.region, tone: "good" },
                    { label: "Tried regions", value: String(ldblockshowResultForStudio.attempted_regions?.length ?? 0), tone: "neutral" },
                    { label: "Site rows", value: String(ldblockshowResultForStudio.site_row_count ?? 0), tone: "neutral" },
                    { label: "Triangle pairs", value: String(ldblockshowResultForStudio.triangle_pair_count ?? 0), tone: "neutral" },
                    { label: "Warnings", value: String(ldblockshowResultForStudio.warnings.length), tone: "neutral" },
                    { label: "Tool", value: ldblockshowResultForStudio.tool, tone: "neutral" },
                  ]}
                />
                <div className="resultList">
                  {ldblockshowResultForStudio.attempted_regions?.length ? (
                    <article className="resultListItem resultListStatic">
                      <strong>Attempted regions</strong>
                      <span>{ldblockshowResultForStudio.attempted_regions.join(" -> ")}</span>
                    </article>
                  ) : null}
                </div>
                <WarningListCard
                  warnings={ldblockshowResultForStudio.warnings}
                  emptyLabel="No LDBlockShow warnings were reported."
                  emptyAsParagraph
                />
                <ArtifactLinksRow
                  items={[
                    ...(ldblockshowResultForStudio.svg_path
                      ? [
                          {
                            label: "Open LD SVG",
                            href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(ldblockshowResultForStudio.svg_path)}`,
                          },
                        ]
                      : []),
                    ...(ldblockshowResultForStudio.block_path
                      ? [
                          {
                            label: "Open block table",
                            href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(ldblockshowResultForStudio.block_path)}`,
                          },
                        ]
                      : []),
                    ...(ldblockshowResultForStudio.site_path
                      ? [
                          {
                            label: "Open site table",
                            href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(ldblockshowResultForStudio.site_path)}`,
                          },
                        ]
                      : []),
                  ]}
                />
              </>
            ) : (
              <p className="emptyState">No LDBlockShow result is available for the current analysis.</p>
            )}
          </div>
        </section>
      ) : null,
    candidates: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>Candidate Variants</h2>
          </div>
          <div className="studioCanvasBody">
            <div className="resultList">
              {candidateVariants.map(({ item, score, inRoh }) => (
                <button
                  type="button"
                  key={`${item.contig}-${item.pos_1based}-${item.rsid}-candidate`}
                  className="resultListItem"
                  onClick={() => {
                    const nextIndex = searchedAnnotations.findIndex(
                      (candidate) =>
                        candidate.contig === item.contig &&
                        candidate.pos_1based === item.pos_1based &&
                        candidate.rsid === item.rsid,
                    );
                    setSelectedAnnotationIndex(nextIndex >= 0 ? nextIndex : 0);
                    setActiveStudioView("annotations");
                  }}
                >
                  <strong>
                    {item.gene || "Unknown"} | {item.contig}:{item.pos_1based}
                  </strong>
                  <span>
                    Score {score} | {summarizeLabel(item.consequence, "Unclassified")} |{" "}
                    {summarizeLabel(item.clinical_significance, "Unreviewed")}
                    {inRoh ? " | inside ROH" : ""}
                    {item.cadd_phred != null ? ` | CADD ${item.cadd_phred.toFixed(1)}` : ""}
                    {item.revel_score != null ? ` | REVEL ${item.revel_score.toFixed(3)}` : ""}
                    {" | "}gnomAD {item.gnomad_af || "n/a"}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </section>
      ) : null,
    acmg: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>ACMG Review</h2>
          </div>
          <div className="studioCanvasBody">
            <p className="emptyState acmgNote">
              This is a triage view with ACMG-style evidence hints. It is not a final clinical classification.
            </p>
            <div className="resultList">
              {candidateVariants.slice(0, 6).map(({ item }) => (
                <article
                  key={`${item.contig}-${item.pos_1based}-${item.rsid}-acmg`}
                  className="resultListItem resultListStatic"
                >
                  <strong>
                    {item.gene || "Unknown"} | {item.rsid || `${item.contig}:${item.pos_1based}`}
                  </strong>
                  <span>
                    {summarizeLabel(item.consequence, "Unclassified")} |{" "}
                    {summarizeLabel(item.clinical_significance, "Unreviewed")}
                  </span>
                  <ul className="hintList">
                    {buildAcmgHints(item).map((hint) => (
                      <li key={hint}>{hint}</li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
          </div>
        </section>
      ) : null,
    plink: () =>
      analysis || plinkResultForStudio ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>PLINK</h2>
          </div>
          <div className="studioCanvasBody">
            <div className="resultMetricGrid">
              <MetricTile label="Mode" value={plinkConfig.mode === "score" ? "Score" : "QC"} tone="good" />
              <MetricTile
                label="Source"
                value={analysis?.facts.file_name ?? activeSource?.file_name ?? attachedFile?.name ?? "n/a"}
                tone="neutral"
              />
              <MetricTile
                label="Existing result"
                value={plinkResultForStudio ? "Available" : "Not run yet"}
                tone={plinkResultForStudio ? "good" : "neutral"}
              />
              <MetricTile
                label="Runner"
                value={plinkRunning ? "Running" : "Ready"}
                tone={plinkRunning ? "warn" : "good"}
              />
            </div>
            <div className="resultList">
              <article className="resultListItem resultListStatic">
                <strong>Run configuration</strong>
                <div className="annotationMetaGrid">
                  <label className="field compactField">
                    <span>Mode</span>
                    <select
                      value={plinkConfig.mode}
                      onChange={(event) => setPlinkConfig((current) => ({ ...current, mode: event.target.value }))}
                    >
                      <option value="qc">qc</option>
                      <option value="score">score</option>
                    </select>
                  </label>
                  <label className="field compactField">
                    <span>Output prefix</span>
                    <input
                      type="text"
                      value={plinkConfig.outputPrefix}
                      onChange={(event) => setPlinkConfig((current) => ({ ...current, outputPrefix: event.target.value }))}
                    />
                  </label>
                  <label className="field compactField">
                    <span>Frequency summary</span>
                    <input
                      type="checkbox"
                      checked={plinkConfig.runFreq}
                      disabled={plinkConfig.mode === "score"}
                      onChange={(event) => setPlinkConfig((current) => ({ ...current, runFreq: event.target.checked }))}
                    />
                  </label>
                  <label className="field compactField">
                    <span>Missingness summary</span>
                    <input
                      type="checkbox"
                      checked={plinkConfig.runMissing}
                      disabled={plinkConfig.mode === "score"}
                      onChange={(event) => setPlinkConfig((current) => ({ ...current, runMissing: event.target.checked }))}
                    />
                  </label>
                  <label className="field compactField">
                    <span>Hardy-Weinberg summary</span>
                    <input
                      type="checkbox"
                      checked={plinkConfig.runHardy}
                      disabled={plinkConfig.mode === "score"}
                      onChange={(event) => setPlinkConfig((current) => ({ ...current, runHardy: event.target.checked }))}
                    />
                  </label>
                  <label className="field compactField">
                    <span>Allow extra chr labels</span>
                    <input
                      type="checkbox"
                      checked={plinkConfig.allowExtraChr}
                      onChange={(event) => setPlinkConfig((current) => ({ ...current, allowExtraChr: event.target.checked }))}
                    />
                  </label>
                </div>
                {plinkConfig.mode === "score" ? (
                  <p className="resultNote">
                    Score mode uses the latest PRS prep score file. Run <code>@skill prs_prep</code> on a summary-statistics source first, then upload a target genotype VCF and run <code>@plink score</code>.
                  </p>
                ) : null}
              </article>
              <article className="resultListItem resultListStatic">
                <strong>Command preview</strong>
                <pre className="codeBlock">{plinkCommandPreview}</pre>
              </article>
            </div>
            <div className="resultActionRow">
              <button
                className="sourceAddButton"
                type="button"
                onClick={() => void handleRunPlink()}
                disabled={plinkRunning || !(analysis?.source_vcf_path || (activeSource?.source_type === "vcf" ? activeSource.source_path : null))}
              >
                {plinkRunning ? "Running PLINK..." : "Run PLINK"}
              </button>
            </div>
            {plinkResultForStudio ? (
              plinkResultForStudio.mode === "score" ? (
                <>
                  <div className="resultMetricGrid">
                    <MetricTile
                      label="Samples scored"
                      value={
                        plinkResultForStudio.sample_count != null
                          ? String(plinkResultForStudio.sample_count)
                          : String(plinkResultForStudio.score_rows.length)
                      }
                      tone="good"
                    />
                    <MetricTile
                      label="Mean score"
                      value={plinkResultForStudio.score_mean != null ? plinkResultForStudio.score_mean.toFixed(4) : "n/a"}
                      tone="neutral"
                    />
                    <MetricTile
                      label="Min score"
                      value={plinkResultForStudio.score_min != null ? plinkResultForStudio.score_min.toFixed(4) : "n/a"}
                      tone="neutral"
                    />
                    <MetricTile
                      label="Max score"
                      value={plinkResultForStudio.score_max != null ? plinkResultForStudio.score_max.toFixed(4) : "n/a"}
                      tone="neutral"
                    />
                    <MetricTile label="Preview rows" value={String(plinkResultForStudio.score_rows.length)} tone="good" />
                    <MetricTile label="Warnings" value={String(plinkResultForStudio.warnings.length)} tone="neutral" />
                  </div>
                  <div className="resultList">
                    <article className="resultListItem resultListStatic">
                      <strong>PRS score inputs</strong>
                      <span>Target genotype: {plinkResultForStudio.input_path}</span>
                      <span>Score file: {plinkResultForStudio.score_file_path || "n/a"}</span>
                    </article>
                    {plinkResultForStudio.score_rows.slice(0, 10).map((row, index) => (
                      <article key={`plink-score-${row.sample_id}-${index}`} className="resultListItem resultListStatic">
                        <strong>{row.sample_id}</strong>
                        <span>
                          allele ct {row.allele_ct != null ? row.allele_ct.toFixed(2) : "n/a"} | dosage sum{" "}
                          {row.named_allele_dosage_sum != null ? row.named_allele_dosage_sum.toFixed(2) : "n/a"} | score{" "}
                          {row.score_sum != null ? row.score_sum.toFixed(4) : "n/a"}
                        </span>
                      </article>
                    ))}
                    {plinkResultForStudio.warnings.map((warning, index) => (
                      <article key={`plink-warning-${index}`} className="resultListItem resultListStatic">
                        <strong>Warning {index + 1}</strong>
                        <span>{warning}</span>
                      </article>
                    ))}
                  </div>
                  <ArtifactLinksRow
                    items={[
                      ...(plinkResultForStudio.score_output_path
                        ? [
                            {
                              label: "Open output",
                              href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.score_output_path)}`,
                            },
                          ]
                        : []),
                      ...(plinkResultForStudio.log_path
                        ? [
                            {
                              label: "Open log",
                              href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.log_path)}`,
                            },
                          ]
                        : []),
                    ]}
                  />
                </>
              ) : (
                <>
                  <div className="resultMetricGrid">
                    <MetricTile
                      label="Samples"
                      value={plinkResultForStudio.sample_count != null ? String(plinkResultForStudio.sample_count) : "n/a"}
                      tone="neutral"
                    />
                    <MetricTile
                      label="Variants"
                      value={plinkResultForStudio.variant_count != null ? String(plinkResultForStudio.variant_count) : "n/a"}
                      tone="neutral"
                    />
                    <MetricTile label="Freq rows" value={String(plinkResultForStudio.freq_rows.length)} tone="good" />
                    <MetricTile label="Missing rows" value={String(plinkResultForStudio.missing_rows.length)} tone="good" />
                    <MetricTile label="Hardy rows" value={String(plinkResultForStudio.hardy_rows.length)} tone="good" />
                    <MetricTile label="Warnings" value={String(plinkResultForStudio.warnings.length)} tone="neutral" />
                  </div>
                  <div className="resultList">
                    {plinkResultForStudio.freq_rows.slice(0, 5).map((row, index) => (
                      <article key={`plink-freq-${row.variant_id}-${index}`} className="resultListItem resultListStatic">
                        <strong>
                          {row.chrom}:{row.variant_id}
                        </strong>
                        <span>
                          {row.ref_allele}&gt;{row.alt_allele} | AF {row.alt_freq != null ? row.alt_freq.toFixed(4) : "n/a"} | OBS{" "}
                          {row.observation_count ?? "n/a"}
                        </span>
                      </article>
                    ))}
                    {plinkResultForStudio.missing_rows.slice(0, 3).map((row, index) => (
                      <article key={`plink-missing-${row.sample_id}-${index}`} className="resultListItem resultListStatic">
                        <strong>Missingness {row.sample_id}</strong>
                        <span>
                          {row.missing_genotype_count} missing / {row.observation_count} obs | rate {(row.missing_rate * 100).toFixed(2)}%
                        </span>
                      </article>
                    ))}
                    {plinkResultForStudio.hardy_rows.slice(0, 5).map((row, index) => (
                      <article key={`plink-hardy-${row.variant_id}-${index}`} className="resultListItem resultListStatic">
                        <strong>
                          Hardy {row.chrom}:{row.variant_id}
                        </strong>
                        <span>
                          obs het {row.observed_hets ?? "n/a"} | exp het{" "}
                          {row.expected_hets != null ? row.expected_hets.toFixed(2) : "n/a"} | p{" "}
                          {row.p_value != null ? row.p_value.toExponential(3) : "n/a"}
                        </span>
                      </article>
                    ))}
                    {plinkResultForStudio.warnings.map((warning, index) => (
                      <article key={`plink-warning-${index}`} className="resultListItem resultListStatic">
                        <strong>Warning {index + 1}</strong>
                        <span>{warning}</span>
                      </article>
                    ))}
                  </div>
                  <ArtifactLinksRow
                    items={[
                      ...(plinkResultForStudio.freq_path
                        ? [
                            {
                              label: "Open afreq",
                              href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.freq_path)}`,
                            },
                          ]
                        : []),
                      ...(plinkResultForStudio.missing_path
                        ? [
                            {
                              label: "Open smiss",
                              href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.missing_path)}`,
                            },
                          ]
                        : []),
                      ...(plinkResultForStudio.hardy_path
                        ? [
                            {
                              label: "Open hardy",
                              href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.hardy_path)}`,
                            },
                          ]
                        : []),
                      ...(plinkResultForStudio.log_path
                        ? [
                            {
                              label: "Open log",
                              href: `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(plinkResultForStudio.log_path)}`,
                            },
                          ]
                        : []),
                    ]}
                  />
                </>
              )
            ) : (
              <p className="emptyState">No PLINK result is available yet. Configure the run and execute it from this card.</p>
            )}
          </div>
        </section>
      ) : null,
    table: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>Variant Table</h2>
            <span className="pill">{searchedAnnotations.length} rows</span>
          </div>
          <div className="studioCanvasBody">
            <StudioSimpleList
              items={filteringSummary.map((item) => ({
                label: item.label,
                detail: item.detail,
              }))}
            />
            <div className="oeAnnotationControls">
              <label className="field">
                <span>Search gene / consequence / ClinVar</span>
                <input
                  value={annotationSearch}
                  onChange={(event) => {
                    setAnnotationSearch(event.target.value);
                    setSelectedAnnotationIndex(0);
                  }}
                  placeholder="e.g. PALMD, missense_variant, benign"
                />
              </label>
            </div>
            <VariantTable
              items={searchedAnnotations}
              onSelect={(item) => {
                const nextIndex = searchedAnnotations.findIndex(
                  (candidate) =>
                    candidate.contig === item.contig &&
                    candidate.pos_1based === item.pos_1based &&
                    candidate.rsid === item.rsid,
                );
                setSelectedAnnotationIndex(nextIndex >= 0 ? nextIndex : 0);
                setActiveStudioView("annotations");
              }}
            />
          </div>
        </section>
      ) : null,
    symbolic: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>Symbolic ALT Review</h2>
          </div>
          <div className="studioCanvasBody">
            <p className="emptyState acmgNote">
              Symbolic ALT records are separated here so they are not over-interpreted as ordinary SNV/indel calls.
            </p>
            {symbolicAnnotations.length ? (
              <div className="resultList">
                {symbolicAnnotations.map((item) => (
                  <button
                    type="button"
                    key={`${item.contig}-${item.pos_1based}-${item.rsid}-symbolic`}
                    className="resultListItem"
                    onClick={() => {
                      const nextIndex = searchedAnnotations.findIndex(
                        (candidate) =>
                          candidate.contig === item.contig &&
                          candidate.pos_1based === item.pos_1based &&
                          candidate.rsid === item.rsid,
                      );
                      setSelectedAnnotationIndex(nextIndex >= 0 ? nextIndex : 0);
                      setActiveStudioView("annotations");
                    }}
                  >
                    <strong>
                      {item.contig}:{item.pos_1based} | {item.gene || "Unknown"} | {item.alts.join(",")}
                    </strong>
                    <span>
                      {summarizeLabel(item.consequence, "Unclassified")} | genotype {item.genotype}
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <p className="emptyState">No symbolic ALT records are present in the current annotated subset.</p>
            )}
          </div>
        </section>
      ) : null,
    roh: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>ROH / Recessive Review</h2>
          </div>
          <div className="studioCanvasBody">
            <p className="emptyState acmgNote">
              This is a homozygous-alt review and ROH-style heuristic from the current annotated subset. A full
              production workflow should add `bcftools roh` on the complete callset.
            </p>
            <div className="resultSectionSplit">
              <article className="miniCard">
                <h3>ROH-style segments</h3>
                {rohCandidates.segments.length ? (
                  <div className="resultList">
                    {rohCandidates.segments.map((segment) => (
                      <article key={segment.label} className="resultListItem resultListStatic">
                        <strong>{segment.label}</strong>
                        <span>
                          {segment.count} markers | span {segment.spanMb}
                          {segment.quality != null ? ` | quality ${segment.quality.toFixed(1)}` : ""}
                          {segment.sample ? ` | ${segment.sample}` : ""}
                        </span>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="emptyState">No multi-site homozygous stretches were detected in the current subset.</p>
                )}
              </article>
              <article className="miniCard">
                <h3>Recessive-model candidates</h3>
                {recessiveShortlist.length ? (
                  <div className="resultList">
                    {recessiveShortlist.map(({ item, score, inRoh }) => (
                      <button
                        type="button"
                        key={`${item.contig}-${item.pos_1based}-${item.rsid}-roh`}
                        className="resultListItem"
                        onClick={() => {
                          const nextIndex = searchedAnnotations.findIndex(
                            (candidate) =>
                              candidate.contig === item.contig &&
                              candidate.pos_1based === item.pos_1based &&
                              candidate.rsid === item.rsid,
                          );
                          setSelectedAnnotationIndex(nextIndex >= 0 ? nextIndex : 0);
                          setActiveStudioView("annotations");
                        }}
                      >
                        <strong>
                          {item.gene || "Unknown"} | {item.contig}:{item.pos_1based}
                        </strong>
                        <span>
                          score {score} | genotype {item.genotype}
                          {inRoh ? " | inside ROH" : ""}
                          {" | "}
                          {summarizeLabel(item.consequence, "Unclassified")} | gnomAD {item.gnomad_af || "n/a"}
                        </span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="emptyState">No homozygous alternate candidates are present in the current annotated subset.</p>
                )}
              </article>
            </div>
          </div>
        </section>
      ) : null,
    clinvar: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>ClinVar Review</h2>
          </div>
          <div className="studioCanvasBody">
            <div className="resultSectionSplit">
              <article className="miniCard">
                <h3>Clinical significance mix</h3>
                <DistributionList items={clinvarCounts} emptyLabel="No ClinVar-style labels were found." />
              </article>
              <article className="miniCard">
                <h3>Representative records</h3>
                <div className="resultList">
                  {analysis.annotations.slice(0, 8).map((item) => (
                    <button
                      type="button"
                      key={`${item.contig}-${item.pos_1based}-${item.rsid}-clinvar`}
                      className="resultListItem"
                      onClick={() => {
                        const nextIndex = searchedAnnotations.findIndex(
                          (candidate) =>
                            candidate.contig === item.contig &&
                            candidate.pos_1based === item.pos_1based &&
                            candidate.rsid === item.rsid,
                        );
                        setSelectedAnnotationIndex(nextIndex >= 0 ? nextIndex : 0);
                        setActiveStudioView("annotations");
                      }}
                    >
                      <strong>
                        {item.gene || "Unknown"} | {item.rsid || `${item.contig}:${item.pos_1based}`}
                      </strong>
                      <span>
                        {summarizeLabel(item.clinical_significance, "Unreviewed")} |{" "}
                        {summarizeLabel(item.clinvar_conditions, "No condition")}
                      </span>
                    </button>
                  ))}
                </div>
              </article>
            </div>
          </div>
        </section>
      ) : null,
    vep: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>VEP Consequence</h2>
          </div>
          <div className="studioCanvasBody">
            <div className="resultSectionSplit">
              <article className="miniCard">
                <h3>Top consequences</h3>
                <DistributionList items={consequenceCounts} emptyLabel="No consequence labels were found." />
              </article>
              <article className="miniCard">
                <h3>Gene burden</h3>
                <DistributionList items={geneCounts} emptyLabel="No gene burden summary is available." />
              </article>
            </div>
          </div>
        </section>
      ) : null,
    references: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>References</h2>
          </div>
          <div className="studioCanvasBody">
            <ReferenceListCard items={analysis.references} />
          </div>
        </section>
      ) : null,
    igv: () =>
      analysis ? (
        <section className="studioCanvasPanel">
          <IgvBrowser
            buildGuess={analysis.facts.genome_build_guess ?? null}
            annotations={searchedAnnotations}
            selectedIndex={safeSelectedIndex}
          />
        </section>
      ) : null,
    annotations: () =>
      analysis ? (
        <section className="notebookPanel studioCanvasPanel">
          <div className="notebookHeader">
            <h2>Annotations</h2>
          </div>
          <div className="studioCanvasBody">
            <div className="oeAnnotationControls">
              <label className="field">
                <span>Search gene / consequence / ClinVar</span>
                <input
                  value={annotationSearch}
                  onChange={(event) => {
                    setAnnotationSearch(event.target.value);
                    setSelectedAnnotationIndex(0);
                  }}
                  placeholder="e.g. PALMD, missense_variant, benign"
                />
              </label>
              <label className="field">
                <span>Annotation dropdown</span>
                <select
                  value={safeSelectedIndex}
                  onChange={(event) => setSelectedAnnotationIndex(Number(event.target.value))}
                  disabled={!searchedAnnotations.length}
                >
                  {searchedAnnotations.length ? (
                    searchedAnnotations.map((item, index) => (
                      <option key={`${item.contig}-${item.pos_1based}-${item.rsid}-${index}`} value={index}>
                        {item.gene || "Unknown"} | {item.contig}:{item.pos_1based} | {item.rsid || "no-rsID"} | {item.consequence}
                      </option>
                    ))
                  ) : (
                    <option value={0}>No annotations matched the search</option>
                  )}
                </select>
              </label>
            </div>
            {selectedAnnotation ? <AnnotationDetailCard item={selectedAnnotation} /> : <p className="emptyState">No annotation is available for the current selection.</p>}
          </div>
        </section>
      ) : null,
  };
  const registeredStudioRenderer = activeStudioView ? studioRendererRegistry[activeStudioView] : null;
  const studioContext = useMemo(() => {
    if (!analysis) {
      return {};
    }
    return {
      active_view: activeStudioView,
      qc_summary: {
        pass_rate: qcMetrics?.pass_rate,
        ti_tv: qcMetrics?.transition_transversion_ratio,
        missing_gt_rate: qcMetrics?.missing_gt_rate,
        het_hom_alt_ratio: qcMetrics?.het_hom_alt_ratio,
      },
      clinical_coverage: clinicalCoverage.slice(0, 5),
      filtering_summary: filteringSummary.slice(0, 4),
      symbolic_alt_review: {
        count: symbolicAnnotations.length,
        examples: symbolicAnnotations.slice(0, 5).map((item) => ({
          locus: `${item.contig}:${item.pos_1based}`,
          gene: item.gene,
          alts: item.alts,
          consequence: item.consequence,
          genotype: item.genotype,
        })),
      },
      roh_review: {
        segment_count: analysis.roh_segments?.length ?? 0,
        segments: (analysis.roh_segments ?? []).slice(0, 5).map((segment) => ({
          sample: segment.sample,
          contig: segment.contig,
          start_1based: segment.start_1based,
          end_1based: segment.end_1based,
          length_bp: segment.length_bp,
          marker_count: segment.marker_count,
          quality: segment.quality,
        })),
        recessive_shortlist: recessiveShortlist.slice(0, 6).map(({ item, score, inRoh }) => ({
          locus: `${item.contig}:${item.pos_1based}`,
          gene: item.gene,
          rsid: item.rsid,
          consequence: item.consequence,
          genotype: item.genotype,
          gnomad_af: item.gnomad_af,
          score,
          in_roh: inRoh,
        })),
      },
      candidate_variants: candidateVariants.slice(0, 6).map(({ item, score, inRoh }) => ({
        locus: `${item.contig}:${item.pos_1based}`,
        gene: item.gene,
        rsid: item.rsid,
        consequence: item.consequence,
        clinical_significance: item.clinical_significance,
        gnomad_af: item.gnomad_af,
        score,
        in_roh: inRoh,
      })),
      clinvar_review: clinvarCounts.slice(0, 8),
      vep_consequence: consequenceCounts.slice(0, 10),
      snpeff_preview: analysis?.snpeff_result
        ? {
            genome: analysis.snpeff_result.genome,
            parsed_records: analysis.snpeff_result.parsed_records.slice(0, 5).map((record) => ({
              locus: `${record.contig}:${record.pos_1based}`,
              ref: record.ref,
              alt: record.alt,
              ann: record.ann.slice(0, 2).map((ann) => ({
                annotation: ann.annotation,
                impact: ann.impact,
                gene_name: ann.gene_name,
                hgvs_c: ann.hgvs_c,
                hgvs_p: ann.hgvs_p,
              })),
            })),
          }
        : null,
      liftover_preview: analysis?.liftover_result
        ? {
            source_build: analysis.liftover_result.source_build,
            target_build: analysis.liftover_result.target_build,
            lifted_record_count: analysis.liftover_result.lifted_record_count,
            rejected_record_count: analysis.liftover_result.rejected_record_count,
            warnings: analysis.liftover_result.warnings.slice(0, 6),
          }
        : null,
      ldblockshow_preview: analysis?.ldblockshow_result
        ? {
            region: analysis.ldblockshow_result.region,
            svg_path: analysis.ldblockshow_result.svg_path,
            warnings: analysis.ldblockshow_result.warnings.slice(0, 6),
          }
        : null,
      plink_preview: analysis?.plink_result
        ? {
            output_prefix: analysis.plink_result.output_prefix,
            sample_count: analysis.plink_result.sample_count,
            variant_count: analysis.plink_result.variant_count,
            warnings: analysis.plink_result.warnings.slice(0, 6),
          }
        : null,
      selected_annotation: selectedAnnotation
        ? {
            locus: `${selectedAnnotation.contig}:${selectedAnnotation.pos_1based}`,
            gene: selectedAnnotation.gene,
            rsid: selectedAnnotation.rsid,
            consequence: selectedAnnotation.consequence,
            clinical_significance: selectedAnnotation.clinical_significance,
            gnomad_af: selectedAnnotation.gnomad_af,
            hgvsc: selectedAnnotation.hgvsc,
            hgvsp: selectedAnnotation.hgvsp,
          }
        : null,
    };
  }, [
    activeStudioView,
    analysis,
    candidateVariants,
    clinicalCoverage,
    filteringSummary,
    clinvarCounts,
    consequenceCounts,
    qcMetrics,
    recessiveShortlist,
    selectedAnnotation,
    symbolicAnnotations,
  ]);

  function openStudioView(view: StudioView) {
    setActiveStudioView(view);
    window.setTimeout(() => {
      studioCanvasRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 0);
  }

  useEffect(() => {
    const node = chatStreamRef.current;
    if (!node) {
      return;
    }
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, [chatTurns.length]);

  return (
    <main className="shell notebookShell">
      <header className="appTopbar">
        <div className="appBrand">
          <img src="/chatgenome-dna.svg" alt="" className="appBrandIconImage" />
          <span className="appBrandName">ChatGenome</span>
        </div>
        <div className="appCopyright">Copyright 2026. BISPL@KAIST AI, All rights reserved.</div>
      </header>
      <div className="notebookGrid">
        <aside className="notebookPanel sourcePanel">
          <div className="notebookHeader">
            <h2>Sources</h2>
          </div>
          <div className="sourcePanelBody">
            <div className="sourceSplitPanel">
              <section className="sourceSplitSection">
                <div className="sourceSectionHeader">
                  <h3>Sources</h3>
                </div>
                <div className="sourceSectionBody">
                  {sessionMode === "prs" ? (
                    <div className="resultActionRow">
                      <button type="button" className="sourceAddButton" onClick={() => handleAttachClick("prs_summary")}>
                        + Add summary stats
                      </button>
                      <button type="button" className="sourceAddButton" onClick={() => handleAttachClick("prs_target")}>
                        + Add target genotype
                      </button>
                    </div>
                  ) : (
                    <button type="button" className="sourceAddButton" onClick={() => handleAttachClick()}>
                      + Add genomics source
                    </button>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    onChange={handleFileChange}
                    className="hiddenInput"
                  />
                  <div className="sourceList">
                    {sessionMode === "prs" ? (
                      <>
                        <article className={`sourceItem ${prsSummaryFile ? "sourceItemActive" : ""}`}>
                          <div>
                            <strong>{prsSummaryFile?.name ?? "Summary stats slot"}</strong>
                            <p>{prsSummaryFile ? "PRS summary-statistics source" : "Upload a summary statistics file"}</p>
                          </div>
                          <span className="sourceBadge">S</span>
                        </article>
                        <article className={`sourceItem ${prsTargetFile ? "sourceItemActive" : ""}`}>
                          <div>
                            <strong>{prsTargetFile?.name ?? "Target genotype slot"}</strong>
                            <p>{prsTargetFile ? "PRS target genotype VCF" : "Upload a target genotype VCF"}</p>
                          </div>
                          <span className="sourceBadge">T</span>
                        </article>
                      </>
                    ) : attachedFile ? (
                      <article className="sourceItem sourceItemActive">
                        <div>
                          <strong>{attachedFile.name}</strong>
                          <p>
                            {isRawQcFileName(attachedFile.name)
                              ? "Active raw sequencing source"
                              : isSummaryStatsFileName(attachedFile.name) && !isVcfFileName(attachedFile.name)
                                ? "Active summary statistics source"
                                : "Active VCF source"}
                          </p>
                        </div>
                        <span className="sourceBadge">1</span>
                      </article>
                    ) : (
                      <div className="sourceEmpty">
                        <p>Select a session mode with `@mode prs`, `@mode vcf_analysis`, or `@mode raw_sequence`.</p>
                      </div>
                    )}
                  </div>
                  <div className="sourceMeta">
                    <span>Status</span>
                    <strong>{status}</strong>
                  </div>
                  {sourceStatusDetail ? <p className="sourceHint">{sourceStatusDetail}</p> : null}
                  {error ? <p className="errorText">{error}</p> : null}
                </div>
              </section>

              <section className="sourceSplitSection">
                <div className="sourceSectionHeader">
                  <h3>Tool Registry</h3>
                </div>
                <div className="sourceSectionBody">
                <div className="toolRegistryDetails">
                    <button
                      type="button"
                      className="toolRegistrySummary"
                      onClick={() => setToolRegistryOpen((current) => !current)}
                    >
                      Available tools
                      <span className="toolRegistryCount">
                        {activeToolRegistry?.length
                          ? activeToolRegistry.length
                          : toolRegistryLoading
                            ? "…"
                            : 0}
                      </span>
                    </button>
                    {toolRegistryOpen ? (
                      <div className="toolRegistryMenu">
                        {activeToolRegistry?.length ? (
                          activeToolRegistry.map((tool) => (
                            <div key={tool.name} className="toolRegistryItem" title={tool.description}>
                              <span className="toolRegistryName">{displayToolAlias(tool.name)}</span>
                              <span className="toolRegistryTask">{tool.task}</span>
                            </div>
                          ))
                        ) : toolRegistryLoading ? (
                          <p className="toolRegistryEmpty">Loading tool registry from the local backend...</p>
                        ) : (
                          <p className="toolRegistryEmpty">Tool registry is currently unavailable from the local backend.</p>
                        )}
                      </div>
                    ) : null}
                  </div>

                  <div className="toolUsageLog">
                    <button
                      type="button"
                      className="toolRegistrySummary"
                      onClick={() => setSkillRegistryOpen((current) => !current)}
                    >
                      Available Skills
                      <span className="toolRegistryCount">{availableWorkflows.length}</span>
                    </button>
                    {skillRegistryOpen ? (
                      <div className="toolRegistryMenu">
                        {availableWorkflows.length ? (
                          availableWorkflows.map((workflow) => (
                            <div key={workflow.name} className="toolRegistryItem" title={workflow.description}>
                              <span className="toolRegistryName">{`@skill ${workflow.name}`}</span>
                              <span className="toolRegistryTask">{workflow.source_type}</span>
                            </div>
                          ))
                        ) : (
                          <p className="toolRegistryEmpty">No available skill is registered for the current source.</p>
                        )}
                      </div>
                    ) : null}
                  </div>
                </div>
              </section>
            </div>
          </div>
        </aside>

        <section className="notebookPanel chatPanel">
          <div className="notebookHeader">
            <h2>Chat</h2>
            <span className="pill">{chatHeaderStatus}</span>
          </div>
          <div className="chatPanelBody">
            <div ref={chatStreamRef} className="chatStream">
              {chatTurns.length ? (
                chatTurns.map((turn, index) => (
                  <article
                    key={`chat-turn-${index}`}
                    className={turn.role === "user" ? "nbUserPrompt" : "nbAssistantAnswer"}
                  >
                    {turn.role === "user" ? (
                      <p className="summaryText nbAnswerText">{renderUserPromptInline(turn.content)}</p>
                    ) : (
                      <MarkdownAnswer content={turn.content} />
                    )}
                  </article>
                ))
              ) : (
                <div className="chatEmptyState">
                  <h3>Select a mode first</h3>
                  <p>Use <code>@mode prs</code>, <code>@mode vcf_analysis</code>, or <code>@mode raw_sequence</code>, then upload the required source files on the left.</p>
                </div>
              )}
            </div>

            <div className="chatComposerDock">
              <input
                className={composerInputClass}
                value={composerText}
                onChange={(event) => setComposerText(event.target.value)}
                onCompositionStart={() => setIsComposing(true)}
                onCompositionEnd={() => setIsComposing(false)}
                placeholder={
                  hasAttachedSource
                    ? "Start typing a follow-up question..."
                    : sessionMode
                      ? "Upload the required source files for the selected mode"
                      : "Use @mode prs, @mode vcf_analysis, or @mode raw_sequence first"
                }
                onKeyDown={(event) => {
                  if (isComposing || event.nativeEvent.isComposing) {
                    return;
                  }
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void handleComposerSubmit();
                  }
                }}
              />
              <button type="button" className="chatSendButton" onClick={() => void handleComposerSubmit()}>
                →
              </button>
            </div>
            {detectedGroundingTriggers.length || detectedToolTriggers.length ? (
              <div className="composerTriggerBar">
                {detectedGroundingTriggers.length ? <span className="composerTriggerLabel">Grounded mode</span> : null}
                {detectedGroundingTriggers.map((token) => (
                  <span key={token} className="composerTriggerChip">
                    {token}
                  </span>
                ))}
                {detectedToolTriggers.length ? <span className="composerTriggerLabel">Tool mode</span> : null}
                {detectedToolTriggers.map((token) => (
                  <button key={token} type="button" className="composerToolChip" tabIndex={-1}>
                    {token}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </section>

        <aside className="notebookPanel studioPanel">
          <div className="notebookHeader">
            <h2>Studio</h2>
          </div>
          <div className="studioPanelBody">
            {hasStudioState ? (
              <div className="studioGrid">
                {studioCards.map((card) => (
                  <button
                    type="button"
                    key={card.id}
                    className={`studioCard ${activeStudioView === card.id ? "studioCardActive" : ""}`}
                    onClick={() => openStudioView(card.id)}
                  >
                    <strong>{card.title}</strong>
                    <span>{card.subtitle}</span>
                  </button>
                ))}
              </div>
            ) : null}
            <div className="studioHint">
              {hasStudioState ? "Choose a card to open a result view." : "Studio cards will appear after tool-driven analysis results are ready."}
            </div>
          </div>
        </aside>
      </div>

      {hasStudioState && activeStudioView ? (
        <section ref={studioCanvasRef} className="studioCanvas">
          {registeredStudioRenderer ? registeredStudioRenderer() : null}
        </section>
      ) : null}
    </main>
  );
}
