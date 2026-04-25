"use client";

import { useEffect, useMemo, useRef, useState, type UIEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  buildStudioRendererRegistry,
  resolveStudioDispatchFromPayload,
  resolveStudioRendererKey,
  type StudioRendererDispatch,
} from "./components/studioRenderers";

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
  requested_view?: string | null;
  studio?: { renderer?: string | null } | null;
  draft_answer: string;
  facts: {
    file_name: string;
    record_count: number;
    samples: string[];
    genome_build_guess: string | null;
    warnings: string[];
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

type ToolRunResponse = {
  tool_name: string;
  alias: string;
  result: Record<string, any>;
  studio?: Record<string, any> | null;
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
  result_kind?: string | null;
  requested_view?: StudioView | null;
  studio?: { renderer?: string | null } | null;
  analysis?: SummaryStatsResponse | null;
  qqman_result?: RPlotResponse | null;
  prs_prep_result?: PrsPrepResponse | null;
};

type TextSourceResponse = {
  analysis_id: string;
  source_text_path?: string | null;
  file_name: string;
  media_type: string;
  char_count: number;
  word_count: number;
  line_count: number;
  preview_lines: string[];
  warnings: string[];
  draft_answer: string;
  used_tools: string[];
  tool_registry: AnalysisResponse["tool_registry"];
};

type TextChatResponse = {
  answer: string;
  citations: string[];
  used_fallback: boolean;
  result_kind?: string | null;
  requested_view?: StudioView | null;
  studio?: { renderer?: string | null } | null;
  analysis?: TextSourceResponse | null;
};

type SpreadsheetSourceResponse = {
  analysis_id: string;
  source_spreadsheet_path?: string | null;
  file_name: string;
  workbook_format: string;
  sheet_names: string[];
  selected_sheet?: string | null;
  sheet_count: number;
  sheet_details: Array<Record<string, any>>;
  studio_cards: Array<Record<string, any>>;
  artifacts: Record<string, any>;
  warnings: string[];
  draft_answer: string;
  used_tools: string[];
  tool_registry: AnalysisResponse["tool_registry"];
};

type SpreadsheetChatResponse = {
  answer: string;
  citations: string[];
  used_fallback: boolean;
  result_kind?: string | null;
  requested_view?: StudioView | null;
  studio?: { renderer?: string | null } | null;
  analysis?: SpreadsheetSourceResponse | null;
};

type DicomSourceResponse = {
  analysis_id: string;
  source_dicom_path?: string | null;
  file_name: string;
  file_kind: string;
  metadata_items: Array<Record<string, any>>;
  series: Array<Record<string, any>>;
  studio_cards: Array<Record<string, any>>;
  artifacts: Record<string, any>;
  warnings: string[];
  draft_answer: string;
  used_tools: string[];
  tool_registry: AnalysisResponse["tool_registry"];
};

type DicomChatResponse = {
  answer: string;
  citations: string[];
  used_fallback: boolean;
  result_kind?: string | null;
  requested_view?: StudioView | null;
  studio?: { renderer?: string | null } | null;
  analysis?: DicomSourceResponse | null;
};

type ImageSourceResponse = {
  analysis_id: string;
  source_image_path?: string | null;
  file_name: string;
  file_kind: string;
  width: number;
  height: number;
  format_name: string;
  color_mode: string;
  bit_depth?: number | null;
  exif_data: Record<string, any>;
  metadata_items: Array<Record<string, any>>;
  studio_cards: Array<Record<string, any>>;
  artifacts: Record<string, any>;
  warnings: string[];
  preview_data_url?: string | null;
  draft_answer: string;
  used_tools: string[];
  tool_registry: AnalysisResponse["tool_registry"];
};

type ImageChatResponse = {
  answer: string;
  citations: string[];
  used_fallback: boolean;
  result_kind?: string | null;
  requested_view?: StudioView | null;
  studio?: { renderer?: string | null } | null;
  analysis?: ImageSourceResponse | null;
};

type FhirSourceResponse = {
  analysis_id: string;
  source_fhir_path?: string | null;
  file_name: string;
  file_kind: string;
  resource_type: string;
  resource_count: number;
  patient_summary: string;
  metadata_items: Array<Record<string, any>>;
  studio_cards: Array<Record<string, any>>;
  artifacts: Record<string, any>;
  warnings: string[];
  draft_answer: string;
  used_tools: string[];
  tool_registry: AnalysisResponse["tool_registry"];
};

type FhirChatResponse = {
  answer: string;
  citations: string[];
  used_fallback: boolean;
  result_kind?: string | null;
  requested_view?: StudioView | null;
  studio?: { renderer?: string | null } | null;
  analysis?: FhirSourceResponse | null;
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
  result_kind?: string | null;
  requested_view?: StudioView | null;
  studio?: { renderer?: string | null } | null;
  analysis?: AnalysisResponse | null;
  plink_result?: AnalysisResponse["plink_result"];
  liftover_result?: AnalysisResponse["liftover_result"];
  ldblockshow_result?: AnalysisResponse["ldblockshow_result"];
};

type RawQcChatResponse = {
  answer: string;
  citations: string[];
  used_fallback: boolean;
  result_kind?: string | null;
  requested_view?: StudioView | null;
  studio?: { renderer?: string | null } | null;
  analysis?: RawQcResponse | null;
  samtools_result?: RawQcResponse["samtools_result"];
};

type SourceReadyResponse = {
  source_type: "vcf" | "raw_qc" | "summary_stats" | "text" | "spreadsheet" | "dicom" | "image" | "nifti" | "fhir";
  file_name: string;
  source_path: string;
  file_kind?: string | null;
};

type UploadedSourceItem = {
  id: string;
  name: string;
  sourceType: SourceReadyResponse["source_type"];
  sourcePath: string;
  fileKind?: string | null;
  timestamp: number;
};

type SessionMode = "prs" | "vcf_analysis" | "raw_sequence" | "text_review" | "spreadsheet_review" | "imaging_review" | "image_review" | "fhir_review";

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
  | "dicom_review"
  | "text"
  | "cohort_browser"
  | `sheet::${string}::cohort_browser`
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
  | "annotations"
  | "image_review"
  | "nifti_review"
  | "fhir_browser";

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

function isTextFileName(fileName: string) {
  const lowered = fileName.toLowerCase();
  return (
    lowered.endsWith(".md") ||
    lowered.endsWith(".markdown") ||
    lowered.endsWith(".text") ||
    lowered.endsWith(".note") ||
    lowered.endsWith(".log")
  );
}

function isSpreadsheetFileName(fileName: string) {
  const lowered = fileName.toLowerCase();
  return lowered.endsWith(".xlsx") || lowered.endsWith(".xlsm");
}

function isNiftiFileName(fileName: string) {
  const lowered = fileName.toLowerCase();
  return lowered.endsWith(".nii") || lowered.endsWith(".nii.gz");
}

function isDicomFileName(fileName: string) {
  const lowered = fileName.toLowerCase();
  return lowered.endsWith(".dcm") || lowered.endsWith(".dicom");
}

function isImageFileName(fileName: string) {
  const lowered = fileName.toLowerCase();
  return (
    lowered.endsWith(".png") ||
    lowered.endsWith(".jpg") ||
    lowered.endsWith(".jpeg") ||
    lowered.endsWith(".tiff") ||
    lowered.endsWith(".tif") ||
    lowered.endsWith(".bmp") ||
    lowered.endsWith(".webp")
  );
}

function isFhirFileName(fileName: string) {
  const lowered = fileName.toLowerCase();
  if (lowered.endsWith(".fhir.json") || lowered.endsWith(".fhir.xml") || lowered.endsWith(".ndjson")) {
    return true;
  }
  // Match any JSON/XML file with FHIR-related keywords in the name
  if (lowered.endsWith(".json") || lowered.endsWith(".xml")) {
    return lowered.includes("fhir") || lowered.includes("bundle") || lowered.includes("patient");
  }
  return false;
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
      content: "",
    },
  ]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [rawQcAnalysis, setRawQcAnalysis] = useState<RawQcResponse | null>(null);
  const [summaryStatsAnalysis, setSummaryStatsAnalysis] = useState<SummaryStatsResponse | null>(null);
  const [dicomAnalysis, setDicomAnalysis] = useState<DicomSourceResponse | null>(null);
  const [spreadsheetAnalysis, setSpreadsheetAnalysis] = useState<SpreadsheetSourceResponse | null>(null);
  const [textAnalysis, setTextAnalysis] = useState<TextSourceResponse | null>(null);
  const [imageAnalysis, setImageAnalysis] = useState<ImageSourceResponse | null>(null);
  const [niftiAnalysis, setNiftiAnalysis] = useState<any>(null);
  const [fhirAnalysis, setFhirAnalysis] = useState<FhirSourceResponse | null>(null);
  const [summaryStatsGridRows, setSummaryStatsGridRows] = useState<Array<Record<string, string>>>([]);
  const [summaryStatsHasMore, setSummaryStatsHasMore] = useState(false);
  const [summaryStatsRowsLoading, setSummaryStatsRowsLoading] = useState(false);
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [attachedSourceType, setAttachedSourceType] = useState<SourceReadyResponse["source_type"] | null>(null);
  const [activeSource, setActiveSource] = useState<SourceReadyResponse | null>(null);
  const [uploadedSources, setUploadedSources] = useState<UploadedSourceItem[]>([]);
  const [imageAnalysesByPath, setImageAnalysesByPath] = useState<Record<string, ImageSourceResponse>>({});
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const sessionMode: SessionMode | null = useMemo(() => {
    const src = attachedSourceType;
    if (!src) return null;
    if (src === "vcf") return "vcf_analysis";
    if (src === "raw_qc") return "raw_sequence";
    if (src === "summary_stats") return "prs";
    if (src === "text") return "text_review";
    if (src === "spreadsheet") return "spreadsheet_review";
    if (src === "dicom") return "imaging_review";
    if (src === "image") return "image_review";
    if (src === "fhir") return "fhir_review";
    return null;
  }, [attachedSourceType]);
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
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);
  const [selectedAnnotationIndex, setSelectedAnnotationIndex] = useState(0);
  const [annotationSearch, setAnnotationSearch] = useState("");
  const [composerText, setComposerText] = useState("");
  const [isComposing, setIsComposing] = useState(false);
  const [analysisQa, setAnalysisQa] = useState<AnalysisQuestionTurn[]>([]);
  const [followUpAnswer, setFollowUpAnswer] = useState<string | null>(null);
  const [activeStudioView, setActiveStudioView] = useState<StudioView | null>(null);
  const [studioDispatch, setStudioDispatch] = useState<StudioRendererDispatch>({});
  const [sourceRenderers, setSourceRenderers] = useState<Record<string, StudioRendererDispatch>>({});
  const [igvUnlocked, setIgvUnlocked] = useState(false);
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
  const latestImageUploadRequestRef = useRef(0);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const studioCanvasRef = useRef<HTMLElement | null>(null);
  const chatStreamRef = useRef<HTMLDivElement | null>(null);
  const summaryStatsGridRef = useRef<HTMLDivElement | null>(null);
  const sessionModeModality =
    sessionMode === "vcf_analysis" || sessionMode === "raw_sequence" || sessionMode === "prs" ? "genomics"
    : sessionMode === "imaging_review" || sessionMode === "image_review" ? "medical-image"
    : sessionMode === "spreadsheet_review" ? "spreadsheet"
    : sessionMode === "text_review" ? "text"
    : sessionMode === "fhir_review" ? "clinical-message"
    : null;
  const activeToolRegistry = (() => {
    const base =
      analysis?.tool_registry?.length
        ? analysis.tool_registry
        : rawQcAnalysis?.tool_registry?.length
          ? rawQcAnalysis.tool_registry
          : summaryStatsAnalysis?.tool_registry?.length
            ? summaryStatsAnalysis.tool_registry
          : dicomAnalysis?.tool_registry?.length
            ? dicomAnalysis.tool_registry
          : spreadsheetAnalysis?.tool_registry?.length
            ? spreadsheetAnalysis.tool_registry
          : textAnalysis?.tool_registry?.length
            ? textAnalysis.tool_registry
          : imageAnalysis?.tool_registry?.length
            ? imageAnalysis.tool_registry
          : niftiAnalysis?.tool_registry?.length
            ? niftiAnalysis.tool_registry
          : fhirAnalysis?.tool_registry?.length
            ? fhirAnalysis.tool_registry
          : toolRegistry;
    if (!analysis && !rawQcAnalysis && !summaryStatsAnalysis && !dicomAnalysis && !spreadsheetAnalysis && !textAnalysis && !imageAnalysis && !niftiAnalysis && !fhirAnalysis && sessionModeModality) {
      return (base ?? []).filter((t: any) => t.modality === sessionModeModality);
    }
    return base;
  })();

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

    void loadToolRegistry();
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

  useEffect(() => {
    let cancelled = false;
    async function loadWelcome() {
      try {
        const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/welcome`);
        if (!response.ok) return;
        const payload = (await response.json()) as { message: string };
        if (!cancelled && payload.message) {
          setMessages((prev) => {
            if (prev.length === 1 && prev[0].role === "assistant" && !prev[0].content) {
              return [{ role: "assistant", content: payload.message }];
            }
            return prev;
          });
        }
      } catch {
        // Keep fallback empty message if backend unavailable.
      }
    }
    void loadWelcome();
    return () => { cancelled = true; };
  }, [apiBase]);

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

  function handleSelectUploadedSource(source: UploadedSourceItem) {
    setActiveSource({
      source_type: source.sourceType,
      file_name: source.name,
      source_path: source.sourcePath,
      file_kind: source.fileKind,
    });
    setAttachedSourceType(source.sourceType);
    setAttachedFile(null);
    if (source.sourceType === "image") {
      const cached = source.sourcePath ? imageAnalysesByPath[source.sourcePath] ?? null : null;
      setImageAnalysis(cached);
      if (cached) {
        activateStudioFromPayload(cached, "image_review", "image");
      }
    }
    setStatus(`Active source switched to ${source.name}`);
    setSourcesExpanded(false);
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
      isFhirFileName(file.name)
        ? "fhir"
        : isNiftiFileName(file.name)
        ? "nifti"
        : isRawQcFileName(file.name)
        ? "raw_qc"
        : isDicomFileName(file.name)
          ? "dicom"
        : isImageFileName(file.name)
          ? "image"
        : isSpreadsheetFileName(file.name)
          ? "spreadsheet"
        : isTextFileName(file.name)
          ? "text"
        : isSummaryStatsFileName(file.name) && !isVcfFileName(file.name)
          ? "summary_stats"
          : "vcf";
    const uploadId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    const imageUploadRequestId = guessedSourceType === "image" ? ++latestImageUploadRequestRef.current : null;
    const finalizeUploadedSource = (sourcePath: string, fileKind?: string | null) => {
      setUploadedSources((prev) =>
        prev.map((src) =>
          src.id === uploadId
            ? {
                ...src,
                sourcePath,
                fileKind,
              }
            : src,
        ),
      );
    };
    const removeUploadedSource = () => {
      setUploadedSources((prev) => prev.filter((src) => src.id !== uploadId));
    };
    setAttachedFile(file);
    setAttachedSourceType(guessedSourceType);
    setActiveSource(null);
    // Track all uploaded sources for the source panel
    setUploadedSources((prev) => [
      ...prev,
      {
        id: uploadId,
        name: file.name,
        sourceType: guessedSourceType,
        sourcePath: "",
        timestamp: Date.now(),
      },
    ]);
    setStatus(
      guessedSourceType === "fhir"
        ? "Uploading FHIR source..."
        : guessedSourceType === "spreadsheet"
        ? "Uploading spreadsheet source..."
        : guessedSourceType === "dicom"
          ? "Uploading DICOM source..."
        : guessedSourceType === "image"
          ? "Uploading image source..."
        : guessedSourceType === "nifti"
          ? "Uploading NIfTI source..."
        : guessedSourceType === "text"
          ? "Uploading text source..."
          : guessedSourceType === "summary_stats"
            ? "Uploading summary statistics source..."
            : guessedSourceType === "raw_qc"
              ? "Uploading raw sequencing source..."
              : "Uploading VCF source...",
    );
    // Multimodal: only clear the SAME source type being re-uploaded,
    // keeping all other analyses intact for cross-source grounded chat.
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
    } else if (guessedSourceType === "dicom") {
      setDicomAnalysis(null);
    } else if (guessedSourceType === "spreadsheet") {
      setSpreadsheetAnalysis(null);
    } else if (guessedSourceType === "text") {
      setTextAnalysis(null);
    } else if (guessedSourceType === "image") {
      setImageAnalysis(null);
    } else if (guessedSourceType === "nifti") {
      setNiftiAnalysis(null);
    } else if (guessedSourceType === "fhir") {
      setFhirAnalysis(null);
    }
    setFollowUpAnswer(null);
    // Multimodal: preserve chat history and studio view across source uploads.
    // Only reset annotation selection within the same source type.
    if (guessedSourceType === "vcf") {
      setSelectedAnnotationIndex(0);
      setAnnotationSearch("");
    }
    setError(null);
    try {
      if (guessedSourceType === "fhir") {
        const payload = await handleStartFhirReview(file, { silent: true });
        if (!payload) {
          removeUploadedSource();
          event.target.value = "";
          setPendingUploadRole("default");
          return;
        }
        finalizeUploadedSource(payload.source_fhir_path ?? "", payload.file_kind);
        setActiveSource({
          source_type: "fhir",
          file_name: payload.file_name,
          source_path: payload.source_fhir_path ?? "",
          file_kind: payload.file_kind,
        });
        setStatus("FHIR review ready");
        addMessage({
          role: "assistant",
          content: `FHIR source \`${file.name}\` is loaded and reviewed automatically. Open the Studio FHIR Browser card to inspect patient, medications, labs, and care team.`,
        });
        event.target.value = "";
        setPendingUploadRole("default");
        return;
      }

      if (guessedSourceType === "image") {
        const payload = await handleStartImageReview(file, {
          silent: true,
          requestId: imageUploadRequestId ?? undefined,
        });
        if (!payload) {
          removeUploadedSource();
          event.target.value = "";
          setPendingUploadRole("default");
          return;
        }
        finalizeUploadedSource(payload.source_image_path ?? "", payload.file_kind);
        if (imageUploadRequestId !== null && imageUploadRequestId !== latestImageUploadRequestRef.current) {
          event.target.value = "";
          setPendingUploadRole("default");
          return;
        }
        setActiveSource({
          source_type: "image",
          file_name: payload.file_name,
          source_path: payload.source_image_path ?? "",
          file_kind: payload.file_kind,
        });
        setStatus("Image review ready");
        addMessage({
          role: "assistant",
          content: `Image source \`${file.name}\` is loaded and reviewed automatically. Open the Studio Image Review card to inspect metadata, EXIF data, and thumbnail preview.`,
        });
        event.target.value = "";
        setPendingUploadRole("default");
        return;
      }

      if (guessedSourceType === "nifti") {
        const payload = await handleStartNiftiReview(file, { silent: true });
        if (!payload) {
          removeUploadedSource();
          event.target.value = "";
          setPendingUploadRole("default");
          return;
        }
        finalizeUploadedSource(payload.source_nifti_path ?? "", payload.file_kind);
        setActiveSource({
          source_type: "nifti",
          file_name: payload.file_name,
          source_path: payload.source_nifti_path ?? "",
          file_kind: payload.file_kind,
        });
        setStatus("NIfTI review ready");
        addMessage({
          role: "assistant",
          content: `NIfTI source \`${file.name}\` is loaded and reviewed automatically. Open the Studio NIfTI Review card to inspect volume metadata and interactive viewer.`,
        });
        event.target.value = "";
        setPendingUploadRole("default");
        return;
      }

      if (guessedSourceType === "text") {
        const payload = await handleStartTextReview(file, { silent: true });
        if (!payload) {
          removeUploadedSource();
          event.target.value = "";
          setPendingUploadRole("default");
          return;
        }
        finalizeUploadedSource(payload.source_text_path ?? "");
        setActiveSource({
          source_type: "text",
          file_name: payload.file_name,
          source_path: payload.source_text_path ?? "",
        });
        setStatus("Text review ready");
        addMessage({
          role: "assistant",
          content: `Text source \`${file.name}\` is loaded and reviewed automatically. Open the Studio text review card to inspect the rendered document.`,
        });
        event.target.value = "";
        setPendingUploadRole("default");
        return;
      }

      if (guessedSourceType === "spreadsheet") {
        const payload = await handleStartSpreadsheetReview(file, { silent: true });
        if (!payload) {
          removeUploadedSource();
          event.target.value = "";
          setPendingUploadRole("default");
          return;
        }
        finalizeUploadedSource(payload.source_spreadsheet_path ?? "");
        setActiveSource({
          source_type: "spreadsheet",
          file_name: payload.file_name,
          source_path: payload.source_spreadsheet_path ?? "",
        });
        setStatus("Spreadsheet review ready");
        addMessage({
          role: "assistant",
          content: `Spreadsheet source \`${file.name}\` is loaded and reviewed automatically. Open the Studio cohort browser card to inspect sheets, schema, and missingness.`,
        });
        event.target.value = "";
        setPendingUploadRole("default");
        return;
      }

      if (guessedSourceType === "dicom") {
        const payload = await handleStartDicomReview(file, { silent: true });
        if (!payload) {
          removeUploadedSource();
          event.target.value = "";
          setPendingUploadRole("default");
          return;
        }
        finalizeUploadedSource(payload.source_dicom_path ?? "", payload.file_kind);
        setActiveSource({
          source_type: "dicom",
          file_name: payload.file_name,
          source_path: payload.source_dicom_path ?? "",
          file_kind: payload.file_kind,
        });
        setStatus("DICOM review ready");
        addMessage({
          role: "assistant",
          content: `DICOM source \`${file.name}\` is loaded and reviewed automatically. Open the Studio DICOM Review card to inspect metadata and the interactive viewer.`,
        });
        event.target.value = "";
        setPendingUploadRole("default");
        return;
      }

      if (guessedSourceType === "summary_stats") {
        const payload = await handleStartSummaryStats(file, { silent: true });
        if (!payload) {
          removeUploadedSource();
          event.target.value = "";
          setPendingUploadRole("default");
          return;
        }
        finalizeUploadedSource(payload.source_stats_path ?? "");
        setActiveSource({
          source_type: "summary_stats",
          file_name: payload.file_name,
          source_path: payload.source_stats_path ?? "",
        });
        setStatus("Summary statistics review ready");
        addMessage({
          role: "assistant",
          content: `Summary statistics source \`${file.name}\` is loaded and reviewed automatically. Open the Studio summary card to inspect preview rows, warnings, and schema clues.`,
        });
        event.target.value = "";
        setPendingUploadRole("default");
        return;
      }

      if (guessedSourceType === "raw_qc") {
        const payload = await handleStartRawQc(file, { silent: true });
        if (!payload) {
          removeUploadedSource();
          event.target.value = "";
          setPendingUploadRole("default");
          return;
        }
        finalizeUploadedSource(payload.source_raw_path ?? "", payload.facts.file_kind);
        setActiveSource({
          source_type: "raw_qc",
          file_name: payload.facts.file_name,
          source_path: payload.source_raw_path ?? "",
          file_kind: payload.facts.file_kind,
        });
        setStatus("Raw QC review ready");
        addMessage({
          role: "assistant",
          content: `Raw sequencing source \`${file.name}\` is loaded and reviewed automatically. Open the Studio FastQC Review card to inspect QC modules and report artifacts.`,
        });
        event.target.value = "";
        setPendingUploadRole("default");
        return;
      }

      const source = await uploadActiveSource(file);
      finalizeUploadedSource(source.source_path, source.file_kind);
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
      removeUploadedSource();
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
            ? `Summary statistics source \`${file.name}\` is loaded into the PRS session. Upload a target genotype VCF into the second slot, then run \`@prs_prep\`.`
            : hadPreparedPrsScoreFile
              ? `Target genotype VCF source \`${file.name}\` is loaded into the PRS session. You can run \`@plink score\` now.`
              : `Target genotype VCF source \`${file.name}\` is loaded into the PRS session. Prepare the summary-statistics source with \`@prs_prep\`, then run \`@plink score\`.`,
      });
      event.target.value = "";
      setPendingUploadRole("default");
      return;
    }
    else if (!hadPreparedPrsScoreFile) {
      const vcfPayload = await handleStartAnalysis("representative", annotationLimit, file, { silent: true });
      if (vcfPayload) {
        addMessage({
          role: "assistant",
          content: `VCF source \`${file.name}\` is loaded and analyzed automatically. Open the Studio QC Summary card to inspect quality metrics and annotations.`,
        });
      }
    }
    else {
      setStatus("VCF source ready");
      addMessage({
        role: "assistant",
        content: `VCF source \`${file.name}\` is loaded as a target genotype source. You can run \`@plink score\` now.`,
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
    if (normalized === "vcfinterpret" || normalized === "vcfinterpretation" || normalized === "vcf_interpret" || normalized === "vcf_interpretation") {
      return "Running VCF interpretation...";
    }
    if (normalized === "vcfreview" || normalized === "vcf_review") {
      return "Running VCF review...";
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
    if (normalized === "vcfinterpret" || normalized === "vcfinterpretation" || normalized === "vcf_interpret" || normalized === "vcf_interpretation") {
      return "VCF interpretation ready";
    }
    if (normalized === "vcfreview" || normalized === "vcf_review") {
      return "VCF review ready";
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
    if (normalized === "vcfinterpret" || normalized === "vcfinterpretation" || normalized === "vcf_interpret" || normalized === "vcf_interpretation") {
      return "VCF interpretation failed";
    }
    if (normalized === "vcfreview" || normalized === "vcf_review") {
      return "VCF review failed";
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

    if (alias === "vcfqc" || alias === "vcf_qc") {
      const vcfPath = analysis?.source_vcf_path ?? (preAnalysisSource.source_type === "vcf" ? preAnalysisSource.source_path : null);
      if (!vcfPath) {
        addMessage({ role: "assistant", content: "Active VCF source가 없습니다. VCF 파일을 먼저 업로드해 주세요." });
        return;
      }
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/tools/vcf_qc_tool/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ payload: { vcf_path: vcfPath, max_examples: 8 } }),
      });
      if (!response.ok) throw new Error(await response.text());
      const toolResult = (await response.json()) as ToolRunResponse;
      const facts = toolResult.result?.facts;
      setAnalysis((current) => current ? { ...current, facts: facts ?? current.facts } : current);
      activateStudioFromPayload({ studio: { renderer: "qc" }, requested_view: "qc" }, undefined, "vcf");
      setStatus("VCF QC complete");
      addMessage({
        role: "assistant",
        content: `VCF QC re-run complete for \`${preAnalysisSource.file_name}\`.\n\n- Records: ${facts?.record_count ?? "n/a"}\n- Build: ${facts?.genome_build_guess ?? "unknown"}`,
      });
      return;
    }

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
      activateStudioFromPayload({ result_kind: "liftover_result" }, "liftover", "vcf");
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
      activateStudioFromPayload({ result_kind: "samtools_result" }, "samtools", "raw_qc");
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
            "- Run `@prs_prep`.\n" +
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
      activateStudioFromPayload({ result_kind: "plink_result" }, "plink", "vcf");
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
      activateStudioFromPayload({ result_kind: "snpeff_result" }, "snpeff", "vcf");
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

    if ((alias === "vcfinterpretation" || alias === "vcf_interpretation" || alias === "vcf_interpret" || alias === "vcfinterpret") && preAnalysisSource.source_type === "vcf") {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/tools/vcf_interpretation/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          payload: {
            vcf_path: preAnalysisSource.source_path,
            facts: analysis?.source_vcf_path === preAnalysisSource.source_path ? analysis.facts : undefined,
            scope: "representative",
            ranking_limit: 8,
          },
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as ToolRunResponse;
      setAnalysis((current) => {
        const baseFacts =
          current?.source_vcf_path === preAnalysisSource.source_path && current?.facts
            ? current.facts
            : (payload.result?.facts as AnalysisResponse["facts"] | undefined) ?? current?.facts;
        if (!baseFacts) {
          return current;
        }
        return {
          analysis_id: current?.analysis_id ?? `vcf-interpretation-${Date.now()}`,
          source_vcf_path: preAnalysisSource.source_path,
          draft_answer: current?.draft_answer ?? "",
          facts: baseFacts,
          annotations: (payload.result?.annotations as VariantAnnotation[]) ?? current?.annotations ?? [],
          snpeff_result: current?.snpeff_result ?? null,
          plink_result: current?.plink_result ?? null,
          liftover_result: current?.liftover_result ?? null,
          ldblockshow_result: current?.ldblockshow_result ?? null,
          candidate_variants: (payload.result?.candidate_variants as AnalysisResponse["candidate_variants"]) ?? current?.candidate_variants ?? [],
          clinvar_summary: current?.clinvar_summary ?? [],
          consequence_summary: current?.consequence_summary ?? [],
          clinical_coverage_summary: current?.clinical_coverage_summary ?? [],
          filtering_summary: current?.filtering_summary ?? [],
          symbolic_alt_summary: current?.symbolic_alt_summary,
          roh_segments: (payload.result?.roh_segments as AnalysisResponse["roh_segments"]) ?? current?.roh_segments ?? [],
          references: current?.references ?? [],
          used_tools: [
            ...new Set([...(current?.used_tools ?? []), "vcf_interpretation_tool"]),
          ],
          tool_registry: current?.tool_registry ?? toolRegistry ?? [],
        };
      });
      activateStudioFromPayload({ requested_view: "candidates", studio: { renderer: "candidates" } }, "candidates", "vcf");
      setStatus(toolReadyStatus(alias, remainder));
      addMessage({
        role: "assistant",
        content:
          `VCF interpretation was run for the active VCF.\n\n` +
          `- Annotations: ${payload.result?.annotation_count ?? "unknown"}\n` +
          `- ROH segments: ${payload.result?.roh_segment_count ?? "unknown"}\n` +
          `- Ranked candidates: ${payload.result?.candidate_count ?? "unknown"}\n` +
          `- CADD matched: ${payload.result?.cadd_matched_count ?? 0}\n` +
          `- REVEL matched: ${payload.result?.revel_matched_count ?? 0}`,
      });
      return;
    }

    if ((alias === "vcfreview" || alias === "vcf_review") && preAnalysisSource.source_type === "vcf") {
      if (!analysis?.facts) {
        addMessage({
          role: "assistant",
          content: `VCF QC analysis가 아직 로드되지 않았습니다. 파일을 다시 업로드해 주세요.\n\n현재 상태: source=${preAnalysisSource.source_type}, analysis=${analysis ? "loaded" : "null"}`,
        });
        setStatus(toolReadyStatus(alias, remainder));
        return;
      }
      addMessage({ role: "assistant", content: "VCF review 실행 중... ClinVar, consequence, coverage 분석 중입니다." });
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/tools/vcf_review/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          payload: {
            facts: analysis.facts,
            annotations: analysis.annotations ?? [],
            candidate_variants: analysis.candidate_variants ?? [],
            references: analysis.references ?? [],
          },
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as ToolRunResponse;
      setAnalysis((current) =>
        current
          ? {
              ...current,
              clinvar_summary: (payload.result?.clinvar_summary as AnalysisResponse["clinvar_summary"]) ?? current.clinvar_summary,
              consequence_summary: (payload.result?.consequence_summary as AnalysisResponse["consequence_summary"]) ?? current.consequence_summary,
              clinical_coverage_summary:
                (payload.result?.clinical_coverage_summary as AnalysisResponse["clinical_coverage_summary"]) ??
                current.clinical_coverage_summary,
              symbolic_alt_summary:
                (payload.result?.symbolic_alt_summary as AnalysisResponse["symbolic_alt_summary"]) ?? current.symbolic_alt_summary,
              draft_answer: typeof payload.result?.draft_answer === "string" ? payload.result.draft_answer : current.draft_answer,
              candidate_variants:
                (payload.result?.candidate_variants as AnalysisResponse["candidate_variants"]) ?? current.candidate_variants,
              used_tools: [...new Set([...(current.used_tools ?? []), "vcf_review_tool"])],
            }
          : current,
      );
      activateStudioFromPayload({ requested_view: "clinvar", studio: { renderer: "clinvar" } }, "clinvar", "vcf");
      setStatus(toolReadyStatus(alias, remainder));
      addMessage({
        role: "assistant",
        content:
          `VCF review was run for the active interpretation state.\n\n` +
          `- ClinVar buckets: ${Array.isArray(payload.result?.clinvar_summary) ? payload.result.clinvar_summary.length : 0}\n` +
          `- Consequence buckets: ${Array.isArray(payload.result?.consequence_summary) ? payload.result.consequence_summary.length : 0}\n` +
          `- Coverage rows: ${Array.isArray(payload.result?.clinical_coverage_summary) ? payload.result.clinical_coverage_summary.length : 0}\n` +
          `- Symbolic ALT count: ${payload.result?.symbolic_alt_summary?.count ?? 0}`,
      });
      return;
    }

    if (alias === "igv" && preAnalysisSource.source_type === "vcf") {
      if (!analysis?.facts) {
        addMessage({ role: "assistant", content: "VCF analysis가 아직 로드되지 않았습니다. 파일을 다시 업로드해 주세요." });
        setStatus(toolReadyStatus(alias, remainder));
        return;
      }
      setIgvUnlocked(true);
      setActiveStudioView("igv");
      setStatus(toolReadyStatus(alias, remainder));
      addMessage({ role: "assistant", content: "IGV Plot을 열었습니다." });
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
      activateStudioFromPayload({ result_kind: "ldblockshow_result" }, "ldblockshow", "vcf");
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
      activateStudioFromPayload({ result_kind: "qqman_result" }, "qqman", "summary_stats");
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

    // No frontend handler matched — delegate to backend chat for the active source type.
    const toolCommand = `@${alias}${remainder ? " " + remainder : ""}`;
    if (preAnalysisSource.source_type === "vcf" && analysis) {
      await handleAskAnalysisQuestion(toolCommand, analysis);
      return;
    }
    if (preAnalysisSource.source_type === "raw_qc" && rawQcAnalysis) {
      await handleAskRawQcQuestion(toolCommand, rawQcAnalysis);
      return;
    }
    if (preAnalysisSource.source_type === "summary_stats" && summaryStatsAnalysis) {
      await handleAskSummaryStatsQuestion(toolCommand, summaryStatsAnalysis);
      return;
    }
    if (preAnalysisSource.source_type === "text" && textAnalysis) {
      await handleAskTextQuestion(toolCommand, textAnalysis);
      return;
    }
    if (preAnalysisSource.source_type === "dicom" && dicomAnalysis) {
      await handleAskDicomQuestion(toolCommand, dicomAnalysis);
      return;
    }
    if (preAnalysisSource.source_type === "spreadsheet" && spreadsheetAnalysis) {
      await handleAskSpreadsheetQuestion(toolCommand, spreadsheetAnalysis);
      return;
    }
    if (preAnalysisSource.source_type === "image" && imageAnalysis) {
      await handleAskImageQuestion(toolCommand, imageAnalysis);
      return;
    }
    if (preAnalysisSource.source_type === "nifti" && niftiAnalysis) {
      await handleAskNiftiQuestion(toolCommand, niftiAnalysis);
      return;
    }
    if (preAnalysisSource.source_type === "fhir" && fhirAnalysis) {
      await handleAskFhirQuestion(toolCommand, fhirAnalysis);
      return;
    }

    // If source is active but typed analysis payload is missing, avoid a false incompatibility message.
    addMessage({
      role: "assistant",
      content:
        "The current source is active, but its analysis payload is not ready yet. " +
        "Please wait for auto-review to finish (or re-upload the file) and try the tool again.",
    });
    return;

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
      activateStudioFromPayload({ requested_view: "rawqc" }, "rawqc", "raw_qc");
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
      activateStudioFromPayload({ requested_view: "sumstats" }, "sumstats", "summary_stats");
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

  async function handleStartTextReview(
    file: File,
    options?: { silent?: boolean },
  ): Promise<TextSourceResponse | null> {
    const silent = options?.silent ?? false;
    setError(null);
    if (!silent) {
      addMessage({
        role: "assistant",
        content: "Text note를 읽고 preview와 기본 길이 통계를 만들고 있습니다.",
        kind: "status",
      });
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/text/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload: TextSourceResponse = await response.json();
      setTextAnalysis(payload);
      activateStudioFromPayload({ requested_view: "text" }, "text", "text");
      setStatus("Text review ready");
      setComposerText("");
      return payload;
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Text review failed");
      addMessage({
        role: "assistant",
        content: `Text review 중 오류가 발생했습니다: ${message}`,
      });
      return null;
    }
  }

  async function handleStartDicomReview(
    file: File,
    options?: { silent?: boolean },
  ): Promise<DicomSourceResponse | null> {
    const silent = options?.silent ?? false;
    setError(null);
    setStatus("Running DICOM review...");
    if (!silent) {
      addMessage({
        role: "assistant",
        content: "DICOM 파일을 읽고 metadata, preview, series summary를 만들고 있습니다.",
        kind: "status",
      });
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/dicom/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload: DicomSourceResponse = await response.json();
      setDicomAnalysis(payload);
      activateStudioFromPayload(payload, "dicom_review", "dicom");
      setStatus("DICOM review ready");
      setComposerText("");
      return payload;
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("DICOM review failed");
      addMessage({
        role: "assistant",
        content: `DICOM review 중 오류가 발생했습니다: ${message}`,
      });
      return null;
    }
  }

  async function handleStartImageReview(
    file: File,
    options?: { silent?: boolean; requestId?: number },
  ): Promise<ImageSourceResponse | null> {
    const silent = options?.silent ?? false;
    const isStaleRequest = () =>
      options?.requestId != null && options.requestId !== latestImageUploadRequestRef.current;
    setError(null);
    setStatus("Running image review...");
    if (!silent) {
      addMessage({
        role: "assistant",
        content: "이미지 파일을 읽고 metadata, EXIF, thumbnail을 추출하고 있습니다.",
        kind: "status",
      });
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/image/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload: ImageSourceResponse = await response.json();
      if (payload.source_image_path) {
        setImageAnalysesByPath((current) => ({
          ...current,
          [payload.source_image_path as string]: payload,
        }));
      }
      if (isStaleRequest()) {
        return payload;
      }
      setImageAnalysis(payload);
      activateStudioFromPayload(payload, "image_review", "image");
      setStatus("Image review ready");
      setComposerText("");
      return payload;
    } catch (caught) {
      if (isStaleRequest()) {
        return null;
      }
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Image review failed");
      addMessage({
        role: "assistant",
        content: `Image review 중 오류가 발생했습니다: ${message}`,
      });
      return null;
    }
  }

  async function handleStartNiftiReview(
    file: File,
    options?: { silent?: boolean },
  ): Promise<any | null> {
    const silent = options?.silent ?? false;
    setError(null);
    setStatus("Running NIfTI review...");
    if (!silent) {
      addMessage({
        role: "assistant",
        content: "Reading NIfTI volume and extracting metadata, orientation, and slice previews.",
        kind: "status",
      });
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/nifti/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload = await response.json();
      setNiftiAnalysis(payload);
      activateStudioFromPayload(payload, "nifti_review", "nifti");
      setStatus("NIfTI review ready");
      setComposerText("");
      return payload;
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("NIfTI review failed");
      addMessage({
        role: "assistant",
        content: `NIfTI review failed: ${message}`,
      });
      return null;
    }
  }

  async function handleStartFhirReview(
    file: File,
    options?: { silent?: boolean },
  ): Promise<FhirSourceResponse | null> {
    const silent = options?.silent ?? false;
    setError(null);
    setStatus("Running FHIR review...");
    if (!silent) {
      addMessage({
        role: "assistant",
        content: "FHIR Bundle을 읽고 환자, 약물, 검사, 진료팀 정보를 추출하고 있습니다.",
        kind: "status",
      });
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/fhir/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload: FhirSourceResponse = await response.json();
      setFhirAnalysis(payload);
      activateStudioFromPayload(payload, "fhir_browser", "fhir");
      setStatus("FHIR review ready");
      setComposerText("");
      return payload;
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("FHIR review failed");
      addMessage({
        role: "assistant",
        content: `FHIR review 중 오류가 발생했습니다: ${message}`,
      });
      return null;
    }
  }

  async function handleStartSpreadsheetReview(
    file: File,
    options?: { silent?: boolean },
  ): Promise<SpreadsheetSourceResponse | null> {
    const silent = options?.silent ?? false;
    setError(null);
    setStatus("Running spreadsheet review...");
    if (!silent) {
      addMessage({
        role: "assistant",
        content: "Workbook을 읽고 sheet별 cohort browser artifact를 만들고 있습니다.",
        kind: "status",
      });
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/spreadsheet/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload: SpreadsheetSourceResponse = await response.json();
      setSpreadsheetAnalysis(payload);
      activateStudioFromPayload(payload, "cohort_browser", "spreadsheet");
      setStatus("Spreadsheet review ready");
      setComposerText("");
      return payload;
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Spreadsheet review failed");
      addMessage({
        role: "assistant",
        content: `Spreadsheet review 중 오류가 발생했습니다: ${message}`,
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
      activateStudioFromPayload({ requested_view: "prs_prep", result_kind: "prs_prep_result" }, "prs_prep", "summary_stats");
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
      activateStudioFromPayload(payload, undefined, "vcf");
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

    // @help intercept — show tool guide from SKILL.md (works without a source)
    if (/^@help\s*$/i.test(text)) {
      addMessage({ role: "user", content: text });
      setComposerText("");
      try {
        const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/help`);
        if (!response.ok) throw new Error(await response.text());
        const payload = (await response.json()) as { message: string };
        addMessage({ role: "assistant", content: payload.message });
      } catch (caught) {
        const message = caught instanceof Error ? caught.message : String(caught);
        addMessage({ role: "assistant", content: `@help 조회 중 오류가 발생했습니다: ${message}` });
      }
      return;
    }

    if (!hasAttachedSource) {
      addMessage({ role: "user", content: text });
      addMessage({
        role: "assistant",
        content: "먼저 분석할 소스 파일을 업로드해 주세요. VCF, FASTQ, BAM, DICOM, Excel, TSV, TXT 등 지원됩니다.",
      });
      setComposerText("");
      return;
    }

    // @tool intercept — runs regardless of whether an analysis is already loaded
    const earlyToolMatch = text.match(/^@([A-Za-z0-9_-]+)(?:\s+(.*))?$/);
    if (earlyToolMatch) {
      const alias = earlyToolMatch[1].trim();
      const remainder = (earlyToolMatch[2] ?? "").trim();
      const isHelp = /^(help|--help|-h)(\s+.*)?$/i.test(remainder);
      setComposerText("");

      if (isHelp) {
        addMessage({ role: "user", content: text });
        try {
          const helpText = await fetchToolHelpText(alias);
          addMessage({ role: "assistant", content: helpText });
        } catch (caught) {
          const message = caught instanceof Error ? caught.message : String(caught);
          addMessage({ role: "assistant", content: `\`@${alias} help\` 조회 중 오류가 발생했습니다: ${message}` });
        }
        return;
      }

      addMessage({ role: "user", content: text });
      try {
        await runPreAnalysisTool(alias.toLowerCase(), remainder);
      } catch (caught) {
        const message = caught instanceof Error ? caught.message : String(caught);
        setStatus(toolFailedStatus(alias.toLowerCase(), remainder));
        addMessage({ role: "assistant", content: `\`@${alias}\` 실행 중 오류가 발생했습니다: ${message}` });
      }
      return;
    }

    if (!analysis && !rawQcAnalysis && !summaryStatsAnalysis && !dicomAnalysis && !spreadsheetAnalysis && !textAnalysis && !imageAnalysis && !niftiAnalysis && !fhirAnalysis) {
      addMessage({ role: "user", content: text });
      setComposerText("");
      addMessage({
        role: "assistant",
        content: "소스 파일이 업로드되었지만 아직 분석이 진행되지 않았습니다. 파일 업로드 후 자동 분석이 완료될 때까지 잠시 기다려 주세요.",
      });
      return;
    }

    // Count active sources — use multimodal endpoint when >1 source is loaded
    const activeSourceCount = [analysis, rawQcAnalysis, summaryStatsAnalysis, dicomAnalysis, spreadsheetAnalysis, textAnalysis, imageAnalysis, niftiAnalysis, fhirAnalysis].filter(Boolean).length;

    if (activeSourceCount > 1) {
      setComposerText("");
      await handleAskMultimodalQuestion(text);
      return;
    }

    // Single source fallback (legacy endpoints for backward compatibility)
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

    if (textAnalysis) {
      setComposerText("");
      await handleAskTextQuestion(text);
      return;
    }

    if (dicomAnalysis) {
      setComposerText("");
      await handleAskDicomQuestion(text);
      return;
    }

    if (spreadsheetAnalysis) {
      setComposerText("");
      await handleAskSpreadsheetQuestion(text);
      return;
    }

    if (imageAnalysis) {
      setComposerText("");
      await handleAskImageQuestion(text);
      return;
    }

    if (niftiAnalysis) {
      setComposerText("");
      await handleAskNiftiQuestion(text);
      return;
    }

    if (fhirAnalysis) {
      setComposerText("");
      await handleAskFhirQuestion(text);
      return;
    }

    addMessage({ role: "user", content: text });
    addMessage({
      role: "assistant",
      content: "소스 파일을 업로드해 주세요.",
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
      activateStudioFromPayload(payload, undefined, "vcf");
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
        activateStudioFromPayload({ ...payload, result_kind: "plink_result" }, "plink", "vcf");
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
        activateStudioFromPayload({ ...payload, result_kind: "liftover_result" }, "liftover", "vcf");
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
        activateStudioFromPayload({ ...payload, result_kind: "ldblockshow_result" }, "ldblockshow", "vcf");
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
        throw new Error("Run @prs_prep first and provide a prepared score file before PLINK score mode.");
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
      activateStudioFromPayload({ result_kind: "plink_result" }, "plink", "vcf");
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
        activateStudioFromPayload({ ...payload, result_kind: "samtools_result" }, "samtools", "raw_qc");
      } else if (payload.requested_view) {
        activateStudioFromPayload(payload, undefined, "raw_qc");
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
      activateStudioFromPayload(payload, undefined, "summary_stats");
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

  async function handleAskTextQuestion(questionText?: string, analysisOverride?: TextSourceResponse | null) {
    const text = questionText?.trim() ?? "";
    const activeAnalysis = analysisOverride ?? textAnalysis;
    if (!text || !activeAnalysis) {
      return;
    }

    setStatus("Generating answer...");
    setAnalysisQa((current) => [...current, { role: "user", content: text }]);

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/text`, {
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
      const payload: TextChatResponse = await response.json();
      if (payload.analysis) {
        setTextAnalysis(payload.analysis);
      }
      activateStudioFromPayload(payload, "text", "text");
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

  async function handleAskDicomQuestion(questionText?: string, analysisOverride?: DicomSourceResponse | null) {
    const text = questionText?.trim() ?? "";
    const activeAnalysis = analysisOverride ?? dicomAnalysis;
    if (!text || !activeAnalysis) {
      return;
    }

    setStatus("Generating answer...");
    setAnalysisQa((current) => [...current, { role: "user", content: text }]);

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/dicom`, {
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
      const payload: DicomChatResponse = await response.json();
      if (payload.analysis) {
        setDicomAnalysis(payload.analysis);
      }
      activateStudioFromPayload(payload, "dicom_review", "dicom");
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

  async function handleAskSpreadsheetQuestion(questionText?: string, analysisOverride?: SpreadsheetSourceResponse | null) {
    const text = questionText?.trim() ?? "";
    const activeAnalysis = analysisOverride ?? spreadsheetAnalysis;
    if (!text || !activeAnalysis) {
      return;
    }

    setStatus("Generating answer...");
    setAnalysisQa((current) => [...current, { role: "user", content: text }]);

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/spreadsheet`, {
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
      const payload: SpreadsheetChatResponse = await response.json();
      if (payload.analysis) {
        setSpreadsheetAnalysis(payload.analysis);
      }
      activateStudioFromPayload(payload, "cohort_browser", "spreadsheet");
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

  async function handleAskImageQuestion(questionText?: string, analysisOverride?: ImageSourceResponse | null) {
    const text = questionText?.trim() ?? "";
    const activeAnalysis = analysisOverride ?? imageAnalysis;
    if (!text || !activeAnalysis) {
      return;
    }

    setStatus("Generating answer...");
    setAnalysisQa((current) => [...current, { role: "user", content: text }]);

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/image`, {
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
      const payload: ImageChatResponse = await response.json();
      if (payload.analysis) {
        setImageAnalysis(payload.analysis);
        if (payload.analysis.source_image_path) {
          setImageAnalysesByPath((current) => ({
            ...current,
            [payload.analysis!.source_image_path as string]: payload.analysis!,
          }));
        }
      }
      activateStudioFromPayload(payload, "image_review", "image");
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

  async function handleAskNiftiQuestion(questionText?: string, analysisOverride?: any) {
    const text = questionText?.trim() ?? "";
    const activeAnalysis = analysisOverride ?? niftiAnalysis;
    if (!text || !activeAnalysis) {
      return;
    }

    setStatus("Generating answer...");
    setAnalysisQa((current) => [...current, { role: "user", content: text }]);

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/nifti`, {
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
      const payload = await response.json();
      if (payload.analysis) {
        setNiftiAnalysis(payload.analysis);
      }
      activateStudioFromPayload(payload, "nifti_review", "nifti");
      setAnalysisQa((current) => [...current, { role: "assistant", content: payload.answer }]);
      setFollowUpAnswer(payload.answer);
      setStatus("Answer ready");
    } catch (caught) {
      const msg = caught instanceof Error ? caught.message : String(caught);
      setAnalysisQa((current) => [
        ...current,
        { role: "assistant", content: `Error: ${msg}` },
      ]);
      setStatus("Answer failed");
    }
  }

  async function handleAskFhirQuestion(questionText?: string, analysisOverride?: FhirSourceResponse | null) {
    const text = questionText?.trim() ?? "";
    const activeAnalysis = analysisOverride ?? fhirAnalysis;
    if (!text || !activeAnalysis) {
      return;
    }

    setStatus("Generating answer...");
    setAnalysisQa((current) => [...current, { role: "user", content: text }]);

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/fhir`, {
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
      const payload: FhirChatResponse = await response.json();
      if (payload.analysis) {
        setFhirAnalysis(payload.analysis);
      }
      activateStudioFromPayload(payload, "fhir_browser", "fhir");
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

  async function handleAskMultimodalQuestion(questionText?: string) {
    const text = questionText?.trim() ?? "";
    if (!text) return;

    setStatus("Generating answer...");
    setAnalysisQa((current) => [...current, { role: "user", content: text }]);

    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/multimodal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          vcf_analysis: analysis ?? undefined,
          raw_qc_analysis: rawQcAnalysis ?? undefined,
          summary_stats_analysis: summaryStatsAnalysis ?? undefined,
          text_analysis: textAnalysis ?? undefined,
          spreadsheet_analysis: spreadsheetAnalysis ?? undefined,
          dicom_analysis: dicomAnalysis ?? undefined,
          image_analysis: imageAnalysis ?? undefined,
          nifti_analysis: niftiAnalysis ?? undefined,
          fhir_analysis: fhirAnalysis ?? undefined,
          primary_source_type: (() => {
            // Infer focused source from active studio view
            const v = activeStudioView ?? "";
            if (typeof v === "string") {
              if (v.includes("cohort_browser") || v.startsWith("sheet::")) return "spreadsheet";
              if (v === "dicom_review" || v.startsWith("dicom")) return "dicom";
              if (v === "image_review") return "image";
              if (v === "fhir_browser") return "fhir";
              if (v === "rawqc" || v === "samtools") return "raw_qc";
              if (v === "sumstats" || v === "qqman" || v === "prs_prep") return "summary_stats";
              if (v === "text") return "text";
              if (["qc", "candidates", "clinvar", "annotations", "roh", "vep", "coverage", "symbolic", "table", "liftover", "snpeff", "plink", "ldblockshow", "igv"].includes(v)) return "vcf";
            }
            return attachedSourceType;
          })(),
          history: analysisQa.map((turn) => ({ role: turn.role, content: turn.content })),
          studio_context: studioContext,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      // Update analysis states if the response carries updated data
      if (payload.analysis) {
        setAnalysis(payload.analysis);
      }
      if (payload.plink_result) {
        setAnalysis((current) =>
          current ? { ...current, plink_result: payload.plink_result } : current,
        );
        activateStudioFromPayload({ ...payload, result_kind: "plink_result" }, "plink", "vcf");
      }
      if (payload.liftover_result) {
        setAnalysis((current) =>
          current ? { ...current, liftover_result: payload.liftover_result } : current,
        );
        activateStudioFromPayload({ ...payload, result_kind: "liftover_result" }, "liftover", "vcf");
      }
      if (payload.ldblockshow_result) {
        setAnalysis((current) =>
          current ? { ...current, ldblockshow_result: payload.ldblockshow_result } : current,
        );
        activateStudioFromPayload({ ...payload, result_kind: "ldblockshow_result" }, "ldblockshow", "vcf");
      }
      if (payload.samtools_result) {
        activateStudioFromPayload({ ...payload, result_kind: "samtools_result" }, "samtools", "raw_qc");
      }
      activateStudioFromPayload(payload);
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
      : dicomAnalysis
        ? dicomAnalysis.draft_answer
      : spreadsheetAnalysis
        ? spreadsheetAnalysis.draft_answer
      : textAnalysis
        ? textAnalysis.draft_answer
      : imageAnalysis
        ? imageAnalysis.draft_answer
      : niftiAnalysis
        ? niftiAnalysis.draft_answer
      : fhirAnalysis
        ? fhirAnalysis.draft_answer
      : null;
  const displayedAnswer = followUpAnswer ?? summaryText;
  const hasInteractiveState = Boolean(attachedFile || analysis || rawQcAnalysis || summaryStatsAnalysis || dicomAnalysis || spreadsheetAnalysis || textAnalysis || imageAnalysis || niftiAnalysis || fhirAnalysis || messages.length > 1);
  const latestStatusMessage =
    [...messages].reverse().find((message) => message.kind === "status" || message.kind === "summary")?.content ?? "";
  const sourceStatusDetail = useMemo(() => {
    if (status === "Generating answer...") {
      return "ChatClinic is reading the current analysis and Studio results to prepare a grounded response.";
    }
    if (status === "Preparing analysis...") {
      return "The VCF is attached. ChatClinic is starting the default representative analysis run.";
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
    if (status === "Text review ready") {
      return "The uploaded text note has been summarized into a preview-oriented Studio review card.";
    }
    if (status === "Uploading DICOM source...") {
      return "The DICOM source is being uploaded and prepared for metadata and preview review.";
    }
    if (status === "Running DICOM review...") {
      return "The DICOM upload is complete. ChatClinic is extracting metadata, preview state, and series summary.";
    }
    if (status === "Uploading spreadsheet source...") {
      return "The workbook is being uploaded and prepared for sheet-level cohort review.";
    }
    if (status === "Running spreadsheet review...") {
      return "The workbook upload is complete. ChatClinic is reading sheets and building cohort browser artifacts.";
    }
    if (status === "Uploading text source...") {
      return "The text source is being uploaded for note review.";
    }
    if (status === "Uploading summary statistics source...") {
      return "The summary statistics source is being uploaded and prepared for intake.";
    }
    if (status === "Uploading raw sequencing source...") {
      return "The sequencing file is being uploaded for QC intake.";
    }
    if (status === "Uploading VCF source...") {
      return "The VCF source is being uploaded and prepared for analysis.";
    }
    if (status === "Spreadsheet source ready") {
      return "A workbook source is attached. Run the spreadsheet review workflow to build cohort-style Studio cards.";
    }
    if (status === "DICOM source ready") {
      return "A DICOM source is attached. Run the DICOM review workflow to build imaging Studio cards.";
    }
    if (status === "Spreadsheet review ready") {
      return "The uploaded workbook has been converted into sheet-level cohort browser artifacts in Studio.";
    }
    if (status === "DICOM review ready") {
      return "The uploaded DICOM file has been converted into metadata, preview, and series review artifacts in Studio.";
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
    if (status === "Spreadsheet review failed") {
      return "Spreadsheet intake failed. Check the workbook format and whether the required parser dependency is installed.";
    }
    if (status === "Uploading NIfTI source...") {
      return "The NIfTI volume is being uploaded and prepared for review.";
    }
    if (status === "Running NIfTI review...") {
      return "Extracting volume metadata, orientation, voxel dimensions, and generating slice previews.";
    }
    if (status === "Uploading FHIR source...") {
      return "The FHIR source is being uploaded and prepared for clinical review.";
    }
    if (status === "Running FHIR review...") {
      return "The FHIR upload is complete. ChatClinic is extracting patient, observation, medication, and care team resources.";
    }
    if (status === "FHIR review ready") {
      return "The uploaded FHIR bundle has been converted into a clinical browser surface in Studio.";
    }
    if (status === "FHIR review failed") {
      return "FHIR intake failed. Check the file format, encoding, and whether it contains a valid FHIR Bundle.";
    }
    if (status === "DICOM review failed") {
      return "DICOM intake failed. Check the file validity and preview dependencies such as pydicom, numpy, and Pillow.";
    }
    if (status === "Answer failed") {
      return "The last chat response failed. Retry the question and ChatClinic will attempt the grounded explanation again.";
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
    status === "Uploading spreadsheet source..." ||
    status === "Uploading DICOM source..." ||
    status === "Running spreadsheet review..." ||
    status === "Running DICOM review..." ||
    status === "Uploading text source..." ||
    status === "Uploading summary statistics source..." ||
    status === "Uploading raw sequencing source..." ||
    status === "Uploading VCF source..." ||
    status === "Uploading NIfTI source..." ||
    status === "Running NIfTI review..." ||
    status === "Uploading FHIR source..." ||
    status === "Running FHIR review..." ||
    status === "Running FastQC..." ||
    status === "Loading summary statistics..." ||
    status === "Running Liftover..." ||
    status === "Running qqman..." ||
    status === "Running samtools..." ||
    status === "Running SnpEff..." ||
    status === "Running LDBlockShow..." ||
    status === "Running PLINK..." ||
    status === "Running PLINK score..." ||
    status === "Running VCF interpretation..." ||
    status === "Running VCF review..." ||
    status === "Running tool..." ||
    status === "Raw QC failed" ||
    status === "Summary stats failed" ||
    status === "Spreadsheet review failed" ||
    status === "DICOM review failed" ||
    status === "FHIR review failed" ||
    status === "Answer failed" ||
    status === "Liftover failed" ||
    status === "qqman failed" ||
    status === "samtools failed" ||
    status === "SnpEff failed" ||
    status === "LDBlockShow failed" ||
    status === "PLINK failed" ||
    status === "PLINK score failed" ||
    status === "VCF interpretation failed" ||
    status === "VCF review failed" ||
    status === "Tool failed" ||
    status === "Liftover ready" ||
    status === "qqman ready" ||
    status === "samtools ready" ||
    status === "SnpEff ready" ||
    status === "LDBlockShow ready" ||
    status === "PLINK ready" ||
    status === "PLINK score ready" ||
    status === "VCF interpretation ready" ||
    status === "VCF review ready" ||
    status === "Tool ready"
      ? status
      : analysis || rawQcAnalysis || summaryStatsAnalysis || dicomAnalysis || spreadsheetAnalysis || textAnalysis || imageAnalysis || niftiAnalysis || fhirAnalysis
        ? analysis
          ? "Analysis ready"
          : rawQcAnalysis
            ? "Raw QC ready"
            : summaryStatsAnalysis
              ? "Summary stats ready"
              : dicomAnalysis
                ? "DICOM review ready"
                : spreadsheetAnalysis
                  ? "Spreadsheet review ready"
                  : textAnalysis
                    ? "Text review ready"
                    : imageAnalysis
                      ? "Image review ready"
                      : niftiAnalysis
                        ? "NIfTI review ready"
                      : fhirAnalysis
                        ? "FHIR review ready"
                        : "Analysis ready"
        : status;
  const summaryTurn =
    analysis || rawQcAnalysis || summaryStatsAnalysis || dicomAnalysis || spreadsheetAnalysis || imageAnalysis || niftiAnalysis || fhirAnalysis
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
  const chatTurns = [...messageTurns, ...summaryTurn, ...analysisQa];
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
      spreadsheetAnalysis ||
      dicomAnalysis ||
      textAnalysis ||
      imageAnalysis ||
      niftiAnalysis ||
      fhirAnalysis ||
      prsPrepResultForStudio ||
      qqmanResultForStudio ||
      snpeffResultForStudio ||
      plinkResultForStudio ||
      liftoverResultForStudio ||
      ldblockshowResultForStudio ||
      samtoolsResultForStudio,
  );
  // Multimodal: accumulate studio cards from ALL active sources.
  const studioCards: Array<{ id: StudioView; title: string; subtitle: string }> = (() => {
    const cards: Array<{ id: StudioView; title: string; subtitle: string }> = [];

    // VCF analysis cards
    if (analysis) {
      const activeRenderer = sourceRenderers.vcf?.renderer ?? studioDispatch?.renderer ?? analysis.studio?.renderer ?? null;
      const hasAnnotations = (analysis.annotations?.length ?? 0) > 0;
      const hasCandidates = (analysis.candidate_variants?.length ?? 0) > 0;
      const hasRoh = (analysis.roh_segments?.length ?? 0) > 0;
      const igvCard = igvUnlocked ? [{ id: "igv" as StudioView, title: "IGV Plot", subtitle: "Locus visualization" }] : [];
      if (activeRenderer === "qc") {
        cards.push(
          { id: "qc" as StudioView, title: "QC Summary", subtitle: "PASS, Ti/Tv, GT quality" },
          ...(liftoverResultForStudio ? [{ id: "liftover" as StudioView, title: "LiftOver Review", subtitle: "Genome build conversion result" }] : []),
          ...(snpeffResultForStudio ? [{ id: "snpeff" as StudioView, title: "SnpEff Review", subtitle: "Local effect annotation preview" }] : []),
          ...(plinkResultForStudio ? [{ id: "plink" as StudioView, title: "PLINK", subtitle: "QC command runner and result review" }] : []),
          ...(ldblockshowResultForStudio ? [{ id: "ldblockshow" as StudioView, title: "LD Block Review", subtitle: "Locus-level LD heatmap" }] : []),
          ...(hasCandidates ? [{ id: "candidates" as StudioView, title: "Candidate Variants", subtitle: "Ranked review shortlist" }] : []),
          ...(hasAnnotations ? [{ id: "annotations" as StudioView, title: "Annotation Cards", subtitle: "Variant detail cards" }] : []),
          ...(hasRoh ? [{ id: "roh" as StudioView, title: "ROH / Recessive", subtitle: "Hom-alt and ROH-style review" }] : []),
          ...igvCard,
        );
      } else if (activeRenderer === "candidates") {
        cards.push(
          ...(hasCandidates ? [{ id: "candidates" as StudioView, title: "Candidate Variants", subtitle: "Ranked review shortlist" }] : []),
          ...(hasAnnotations ? [{ id: "annotations" as StudioView, title: "Annotation Cards", subtitle: "Variant detail cards" }] : []),
          ...(hasAnnotations ? [{ id: "vep" as StudioView, title: "VEP Consequence", subtitle: "Consequence and gene burden" }] : []),
          ...(hasRoh ? [{ id: "roh" as StudioView, title: "ROH / Recessive", subtitle: "Hom-alt and ROH-style review" }] : []),
          { id: "table" as StudioView, title: "Filtering View", subtitle: "Searchable variant triage" },
          ...igvCard,
        );
      } else if (activeRenderer === "clinvar") {
        cards.push(
          { id: "clinvar" as StudioView, title: "ClinVar Review", subtitle: "Clinical significance mix" },
          { id: "vep" as StudioView, title: "VEP Consequence", subtitle: "Consequence and gene burden" },
          { id: "coverage" as StudioView, title: "Clinical Coverage", subtitle: "Annotation completeness view" },
          { id: "symbolic" as StudioView, title: "Symbolic ALT Review", subtitle: "Structural-style records split out" },
          ...(hasRoh ? [{ id: "roh" as StudioView, title: "ROH / Recessive", subtitle: "Hom-alt and ROH-style review" }] : []),
          ...(hasCandidates ? [{ id: "candidates" as StudioView, title: "Candidate Variants", subtitle: "Ranked review shortlist" }] : []),
          { id: "table" as StudioView, title: "Filtering View", subtitle: "Searchable variant triage" },
          ...igvCard,
        );
      } else {
        cards.push(
          { id: "qc" as StudioView, title: "QC Summary", subtitle: "PASS, Ti/Tv, GT quality" },
          ...(liftoverResultForStudio ? [{ id: "liftover" as StudioView, title: "LiftOver Review", subtitle: "Genome build conversion result" }] : []),
          ...(snpeffResultForStudio ? [{ id: "snpeff" as StudioView, title: "SnpEff Review", subtitle: "Local effect annotation preview" }] : []),
          ...(plinkResultForStudio ? [{ id: "plink" as StudioView, title: "PLINK", subtitle: "QC command runner and result review" }] : []),
          ...(ldblockshowResultForStudio ? [{ id: "ldblockshow" as StudioView, title: "LD Block Review", subtitle: "Locus-level LD heatmap" }] : []),
          ...(hasCandidates ? [{ id: "candidates" as StudioView, title: "Candidate Variants", subtitle: "Ranked review shortlist" }] : []),
          ...(hasAnnotations ? [{ id: "annotations" as StudioView, title: "Annotation Cards", subtitle: "Variant detail cards" }] : []),
          ...(hasRoh ? [{ id: "roh" as StudioView, title: "ROH / Recessive", subtitle: "Hom-alt and ROH-style review" }] : []),
          ...igvCard,
        );
      }
    }

    // Raw QC cards
    if (rawQcAnalysis) {
      cards.push({ id: "rawqc", title: "FastQC Review", subtitle: "Raw sequencing module summary" });
      if (samtoolsResultForStudio) {
        cards.push({ id: "samtools" as StudioView, title: "Samtools Review", subtitle: "Alignment QC summary" });
      }
    }

    // Summary stats cards
    if (summaryStatsAnalysis) {
      cards.push({ id: "sumstats" as StudioView, title: "Summary Stats Review", subtitle: "Post-GWAS intake and column mapping" });
      if (prsPrepResultForStudio) {
        cards.push({ id: "prs_prep" as StudioView, title: "PRS Prep Review", subtitle: "Build check, harmonization, and score-file readiness" });
      }
      if (qqmanResultForStudio) {
        cards.push({ id: "qqman" as StudioView, title: "qqman Plots", subtitle: "Manhattan and QQ visualization" });
      }
    }

    // Spreadsheet cards
    if (spreadsheetAnalysis) {
      if (spreadsheetAnalysis.studio_cards?.length) {
        spreadsheetAnalysis.studio_cards.forEach((card) => {
          cards.push({
            id: String(card.id ?? card.base_id ?? "cohort_browser") as StudioView,
            title: String(card.title ?? "Cohort Browser"),
            subtitle: String(card.subtitle ?? "Sheet-level cohort overview and grid preview"),
          });
        });
      } else {
        cards.push({ id: "cohort_browser" as StudioView, title: "Cohort Browser", subtitle: "Sheet-level cohort overview and grid preview" });
      }
    }

    // DICOM cards
    if (dicomAnalysis) {
      if (dicomAnalysis.studio_cards?.length) {
        dicomAnalysis.studio_cards.forEach((card) => {
          cards.push({
            id: String(card.id ?? "dicom_review") as StudioView,
            title: String(card.title ?? "DICOM Review"),
            subtitle: String(card.subtitle ?? "Metadata, preview, and series summary"),
          });
        });
      } else {
        cards.push({ id: "dicom_review" as StudioView, title: "DICOM Review", subtitle: "Metadata, preview, and series summary" });
      }
    }

    // Text cards
    if (textAnalysis) {
      cards.push({ id: "text" as StudioView, title: "Text Review", subtitle: "Preview and note-length summary" });
    }

    // Image cards
    if (imageAnalysis) {
      if (imageAnalysis.studio_cards?.length) {
        imageAnalysis.studio_cards.forEach((card) => {
          cards.push({
            id: String(card.id ?? "image_review") as StudioView,
            title: String(card.title ?? "Image Review"),
            subtitle: String(card.subtitle ?? "Metadata, EXIF, and thumbnail preview"),
          });
        });
      } else {
        cards.push({ id: "image_review" as StudioView, title: "Image Review", subtitle: "Metadata, EXIF, and thumbnail preview" });
      }
    }

    // NIfTI cards
    if (niftiAnalysis) {
      if (niftiAnalysis.studio_cards?.length) {
        niftiAnalysis.studio_cards.forEach((card: any) => {
          cards.push({
            id: String(card.id ?? "nifti_review") as StudioView,
            title: String(card.title ?? "NIfTI Review"),
            subtitle: String(card.subtitle ?? "Volume metadata and interactive viewer"),
          });
        });
      } else {
        cards.push({ id: "nifti_review" as StudioView, title: "NIfTI Review", subtitle: "Volume metadata and interactive viewer" });
      }
    }

    // FHIR cards
    if (fhirAnalysis) {
      if (fhirAnalysis.studio_cards?.length) {
        fhirAnalysis.studio_cards.forEach((card) => {
          cards.push({
            id: String(card.id ?? "fhir_browser") as StudioView,
            title: String(card.title ?? "FHIR Browser"),
            subtitle: String(card.subtitle ?? "Patient, medications, labs, and care team"),
          });
        });
      } else {
        cards.push({ id: "fhir_browser" as StudioView, title: "FHIR Browser", subtitle: "Patient, medications, labs, and care team" });
      }
    }

    // Orphan direct tool results (no parent analysis loaded)
    if (!analysis && !rawQcAnalysis) {
      if (samtoolsResultForStudio) cards.push({ id: "samtools" as StudioView, title: "Samtools Review", subtitle: "Alignment QC summary" });
      if (snpeffResultForStudio) cards.push({ id: "snpeff" as StudioView, title: "SnpEff Review", subtitle: "Local effect annotation preview" });
      if (plinkResultForStudio) cards.push({ id: "plink" as StudioView, title: "PLINK", subtitle: "QC command runner and result review" });
      if (liftoverResultForStudio) cards.push({ id: "liftover" as StudioView, title: "LiftOver Review", subtitle: "Genome build conversion result" });
      if (ldblockshowResultForStudio) cards.push({ id: "ldblockshow" as StudioView, title: "LD Block Review", subtitle: "Locus-level LD heatmap" });
    }
    if (!summaryStatsAnalysis) {
      if (qqmanResultForStudio) cards.push({ id: "qqman" as StudioView, title: "qqman Plots", subtitle: "Manhattan and QQ visualization" });
      if (prsPrepResultForStudio) cards.push({ id: "prs_prep" as StudioView, title: "PRS Prep Review", subtitle: "Build check, harmonization, and score-file readiness" });
    }

    return cards;
  })();
  const externalStudioRendererRegistry = buildStudioRendererRegistry({
    apiBase,
    activeStudioView,
    analysis,
    rawQcAnalysis,
    summaryStatsAnalysis,
    dicomAnalysis,
    spreadsheetAnalysis,
    textAnalysis,
    imageAnalysis,
    niftiAnalysis,
    fhirAnalysis,
    prsPrepResultForStudio,
    qqmanResultForStudio,
    samtoolsResultForStudio,
    snpeffResultForStudio,
    plinkResultForStudio,
    liftoverResultForStudio,
    ldblockshowResultForStudio,
    summaryStatsGridRows,
    summaryStatsRowsLoading,
    summaryStatsHasMore,
    summaryStatsGridRef,
    handleSummaryStatsGridScroll,
    loadMoreSummaryStatsRows,
    candidateVariants,
    searchedAnnotations,
    setSelectedAnnotationIndex,
    setActiveStudioView,
    buildAcmgHints,
    annotationScope,
    annotationLimit,
    qcMetrics,
    clinicalCoverage,
    plinkConfig,
    setPlinkConfig,
    plinkCommandPreview,
    handleRunPlink,
    plinkRunning,
    activeSource,
    attachedFile,
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
    components: {
      StudioMetricGrid,
      StudioPreviewTable,
      WarningListCard,
      ArtifactLinksRow,
      StudioSimpleList,
      DistributionList,
      VariantTable,
      MetricTile,
      ReferenceListCard,
      AnnotationDetailCard,
    },
    helpers: {
      formatPercent,
      formatNumber,
      summarizeLabel,
    },
  });
  const resolvedStudioRendererKey = resolveStudioRendererKey({
    activeView: activeStudioView,
    dispatch: studioDispatch,
  });
  const registeredStudioRenderer = resolvedStudioRendererKey ? externalStudioRendererRegistry[resolvedStudioRendererKey] : null;

  function activateStudioFromPayload(payload: Record<string, unknown> | null | undefined, fallbackView?: StudioView | null, sourceType?: string) {
    const dispatch = resolveStudioDispatchFromPayload(payload);
    // Only update studio dispatch when the payload carries explicit studio
    // metadata (result_kind, requested_view, or studio.renderer).  Plain
    // grounded-chat responses contain none of these — updating dispatch in
    // that case would reset the renderer to null and show the full card
    // fallback list.
    const hasStudioIntent =
      Boolean(dispatch.resultKind) ||
      Boolean(dispatch.requestedView) ||
      Boolean(dispatch.renderer);
    if (hasStudioIntent) {
      setStudioDispatch(dispatch);
      // Save per-source renderer so other sources don't overwrite it
      if (sourceType) {
        setSourceRenderers((prev) => ({ ...prev, [sourceType]: dispatch }));
      }
    }
    const resolvedView = resolveStudioRendererKey({
      activeView: fallbackView,
      dispatch: hasStudioIntent ? dispatch : undefined,
    });
    if (resolvedView) {
      setActiveStudioView(resolvedView as StudioView);
      return;
    }
    if (fallbackView) {
      setActiveStudioView(fallbackView);
    }
  }
  const studioContext = useMemo(() => {
    // Merge studio context from ALL active sources (multimodal-aware)
    const merged: Record<string, any> = { active_view: activeStudioView };
    const mergedExtra: Record<string, any> = {};
    const allWarnings: any[] = [];

    // --- VCF source ---
    if (analysis) {
      merged.current_card = selectedAnnotation
        ? {
            locus: `${selectedAnnotation.contig}:${selectedAnnotation.pos_1based}`,
            gene: selectedAnnotation.gene,
            rsid: selectedAnnotation.rsid,
            consequence: selectedAnnotation.consequence,
          }
        : null;
      merged.current_summary = {
        qc_summary: {
          pass_rate: qcMetrics?.pass_rate,
          ti_tv: qcMetrics?.transition_transversion_ratio,
          missing_gt_rate: qcMetrics?.missing_gt_rate,
          het_hom_alt_ratio: qcMetrics?.het_hom_alt_ratio,
        },
        candidate_count: candidateVariants.length,
        roh_segment_count: analysis.roh_segments?.length ?? 0,
      };
      merged.current_preview = {
        columns: ["locus", "gene", "rsid", "consequence", "clinical_significance", "gnomad_af", "score", "in_roh"],
        rows: candidateVariants.slice(0, 6).map(({ item, score, inRoh }) => ({
          locus: `${item.contig}:${item.pos_1based}`,
          gene: item.gene,
          rsid: item.rsid,
          consequence: item.consequence,
          clinical_significance: item.clinical_significance,
          gnomad_af: item.gnomad_af,
          score,
          in_roh: inRoh,
        })),
      };
      allWarnings.push(...(analysis.facts?.warnings?.slice(0, 12) ?? []));
      Object.assign(mergedExtra, {
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
      });
    }

    // --- DICOM source ---
    if (dicomAnalysis) {
      const dicomCard = dicomAnalysis.artifacts?.dicom_review ?? null;
      const metadata = Array.isArray(dicomAnalysis.metadata_items) ? dicomAnalysis.metadata_items[0] ?? null : null;
      const preview = metadata?.preview ?? dicomCard?.preview ?? null;
      if (!analysis) {
        // Only set current_card/current_summary if VCF didn't already
        merged.current_card = dicomCard;
        merged.current_summary = metadata
          ? {
              modality: metadata.modality ?? null,
              patient_id: metadata.patient_id ?? null,
              study_description: metadata.study_description ?? null,
              series_description: metadata.series_description ?? null,
            }
          : null;
        merged.current_preview = preview
          ? {
              columns: ["preview_state"],
              rows: [{ preview_state: String(preview.available ? "available" : preview.message ?? "not available") }],
            }
          : null;
      }
      allWarnings.push(...(Array.isArray(dicomAnalysis.warnings) ? dicomAnalysis.warnings.slice(0, 12) : []));
      mergedExtra.dicom = {
        metadata_items: dicomAnalysis.metadata_items,
        series: dicomAnalysis.series,
        preview,
        current_card: dicomCard,
        current_summary: metadata
          ? {
              modality: metadata.modality ?? null,
              patient_id: metadata.patient_id ?? null,
              study_description: metadata.study_description ?? null,
              series_description: metadata.series_description ?? null,
            }
          : null,
      };
    }

    // --- Spreadsheet source ---
    if (spreadsheetAnalysis) {
      const selectedSpreadsheetSheet =
        typeof activeStudioView === "string" && activeStudioView.startsWith("sheet::") && activeStudioView.endsWith("::cohort_browser")
          ? activeStudioView.slice("sheet::".length, -"::cohort_browser".length)
          : spreadsheetAnalysis.selected_sheet;
      const selectedSpreadsheetArtifact =
        selectedSpreadsheetSheet && spreadsheetAnalysis.artifacts
          ? spreadsheetAnalysis.artifacts[`sheet::${selectedSpreadsheetSheet}::cohort_browser`] ?? null
          : null;
      if (!analysis && !dicomAnalysis) {
        merged.current_card = selectedSpreadsheetArtifact;
        merged.current_summary = selectedSpreadsheetArtifact
          ? {
              selected_sheet: selectedSpreadsheetSheet,
              overview: selectedSpreadsheetArtifact.overview ?? null,
              intake: selectedSpreadsheetArtifact.intake ?? null,
              composition: selectedSpreadsheetArtifact.composition ?? null,
            }
          : null;
        merged.current_schema =
          selectedSpreadsheetArtifact && Array.isArray(selectedSpreadsheetArtifact.schema_highlights)
            ? selectedSpreadsheetArtifact.schema_highlights.slice(0, 24)
            : [];
        merged.current_preview = selectedSpreadsheetArtifact
          ? {
              columns: selectedSpreadsheetArtifact.grid?.columns ?? [],
              rows: Array.isArray(selectedSpreadsheetArtifact.grid?.rows)
                ? selectedSpreadsheetArtifact.grid.rows.slice(0, 60)
                : [],
            }
          : null;
      }
      allWarnings.push(
        ...(Array.isArray(selectedSpreadsheetArtifact?.warnings)
          ? selectedSpreadsheetArtifact.warnings.slice(0, 12)
          : Array.isArray(spreadsheetAnalysis.warnings)
            ? spreadsheetAnalysis.warnings.slice(0, 12)
            : []),
      );
      mergedExtra.spreadsheet = {
        sheet_count: spreadsheetAnalysis.sheet_count,
        selected_sheet: selectedSpreadsheetSheet,
        sheet_names: spreadsheetAnalysis.sheet_names,
        sheet_details: spreadsheetAnalysis.sheet_details?.slice(0, 8),
        current_sheet: selectedSpreadsheetArtifact
          ? {
              overview: selectedSpreadsheetArtifact.overview ?? null,
              intake: selectedSpreadsheetArtifact.intake ?? null,
              schema_highlights: Array.isArray(selectedSpreadsheetArtifact.schema_highlights)
                ? selectedSpreadsheetArtifact.schema_highlights.slice(0, 24)
                : [],
              missingness: selectedSpreadsheetArtifact.missingness ?? null,
              composition: selectedSpreadsheetArtifact.composition ?? null,
              preview_columns: selectedSpreadsheetArtifact.grid?.columns ?? [],
              preview_rows: Array.isArray(selectedSpreadsheetArtifact.grid?.rows)
                ? selectedSpreadsheetArtifact.grid.rows.slice(0, 60)
                : [],
            }
          : null,
      };
    }

    // --- Image source ---
    if (imageAnalysis) {
      if (!analysis && !dicomAnalysis && !spreadsheetAnalysis) {
        merged.current_card = {
          file_name: imageAnalysis.file_name,
          format: imageAnalysis.format_name,
          dimensions: `${imageAnalysis.width}×${imageAnalysis.height}`,
        };
        merged.current_summary = {
          format_name: imageAnalysis.format_name,
          color_mode: imageAnalysis.color_mode,
          width: imageAnalysis.width,
          height: imageAnalysis.height,
          bit_depth: imageAnalysis.bit_depth,
        };
      }
      allWarnings.push(...(Array.isArray(imageAnalysis.warnings) ? imageAnalysis.warnings.slice(0, 12) : []));
      mergedExtra.image = {
        file_name: imageAnalysis.file_name,
        format_name: imageAnalysis.format_name,
        color_mode: imageAnalysis.color_mode,
        width: imageAnalysis.width,
        height: imageAnalysis.height,
        bit_depth: imageAnalysis.bit_depth,
        exif_data: imageAnalysis.exif_data,
        metadata_items: imageAnalysis.metadata_items,
      };
    }

    // --- NIfTI source ---
    if (niftiAnalysis) {
      if (!analysis && !dicomAnalysis && !spreadsheetAnalysis && !imageAnalysis) {
        merged.current_card = {
          file_name: niftiAnalysis.file_name,
          shape: niftiAnalysis.shape?.join(" × "),
          voxel_dims: niftiAnalysis.voxel_dims?.slice(0, 3).join(" × "),
          orientation: niftiAnalysis.orientation,
        };
        merged.current_summary = {
          shape: niftiAnalysis.shape,
          voxel_dims: niftiAnalysis.voxel_dims,
          orientation: niftiAnalysis.orientation,
          datatype: niftiAnalysis.datatype,
          is_4d: niftiAnalysis.is_4d,
        };
      }
      allWarnings.push(...(Array.isArray(niftiAnalysis.warnings) ? niftiAnalysis.warnings.slice(0, 12) : []));
      mergedExtra.nifti = {
        file_name: niftiAnalysis.file_name,
        shape: niftiAnalysis.shape,
        voxel_dims: niftiAnalysis.voxel_dims,
        orientation: niftiAnalysis.orientation,
        datatype: niftiAnalysis.datatype,
        is_4d: niftiAnalysis.is_4d,
        fov_mm: niftiAnalysis.fov_mm,
        metadata_items: niftiAnalysis.metadata_items,
      };
    }

    // --- FHIR source ---
    if (fhirAnalysis) {
      if (!analysis && !dicomAnalysis && !spreadsheetAnalysis && !imageAnalysis && !niftiAnalysis) {
        merged.current_card = {
          file_name: fhirAnalysis.file_name,
          resource_type: fhirAnalysis.resource_type,
          resource_count: fhirAnalysis.resource_count,
          patient_summary: fhirAnalysis.patient_summary,
        };
        merged.current_summary = {
          resource_type: fhirAnalysis.resource_type,
          resource_count: fhirAnalysis.resource_count,
          patient_summary: fhirAnalysis.patient_summary,
        };
      }
      allWarnings.push(...(Array.isArray(fhirAnalysis.warnings) ? fhirAnalysis.warnings.slice(0, 12) : []));
      mergedExtra.fhir = {
        file_name: fhirAnalysis.file_name,
        file_kind: fhirAnalysis.file_kind,
        resource_type: fhirAnalysis.resource_type,
        resource_count: fhirAnalysis.resource_count,
        patient_summary: fhirAnalysis.patient_summary,
        metadata_items: fhirAnalysis.metadata_items,
      };
    }

    if (!analysis && !dicomAnalysis && !spreadsheetAnalysis && !imageAnalysis && !niftiAnalysis && !fhirAnalysis) {
      return {};
    }

    merged.current_schema = merged.current_schema ?? [];
    merged.current_warnings = allWarnings;
    merged.extra = mergedExtra;
    return merged;
  }, [
    activeStudioView,
    analysis,
    dicomAnalysis,
    spreadsheetAnalysis,
    imageAnalysis,
    niftiAnalysis,
    fhirAnalysis,
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
          <img src="/chatclinic-logo.svg" alt="" className="appBrandIconImage" />
          <span className="appBrandName">ChatClinic</span>
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
                      + Add source files
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
                    ) : uploadedSources.length > 0 ? (
                      <>
                        <article
                          className="sourceItem sourceItemActive"
                          style={{ cursor: uploadedSources.length > 1 ? "pointer" : undefined }}
                          onClick={() => uploadedSources.length > 1 && setSourcesExpanded((v) => !v)}
                        >
                          <div>
                            <strong>{activeSource?.file_name ?? attachedFile?.name ?? uploadedSources[uploadedSources.length - 1].name}</strong>
                            <p>
                              {uploadedSources.length > 1
                                ? `${uploadedSources.length} sources loaded`
                                : (() => {
                                    const st = uploadedSources[0].sourceType;
                                    return st === "raw_qc" ? "Active raw sequencing source"
                                      : st === "spreadsheet" ? "Active spreadsheet source"
                                      : st === "dicom" ? "Active DICOM source"
                                      : st === "text" ? "Active text source"
                                      : st === "summary_stats" ? "Active summary statistics source"
                                      : st === "image" ? "Active image source"
                                      : st === "nifti" ? "Active NIfTI source"
                                      : st === "fhir" ? "Active FHIR source"
                                      : "Active VCF source";
                                  })()}
                            </p>
                          </div>
                          <span className="sourceBadge">{uploadedSources.length}</span>
                        </article>
                        {sourcesExpanded && uploadedSources.length > 1 && (
                          <div className="sourceDropdownList" style={{ marginTop: 4, paddingLeft: 8 }}>
                            {uploadedSources.map((src, idx) => {
                              const isActive =
                                activeSource?.source_type === src.sourceType &&
                                activeSource?.file_name === src.name &&
                                (!src.sourcePath || activeSource?.source_path === src.sourcePath);
                              const typeLabel =
                                src.sourceType === "vcf" ? "VCF"
                                : src.sourceType === "raw_qc" ? "Raw QC"
                                : src.sourceType === "spreadsheet" ? "Spreadsheet"
                                : src.sourceType === "dicom" ? "DICOM"
                                : src.sourceType === "image" ? "Image"
                                : src.sourceType === "text" ? "Text"
                                : src.sourceType === "summary_stats" ? "Summary Stats"
                                : src.sourceType === "nifti" ? "NIfTI"
                                : src.sourceType;
                              return (
                                <article
                                  key={src.id || `src-${idx}`}
                                  className={`sourceItem ${isActive ? "sourceItemActive" : ""}`}
                                  style={{ opacity: 0.9, marginBottom: 4, cursor: "pointer" }}
                                  onClick={() => handleSelectUploadedSource(src)}
                                >
                                  <div>
                                    <strong style={{ fontSize: "0.85em" }}>{src.name}</strong>
                                    <p style={{ fontSize: "0.8em", margin: 0 }}>
                                      {isActive ? `Active ${typeLabel}` : typeLabel}
                                    </p>
                                  </div>
                                </article>
                              );
                            })}
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="sourceEmpty">
                        <p>Upload a source file to begin. The session type is detected automatically.</p>
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
                  <h3>Upload a source file</h3>
                  <p>Upload a file on the left to start a session. The source type and tools are detected automatically.</p>
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
                    : "Upload a source file to begin"
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
