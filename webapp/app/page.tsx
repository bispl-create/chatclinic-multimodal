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
    input_path: string;
    command_preview: string;
    output_prefix: string;
    log_path?: string | null;
    freq_path?: string | null;
    missing_path?: string | null;
    hardy_path?: string | null;
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
};

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

export default function Page() {
  const [apiBase, setApiBase] = useState("http://127.0.0.1:8001");
  const [toolRegistry, setToolRegistry] = useState<AnalysisResponse["tool_registry"]>([]);
  const [toolRegistryLoading, setToolRegistryLoading] = useState(true);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "VCF를 올리면 variant interpretation workflow를 진행하고, FASTQ/BAM/SAM을 올리면 FastQC 기반 raw sequencing QC를 진행합니다. Summary statistics를 올리면 post-GWAS review intake를 시작합니다.",
    },
  ]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [rawQcAnalysis, setRawQcAnalysis] = useState<RawQcResponse | null>(null);
  const [summaryStatsAnalysis, setSummaryStatsAnalysis] = useState<SummaryStatsResponse | null>(null);
  const [summaryStatsGridRows, setSummaryStatsGridRows] = useState<Array<Record<string, string>>>([]);
  const [summaryStatsHasMore, setSummaryStatsHasMore] = useState(false);
  const [summaryStatsRowsLoading, setSummaryStatsRowsLoading] = useState(false);
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [annotationScope, setAnnotationScope] = useState<"representative" | "all">("representative");
  const [annotationLimit, setAnnotationLimit] = useState("200");
  const [status, setStatus] = useState("Waiting for a genomics source");
  const [error, setError] = useState<string | null>(null);
  const [selectedAnnotationIndex, setSelectedAnnotationIndex] = useState(0);
  const [annotationSearch, setAnnotationSearch] = useState("");
  const [composerText, setComposerText] = useState("");
  const [isComposing, setIsComposing] = useState(false);
  const [analysisQa, setAnalysisQa] = useState<AnalysisQuestionTurn[]>([]);
  const [followUpAnswer, setFollowUpAnswer] = useState<string | null>(null);
  const [initialGroundedAnswer, setInitialGroundedAnswer] = useState<string | null>(null);
  const [activeStudioView, setActiveStudioView] = useState<StudioView | null>(null);
  const [plinkRunning, setPlinkRunning] = useState(false);
  const [plinkConfig, setPlinkConfig] = useState({
    runFreq: true,
    runMissing: true,
    runHardy: true,
    allowExtraChr: true,
    outputPrefix: "",
  });
  const [toolRegistryOpen, setToolRegistryOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const studioCanvasRef = useRef<HTMLElement | null>(null);
  const chatStreamRef = useRef<HTMLDivElement | null>(null);
  const summaryStatsGridRef = useRef<HTMLDivElement | null>(null);
  const initialSummaryRequestRef = useRef<string | null>(null);
  const activeToolRegistry =
    analysis?.tool_registry?.length
      ? analysis.tool_registry
      : rawQcAnalysis?.tool_registry?.length
        ? rawQcAnalysis.tool_registry
        : summaryStatsAnalysis?.tool_registry?.length
          ? summaryStatsAnalysis.tool_registry
        : toolRegistry;
  const plinkCommandPreview = useMemo(() => {
    const inputPath = analysis?.source_vcf_path || "<source.vcf.gz>";
    const outputPrefix = (plinkConfig.outputPrefix || `${analysis?.analysis_id ?? "analysis"}-plink`).trim();
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
  }, [analysis, plinkConfig]);
  const groundingTokens = ["$studio", "$current analysis", "$current card", "$grounded"];
  const normalizedComposerText = typeof composerText === "string" ? composerText.toLowerCase() : "";
  const detectedGroundingTriggers = groundingTokens.filter((token) => normalizedComposerText.includes(token));

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

  function handleAttachClick() {
    fileInputRef.current?.click();
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (!file) {
      return;
    }
    setAttachedFile(file);
    setAnalysis(null);
    setRawQcAnalysis(null);
    setSummaryStatsAnalysis(null);
    setFollowUpAnswer(null);
    setInitialGroundedAnswer(null);
    setAnalysisQa([]);
    setActiveStudioView(null);
    setSelectedAnnotationIndex(0);
    setAnnotationSearch("");
    initialSummaryRequestRef.current = null;
    setError(null);
    if (isRawQcFileName(file.name)) {
      setStatus("Running FastQC...");
      void handleStartRawQc(file);
    } else if (isSummaryStatsFileName(file.name) && !isVcfFileName(file.name)) {
      setStatus("Loading summary statistics...");
      void handleStartSummaryStats(file);
    } else {
      setStatus("Preparing analysis...");
      void handleStartAnalysis("representative", annotationLimit, file);
    }
    event.target.value = "";
  }

  async function handleStartRawQc(file: File) {
    setError(null);
    addMessage({
      role: "assistant",
      content: "FastQC로 raw sequencing QC를 실행하고 있습니다.",
      kind: "status",
    });

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
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Raw QC failed");
      addMessage({
        role: "assistant",
        content: `FastQC 실행 중 오류가 발생했습니다: ${message}`,
      });
    }
  }

  async function handleStartSummaryStats(file: File) {
    setError(null);
    addMessage({
      role: "assistant",
      content: "Summary statistics 파일을 읽고 컬럼과 기본 QC를 확인하고 있습니다.",
      kind: "status",
    });

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
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Summary stats failed");
      addMessage({
        role: "assistant",
        content: `Summary statistics intake 중 오류가 발생했습니다: ${message}`,
      });
    }
  }

  async function handleStartAnalysis(
    parsedScope?: "representative" | "all",
    parsedLimit?: string,
    fileOverride?: File | null,
  ) {
    const inputFile = fileOverride ?? attachedFile;
    if (!inputFile) {
      setError("먼저 + 버튼으로 VCF 파일을 첨부해 주세요.");
      return;
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
      setInitialGroundedAnswer(null);
      setAnalysisQa([]);
      setActiveStudioView(null);
      setSelectedAnnotationIndex(0);
      setComposerText("");
      setStatus("Preparing grounded summary...");
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setStatus("Analysis failed");
      addMessage({
        role: "assistant",
        content: `분석 중 오류가 발생했습니다: ${message}`,
      });
    }
  }

  async function handleComposerSubmit() {
    const text = composerText.trim();
    if (!text) {
      return;
    }

    if (!attachedFile) {
      addMessage({ role: "user", content: text });
      addMessage({
        role: "assistant",
        content: "먼저 + 버튼으로 VCF 파일을 첨부해 주세요.",
      });
      setComposerText("");
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
      content: "먼저 VCF 또는 raw sequencing file을 첨부해 주세요.",
    });
    setComposerText("");
  }

  async function handleAskAnalysisQuestion(questionText?: string) {
    const text = questionText?.trim() ?? "";
    if (!text || !analysis) {
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
          analysis,
          history: analysisQa.map((turn) => ({ role: turn.role, content: turn.content })),
          studio_context: studioContext,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      setAnalysisQa((current) => [...current, { role: "assistant", content: payload.answer }]);
      if (payload.requested_view === "plink") {
        setActiveStudioView("plink");
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
    if (!analysis?.source_vcf_path) {
      setError("현재 분석에는 실행 가능한 source VCF path가 없습니다.");
      return;
    }

    setPlinkRunning(true);
    setStatus("Running PLINK...");
    setError(null);
    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/plink/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vcf_path: analysis.source_vcf_path,
          output_prefix: (plinkConfig.outputPrefix || `${analysis.analysis_id}-plink`).trim(),
          allow_extra_chr: plinkConfig.allowExtraChr,
          freq_limit: plinkConfig.runFreq ? 12 : 0,
          missing_limit: plinkConfig.runMissing ? 12 : 0,
          hardy_limit: plinkConfig.runHardy ? 12 : 0,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      setAnalysis((current) =>
        current
          ? {
              ...current,
              plink_result: payload,
              used_tools: ["plink_execution_tool"],
            }
          : current,
      );
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

  async function handleAskRawQcQuestion(questionText?: string) {
    const text = questionText?.trim() ?? "";
    if (!text || !rawQcAnalysis) {
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
          analysis: rawQcAnalysis,
          history: analysisQa.map((turn) => ({ role: turn.role, content: turn.content })),
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
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

  async function handleAskSummaryStatsQuestion(questionText?: string) {
    const text = questionText?.trim() ?? "";
    if (!text || !summaryStatsAnalysis) {
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
          analysis: summaryStatsAnalysis,
          history: analysisQa.map((turn) => ({ role: turn.role, content: turn.content })),
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload: SummaryStatsChatResponse = await response.json();
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
    ? formatSummaryWithCitations(initialGroundedAnswer ?? analysis.draft_answer, analysis.references)
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
    if (status === "Preparing grounded summary...") {
      return "The VCF analysis is finished. ChatGenome is now composing the first grounded summary from the current Studio results.";
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
    if (status === "Answer ready") {
      return "The latest answer is ready in Chat and grounded against the current analysis context.";
    }
    if (status === "Grounded summary ready") {
      return "The initial grounded summary is ready. You can continue asking questions about any Studio result.";
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
    return latestStatusMessage;
  }, [latestStatusMessage, status]);
  const chatHeaderStatus =
    status === "Generating answer..." ||
    status === "Preparing analysis..." ||
    status === "Preparing grounded summary..." ||
    status === "Analyzing" ||
    status === "Running FastQC..." ||
    status === "Loading summary statistics..." ||
    status === "Raw QC failed" ||
    status === "Summary stats failed" ||
    status === "Answer failed"
      ? status
      : analysis || rawQcAnalysis || summaryStatsAnalysis
        ? analysis
          ? "Analysis ready"
          : rawQcAnalysis
            ? "Raw QC ready"
            : "Summary stats ready"
        : status;
  const chatTurns = [
    ...messages
      .filter((message) => message.role === "user")
      .map((message) => ({ role: message.role, content: message.content })),
    ...(analysis || rawQcAnalysis || summaryStatsAnalysis
      ? [
          {
            role: "assistant" as const,
            content:
              summaryText ??
              "분석이 완료되면 grounded summary가 여기에 표시됩니다.",
          },
        ]
      : []),
    ...analysisQa,
  ];
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
  const studioCards: Array<{ id: StudioView; title: string; subtitle: string }> = rawQcAnalysis
    ? [
        { id: "rawqc", title: "FastQC Review", subtitle: "Raw sequencing module summary" },
        ...(rawQcAnalysis.samtools_result
          ? [{ id: "samtools" as StudioView, title: "Samtools Review", subtitle: "Alignment QC summary" }]
          : []),
      ]
    : summaryStatsAnalysis
      ? [{ id: "sumstats" as StudioView, title: "Summary Stats Review", subtitle: "Post-GWAS intake and column mapping" }]
    : [
        { id: "provenance", title: "Workflow Setup", subtitle: "Tools, scope, and run policy" },
        { id: "qc", title: "QC Summary", subtitle: "PASS, Ti/Tv, GT quality" },
        { id: "coverage", title: "Clinical Coverage", subtitle: "Annotation completeness view" },
        { id: "snpeff", title: "SnpEff Review", subtitle: "Local effect annotation preview" },
        { id: "plink", title: "PLINK", subtitle: "QC command runner and result review" },
        ...(analysis?.liftover_result
          ? [{ id: "liftover" as StudioView, title: "LiftOver Review", subtitle: "Genome build conversion result" }]
          : []),
        ...(analysis?.ldblockshow_result
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
      ];
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

  useEffect(() => {
    async function generateInitialGroundedSummary() {
      if (!analysis) {
        return;
      }
      setStatus("Preparing grounded summary...");
      try {
        const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/chat/analysis`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question:
              "$studio Provide an initial grounded summary of this VCF analysis. Combine the backend draft_answer with the current studio_context explicitly, including QC, clinical coverage, symbolic ALT review, ROH/recessive review, candidate variants, ClinVar review, and VEP consequence.",
            analysis,
            history: [],
            studio_context: studioContext,
          }),
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const payload = await response.json();
        setInitialGroundedAnswer(payload.answer);
        setStatus("Grounded summary ready");
      } catch {
        setInitialGroundedAnswer(null);
        setStatus("Grounded summary ready");
      }
    }

    if (!analysis) {
      return;
    }
    if (initialSummaryRequestRef.current === analysis.analysis_id) {
      return;
    }
    initialSummaryRequestRef.current = analysis.analysis_id;
    void generateInitialGroundedSummary();
  }, [analysis, apiBase, studioContext]);

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
                  <button type="button" className="sourceAddButton" onClick={handleAttachClick}>
                    + Add genomics source
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    onChange={handleFileChange}
                    className="hiddenInput"
                  />
                  <div className="sourceList">
                    {attachedFile ? (
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
                        <p>Attach a VCF, a raw sequencing file such as FASTQ/BAM/SAM, or a summary statistics file to start analysis.</p>
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
                              <span className="toolRegistryName">{tool.name}</span>
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
                    <span>Used tools</span>
                    {(analysis?.used_tools?.length || rawQcAnalysis?.used_tools?.length || summaryStatsAnalysis?.used_tools?.length) ? (
                      <ul className="toolUsageList">
                        {(analysis?.used_tools ?? rawQcAnalysis?.used_tools ?? summaryStatsAnalysis?.used_tools ?? []).map((toolName, index) => (
                          <li key={`${toolName}-${index}`}>{toolName}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="toolUsageEmpty">No tool has been logged for this run yet.</p>
                    )}
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
                      <p className="summaryText nbAnswerText">{turn.content}</p>
                    ) : (
                      <MarkdownAnswer content={turn.content} />
                    )}
                  </article>
                ))
              ) : (
                <div className="chatEmptyState">
                  <h3>Start with one genomics source</h3>
                  <p>Upload a VCF, a raw sequencing file, or a summary statistics file on the left, then continue the workflow in this chat.</p>
                </div>
              )}
            </div>

            <div className="chatComposerDock">
              <input
                className={detectedGroundingTriggers.length ? "composerInput composerInputTriggered" : "composerInput"}
                value={composerText}
                onChange={(event) => setComposerText(event.target.value)}
                onCompositionStart={() => setIsComposing(true)}
                onCompositionEnd={() => setIsComposing(false)}
                placeholder={
                  attachedFile
                    ? "Start typing a follow-up question..."
                    : "Upload a VCF, raw sequencing file, or summary statistics file first"
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
            {detectedGroundingTriggers.length ? (
              <div className="composerTriggerBar">
                <span className="composerTriggerLabel">Grounded mode</span>
                {detectedGroundingTriggers.map((token) => (
                  <span key={token} className="composerTriggerChip">
                    {token}
                  </span>
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
            {analysis || rawQcAnalysis || summaryStatsAnalysis ? (
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
              {analysis || rawQcAnalysis || summaryStatsAnalysis ? "Choose a card to open a result view." : "Studio cards will appear after tool-driven analysis results are ready."}
            </div>
          </div>
        </aside>
      </div>

      {(analysis || rawQcAnalysis || summaryStatsAnalysis) && activeStudioView ? (
        <section ref={studioCanvasRef} className="studioCanvas">
          {summaryStatsAnalysis && activeStudioView === "sumstats" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>Summary Stats Review</h2>
              </div>
              <div className="studioCanvasBody">
                <div className="resultMetricGrid">
                  <MetricTile label="Rows" value={String(summaryStatsAnalysis.row_count)} tone="good" />
                  <MetricTile label="Columns" value={String(summaryStatsAnalysis.detected_columns.length)} tone="neutral" />
                  <MetricTile label="Build" value={summaryStatsAnalysis.genome_build} tone="neutral" />
                  <MetricTile label="Trait" value={summaryStatsAnalysis.trait_type} tone="neutral" />
                  <MetricTile label="Delimiter" value={summaryStatsAnalysis.delimiter} tone="neutral" />
                  <MetricTile label="Warnings" value={String(summaryStatsAnalysis.warnings.length)} tone="neutral" />
                </div>
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
                  <div ref={summaryStatsGridRef} className="variantTableWrap summaryStatsTableWrap" onScroll={handleSummaryStatsGridScroll}>
                    <table className="variantTable summaryStatsTable">
                      <thead>
                        <tr>
                          <th className="summaryStatsRowHeader">#</th>
                          {summaryStatsAnalysis.detected_columns.map((column) => (
                            <th key={column}>{column}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {summaryStatsGridRows.map((row, index) => (
                          <tr key={`sumstats-preview-${index}`}>
                            <td className="summaryStatsRowHeader">{index + 1}</td>
                            {summaryStatsAnalysis.detected_columns.map((column) => (
                              <td key={`${index}-${column}`}>{row[column] || ""}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
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
                  </div>
                </article>
                {summaryStatsAnalysis.warnings.length ? (
                  <div className="resultList">
                    {summaryStatsAnalysis.warnings.map((warning, index) => (
                      <article key={`sumstats-warning-${index}`} className="resultListItem resultListStatic">
                        <strong>Warning {index + 1}</strong>
                        <span>{warning}</span>
                      </article>
                    ))}
                  </div>
                ) : null}
              </div>
            </section>
          ) : null}

          {rawQcAnalysis && activeStudioView === "rawqc" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>FastQC Review</h2>
              </div>
              <div className="studioCanvasBody">
                <div className="resultMetricGrid">
                  <MetricTile
                    label="Total sequences"
                    value={rawQcAnalysis.facts.total_sequences != null ? String(rawQcAnalysis.facts.total_sequences) : "n/a"}
                    tone="good"
                  />
                  <MetricTile label="Sequence length" value={rawQcAnalysis.facts.sequence_length ?? "n/a"} tone="neutral" />
                  <MetricTile
                    label="%GC"
                    value={rawQcAnalysis.facts.gc_content != null ? `${rawQcAnalysis.facts.gc_content.toFixed(1)}%` : "n/a"}
                    tone="neutral"
                  />
                  <MetricTile label="Encoding" value={rawQcAnalysis.facts.encoding ?? "n/a"} tone="neutral" />
                </div>
                <div className="resultList">
                  {rawQcAnalysis.modules.map((module) => (
                    <article key={module.name} className="miniCard">
                      <h3>{module.name}</h3>
                      <p>Status: {module.status}</p>
                      {module.detail ? <p>{module.detail}</p> : null}
                    </article>
                  ))}
                </div>
                <div className="resultActionRow">
                  {rawQcAnalysis.report_html_path ? (
                    <a
                      className="sourceAddButton"
                      href={`${apiBase.replace(/\/$/, "")}/api/v1/raw-qc/report?path=${encodeURIComponent(rawQcAnalysis.report_html_path)}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open HTML report
                    </a>
                  ) : null}
                  {rawQcAnalysis.report_zip_path ? (
                    <a
                      className="sourceAddButton"
                      href={`${apiBase.replace(/\/$/, "")}/api/v1/raw-qc/report?path=${encodeURIComponent(rawQcAnalysis.report_zip_path)}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Download ZIP
                    </a>
                  ) : null}
                </div>
              </div>
            </section>
          ) : null}
          {rawQcAnalysis && activeStudioView === "samtools" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>Samtools Review</h2>
              </div>
              <div className="studioCanvasBody">
                {rawQcAnalysis.samtools_result ? (
                  <>
                    <div className="resultMetricGrid">
                      <MetricTile label="File kind" value={rawQcAnalysis.samtools_result.file_kind} tone="good" />
                      <MetricTile
                        label="Total reads"
                        value={
                          rawQcAnalysis.samtools_result.total_reads != null
                            ? String(rawQcAnalysis.samtools_result.total_reads)
                            : "n/a"
                        }
                        tone="neutral"
                      />
                      <MetricTile
                        label="Mapped"
                        value={
                          rawQcAnalysis.samtools_result.mapped_reads != null
                            ? `${rawQcAnalysis.samtools_result.mapped_reads}${
                                rawQcAnalysis.samtools_result.mapped_rate != null
                                  ? ` (${rawQcAnalysis.samtools_result.mapped_rate.toFixed(2)}%)`
                                  : ""
                              }`
                            : "n/a"
                        }
                        tone="good"
                      />
                      <MetricTile
                        label="Properly paired"
                        value={
                          rawQcAnalysis.samtools_result.properly_paired_reads != null
                            ? `${rawQcAnalysis.samtools_result.properly_paired_reads}${
                                rawQcAnalysis.samtools_result.properly_paired_rate != null
                                  ? ` (${rawQcAnalysis.samtools_result.properly_paired_rate.toFixed(2)}%)`
                                  : ""
                              }`
                            : "n/a"
                        }
                        tone="neutral"
                      />
                      <MetricTile
                        label="Quickcheck"
                        value={rawQcAnalysis.samtools_result.quickcheck_ok ? "PASS" : "Issue detected"}
                        tone={rawQcAnalysis.samtools_result.quickcheck_ok ? "good" : "warn"}
                      />
                      <MetricTile
                        label="Index"
                        value={rawQcAnalysis.samtools_result.index_path ? "Created / available" : "n/a"}
                        tone="neutral"
                      />
                    </div>
                    <div className="resultSectionSplit">
                      <article className="miniCard">
                        <h3>samtools stats highlights</h3>
                        <div className="resultList">
                          {rawQcAnalysis.samtools_result.stats_highlights.map((item) => (
                            <article key={item.label} className="resultListItem resultListStatic">
                              <strong>{item.label}</strong>
                              <span>{item.value}</span>
                            </article>
                          ))}
                        </div>
                      </article>
                      <article className="miniCard">
                        <h3>idxstats preview</h3>
                        <div className="resultList">
                          {rawQcAnalysis.samtools_result.idxstats_rows.map((row) => (
                            <article key={`${row.contig}-${row.length_bp}`} className="resultListItem resultListStatic">
                              <strong>{row.contig}</strong>
                              <span>
                                mapped {row.mapped} | unmapped {row.unmapped} | length {row.length_bp}
                              </span>
                            </article>
                          ))}
                        </div>
                      </article>
                    </div>
                    {rawQcAnalysis.samtools_result.warnings.length ? (
                      <div className="resultList">
                        {rawQcAnalysis.samtools_result.warnings.map((warning, index) => (
                          <article key={`${warning}-${index}`} className="miniCard">
                            <h3>Warning</h3>
                            <p>{warning}</p>
                          </article>
                        ))}
                      </div>
                    ) : null}
                  </>
                ) : (
                  <p className="emptyState">No samtools result is available for the current raw-QC session.</p>
                )}
              </div>
            </section>
          ) : null}
          {analysis && activeStudioView === "candidates" ? (
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
          ) : null}

          {analysis && activeStudioView === "acmg" ? (
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
          ) : null}

          {analysis && activeStudioView === "provenance" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>Analysis Provenance</h2>
              </div>
              <div className="studioCanvasBody">
                <div className="resultMetricGrid">
                  <MetricTile label="Annotation scope" value={annotationScope} />
                  <MetricTile label="Annotation limit" value={annotationScope === "all" ? annotationLimit || "n/a" : "representative"} />
                  <MetricTile label="References" value={String(analysis.references.length)} />
                  <MetricTile label="Annotations" value={String(analysis.annotations.length)} />
                </div>
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
          ) : null}

          {analysis && activeStudioView === "qc" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>QC Summary</h2>
              </div>
              <div className="studioCanvasBody">
                <div className="resultMetricGrid">
                  <MetricTile label="PASS rate" value={formatPercent(qcMetrics?.pass_rate)} tone="good" />
                  <MetricTile
                    label="Ti/Tv"
                    value={formatNumber(qcMetrics?.transition_transversion_ratio)}
                    tone="neutral"
                  />
                  <MetricTile
                    label="Missing GT"
                    value={formatPercent(qcMetrics?.missing_gt_rate)}
                    tone="warn"
                  />
                  <MetricTile
                    label="Het/HomAlt"
                    value={formatNumber(qcMetrics?.het_hom_alt_ratio)}
                    tone="neutral"
                  />
                  <MetricTile
                    label="Multi-allelic"
                    value={formatPercent(qcMetrics?.multi_allelic_rate)}
                    tone="neutral"
                  />
                  <MetricTile
                    label="Symbolic ALT"
                    value={formatPercent(qcMetrics?.symbolic_alt_rate)}
                    tone="neutral"
                  />
                  <MetricTile label="SNV fraction" value={formatPercent(qcMetrics?.snv_fraction)} tone="good" />
                  <MetricTile label="Indel fraction" value={formatPercent(qcMetrics?.indel_fraction)} tone="neutral" />
                </div>
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
          ) : null}

          {analysis && activeStudioView === "coverage" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>Clinical Annotation Coverage</h2>
              </div>
              <div className="studioCanvasBody">
                <div className="resultList">
                  {clinicalCoverage.map((item) => (
                    <article key={item.label} className="resultListItem resultListStatic">
                      <strong>{item.label}</strong>
                      <span>{item.detail}</span>
                    </article>
                  ))}
                </div>
              </div>
            </section>
          ) : null}

          {analysis && activeStudioView === "snpeff" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>SnpEff Review</h2>
              </div>
              <div className="studioCanvasBody">
                {analysis.snpeff_result ? (
                  <>
                    <div className="resultMetricGrid">
                      <MetricTile label="Genome DB" value={analysis.snpeff_result.genome} tone="good" />
                      <MetricTile label="Preview rows" value={String(analysis.snpeff_result.parsed_records.length)} tone="neutral" />
                      <MetricTile label="Tool" value={analysis.snpeff_result.tool} tone="neutral" />
                    </div>
                    <div className="resultList">
                      {analysis.snpeff_result.parsed_records.map((record, index) => (
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
                    <div className="resultActionRow">
                      <a
                        className="sourceAddButton"
                        href={`file://${analysis.snpeff_result.output_path}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open annotated VCF
                      </a>
                    </div>
                  </>
                ) : (
                  <p className="emptyState">No auxiliary SnpEff result is available for the current analysis.</p>
                )}
              </div>
            </section>
          ) : null}

          {analysis && activeStudioView === "plink" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>PLINK</h2>
              </div>
              <div className="studioCanvasBody">
                <div className="resultMetricGrid">
                  <MetricTile label="Mode" value="QC" tone="good" />
                  <MetricTile label="Source" value={analysis.facts.file_name} tone="neutral" />
                  <MetricTile
                    label="Existing result"
                    value={analysis.plink_result ? "Available" : "Not run yet"}
                    tone={analysis.plink_result ? "good" : "neutral"}
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
                        <span>Output prefix</span>
                        <input
                          type="text"
                          value={plinkConfig.outputPrefix}
                          onChange={(event) =>
                            setPlinkConfig((current) => ({ ...current, outputPrefix: event.target.value }))
                          }
                        />
                      </label>
                      <label className="field compactField">
                        <span>Frequency summary</span>
                        <input
                          type="checkbox"
                          checked={plinkConfig.runFreq}
                          onChange={(event) =>
                            setPlinkConfig((current) => ({ ...current, runFreq: event.target.checked }))
                          }
                        />
                      </label>
                      <label className="field compactField">
                        <span>Missingness summary</span>
                        <input
                          type="checkbox"
                          checked={plinkConfig.runMissing}
                          onChange={(event) =>
                            setPlinkConfig((current) => ({ ...current, runMissing: event.target.checked }))
                          }
                        />
                      </label>
                      <label className="field compactField">
                        <span>Hardy-Weinberg summary</span>
                        <input
                          type="checkbox"
                          checked={plinkConfig.runHardy}
                          onChange={(event) =>
                            setPlinkConfig((current) => ({ ...current, runHardy: event.target.checked }))
                          }
                        />
                      </label>
                      <label className="field compactField">
                        <span>Allow extra chr labels</span>
                        <input
                          type="checkbox"
                          checked={plinkConfig.allowExtraChr}
                          onChange={(event) =>
                            setPlinkConfig((current) => ({ ...current, allowExtraChr: event.target.checked }))
                          }
                        />
                      </label>
                    </div>
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
                    disabled={plinkRunning || !analysis.source_vcf_path}
                  >
                    {plinkRunning ? "Running PLINK..." : "Run PLINK"}
                  </button>
                </div>
                {analysis.plink_result ? (
                  <>
                    <div className="resultMetricGrid">
                      <MetricTile
                        label="Samples"
                        value={
                          analysis.plink_result.sample_count != null ? String(analysis.plink_result.sample_count) : "n/a"
                        }
                        tone="neutral"
                      />
                      <MetricTile
                        label="Variants"
                        value={
                          analysis.plink_result.variant_count != null ? String(analysis.plink_result.variant_count) : "n/a"
                        }
                        tone="neutral"
                      />
                      <MetricTile label="Freq rows" value={String(analysis.plink_result.freq_rows.length)} tone="good" />
                      <MetricTile
                        label="Missing rows"
                        value={String(analysis.plink_result.missing_rows.length)}
                        tone="good"
                      />
                      <MetricTile label="Hardy rows" value={String(analysis.plink_result.hardy_rows.length)} tone="good" />
                      <MetricTile label="Warnings" value={String(analysis.plink_result.warnings.length)} tone="neutral" />
                    </div>
                    <div className="resultList">
                      {analysis.plink_result.freq_rows.slice(0, 5).map((row, index) => (
                        <article key={`plink-freq-${row.variant_id}-${index}`} className="resultListItem resultListStatic">
                          <strong>
                            {row.chrom}:{row.variant_id}
                          </strong>
                          <span>
                            {row.ref_allele}&gt;{row.alt_allele} | AF{" "}
                            {row.alt_freq != null ? row.alt_freq.toFixed(4) : "n/a"} | OBS{" "}
                            {row.observation_count ?? "n/a"}
                          </span>
                        </article>
                      ))}
                      {analysis.plink_result.missing_rows.slice(0, 3).map((row, index) => (
                        <article
                          key={`plink-missing-${row.sample_id}-${index}`}
                          className="resultListItem resultListStatic"
                        >
                          <strong>Missingness {row.sample_id}</strong>
                          <span>
                            {row.missing_genotype_count} missing / {row.observation_count} obs | rate{" "}
                            {(row.missing_rate * 100).toFixed(2)}%
                          </span>
                        </article>
                      ))}
                      {analysis.plink_result.hardy_rows.slice(0, 5).map((row, index) => (
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
                      {analysis.plink_result.warnings.map((warning, index) => (
                        <article key={`plink-warning-${index}`} className="resultListItem resultListStatic">
                          <strong>Warning {index + 1}</strong>
                          <span>{warning}</span>
                        </article>
                      ))}
                    </div>
                    <div className="resultActionRow">
                      {analysis.plink_result.freq_path ? (
                        <a
                          className="sourceAddButton"
                          href={`${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(
                            analysis.plink_result.freq_path,
                          )}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open afreq
                        </a>
                      ) : null}
                      {analysis.plink_result.missing_path ? (
                        <a
                          className="sourceAddButton"
                          href={`${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(
                            analysis.plink_result.missing_path,
                          )}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open smiss
                        </a>
                      ) : null}
                      {analysis.plink_result.hardy_path ? (
                        <a
                          className="sourceAddButton"
                          href={`${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(
                            analysis.plink_result.hardy_path,
                          )}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open hardy
                        </a>
                      ) : null}
                      {analysis.plink_result.log_path ? (
                        <a
                          className="sourceAddButton"
                          href={`${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(
                            analysis.plink_result.log_path,
                          )}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open log
                        </a>
                      ) : null}
                    </div>
                  </>
                ) : (
                  <p className="emptyState">No PLINK result is available yet. Configure the run and execute it from this card.</p>
                )}
              </div>
            </section>
          ) : null}

          {analysis && activeStudioView === "liftover" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>LiftOver Review</h2>
              </div>
              <div className="studioCanvasBody">
                {analysis.liftover_result ? (
                  <>
                    <div className="resultMetricGrid">
                      <MetricTile label="Source build" value={analysis.liftover_result.source_build ?? "unknown"} tone="neutral" />
                      <MetricTile label="Target build" value={analysis.liftover_result.target_build ?? "unknown"} tone="good" />
                      <MetricTile label="Lifted records" value={String(analysis.liftover_result.lifted_record_count ?? 0)} tone="good" />
                      <MetricTile label="Rejected records" value={String(analysis.liftover_result.rejected_record_count ?? 0)} tone="neutral" />
                      <MetricTile label="Warnings" value={String(analysis.liftover_result.warnings.length)} tone="neutral" />
                      <MetricTile label="Tool" value={analysis.liftover_result.tool} tone="neutral" />
                    </div>
                    <div className="resultList">
                      {analysis.liftover_result.parsed_records.length ? (
                        analysis.liftover_result.parsed_records.map((record, index) => (
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
                      {analysis.liftover_result.warnings.length
                        ? analysis.liftover_result.warnings.map((warning, index) => (
                            <article key={`liftover-warning-${index}`} className="resultListItem resultListStatic">
                              <strong>Warning {index + 1}</strong>
                              <span>{warning}</span>
                            </article>
                          ))
                        : null}
                    </div>
                    <div className="resultActionRow">
                      <a
                        className="sourceAddButton"
                        href={`${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(
                          analysis.liftover_result.output_path,
                        )}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open lifted VCF
                      </a>
                      <a
                        className="sourceAddButton"
                        href={`${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(
                          analysis.liftover_result.reject_path,
                        )}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open reject VCF
                      </a>
                    </div>
                  </>
                ) : (
                  <p className="emptyState">No liftover result is available for the current analysis.</p>
                )}
              </div>
            </section>
          ) : null}

          {analysis && activeStudioView === "ldblockshow" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>LD Block Review</h2>
              </div>
              <div className="studioCanvasBody">
                {analysis.ldblockshow_result ? (
                  <>
                    <div className="resultMetricGrid">
                      <MetricTile label="Region" value={analysis.ldblockshow_result.region} tone="good" />
                      <MetricTile
                        label="Tried regions"
                        value={String(analysis.ldblockshow_result.attempted_regions?.length ?? 0)}
                        tone="neutral"
                      />
                      <MetricTile
                        label="Site rows"
                        value={String(analysis.ldblockshow_result.site_row_count ?? 0)}
                        tone="neutral"
                      />
                      <MetricTile
                        label="Triangle pairs"
                        value={String(analysis.ldblockshow_result.triangle_pair_count ?? 0)}
                        tone="neutral"
                      />
                      <MetricTile
                        label="Warnings"
                        value={String(analysis.ldblockshow_result.warnings.length)}
                        tone="neutral"
                      />
                      <MetricTile label="Tool" value={analysis.ldblockshow_result.tool} tone="neutral" />
                    </div>
                    <div className="resultList">
                      {analysis.ldblockshow_result.attempted_regions?.length ? (
                        <article className="resultListItem resultListStatic">
                          <strong>Attempted regions</strong>
                          <span>{analysis.ldblockshow_result.attempted_regions.join(" -> ")}</span>
                        </article>
                      ) : null}
                      {analysis.ldblockshow_result.warnings.length ? (
                        analysis.ldblockshow_result.warnings.map((warning, index) => (
                          <article key={`ldblockshow-warning-${index}`} className="resultListItem resultListStatic">
                            <strong>Warning {index + 1}</strong>
                            <span>{warning}</span>
                          </article>
                        ))
                      ) : (
                        <p className="emptyState">No LDBlockShow warnings were reported.</p>
                      )}
                    </div>
                    <div className="resultActionRow">
                      {analysis.ldblockshow_result.svg_path ? (
                        <a
                          className="sourceAddButton"
                          href={`${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(
                            analysis.ldblockshow_result.svg_path,
                          )}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open LD SVG
                        </a>
                      ) : null}
                      {analysis.ldblockshow_result.block_path ? (
                        <a
                          className="sourceAddButton"
                          href={`${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(
                            analysis.ldblockshow_result.block_path,
                          )}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open block table
                        </a>
                      ) : null}
                      {analysis.ldblockshow_result.site_path ? (
                        <a
                          className="sourceAddButton"
                          href={`${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(
                            analysis.ldblockshow_result.site_path,
                          )}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open site table
                        </a>
                      ) : null}
                    </div>
                  </>
                ) : (
                  <p className="emptyState">No LDBlockShow result is available for the current analysis.</p>
                )}
              </div>
            </section>
          ) : null}

          {analysis && activeStudioView === "table" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>Variant Table</h2>
                <span className="pill">{searchedAnnotations.length} rows</span>
              </div>
              <div className="studioCanvasBody">
                <div className="resultList">
                  {filteringSummary.map((item) => (
                    <article key={item.label} className="resultListItem resultListStatic">
                      <strong>{item.label}</strong>
                      <span>{item.detail}</span>
                    </article>
                  ))}
                </div>
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
          ) : null}

          {analysis && activeStudioView === "symbolic" ? (
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
          ) : null}

          {analysis && activeStudioView === "roh" ? (
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
          ) : null}

          {analysis && activeStudioView === "clinvar" ? (
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
          ) : null}

          {analysis && activeStudioView === "vep" ? (
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
          ) : null}

          {analysis && activeStudioView === "references" ? (
            <section className="notebookPanel studioCanvasPanel">
              <div className="notebookHeader">
                <h2>References</h2>
              </div>
              <div className="studioCanvasBody">
                <ol className="referenceList">
                  {analysis.references.map((item, index) => (
                    <li key={item.id}>
                      <span className="referenceIndex">[{index + 1}]</span>{" "}
                      <a href={item.url} target="_blank" rel="noreferrer">
                        {item.title}
                      </a>
                    </li>
                  ))}
                </ol>
              </div>
            </section>
          ) : null}

          {analysis && activeStudioView === "igv" ? (
            <section className="studioCanvasPanel">
              <IgvBrowser
                buildGuess={analysis.facts.genome_build_guess ?? null}
                annotations={searchedAnnotations}
                selectedIndex={safeSelectedIndex}
              />
            </section>
          ) : null}

          {analysis && activeStudioView === "annotations" ? (
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
                {selectedAnnotation ? (
                  <AnnotationDetailCard item={selectedAnnotation} />
                ) : (
                  <p className="emptyState">No annotation is available for the current selection.</p>
                )}
              </div>
            </section>
          ) : null}
        </section>
      ) : null}
    </main>
  );
}
