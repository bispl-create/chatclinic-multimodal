from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class FromPathRequest(BaseModel):
    vcf_path: str = Field(..., description="Absolute path to a VCF or VCF.gz file")
    annotation_scope: Literal["representative", "all"] = Field(
        default="representative",
        description="Whether to annotate only representative variants or iterate through the whole file.",
    )
    annotation_limit: Optional[int] = Field(
        default=None,
        description="Optional cap on the number of variants to annotate when scope is 'all'.",
    )


class ReferenceItem(BaseModel):
    id: str
    title: str
    source: str
    url: str
    note: str


class RecommendationItem(BaseModel):
    id: str
    title: str
    rationale: str
    action: str
    priority: str


class VariantExample(BaseModel):
    contig: str
    pos_1based: int
    ref: str
    alts: list[str]
    genotype: str
    variant_class: str


class TranscriptAnnotation(BaseModel):
    transcript_id: str
    transcript_biotype: str
    canonical: str
    exon: str
    intron: str
    hgvsc: str
    hgvsp: str
    protein_id: str
    amino_acids: str
    codons: str


class VariantAnnotation(BaseModel):
    contig: str
    pos_1based: int
    ref: str
    alts: list[str]
    genotype: str
    rsid: str
    gene: str
    consequence: str
    transcript_id: str
    transcript_biotype: str
    canonical: str
    exon: str
    intron: str
    hgvsc: str
    hgvsp: str
    protein_id: str
    amino_acids: str
    codons: str
    transcript_options: list[TranscriptAnnotation]
    clinical_significance: str
    maf: str
    clinvar_accession: str
    clinvar_review_status: str
    clinvar_conditions: str
    gnomad_af: str
    source_url: str
    cadd_raw_score: Optional[float] = None
    cadd_phred: Optional[float] = None
    cadd_lookup_status: Optional[str] = None
    revel_score: Optional[float] = None
    revel_lookup_status: Optional[str] = None


class QualityControlMetrics(BaseModel):
    pass_rate: Optional[float]
    missing_gt_rate: Optional[float]
    multi_allelic_rate: Optional[float]
    symbolic_alt_rate: Optional[float]
    snv_fraction: Optional[float]
    indel_fraction: Optional[float]
    transition_transversion_ratio: Optional[float]
    het_hom_alt_ratio: Optional[float]
    mean_dp: Optional[float]
    mean_gq: Optional[float]
    records_with_dp_rate: Optional[float]
    records_with_gq_rate: Optional[float]


class AnalysisFacts(BaseModel):
    file_name: str
    vcf_version: Optional[str]
    genome_build_guess: Optional[str]
    samples: list[str]
    contigs: list[dict[str, Any]]
    record_count: int
    chrom_counts: dict[str, int]
    variant_types: dict[str, int]
    genotype_counts: dict[str, int]
    filter_counts: dict[str, int]
    qc: QualityControlMetrics
    position_range_1based: list[int]
    example_variants: list[VariantExample]
    warnings: list[str]


class RohSegment(BaseModel):
    sample: str
    contig: str
    start_1based: int
    end_1based: int
    length_bp: int
    marker_count: int
    quality: float


class RankedCandidate(BaseModel):
    item: VariantAnnotation
    score: int
    in_roh: bool


class CountSummaryItem(BaseModel):
    label: str
    count: int


class DetailedCountSummaryItem(BaseModel):
    label: str
    count: int
    detail: str


class SymbolicAltExample(BaseModel):
    locus: str
    gene: str
    alts: list[str]
    consequence: str
    genotype: str


class SymbolicAltSummary(BaseModel):
    count: int
    examples: list[SymbolicAltExample] = []


class ToolInfo(BaseModel):
    name: str
    description: str
    task: str
    modality: str
    approval_required: bool = False
    source: str = "plugin"


class AnalysisResponse(BaseModel):
    analysis_id: str
    facts: AnalysisFacts
    annotations: list[VariantAnnotation]
    roh_segments: list[RohSegment]
    source_vcf_path: Optional[str] = None
    snpeff_result: Optional[SnpEffResponse] = None
    plink_result: Optional[PlinkResponse] = None
    liftover_result: Optional[GatkLiftoverVcfResponse] = None
    ldblockshow_result: Optional[LDBlockShowResponse] = None
    candidate_variants: list[RankedCandidate] = []
    clinvar_summary: list[CountSummaryItem] = []
    consequence_summary: list[CountSummaryItem] = []
    clinical_coverage_summary: list[DetailedCountSummaryItem] = []
    filtering_summary: list[DetailedCountSummaryItem] = []
    symbolic_alt_summary: Optional[SymbolicAltSummary] = None
    references: list[ReferenceItem]
    recommendations: list[RecommendationItem]
    ui_cards: list[dict[str, Any]]
    draft_answer: str
    used_tools: list[str] = []
    tool_registry: list[ToolInfo] = []


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[AnalysisResponse] = None
    error: Optional[str] = None


class ChatTurn(BaseModel):
    role: str
    content: str


class AnalysisChatRequest(BaseModel):
    question: str
    analysis: AnalysisResponse
    history: list[ChatTurn] = []
    studio_context: dict[str, Any] = {}


class AnalysisChatResponse(BaseModel):
    answer: str
    citations: list[str]
    used_fallback: bool
    used_tools: list[str] = []
    requested_view: Optional[str] = None
    plink_result: Optional[PlinkResponse] = None
    liftover_result: Optional[GatkLiftoverVcfResponse] = None
    ldblockshow_result: Optional[LDBlockShowResponse] = None


class RawQcFacts(BaseModel):
    file_name: str
    file_kind: str
    total_sequences: Optional[int] = None
    filtered_sequences: Optional[int] = None
    poor_quality_sequences: Optional[int] = None
    sequence_length: Optional[str] = None
    gc_content: Optional[float] = None
    encoding: Optional[str] = None


class RawQcModule(BaseModel):
    name: str
    status: str
    detail: str = ""


class RawQcResponse(BaseModel):
    analysis_id: str
    source_raw_path: Optional[str] = None
    facts: RawQcFacts
    modules: list[RawQcModule]
    samtools_result: Optional[SamtoolsResponse] = None
    draft_answer: str
    report_html_path: Optional[str] = None
    report_zip_path: Optional[str] = None
    used_tools: list[str] = []
    tool_registry: list[ToolInfo] = []


class RawQcChatRequest(BaseModel):
    question: str
    analysis: RawQcResponse
    history: list[ChatTurn] = []


class RawQcChatResponse(BaseModel):
    answer: str
    citations: list[str]
    used_fallback: bool
    samtools_result: Optional[SamtoolsResponse] = None


class SummaryStatsFieldMapping(BaseModel):
    chrom: Optional[str] = None
    pos: Optional[str] = None
    rsid: Optional[str] = None
    effect_allele: Optional[str] = None
    other_allele: Optional[str] = None
    beta_or: Optional[str] = None
    standard_error: Optional[str] = None
    p_value: Optional[str] = None
    n: Optional[str] = None
    eaf: Optional[str] = None


class SummaryStatsResponse(BaseModel):
    analysis_id: str
    source_stats_path: Optional[str] = None
    file_name: str
    genome_build: str = "unknown"
    trait_type: str = "unknown"
    delimiter: str = "tab"
    detected_columns: list[str]
    mapped_fields: SummaryStatsFieldMapping
    row_count: int = 0
    preview_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    draft_answer: str
    used_tools: list[str] = []
    tool_registry: list[ToolInfo] = []


class SummaryStatsRowsRequest(BaseModel):
    source_stats_path: str
    offset: int = 0
    limit: int = 200


class SummaryStatsRowsResponse(BaseModel):
    rows: list[dict[str, str]]
    offset: int
    limit: int
    returned: int
    has_more: bool


class WorkflowStartRequest(BaseModel):
    file_name: str


class WorkflowReplyRequest(BaseModel):
    file_name: str
    message: str


class WorkflowAgentResponse(BaseModel):
    assistant_message: str
    should_start_analysis: bool
    parsed_scope: Literal["representative", "all"]
    parsed_limit: Optional[int] = None
    used_fallback: bool
    model: str


class FilterRequest(BaseModel):
    vcf_path: str = Field(..., description="Absolute path to the input VCF or VCF.gz")
    tool: Literal["bcftools", "gatk"]
    expression: str = Field(..., description="Filter expression for bcftools or GATK VariantFiltration")
    filter_name: str = Field(default="LowQual", description="FILTER label to attach when soft filtering")
    mode: Literal["soft_filter", "include", "exclude"] = Field(
        default="soft_filter",
        description="Filtering mode. GATK currently supports only soft_filter.",
    )
    output_prefix: Optional[str] = Field(
        default=None,
        description="Optional output prefix. Files are written under the app filter output directory.",
    )


class FilterResponse(BaseModel):
    tool: str
    input_path: str
    output_path: str
    index_path: Optional[str]
    command_preview: str


class SnpEffRequest(BaseModel):
    vcf_path: str = Field(..., description="Absolute path to the input VCF or VCF.gz")
    genome: str = Field(default="GRCh37.75", description="SnpEff genome database key")
    output_prefix: Optional[str] = Field(
        default=None,
        description="Optional output prefix. Files are written under the app snpEff output directory.",
    )
    parse_limit: int = Field(default=25, description="Maximum number of annotated records to parse for preview")


class SnpEffAnnEntry(BaseModel):
    allele: str
    annotation: str
    impact: str
    gene_name: str
    gene_id: str
    feature_type: str
    feature_id: str
    transcript_biotype: str
    rank: str
    hgvs_c: str
    hgvs_p: str


class SnpEffAnnotatedRecord(BaseModel):
    contig: str
    pos_1based: int
    ref: str
    alt: str
    ann: list[SnpEffAnnEntry]


class SnpEffResponse(BaseModel):
    tool: str
    genome: str
    input_path: str
    output_path: str
    index_path: Optional[str]
    command_preview: str
    parsed_records: list[SnpEffAnnotatedRecord]


class SamtoolsRequest(BaseModel):
    raw_path: str = Field(..., description="Absolute path to the input BAM, SAM, or CRAM file")
    original_name: Optional[str] = Field(default=None, description="Optional original file name for display")
    create_index_if_possible: bool = Field(
        default=True,
        description="Create an index for BAM or CRAM inputs when no index is already present.",
    )
    stats_limit: int = Field(default=12, description="Maximum number of samtools stats summary lines to keep")
    idxstats_limit: int = Field(default=12, description="Maximum number of idxstats rows to keep")


class SamtoolsStatsItem(BaseModel):
    label: str
    value: str


class SamtoolsIdxstatsRow(BaseModel):
    contig: str
    length_bp: int
    mapped: int
    unmapped: int


class SamtoolsResponse(BaseModel):
    tool: str
    input_path: str
    display_name: str
    file_kind: str
    command_preview: str
    quickcheck_ok: Optional[bool] = None
    total_reads: Optional[int] = None
    mapped_reads: Optional[int] = None
    mapped_rate: Optional[float] = None
    paired_reads: Optional[int] = None
    properly_paired_reads: Optional[int] = None
    properly_paired_rate: Optional[float] = None
    singleton_reads: Optional[int] = None
    index_path: Optional[str] = None
    stats_highlights: list[SamtoolsStatsItem] = []
    idxstats_rows: list[SamtoolsIdxstatsRow] = []
    warnings: list[str] = []


class PlinkRequest(BaseModel):
    vcf_path: str = Field(..., description="Absolute path to the input VCF or VCF.gz")
    output_prefix: Optional[str] = Field(
        default=None,
        description="Optional output prefix. Files are written under the app PLINK output directory.",
    )
    allow_extra_chr: bool = Field(default=True, description="Allow noncanonical chromosome labels such as chr1.")
    freq_limit: int = Field(default=12, description="Maximum number of allele frequency rows to keep")
    missing_limit: int = Field(default=12, description="Maximum number of missingness rows to keep")
    hardy_limit: int = Field(default=12, description="Maximum number of Hardy-Weinberg rows to keep")


class PlinkFreqRow(BaseModel):
    chrom: str
    variant_id: str
    ref_allele: str
    alt_allele: str
    alt_freq: Optional[float] = None
    observation_count: Optional[int] = None


class PlinkMissingRow(BaseModel):
    sample_id: str
    missing_genotype_count: int
    observation_count: int
    missing_rate: float


class PlinkHardyRow(BaseModel):
    chrom: str
    variant_id: str
    observed_hets: Optional[int] = None
    expected_hets: Optional[float] = None
    p_value: Optional[float] = None


class PlinkResponse(BaseModel):
    tool: str
    input_path: str
    command_preview: str
    output_prefix: str
    log_path: Optional[str] = None
    freq_path: Optional[str] = None
    missing_path: Optional[str] = None
    hardy_path: Optional[str] = None
    variant_count: Optional[int] = None
    sample_count: Optional[int] = None
    freq_rows: list[PlinkFreqRow] = []
    missing_rows: list[PlinkMissingRow] = []
    hardy_rows: list[PlinkHardyRow] = []
    warnings: list[str] = []


class GatkLiftoverVcfRequest(BaseModel):
    vcf_path: str = Field(..., description="Absolute path to the input VCF or VCF.gz")
    target_reference_fasta: str = Field(..., description="Absolute path to the target reference FASTA")
    chain_file: str = Field(..., description="Absolute path to the chain file used for liftover")
    source_build: Optional[str] = Field(default=None, description="Optional source build label such as GRCh37")
    target_build: Optional[str] = Field(default=None, description="Optional target build label such as GRCh38")
    output_prefix: Optional[str] = Field(
        default=None,
        description="Optional output prefix. Files are written under the app liftover output directory.",
    )
    parse_limit: int = Field(default=12, description="Maximum number of lifted records to parse for preview")


class GatkLiftoverRecord(BaseModel):
    contig: str
    pos_1based: int
    ref: str
    alts: list[str]


class GatkLiftoverVcfResponse(BaseModel):
    tool: str
    input_path: str
    source_build: Optional[str] = None
    target_build: Optional[str] = None
    target_reference_fasta: str
    chain_file: str
    output_path: str
    output_index_path: Optional[str] = None
    reject_path: str
    reject_index_path: Optional[str] = None
    command_preview: str
    lifted_record_count: Optional[int] = None
    rejected_record_count: Optional[int] = None
    parsed_records: list[GatkLiftoverRecord] = []
    warnings: list[str] = []


class LDBlockShowRequest(BaseModel):
    vcf_path: str = Field(..., description="Absolute path to the input VCF or VCF.gz")
    region: str = Field(..., description="Target locus in chr:start:end format")
    output_prefix: Optional[str] = Field(
        default=None,
        description="Optional output prefix. Results are written under the app LDBlockShow output directory.",
    )
    sele_var: Literal[1, 2, 3, 4] = Field(default=2, description="LD statistic mode for LDBlockShow")
    block_type: Literal[1, 2, 3, 4, 5] = Field(
        default=5,
        description="Block display mode. Default 5 avoids extra PLINK-based block calling.",
    )
    subgroup_path: Optional[str] = Field(default=None, description="Optional sample subset file")
    gwas_path: Optional[str] = Field(default=None, description="Optional chr pos pvalue file")
    gff_path: Optional[str] = Field(default=None, description="Optional GFF3 annotation file")
    out_png: bool = Field(default=False, description="Request PNG conversion")
    out_pdf: bool = Field(default=False, description="Request PDF conversion")


class LDBlockShowResponse(BaseModel):
    tool: str
    input_path: str
    region: str
    output_prefix: str
    command_preview: str
    svg_path: Optional[str] = None
    png_path: Optional[str] = None
    pdf_path: Optional[str] = None
    block_path: Optional[str] = None
    site_path: Optional[str] = None
    triangle_path: Optional[str] = None
    attempted_regions: list[str] = []
    site_row_count: int = 0
    block_row_count: int = 0
    triangle_pair_count: int = 0
    warnings: list[str] = []


class RPlotRequest(BaseModel):
    vcf_path: str = Field(..., description="Absolute path to the input VCF or VCF.gz")
    output_prefix: Optional[str] = Field(
        default=None,
        description="Optional output prefix. Files are written under the app R plot output directory.",
    )
    density_bin_size: int = Field(default=1_000_000, description="Bin size for CMplot density rendering")


class CmplotAssociationRequest(BaseModel):
    association_path: str = Field(
        ...,
        description="Absolute path to a TSV/CSV table with GWAS-style association columns such as SNP/CHR/BP/P.",
    )
    output_prefix: Optional[str] = Field(
        default=None,
        description="Optional output prefix. Files are written under the app R plot output directory.",
    )


class RPlotArtifact(BaseModel):
    plot_type: str
    title: str
    image_path: str
    api_path: str
    note: str


class RPlotResponse(BaseModel):
    tool: str
    input_path: str
    output_dir: str
    command_preview: str
    artifacts: list[RPlotArtifact]
    warnings: list[str]
