"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";

import DicomInteractiveViewer from "./DicomInteractiveViewer";
import IgvBrowser from "./IgvBrowser";

const NiivueViewer = dynamic(() => import("./NiivueViewer"), { ssr: false });
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

function DicomReviewCard({
  analysis,
  components,
}: {
  analysis: any;
  components: StudioRendererBuilderArgs["components"];
  helpers: StudioRendererBuilderArgs["helpers"];
}) {
  const { StudioMetricGrid, WarningListCard } = components;
  const metadata = Array.isArray(analysis?.metadata_items) ? analysis.metadata_items[0] ?? null : null;
  const preview = metadata?.preview ?? analysis?.artifacts?.dicom_review?.preview ?? null;
  const series = Array.isArray(analysis?.series) ? analysis.series : [];
  const seriesGroups = useMemo(
    () =>
      series.map((item: any, index: number) => ({
        id: String(item.series_instance_uid ?? `series-${index}`),
        label: String(item.series_description ?? item.modality ?? `Series ${index + 1}`),
        files: [
          {
            fileName: String((item.example_files ?? [])[0] ?? metadata?.file_name ?? analysis?.file_name ?? `dicom-${index + 1}.dcm`),
            preview: item.preview ?? preview ?? null,
            preview_presets: item.preview_presets ?? undefined,
          },
        ],
      })),
    [analysis?.file_name, metadata?.file_name, preview, series],
  );
  const fallbackSeriesGroups =
    seriesGroups.length > 0
      ? seriesGroups
      : [
          {
            id: String(metadata?.series_instance_uid ?? "single-series"),
            label: String(metadata?.series_description ?? metadata?.modality ?? "DICOM file"),
            files: [
              {
                fileName: String(metadata?.file_name ?? analysis?.file_name ?? "dicom.dcm"),
                preview: preview ?? null,
              },
            ],
          },
        ];

  return (
    <section className="notebookPanel studioCanvasPanel">
      <div className="notebookHeader">
        <h2>DICOM Review</h2>
        <span className="pill">{analysis?.file_name ?? "dicom"}</span>
      </div>
      <div className="studioCanvasBody">
        <StudioMetricGrid
          items={[
            { label: "Modality", value: String(metadata?.modality ?? "n/a"), tone: "good" },
            { label: "Patient", value: String(metadata?.patient_id ?? "n/a"), tone: "neutral" },
            { label: "Rows", value: String(metadata?.rows ?? "n/a"), tone: "neutral" },
            { label: "Columns", value: String(metadata?.columns ?? "n/a"), tone: "neutral" },
            { label: "Instance", value: String(metadata?.instance_number ?? "n/a"), tone: "neutral" },
            { label: "Series", value: String(series.length), tone: "good" },
          ]}
        />
        <div className="resultSectionSplit" style={{ gridTemplateColumns: "minmax(0, 1.8fr) minmax(280px, 0.9fr)", alignItems: "start" }}>
          <article className="miniCard">
            <h3>Interactive viewer</h3>
            <DicomInteractiveViewer seriesGroups={fallbackSeriesGroups} />
          </article>
          <article className="miniCard" style={{ maxWidth: "100%", overflow: "hidden" }}>
            <h3>Metadata</h3>
            <div className="variantTableWrap summaryStatsTableWrap">
              <table className="variantTable summaryStatsTable">
                <tbody>
                  <tr>
                    <th>Study</th>
                    <td>{String(metadata?.study_description ?? "not available")}</td>
                  </tr>
                  <tr>
                    <th>Series</th>
                    <td>{String(metadata?.series_description ?? "not available")}</td>
                  </tr>
                  <tr>
                    <th>Study UID</th>
                    <td>{String(metadata?.study_instance_uid ?? "not available")}</td>
                  </tr>
                  <tr>
                    <th>Series UID</th>
                    <td>{String(metadata?.series_instance_uid ?? "not available")}</td>
                  </tr>
                  <tr>
                    <th>File</th>
                    <td>{String(metadata?.file_name ?? analysis?.file_name ?? "not available")}</td>
                  </tr>
                  <tr>
                    <th>Series summary</th>
                    <td>
                      {series.length
                        ? series
                            .map((item: any) => `${item.series_description ?? item.modality ?? "Series"} (${item.modality ?? "unknown"}, ${item.instance_count ?? 0} instance(s))`)
                            .join(" | ")
                        : "No series summary is available."}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </article>
        </div>
        <WarningListCard
          warnings={[
            ...(Array.isArray(analysis?.warnings) ? analysis.warnings : []),
            ...(!preview?.available && preview?.message ? [String(preview.message)] : []),
          ]}
          emptyLabel="No DICOM warnings."
        />
      </div>
    </section>
  );
}

function ImageReviewCard({
  analysis,
  components,
}: {
  analysis: any;
  components: StudioRendererBuilderArgs["components"];
}) {
  const { StudioMetricGrid, WarningListCard } = components;
  const exifEntries = Object.entries(analysis?.exif_data ?? {});
  const unsbArtifact = analysis?.artifacts?.unsb_enhancement ?? null;
  const unsbPreview = typeof unsbArtifact?.preview_data_url === "string" ? unsbArtifact.preview_data_url : null;

  return (
    <section className="notebookPanel studioCanvasPanel">
      <div className="notebookHeader">
        <h2>Image Review</h2>
        <span className="pill">{analysis?.file_name ?? "image"}</span>
      </div>
      <div className="studioCanvasBody">
        <StudioMetricGrid
          items={[
            { label: "Format", value: String(analysis?.format_name ?? "n/a"), tone: "good" },
            { label: "Dimensions", value: `${analysis?.width ?? 0}×${analysis?.height ?? 0}`, tone: "neutral" },
            { label: "Color mode", value: String(analysis?.color_mode ?? "n/a"), tone: "neutral" },
            { label: "Bit depth", value: analysis?.bit_depth != null ? String(analysis.bit_depth) : "n/a", tone: "neutral" },
            { label: "File kind", value: String(analysis?.file_kind ?? "IMAGE"), tone: "neutral" },
            { label: "UNSB", value: unsbArtifact ? "Ready" : "Not run", tone: unsbArtifact ? "good" : "neutral" },
          ]}
        />
        <div className="resultSectionSplit" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", alignItems: "start" }}>
          {analysis?.preview_data_url ? (
            <article className="miniCard">
              <h3>Original</h3>
              <div style={{ textAlign: "center", padding: "0.5rem 0" }}>
                <img
                  src={analysis.preview_data_url}
                  alt={analysis.file_name ?? "preview"}
                  style={{ maxWidth: "100%", maxHeight: "400px", borderRadius: "6px", border: "1px solid var(--border-color, #ddd)" }}
                />
              </div>
            </article>
          ) : null}
          {unsbPreview ? (
            <article className="miniCard">
              <h3>UNSB Enhanced</h3>
              <div style={{ textAlign: "center", padding: "0.5rem 0" }}>
                <img
                  src={unsbPreview}
                  alt={`${analysis?.file_name ?? "image"} enhanced`}
                  style={{ maxWidth: "100%", maxHeight: "400px", borderRadius: "6px", border: "1px solid var(--border-color, #ddd)" }}
                />
              </div>
            </article>
          ) : (
            <article className="miniCard">
              <h3>UNSB Enhanced</h3>
              <p className="emptyState">Run <code>@unsb</code> to attach an enhanced preview.</p>
            </article>
          )}
        </div>
        <div className="resultSectionSplit" style={{ gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 0.8fr)", alignItems: "start" }}>
          <article className="miniCard" style={{ maxWidth: "100%", overflow: "hidden" }}>
            <h3>{exifEntries.length ? "EXIF data" : "Metadata"}</h3>
            <div className="variantTableWrap summaryStatsTableWrap">
              <table className="variantTable summaryStatsTable">
                <tbody>
                  <tr><th>File</th><td>{String(analysis?.file_name ?? "n/a")}</td></tr>
                  <tr><th>Format</th><td>{String(analysis?.format_name ?? "n/a")}</td></tr>
                  <tr><th>Size</th><td>{analysis?.width ?? 0}×{analysis?.height ?? 0} px</td></tr>
                  <tr><th>Color</th><td>{String(analysis?.color_mode ?? "n/a")}</td></tr>
                  {exifEntries.slice(0, 12).map(([key, value]) => (
                    <tr key={key}><th>{key}</th><td>{String(value ?? "")}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
          <article className="miniCard" style={{ maxWidth: "100%", overflow: "hidden" }}>
            <h3>UNSB Run</h3>
            <div className="variantTableWrap summaryStatsTableWrap">
              <table className="variantTable summaryStatsTable">
                <tbody>
                  <tr><th>Status</th><td>{unsbArtifact ? "Ready" : "Not run"}</td></tr>
                  <tr><th>Device</th><td>{String(unsbArtifact?.device ?? "n/a")}</td></tr>
                  <tr><th>Checkpoint</th><td>{String(unsbArtifact?.epoch ?? "n/a")}</td></tr>
                  <tr><th>Steps</th><td>{String(unsbArtifact?.num_timesteps ?? "n/a")}</td></tr>
                  <tr><th>Tau</th><td>{String(unsbArtifact?.tau ?? "n/a")}</td></tr>
                  <tr><th>Output</th><td>{String(unsbArtifact?.output_image_path ?? "n/a")}</td></tr>
                </tbody>
              </table>
            </div>
          </article>
        </div>
        <WarningListCard
          warnings={Array.isArray(analysis?.warnings) ? analysis.warnings : []}
          emptyLabel="No image warnings."
        />
      </div>
    </section>
  );
}

function NiftiReviewCard({
  analysis,
  apiBase,
  components,
}: {
  analysis: any;
  apiBase: string;
  components: StudioRendererBuilderArgs["components"];
}) {
  const { StudioMetricGrid, WarningListCard } = components;
  const shapeStr = analysis?.shape?.join(" × ") ?? "n/a";
  const voxelStr = analysis?.voxel_dims?.slice(0, 3).join(" × ") ?? "n/a";
  const fovStr = analysis?.fov_mm?.map((v: number) => v.toFixed(1)).join(" × ") ?? "n/a";
  const niftiFileUrl = analysis?.source_nifti_path
    ? `${apiBase.replace(/\/$/, "")}/api/v1/files?path=${encodeURIComponent(analysis.source_nifti_path)}`
    : null;

  return (
    <section className="notebookPanel studioCanvasPanel">
      <div className="notebookHeader">
        <h2>NIfTI Review</h2>
        <span className="pill">{analysis?.file_name ?? "nifti"}</span>
      </div>
      <div className="studioCanvasBody">
        <StudioMetricGrid
          items={[
            { label: "Shape", value: shapeStr, tone: "good" },
            { label: "Voxel (mm)", value: voxelStr, tone: "neutral" },
            { label: "FOV (mm)", value: fovStr, tone: "neutral" },
            { label: "Orientation", value: String(analysis?.orientation ?? "n/a"), tone: "neutral" },
            { label: "Datatype", value: String(analysis?.datatype ?? "n/a"), tone: "neutral" },
            { label: "4D volume", value: analysis?.is_4d ? "Yes" : "No", tone: analysis?.is_4d ? "warn" : "neutral" },
          ]}
        />
        {niftiFileUrl ? (
          <article className="miniCard">
            <h3>Interactive Viewer</h3>
            <NiivueViewer niftiUrl={niftiFileUrl} />
          </article>
        ) : analysis?.preview_data_url ? (
          <article className="miniCard">
            <h3>Slice Preview (Axial / Coronal / Sagittal)</h3>
            <div style={{ textAlign: "center", padding: "0.5rem 0" }}>
              <img
                src={analysis.preview_data_url}
                alt={analysis.file_name ?? "preview"}
                style={{ maxWidth: "100%", maxHeight: "400px", borderRadius: "6px", border: "1px solid var(--border-color, #ddd)" }}
              />
            </div>
          </article>
        ) : null}
        <article className="miniCard" style={{ maxWidth: "100%", overflow: "hidden" }}>
          <h3>Metadata</h3>
          <div className="variantTableWrap summaryStatsTableWrap">
            <table className="variantTable summaryStatsTable">
              <tbody>
                <tr><th>File</th><td>{String(analysis?.file_name ?? "n/a")}</td></tr>
                <tr><th>Shape</th><td>{shapeStr}</td></tr>
                <tr><th>Voxel size</th><td>{voxelStr} mm</td></tr>
                <tr><th>FOV</th><td>{fovStr} mm</td></tr>
                <tr><th>Orientation</th><td>{String(analysis?.orientation ?? "n/a")}</td></tr>
                <tr><th>Data type</th><td>{String(analysis?.datatype ?? "n/a")}</td></tr>
              </tbody>
            </table>
          </div>
        </article>
        <WarningListCard
          warnings={Array.isArray(analysis?.warnings) ? analysis.warnings : []}
          emptyLabel="No NIfTI warnings."
        />
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// FHIR helpers
// ---------------------------------------------------------------------------

function fhirFormatAge(birthDate?: string) {
  if (!birthDate || birthDate === "n/a") return "n/a";
  const birth = new Date(birthDate);
  if (Number.isNaN(birth.getTime())) return "n/a";
  const now = new Date();
  let age = now.getFullYear() - birth.getFullYear();
  const monthDiff = now.getMonth() - birth.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && now.getDate() < birth.getDate())) age -= 1;
  return String(age);
}

function fhirDateToNumber(value?: string) {
  if (!value || value === "n/a") return null;
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : null;
}

function fhirClampPercent(value: number) {
  return Math.max(0, Math.min(100, value));
}

function FhirMetadataRows({ rows }: { rows: Array<{ label: string; value: string | undefined | null }> }) {
  return (
    <div className="metadataTable">
      {rows.map((row) => (
        <div key={row.label} className="metadataRow">
          <span className="metadataLabel">{row.label}</span>
          <span className="metadataValue">{row.value && String(row.value).trim() ? row.value : "n/a"}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// FHIR Browser Card
// ---------------------------------------------------------------------------

type FhirPatientArtifact = { resource_type?: string; id?: string; full_name?: string; gender?: string; birth_date?: string; active?: string; managing_organization?: string; identifiers?: Array<{ system?: string; value?: string; use?: string }>; telecom?: Array<{ system?: string; value?: string; use?: string }>; addresses?: Array<{ line?: string; city?: string; state?: string; postalCode?: string; country?: string }> };
type FhirAllergyArtifact = { count?: number; items?: Array<{ substance?: string; criticality?: string; clinical_status?: string; verification_status?: string }> };
type FhirVitalArtifact = { items?: Array<{ label?: string; value?: string; effective?: string; status?: string }> };
type FhirMedicationArtifact = { count?: number; items?: Array<{ medication?: string; status?: string; intent?: string; date?: string; dosage?: string; start?: string; end?: string; duration_days?: number | null; current?: boolean }> };
type FhirTimelineArtifact = { events?: Array<{ type?: string; label?: string; start?: string; end?: string; status?: string }> };
type FhirLabArtifact = { series?: Array<{ label?: string; points?: Array<{ date?: string; value?: number; unit?: string; low?: number | null; high?: number | null }> }>; latest?: Array<{ label?: string; value?: number | string; unit?: string; low?: number | null; high?: number | null }> };
type FhirCareTeamArtifact = { practitioners?: Array<{ name?: string; role?: string; contact?: string; organization?: string }>; organizations?: Array<{ name?: string; contact?: string }> };

function FhirBrowserCard({ analysis }: { analysis: any }) {
  const artifacts: Record<string, any> = analysis?.artifacts ?? {};
  const patient = (artifacts.patient as FhirPatientArtifact | undefined) ?? {};
  const allergies = (artifacts.allergies as FhirAllergyArtifact | undefined) ?? {};
  const vitals = (artifacts.vitals as FhirVitalArtifact | undefined) ?? {};
  const medications = (artifacts.medications as FhirMedicationArtifact | undefined) ?? {};
  const timeline = (artifacts.timeline as FhirTimelineArtifact | undefined) ?? {};
  const labs = (artifacts.labs as FhirLabArtifact | undefined) ?? {};
  const careTeam = (artifacts.care_team as FhirCareTeamArtifact | undefined) ?? {};

  const patientInitial = (patient.full_name || "P").charAt(0).toUpperCase();
  const age = fhirFormatAge(patient.birth_date);
  const vitalItems = vitals.items ?? [];
  const allergyItems = allergies.items ?? [];
  const medicationItems = medications.items ?? [];
  const eventItems = timeline.events ?? [];
  const labSeries = labs.series ?? [];
  const carePractitioners = careTeam.practitioners ?? [];
  const careOrganizations = careTeam.organizations ?? [];

  // Timeline extent
  const datedEvents = [...eventItems, ...medicationItems].map((item: any) => fhirDateToNumber("start" in item ? item.start : undefined) ?? fhirDateToNumber("date" in item ? item.date : undefined)).filter((v): v is number => v !== null);
  const timelineMin = datedEvents.length ? Math.min(...datedEvents) : null;
  const timelineMax = datedEvents.length ? Math.max(...datedEvents) : null;
  const timelineSpan = timelineMin !== null && timelineMax !== null && timelineMax > timelineMin ? timelineMax - timelineMin : 1;

  // Lab extent
  const labPoints = labSeries.flatMap((s: any) => s.points ?? []);
  const labDates = labPoints.map((p: any) => fhirDateToNumber(p.date)).filter((v): v is number => v !== null);
  const labValues = labPoints.map((p: any) => p.value).filter((v): v is number => typeof v === "number");
  const labMinDate = labDates.length ? Math.min(...labDates) : null;
  const labMaxDate = labDates.length ? Math.max(...labDates) : null;
  const labMinValue = labValues.length ? Math.min(...labValues) : null;
  const labMaxValue = labValues.length ? Math.max(...labValues) : null;
  const labDateSpan = labMinDate !== null && labMaxDate !== null && labMaxDate > labMinDate ? labMaxDate - labMinDate : 1;
  const labValueSpan = labMinValue !== null && labMaxValue !== null && labMaxValue > labMinValue ? labMaxValue - labMinValue : 1;
  const lineColors = ["#2f80ed", "#f97331", "#16a34a", "#8b5cf6", "#ef4444", "#0f766e"];

  return (
    <section className="notebookPanel studioCanvasPanel">
      <div className="notebookHeader">
        <h2>FHIR Browser</h2>
        <span className="pill">{analysis?.file_name ?? "fhir"}</span>
      </div>
      <div className="studioCanvasBody">
        {/* Summary header: patient + allergies + vitals */}
        <section className="fhirSummaryHeader">
          <article className="artifactCard">
            <strong>FHIR patient overview</strong>
            <div className="patientHero">
              <div className="patientAvatar">{patientInitial}</div>
              <div>
                <div className="patientName">{patient.full_name || "Unknown patient"}</div>
                <div className="mutedText">
                  {patient.gender || "n/a"} / age {age} / Patient ID {patient.id || "n/a"}
                </div>
              </div>
            </div>
            <FhirMetadataRows rows={[
              { label: "Birth Date", value: patient.birth_date },
              { label: "Active", value: patient.active },
              { label: "Managing Org", value: patient.managing_organization },
            ]} />
          </article>

          <article className="artifactCard">
            <strong>AllergyIntolerance alerts</strong>
            {allergyItems.length ? (
              <div className="alertStack">
                {allergyItems.map((item: any, index: number) => (
                  <div key={`allergy-${index}`} className="allergyBadge">
                    <span className="allergyDot" />
                    <div>
                      <strong>{item.substance || "Unknown allergen"}</strong>
                      <p className="mutedText">
                        {item.criticality || "n/a"} / {item.clinical_status || "n/a"}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mutedText">No AllergyIntolerance resources are available.</p>
            )}
          </article>

          <article className="artifactCard">
            <strong>Latest vital signs</strong>
            {vitalItems.length ? (
              <div className="vitalGrid">
                {vitalItems.map((item: any, index: number) => (
                  <div key={`vital-${index}`} className="vitalTile">
                    <span className="vitalLabel">{item.label || "Vital"}</span>
                    <strong>{item.value || "n/a"}</strong>
                    <small>{item.effective || item.status || "n/a"}</small>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mutedText">No latest vital observations are available.</p>
            )}
          </article>
        </section>

        {/* Main grid: timeline + lab chart */}
        <section className="fhirMainGrid">
          <article className="artifactCard">
            <strong>Medical &amp; medication timeline</strong>
            <div className="timelineSection">
              <div className="timelineSectionTitle">Events</div>
              {eventItems.length ? (
                <div className="timelineTrackList">
                  {eventItems.map((item: any, index: number) => {
                    const start = fhirDateToNumber(item.start);
                    const end = fhirDateToNumber(item.end) ?? start;
                    const left = start !== null && timelineMin !== null ? fhirClampPercent(((start - timelineMin) / timelineSpan) * 100) : 0;
                    const width = start !== null && end !== null ? Math.max(8, fhirClampPercent((((end - start) || 1) / timelineSpan) * 100)) : 12;
                    return (
                      <div key={`event-${index}`} className="timelineRow">
                        <div className="timelineLabel">
                          <strong>{item.label || item.type || "Event"}</strong>
                          <small>{item.type || "Event"} / {item.status || "n/a"}</small>
                        </div>
                        <div className="timelineBarArea">
                          <div className="timelineBar eventBar" style={{ left: `${left}%`, width: `${width}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="mutedText">No Encounter or Procedure events are available.</p>
              )}
            </div>

            <div className="timelineSection">
              <div className="timelineSectionTitle">Medications</div>
              {medicationItems.length ? (
                <div className="fhirAccordionList">
                  {medicationItems.map((item: any, index: number) => {
                    const start = fhirDateToNumber(item.start || item.date);
                    const end = fhirDateToNumber(item.end) ?? start;
                    const left = start !== null && timelineMin !== null ? fhirClampPercent(((start - timelineMin) / timelineSpan) * 100) : 0;
                    const width = start !== null && end !== null ? Math.max(8, fhirClampPercent((((end - start) || 1) / timelineSpan) * 100)) : 12;
                    return (
                      <details key={`med-${index}`} className="fhirAccordionItem">
                        <summary>
                          <div className="timelineSummary">
                            <span>{item.medication || "Unknown medication"}</span>
                            <span className={`timelineStatus ${item.current ? "timelineStatusCurrent" : "timelineStatusPast"}`}>
                              {item.current ? "Current" : item.status || "Ended"}
                            </span>
                          </div>
                        </summary>
                        <div className="timelineBarArea timelineBarAreaExpanded">
                          <div className={`timelineBar ${item.current ? "medBarActive" : "medBarPast"}`} style={{ left: `${left}%`, width: `${width}%` }} />
                        </div>
                        <FhirMetadataRows rows={[
                          { label: "Status", value: item.status },
                          { label: "Intent", value: item.intent },
                          { label: "Start", value: item.start || item.date },
                          { label: "End", value: item.end },
                          { label: "Duration (days)", value: item.duration_days != null ? String(item.duration_days) : "n/a" },
                          { label: "Dosage", value: item.dosage },
                        ]} />
                      </details>
                    );
                  })}
                </div>
              ) : (
                <p className="mutedText">No medication history is available.</p>
              )}
            </div>
          </article>

          <article className="artifactCard">
            <strong>Lab graph &amp; insights</strong>
            {labSeries.length ? (
              <>
                <svg className="labChart" viewBox="0 0 560 280" preserveAspectRatio="none">
                  {labSeries.slice(0, 4).map((series: any, seriesIndex: number) => {
                    const points = (series.points ?? [])
                      .map((point: any) => {
                        const xDate = fhirDateToNumber(point.date);
                        if (xDate === null || labMinDate === null || labMaxDate === null || labMinValue === null || labMaxValue === null || typeof point.value !== "number") return null;
                        const x = 30 + ((xDate - labMinDate) / labDateSpan) * 500;
                        const y = 240 - ((point.value - labMinValue) / labValueSpan) * 190;
                        return `${x},${y}`;
                      })
                      .filter(Boolean)
                      .join(" ");
                    if (!points) return null;
                    return <polyline key={`series-${seriesIndex}`} fill="none" stroke={lineColors[seriesIndex % lineColors.length]} strokeWidth="3" points={points} />;
                  })}
                  {(labs.latest ?? []).map((item: any, index: number) =>
                    typeof item.low === "number" && typeof item.high === "number" && labMinValue !== null && labMaxValue !== null ? (
                      <rect
                        key={`range-${index}`}
                        x="30"
                        width="500"
                        y={240 - ((item.high - labMinValue) / labValueSpan) * 190}
                        height={Math.max(6, ((item.high - item.low) / labValueSpan) * 190)}
                        fill="rgba(47,128,237,0.08)"
                      />
                    ) : null,
                  )}
                </svg>
                <div className="insightGrid">
                  {(labs.latest ?? []).slice(0, 4).map((item: any, index: number) => (
                    <div key={`insight-${index}`} className="insightTile">
                      <strong>{item.label || "Lab"}</strong>
                      <span>{item.value} {item.unit || ""}</span>
                      <small>Normal {item.low ?? "n/a"} - {item.high ?? "n/a"}</small>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="mutedText">No numeric Observation trends are available for graph review.</p>
            )}
          </article>
        </section>

        {/* Care team */}
        <article className="artifactCard">
          <strong>Care team map</strong>
          <div className="careTeamGrid">
            <div>
              <div className="timelineSectionTitle">Practitioners</div>
              {carePractitioners.length ? (
                <div className="careCardStack">
                  {carePractitioners.map((item: any, index: number) => (
                    <div key={`prac-${index}`} className="careCard">
                      <strong>{item.name || "Unknown practitioner"}</strong>
                      <small>{item.role || "Practitioner"}</small>
                      <small>{item.contact || "n/a"}</small>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mutedText">No Practitioner resources are available.</p>
              )}
            </div>
            <div>
              <div className="timelineSectionTitle">Organizations</div>
              {careOrganizations.length ? (
                <div className="careCardStack">
                  {careOrganizations.map((item: any, index: number) => (
                    <div key={`org-${index}`} className="careCard">
                      <strong>{item.name || "Unknown organization"}</strong>
                      <small>{item.contact || "n/a"}</small>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mutedText">No Organization resources are available.</p>
              )}
            </div>
          </div>
        </article>
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
  dicomAnalysis,
  spreadsheetAnalysis,
  imageAnalysis,
  niftiAnalysis,
  fhirAnalysis,
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
    dicom_review: () =>
      dicomAnalysis ? (
        <DicomReviewCard analysis={dicomAnalysis} components={components} helpers={helpers} />
      ) : null,
    image_review: () =>
      imageAnalysis ? (
        <ImageReviewCard analysis={imageAnalysis} components={components} />
      ) : null,
    nifti_review: () =>
      niftiAnalysis ? (
        <NiftiReviewCard analysis={niftiAnalysis} apiBase={apiBase} components={components} />
      ) : null,
    fhir_browser: () =>
      fhirAnalysis ? (
        <FhirBrowserCard analysis={fhirAnalysis} />
      ) : null,
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
