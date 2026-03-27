"use client";

import { buildCustomStudioRendererRegistry } from "./customStudioRenderers";
import { buildGenericStudioRendererRegistry } from "./genericStudioRenderers";
import { type StudioRendererBuilderArgs, type StudioRendererRegistry } from "./studioRendererTypes";

export type { StudioRendererBuilderArgs, StudioRendererRegistry } from "./studioRendererTypes";
export type StudioRendererDispatch = {
  requestedView?: string | null;
  resultKind?: string | null;
  renderer?: string | null;
};

const STUDIO_RENDERER_METADATA: Record<string, { requestedViews?: string[]; resultKinds?: string[] }> = {
  rawqc: { requestedViews: ["rawqc"] },
  text: { requestedViews: ["text"] },
  cohort_browser: { requestedViews: ["cohort_browser"] },
  sumstats: { requestedViews: ["sumstats"] },
  samtools: { requestedViews: ["samtools"], resultKinds: ["samtools_result"] },
  prs_prep: { requestedViews: ["prs_prep"], resultKinds: ["prs_prep_result"] },
  qqman: { requestedViews: ["qqman"], resultKinds: ["qqman_result"] },
  provenance: { requestedViews: ["provenance"] },
  qc: { requestedViews: ["qc"] },
  coverage: { requestedViews: ["coverage"] },
  snpeff: { requestedViews: ["snpeff"], resultKinds: ["snpeff_result"] },
  plink: { requestedViews: ["plink"], resultKinds: ["plink_result"] },
  liftover: { requestedViews: ["liftover"], resultKinds: ["liftover_result"] },
  ldblockshow: { requestedViews: ["ldblockshow"], resultKinds: ["ldblockshow_result"] },
  candidates: { requestedViews: ["candidates"] },
  acmg: { requestedViews: ["acmg"] },
  table: { requestedViews: ["table"] },
  symbolic: { requestedViews: ["symbolic"] },
  roh: { requestedViews: ["roh"] },
  clinvar: { requestedViews: ["clinvar"] },
  vep: { requestedViews: ["vep"] },
  references: { requestedViews: ["references"] },
  igv: { requestedViews: ["igv"] },
  annotations: { requestedViews: ["annotations"] },
};

function findRendererKeyByRequestedView(requestedView?: string | null): string | null {
  const normalized = String(requestedView || "").trim();
  if (!normalized) {
    return null;
  }
  if (normalized.startsWith("sheet::") && normalized.endsWith("::cohort_browser")) {
    return "cohort_browser";
  }
  for (const [rendererKey, metadata] of Object.entries(STUDIO_RENDERER_METADATA)) {
    if (metadata.requestedViews?.includes(normalized)) {
      return rendererKey;
    }
  }
  return null;
}

function findRendererKeyByResultKind(resultKind?: string | null): string | null {
  const normalized = String(resultKind || "").trim();
  if (!normalized) {
    return null;
  }
  for (const [rendererKey, metadata] of Object.entries(STUDIO_RENDERER_METADATA)) {
    if (metadata.resultKinds?.includes(normalized)) {
      return rendererKey;
    }
  }
  return null;
}

export function resolveStudioDispatchFromPayload(payload: Record<string, unknown> | null | undefined): StudioRendererDispatch {
    if (!payload) {
        return {};
    }
    const studio =
      payload.studio && typeof payload.studio === "object" && !Array.isArray(payload.studio)
        ? (payload.studio as Record<string, unknown>)
        : null;
    const requestedView =
      typeof payload.requested_view === "string" && payload.requested_view.trim() ? payload.requested_view : null;
    const renderer =
      studio && typeof studio.renderer === "string" && studio.renderer.trim() ? studio.renderer : null;
    const explicitResultKind =
      typeof payload.result_kind === "string" && payload.result_kind.trim() ? payload.result_kind : null;
    if (explicitResultKind) {
      return { requestedView, resultKind: explicitResultKind, renderer };
    }
    for (const metadata of Object.values(STUDIO_RENDERER_METADATA)) {
      const matched = metadata.resultKinds?.find((kind) => payload[kind] != null);
      if (matched) {
        return { requestedView, resultKind: matched, renderer };
      }
    }
    return { requestedView, renderer };
}

export function resolveStudioRendererKey({
  activeView,
  dispatch,
}: {
  activeView?: string | null;
  dispatch?: StudioRendererDispatch | null;
}): string | null {
  return (
    findRendererKeyByRequestedView(dispatch?.renderer) ??
    findRendererKeyByRequestedView(activeView) ??
    findRendererKeyByRequestedView(dispatch?.requestedView) ??
    findRendererKeyByResultKind(dispatch?.resultKind) ??
    null
  );
}

export function buildStudioRendererRegistry(args: StudioRendererBuilderArgs): StudioRendererRegistry {
  return {
    ...buildGenericStudioRendererRegistry(args),
    ...buildCustomStudioRendererRegistry(args),
  };
}
