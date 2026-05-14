"use client";

import { type WheelEvent, useEffect, useMemo, useState } from "react";

type SliceAxis = 0 | 1 | 2;

type NiftiSliceViewerProps = {
  apiBase: string;
  niftiPath?: string | null;
  shape?: number[] | null;
  fileName?: string | null;
};

const AXES: Array<{ axis: SliceAxis; label: string }> = [
  { axis: 2, label: "Axial" },
  { axis: 1, label: "Coronal" },
  { axis: 0, label: "Sagittal" },
];

function coerceSpatialShape(shape?: number[] | null): number[] {
  if (!Array.isArray(shape)) {
    return [];
  }
  return shape.slice(0, 3).map((dim) => {
    const parsed = Number(dim);
    return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : 0;
  });
}

export default function NiftiSliceViewer({ apiBase, niftiPath, shape, fileName }: NiftiSliceViewerProps) {
  const [axis, setAxis] = useState<SliceAxis>(2);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const spatialShape = useMemo(() => coerceSpatialShape(shape), [shape]);
  const maxSlice = Math.max(0, (spatialShape[axis] ?? 1) - 1);
  const boundedSliceIndex = Math.max(0, Math.min(sliceIndex, maxSlice));
  const axisLabel = AXES.find((item) => item.axis === axis)?.label ?? "Axial";

  useEffect(() => {
    setSliceIndex(Math.floor(maxSlice / 2));
    setLoadError(null);
  }, [axis, maxSlice, niftiPath]);

  useEffect(() => {
    setLoadError(null);
  }, [boundedSliceIndex]);

  const imageUrl = useMemo(() => {
    if (!niftiPath || spatialShape.length < 3) {
      return null;
    }
    const base = apiBase.replace(/\/$/, "");
    const params = new URLSearchParams({
      path: niftiPath,
      axis: String(axis),
      index: String(boundedSliceIndex),
    });
    return `${base}/api/v1/nifti/slice?${params.toString()}`;
  }, [apiBase, axis, boundedSliceIndex, niftiPath, spatialShape.length]);

  function updateSlice(nextIndex: number) {
    setSliceIndex(Math.max(0, Math.min(nextIndex, maxSlice)));
  }

  function handleWheel(event: WheelEvent<HTMLDivElement>) {
    if (maxSlice <= 0) {
      return;
    }
    event.preventDefault();
    updateSlice(boundedSliceIndex + (event.deltaY > 0 ? 1 : -1));
  }

  if (!niftiPath || spatialShape.length < 3) {
    return <p className="emptyState">No NIfTI volume is available for slice preview.</p>;
  }

  return (
    <div className="niftiSliceViewerShell">
      <div className="dicomViewerToolbar">
        <div className="dicomViewerControls">
          {AXES.map((item) => (
            <button
              key={item.axis}
              type="button"
              className={`viewerPresetButton ${axis === item.axis ? "viewerPresetButtonActive" : ""}`}
              onClick={() => setAxis(item.axis)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="dicomViewerMeta">
          <span>{fileName ?? "NIfTI volume"}</span>
          <span>
            {axisLabel} {boundedSliceIndex + 1}/{maxSlice + 1}
          </span>
        </div>
      </div>
      <div className="niftiSliceCanvas" onWheel={handleWheel}>
        {imageUrl && !loadError ? (
          <img
            src={imageUrl}
            alt={`${fileName ?? "NIfTI"} ${axisLabel} slice ${boundedSliceIndex + 1}`}
            className="niftiSliceImage"
            draggable={false}
            onError={() => setLoadError("Slice preview could not be loaded.")}
          />
        ) : (
          <div className="dicomPreviewPlaceholder dicomViewerEmpty">{loadError ?? "Slice preview is not available."}</div>
        )}
      </div>
      <label className="niftiSliceSlider">
        <span>Slice</span>
        <input
          type="range"
          min={0}
          max={maxSlice}
          value={boundedSliceIndex}
          onChange={(event) => updateSlice(Number(event.target.value))}
        />
      </label>
      <div className="niftiSliceMeta">
        <span>Shape {spatialShape.join(" x ")}</span>
        <span>Axis {axis}</span>
      </div>
    </div>
  );
}
