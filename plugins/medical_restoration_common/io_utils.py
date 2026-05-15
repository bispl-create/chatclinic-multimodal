from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

THUMB_PX = 512


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo > 0:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return np.nan_to_num(arr, nan=0).astype(np.uint8)


def array_to_data_url(arr: np.ndarray, max_px: int = THUMB_PX) -> str | None:
    try:
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            img = Image.fromarray(normalize_uint8(arr))
        else:
            img = Image.fromarray(normalize_uint8(arr), mode="L")
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception:
        return None


def load_input_as_2d(path: Path) -> np.ndarray:
    """Load a PNG/JPG/DICOM/NIfTI path and return a 2D float array."""
    suffix = "".join(path.suffixes).lower()
    if suffix in (".nii", ".nii.gz") or path.name.lower().endswith(".nii.gz"):
        import nibabel as nib

        vol = np.asanyarray(nib.load(str(path)).dataobj)
        if vol.ndim >= 4:
            vol = vol[..., 0]
        if vol.ndim >= 3:
            # central axial slice
            return vol[:, :, vol.shape[2] // 2].astype(np.float32)
        return vol.astype(np.float32)
    if suffix in (".dcm", ".dicom"):
        import pydicom

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array
        if arr.ndim >= 3:
            arr = arr[arr.shape[0] // 2]
        return arr.astype(np.float32)
    # PIL fallback for PNG/JPG/TIFF
    img = Image.open(str(path))
    if img.mode not in ("L", "F", "I", "I;16"):
        img = img.convert("L")
    return np.asarray(img).astype(np.float32)


def side_by_side(before: np.ndarray, after: np.ndarray, max_px: int = THUMB_PX * 2) -> str | None:
    try:
        b = normalize_uint8(before)
        a = normalize_uint8(after)
        if a.shape != b.shape:
            # resize after to before shape with PIL
            a_img = Image.fromarray(a, mode="L").resize((b.shape[1], b.shape[0]), Image.LANCZOS)
            a = np.asarray(a_img)
        sep = np.full((b.shape[0], 4), 128, dtype=np.uint8)
        mont = np.hstack([b, sep, a])
        img = Image.fromarray(mont, mode="L")
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception:
        return None


def compute_metrics(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    """Reference-free quality metrics for denoising evaluation.

    Since no clean ground truth is available, SNR and noise-level estimation
    are used to measure actual denoising improvement instead of comparing the
    noisy input against the denoised output (which would reward doing nothing).
    """
    try:
        from scipy.ndimage import gaussian_filter, laplace

        def _to_float(arr: np.ndarray) -> np.ndarray:
            arr = arr.astype(np.float64)
            lo, hi = np.nanmin(arr), np.nanmax(arr)
            return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

        b = _to_float(before)
        a = _to_float(after)
        if a.shape != b.shape:
            a_img = Image.fromarray((a * 255.0).astype(np.uint8), mode="L")
            a_img = a_img.resize((b.shape[1], b.shape[0]), Image.LANCZOS)
            a = np.asarray(a_img).astype(np.float64) / 255.0

        def _snr_db(arr: np.ndarray) -> float:
            """Estimate SNR by separating signal (low-pass) from noise (residual)."""
            signal = gaussian_filter(arr, sigma=3.0)
            noise = arr - signal
            rms_s = np.sqrt(np.mean(signal ** 2))
            rms_n = np.sqrt(np.mean(noise ** 2))
            if rms_n < 1e-10:
                return 99.0
            if rms_s < 1e-10:
                return 0.0
            return float(20.0 * np.log10(rms_s / rms_n))

        def _noise_sigma(arr: np.ndarray) -> float:
            return float(np.std(arr - gaussian_filter(arr, sigma=3.0)))

        def _sharpness(arr: np.ndarray) -> float:
            return float(np.var(laplace(arr)))

        snr_b = _snr_db(b)
        snr_a = _snr_db(a)
        return {
            "snr_before_db": round(snr_b, 2),
            "snr_after_db": round(snr_a, 2),
            "snr_improvement_db": round(snr_a - snr_b, 2),
            "noise_sigma_before": round(_noise_sigma(b), 6),
            "noise_sigma_after": round(_noise_sigma(a), 6),
            "sharpness_before": round(_sharpness(b), 4),
            "sharpness_after": round(_sharpness(a), 4),
            # MAE(noisy_input, denoised_output): magnitude of change, not a quality score
            "change_mae": round(float(np.mean(np.abs(b - a))), 6),
        }
    except Exception:
        return {}
