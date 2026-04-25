"""Build a realistic demo input set.

Starts from a few clean medical images (CT DICOM, MR DICOM, MRI NIfTI), applies
realistic degradation models matching what each backend was trained on, and
writes the noisy inputs to ``demo_images/noisy/``. Also produces a CXR
phantom with lung fields + ribs + Poisson-Gaussian low-dose noise for the
TorchXRayVision ResNetAE demo.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pydicom
import nibabel as nib
from PIL import Image


def load_dicom_slice(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / max(hi - lo, 1e-6)


def load_nifti_slice(path: Path, axis: str = "axial", offset: int = 0) -> np.ndarray:
    vol = np.asanyarray(nib.load(str(path)).dataobj)
    if vol.ndim >= 4:
        vol = vol[..., 0]
    cz = vol.shape[2] // 2 + offset
    cy = vol.shape[1] // 2 + offset
    cx = vol.shape[0] // 2 + offset
    if axis == "axial":
        arr = vol[:, :, cz]
    elif axis == "coronal":
        arr = vol[:, cy, :]
    else:
        arr = vol[cx, :, :]
    arr = np.rot90(arr).astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / max(hi - lo, 1e-6)


def save_png(arr: np.ndarray, path: Path) -> None:
    a = np.clip(arr, 0.0, 1.0)
    Image.fromarray((a * 255.0).astype(np.uint8), mode="L").save(str(path))


def add_low_dose_ct_noise(clean: np.ndarray, I0: float = 1e4, seed: int = 0) -> np.ndarray:
    """Simulate low-dose CT: Poisson photon statistics + electronic Gaussian noise.

    This is the same model CoreDiff expects (Mayo 2016 25 % dose)."""
    rng = np.random.default_rng(seed)
    # Normalize to attenuation-like; simulate sinogram Poisson.
    attenuation = (1.0 - clean) * 2.5
    expected_photons = I0 * np.exp(-attenuation)
    measured = rng.poisson(np.maximum(expected_photons, 1.0))
    # Electronic noise
    measured = measured + rng.normal(0, 10.0, size=measured.shape)
    # Reconstruct back to image domain
    noisy = -np.log(np.maximum(measured, 1.0) / I0) / 2.5
    noisy = 1.0 - np.clip(noisy, 0.0, 1.0)
    return noisy


def add_mri_rician_noise(clean: np.ndarray, sigma: float = 0.05, seed: int = 1) -> np.ndarray:
    """Simulate k-space magnitude noise via Rician model (Gaussian on real+imag)."""
    rng = np.random.default_rng(seed)
    real = clean + rng.normal(0, sigma, size=clean.shape)
    imag = rng.normal(0, sigma, size=clean.shape)
    return np.sqrt(real ** 2 + imag ** 2)


def add_lowdose_cxr_noise(clean: np.ndarray, eta: float = 100.0, sigma: float = 0.05, seed: int = 2) -> np.ndarray:
    """Poisson-Gaussian low-dose X-ray noise (matches SharpXR/MIRASOL paper)."""
    rng = np.random.default_rng(seed)
    noisy = rng.poisson(np.maximum(clean * eta, 0)) / eta
    noisy = noisy + rng.normal(0, sigma, size=clean.shape)
    return np.clip(noisy, 0.0, 1.0)


def make_cxr_phantom(size: int = 512, seed: int = 3) -> np.ndarray:
    """A more anatomically realistic chest X-ray phantom.

    Mediastinum + lung fields + ribs + heart shadow — enough structure that the
    denoiser has something to preserve and the improvement is visible.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # Chest silhouette (two lung fields)
    lung_l = np.exp(-(((xx - size * 0.33) / (size * 0.16)) ** 2) - (((yy - size * 0.55) / (size * 0.22)) ** 2))
    lung_r = np.exp(-(((xx - size * 0.67) / (size * 0.16)) ** 2) - (((yy - size * 0.55) / (size * 0.22)) ** 2))
    lungs = lung_l + lung_r
    # Mediastinum / spine column
    spine = np.exp(-(((xx - size * 0.5) / (size * 0.05)) ** 2))
    # Heart shadow
    heart = np.exp(-(((xx - size * 0.45) / (size * 0.1)) ** 2) - (((yy - size * 0.62) / (size * 0.09)) ** 2))
    # Ribs
    ribs = np.zeros_like(lungs)
    for k in range(8):
        y_c = size * (0.28 + 0.06 * k)
        ribs += np.exp(-(((yy - y_c - 0.2 * (xx - size * 0.5)) / 3.0) ** 2)) * 0.18
        ribs += np.exp(-(((yy - y_c + 0.2 * (xx - size * 0.5)) / 3.0) ** 2)) * 0.18
    chest = 0.15 + 0.35 * lungs - 0.25 * spine - 0.2 * heart - 0.25 * ribs * (lung_l + lung_r)
    chest = np.clip(chest, 0.02, 0.95)
    # Add mild texture
    chest = chest + rng.normal(0, 0.01, chest.shape)
    chest = (chest - chest.min()) / (chest.max() - chest.min())
    return chest.astype(np.float32)


def downsample_for_sr(clean: np.ndarray, factor: int = 2, seed: int = 4) -> np.ndarray:
    """Make a low-resolution MRI by downsampling then upsampling with bicubic.

    Fast-DDPM PMUB expects a noisy/blurry input it then refines."""
    h, w = clean.shape
    pil = Image.fromarray((clean * 255.0).astype(np.uint8), mode="L")
    lo = pil.resize((w // factor, h // factor), Image.BICUBIC)
    lo = lo.resize((w, h), Image.BICUBIC)
    return np.asarray(lo).astype(np.float32) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", type=Path,
                    default=Path("/home/ttran/kaist_projects/chatclinic-class/demo_images"))
    args = ap.parse_args()

    clean_dir = args.demo_dir / "clean"
    noisy_dir = args.demo_dir / "noisy"
    noisy_dir.mkdir(parents=True, exist_ok=True)

    # Scenario 1: Low-dose CT from a real CT DICOM
    ct = load_dicom_slice(clean_dir / "ct_head.dcm")
    save_png(ct, noisy_dir / "00_ct_head_clean.png")
    for label, (i0, tag) in [("25% dose", (3e3, "25pct")), ("10% dose", (1e3, "10pct")), ("5% dose", (5e2, "5pct"))]:
        save_png(add_low_dose_ct_noise(ct, I0=i0), noisy_dir / f"01_ct_head_lowdose_{tag}.png")
    print("CT scenarios: 1 clean + 3 low-dose (25%, 10%, 5%)")

    # Scenario 2: MRI denoising + SR — multiple slices from the real NIfTI
    for axis, offset, tag in [("axial", 0, "axial_mid"), ("coronal", 0, "coronal_mid"), ("sagittal", 0, "sagittal_mid")]:
        mri = load_nifti_slice(clean_dir / "mri_4d.nii.gz", axis=axis, offset=offset)
        save_png(mri, noisy_dir / f"02_mri_{tag}_clean.png")
        save_png(add_mri_rician_noise(mri, sigma=0.08), noisy_dir / f"03_mri_{tag}_rician.png")
        save_png(downsample_for_sr(mri, factor=2), noisy_dir / f"04_mri_{tag}_lowres.png")
    print("MRI scenarios: 3 orientations × 3 variants (clean, rician noise, low-res)")

    # Scenario 3: MR abdomen (real DICOM)
    mr_abd = load_dicom_slice(clean_dir / "mr_abdomen.dcm")
    save_png(mr_abd, noisy_dir / "05_mr_abdomen_clean.png")
    save_png(add_mri_rician_noise(mr_abd, sigma=0.06), noisy_dir / "06_mr_abdomen_rician.png")
    save_png(downsample_for_sr(mr_abd, factor=2), noisy_dir / "07_mr_abdomen_lowres.png")
    print("MR abdomen: 1 clean + 1 rician + 1 lowres")

    # Scenario 4: Chest X-ray phantom with realistic anatomy + low-dose noise
    cxr = make_cxr_phantom(size=512)
    save_png(cxr, noisy_dir / "08_cxr_clean.png")
    for label, (eta, tag) in [("low-dose", (100.0, "lowdose")), ("very-low-dose", (30.0, "verylowdose"))]:
        save_png(add_lowdose_cxr_noise(cxr, eta=eta, sigma=0.04), noisy_dir / f"09_cxr_{tag}.png")
    print("CXR scenarios: 1 clean + 2 low-dose levels")

    print(f"\nAll demo inputs in: {noisy_dir}")
    print(f"Total files: {len(list(noisy_dir.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
