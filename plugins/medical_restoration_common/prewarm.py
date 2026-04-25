"""Pre-warm all restoration backends.

Run this after launching uvicorn so every model is resident on the GPU when the
class starts. Works by making a real HTTP round-trip for each tool — the same
path a user click would take — so ``nvidia-smi`` will show the combined VRAM.

Usage:

    python -m plugins.medical_restoration_common.prewarm \
        --backend http://127.0.0.1:8001 \
        --demo-dir /home/ttran/kaist_projects/chatclinic-class/demo_images
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _upload(backend: str, endpoint: str, file_path: Path) -> dict:
    out = subprocess.check_output([
        "curl", "-s", "-X", "POST",
        "-F", f"file=@{file_path}",
        f"{backend}/api/v1/{endpoint}/upload",
    ], timeout=60)
    return json.loads(out.decode())


def _chat(backend: str, endpoint: str, analysis: dict, question: str, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        f"{backend}/api/v1/chat/{endpoint}",
        data=json.dumps({"question": question, "analysis": analysis, "history": []}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")


def _gpu_used_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            timeout=10,
        ).decode()
        return int(out.strip())
    except Exception:
        return -1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="http://127.0.0.1:8001")
    p.add_argument("--demo-dir", required=True, type=Path)
    args = p.parse_args()

    dicom_file = args.demo_dir / "ct_sample.dcm"
    nifti_file = args.demo_dir / "mri_sample.nii.gz"
    cxr_file = args.demo_dir / "cxr_sample.png"
    for f in (dicom_file, nifti_file, cxr_file):
        if not f.exists():
            print(f"ERROR: missing {f}", file=sys.stderr)
            return 1

    print(f"GPU 0 baseline: {_gpu_used_mb()} MiB")

    # Upload each once
    print("Uploading DICOM...")
    dicom_ana = _upload(args.backend, "dicom", dicom_file)
    print("Uploading NIfTI...")
    nifti_ana = _upload(args.backend, "nifti", nifti_file)
    print("Uploading PNG (X-ray)...")
    image_ana = _upload(args.backend, "image", cxr_file)

    # One prewarm call per tool — the first call loads the model
    steps = [
        ("DICOM", "dicom", dicom_ana, "@ct_denoise"),
        ("DICOM", "dicom", dicom_ana, "@ct_artifact"),
        ("NIFTI", "nifti", nifti_ana, "@mri_denoise"),
        ("NIFTI", "nifti", nifti_ana, "@mri_sr"),
        ("NIFTI", "nifti", nifti_ana, "@med_translate"),
        ("IMAGE", "image", image_ana, "@xray_denoise"),
    ]
    for source_label, endpoint, analysis, question in steps:
        t0 = time.time()
        try:
            resp = _chat(args.backend, endpoint, analysis, question, timeout=600)
            ana = resp.get("analysis") or {}
            rest = (ana.get("artifacts") or {}).get("restoration") or {}
            backend = rest.get("backend", "?")
            mode = rest.get("mode", "?")
            dt = time.time() - t0
            gpu = _gpu_used_mb()
            print(f"[{mode:8s}] {source_label} {question:20s} -> {backend:28s}  {dt:5.1f}s  GPU0={gpu} MiB")
        except Exception as e:
            print(f"[FAIL   ] {source_label} {question:20s} -> {e}")

    print(f"\nFinal GPU 0 usage: {_gpu_used_mb()} MiB")
    print("All backends now resident. Demo is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
