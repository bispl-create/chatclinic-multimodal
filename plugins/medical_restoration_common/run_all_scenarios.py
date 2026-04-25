"""Run every restoration tool across the curated scenario set and save outputs.

Writes to ``outputs/<scenario_id>/<tool>/`` with:
- ``before.png`` — the input the tool received
- ``after.png`` — the model's output
- ``compare.png`` — side-by-side (before | after)
- ``metrics.json`` — psnr/ssim/mae/mse before vs after, backend, mode, notes, VRAM, runtime
- ``summary.md`` — one-page human-readable report

Run:
    python -m plugins.medical_restoration_common.run_all_scenarios \
        --noisy-dir /home/ttran/kaist_projects/chatclinic-class/demo_images/noisy \
        --outputs-dir /home/ttran/kaist_projects/chatclinic-class/outputs
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path

from app.services.tool_runner import run_tool


# scenario_id -> (input_file_stem, tool_name, payload_overrides, description)
# Scenarios are picked so each tool is exercised on task-appropriate inputs.
SCENARIOS: list[tuple[str, str, str, dict, str]] = [
    # ----- CT denoising (CoreDiff) -----
    ("s01_ct_lowdose_25pct_corediff",    "01_ct_head_lowdose_25pct.png",  "ct_denoise_tool",             {},                      "Low-dose CT 25% — CoreDiff"),
    ("s02_ct_lowdose_10pct_corediff",    "01_ct_head_lowdose_10pct.png",  "ct_denoise_tool",             {},                      "Low-dose CT 10% — CoreDiff"),
    ("s03_ct_lowdose_5pct_corediff",     "01_ct_head_lowdose_5pct.png",   "ct_denoise_tool",             {},                      "Very low-dose CT 5% — CoreDiff"),
    ("s04_ct_lowdose_10pct_fastddpm",    "01_ct_head_lowdose_10pct.png",  "ct_denoise_tool",             {"backend": "fastddpm"}, "Low-dose CT 10% — Fast-DDPM LDFDCT (comparison)"),
    # ----- CT artifact reduction (CoreDiff) -----
    ("s05_ct_artifact_25pct_corediff",   "01_ct_head_lowdose_25pct.png",  "ct_artifact_reduction_tool",  {},                      "CT artifact reduction 25% — CoreDiff"),
    ("s06_ct_artifact_5pct_corediff",    "01_ct_head_lowdose_5pct.png",   "ct_artifact_reduction_tool",  {},                      "CT artifact reduction 5% — CoreDiff"),
    # ----- MRI denoising (SNRAware default, Fast-DDPM comparison) -----
    ("s07_mri_axial_rician_snraware",    "03_mri_axial_mid_rician.png",   "mri_denoise_tool",            {},                      "MRI axial Rician noise — SNRAware small"),
    ("s08_mri_coronal_rician_snraware",  "03_mri_coronal_mid_rician.png", "mri_denoise_tool",            {},                      "MRI coronal Rician noise — SNRAware small"),
    ("s09_mri_sagittal_rician_snraware", "03_mri_sagittal_mid_rician.png","mri_denoise_tool",            {},                      "MRI sagittal Rician noise — SNRAware small"),
    ("s10_mri_abdomen_rician_snraware",  "06_mr_abdomen_rician.png",      "mri_denoise_tool",            {},                      "MR abdomen Rician noise — SNRAware small"),
    ("s11_mri_abdomen_rician_snra_med",  "06_mr_abdomen_rician.png",      "mri_denoise_tool",            {"size": "medium"},     "MR abdomen Rician noise — SNRAware medium"),
    # ----- MRI super-resolution (Fast-DDPM PMUB) -----
    ("s12_mri_axial_sr_fastddpm",        "04_mri_axial_mid_lowres.png",   "mri_super_resolution_tool",   {},                      "MRI axial SR — Fast-DDPM PMUB"),
    ("s13_mri_coronal_sr_fastddpm",      "04_mri_coronal_mid_lowres.png", "mri_super_resolution_tool",   {},                      "MRI coronal SR — Fast-DDPM PMUB"),
    ("s14_mri_abdomen_sr_fastddpm",      "07_mr_abdomen_lowres.png",      "mri_super_resolution_tool",   {},                      "MR abdomen SR — Fast-DDPM PMUB"),
    # ----- Medical image translation (Fast-DDPM BRATS) -----
    ("s15_mri_axial_translate",          "02_mri_axial_mid_clean.png",    "medical_image_translation_tool", {},                   "MRI axial cross-contrast — Fast-DDPM BRATS"),
    ("s16_mri_coronal_translate",        "02_mri_coronal_mid_clean.png",  "medical_image_translation_tool", {},                   "MRI coronal cross-contrast — Fast-DDPM BRATS"),
    # ----- Chest X-ray denoising (TorchXRayVision ResNetAE) -----
    ("s17_cxr_lowdose_resnetae",         "09_cxr_lowdose.png",            "xray_denoise_tool",           {},                      "Low-dose chest X-ray — TorchXRayVision ResNetAE"),
    ("s18_cxr_verylowdose_resnetae",     "09_cxr_verylowdose.png",        "xray_denoise_tool",           {},                      "Very-low-dose chest X-ray — TorchXRayVision ResNetAE"),
]


def _data_url_to_png(data_url: str | None, out_path: Path) -> bool:
    if not data_url:
        return False
    m = re.match(r"data:image/(png|jpeg);base64,(.+)$", data_url)
    if not m:
        return False
    out_path.write_bytes(base64.b64decode(m.group(2)))
    return True


def _save_scenario(scenario_id: str, input_file: str, tool: str, opts: dict, desc: str,
                   noisy_dir: Path, outputs_dir: Path) -> dict:
    src = noisy_dir / input_file
    out_dir = outputs_dir / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = run_tool(tool, {"image_path": str(src), "file_path": str(src), **opts})
        error = None
    except Exception as exc:
        result = {}
        error = f"{type(exc).__name__}: {exc}"
    dt = time.time() - t0

    analysis = (result.get("analysis") if isinstance(result, dict) else None) or {}
    rest = (analysis.get("artifacts") or {}).get("restoration") or {}
    before_ok = _data_url_to_png(rest.get("before_preview_data_url"), out_dir / "before.png")
    after_ok = _data_url_to_png(rest.get("after_preview_data_url"),   out_dir / "after.png")
    comp_ok = _data_url_to_png(rest.get("compare_preview_data_url"), out_dir / "compare.png")

    metrics = {
        "scenario_id": scenario_id,
        "description": desc,
        "input_file": input_file,
        "tool": tool,
        "payload_overrides": opts,
        "runtime_seconds": round(dt, 2),
        "backend": rest.get("backend"),
        "mode": rest.get("mode"),
        "metrics": rest.get("metrics") or {},
        "notes": rest.get("notes") or [],
        "before_png_saved": before_ok,
        "after_png_saved": after_ok,
        "compare_png_saved": comp_ok,
        "error": error,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [
        f"# {scenario_id}",
        f"_{desc}_",
        "",
        f"- input: `demo_images/noisy/{input_file}`",
        f"- tool: `{tool}`  opts: `{opts or '{}'}`",
        f"- backend: **{metrics['backend']}** (mode: **{metrics['mode']}**)",
        f"- runtime: **{metrics['runtime_seconds']} s**",
    ]
    m = metrics["metrics"]
    if m:
        delta = m.get("snr_improvement_db")
        sign = "+" if delta is not None and delta >= 0 else ""
        lines.append(
            f"- metrics: SNR {m.get('snr_before_db')} dB → {m.get('snr_after_db')} dB "
            f"({sign}{delta} dB), "
            f"noise σ: {m.get('noise_sigma_before')} → {m.get('noise_sigma_after')}"
        )
    if metrics["notes"]:
        lines.append("- notes:")
        for n in metrics["notes"]:
            lines.append(f"  - {n}")
    if metrics["error"]:
        lines.append(f"- **ERROR:** `{metrics['error']}`")
    lines += ["", "![compare](compare.png)"]
    (out_dir / "summary.md").write_text("\n".join(lines))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy-dir", type=Path,
                    default=Path("/home/ttran/kaist_projects/chatclinic-class/demo_images/noisy"))
    ap.add_argument("--outputs-dir", type=Path,
                    default=Path("/home/ttran/kaist_projects/chatclinic-class/outputs"))
    args = ap.parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    for scenario_id, input_file, tool, opts, desc in SCENARIOS:
        print(f"[running] {scenario_id:36s}  {tool:30s}  {input_file}")
        m = _save_scenario(scenario_id, input_file, tool, opts, desc, args.noisy_dir, args.outputs_dir)
        all_metrics.append(m)
        ok = m["mode"] == "backend" and not m["error"]
        status = "OK" if ok else ("FAIL" if m["error"] else "FALLBACK")
        metrics_str = ""
        if m["metrics"]:
            mm = m["metrics"]
            delta = mm.get("snr_improvement_db")
            sign = "+" if delta is not None and delta >= 0 else ""
            metrics_str = f"  SNR {mm.get('snr_before_db')}→{mm.get('snr_after_db')} dB ({sign}{delta} dB)"
        print(f"  -> [{status:8s}] backend={m['backend'] or '?':28s} {m['runtime_seconds']}s{metrics_str}")

    # Master index.md
    idx = ["# Restoration scenario outputs\n",
           "Run via `python -m plugins.medical_restoration_common.run_all_scenarios`.\n",
           "| Scenario | Tool | Backend | Mode | SNR Before (dB) | SNR After (dB) | SNR Improvement (dB) | Runtime (s) |",
           "|---|---|---|---|---|---|---|---|"]
    for m in all_metrics:
        mm = m["metrics"] or {}
        idx.append(
            f"| [{m['scenario_id']}]({m['scenario_id']}/summary.md) | `{m['tool']}` "
            f"| `{m['backend']}` | `{m['mode']}` | {mm.get('snr_before_db','')} "
            f"| {mm.get('snr_after_db','')} | {mm.get('snr_improvement_db','')} | {m['runtime_seconds']} |"
        )
    (args.outputs_dir / "INDEX.md").write_text("\n".join(idx))
    (args.outputs_dir / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"\nWrote {len(all_metrics)} scenario folders + INDEX.md + all_metrics.json to {args.outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
