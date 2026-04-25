# Medical Image Restoration — Integration Notes

Five ChatClinic plugins share this adapter module:

| Plugin | Alias | Task | Primary Backend | Fallback |
|--------|-------|------|-----------------|----------|
| `ct_denoise_tool` | `@ct_denoise` | Low-dose CT denoising | **CoreDiff** | Fast-DDPM (LDFDCT) / classical |
| `ct_artifact_reduction_tool` | `@ct_artifact` | CT artifact reduction | **CoreDiff** | classical |
| `mri_denoise_tool` | `@mri_denoise` | MRI denoising | **SNRAware** (Microsoft) | classical |
| `mri_super_resolution_tool` | `@mri_sr` | MRI super-resolution | **Fast-DDPM** (PMUB) | bicubic |
| `medical_image_translation_tool` | `@med_translate` | Medical image translation / synthesis-restoration | **Fast-DDPM** (BRATS) | classical |

All plugins accept the same payload keys:
- `image_path` / `nifti_path` / `dicom_path` / `file_path` — any one
- task-specific options (e.g. `size`, `backend`) per `tool.json`

They return an `ImageSourceResponse` whose `artifacts["restoration"]` contains
backend name, run mode (`backend` or `classical_fallback`), notes, PSNR/SSIM
between input and output, and `before/after/compare` data URLs that render via
the existing `image_review` Studio card.

## Backend layout (from repo root)

```
external_backends/
  CoreDiff/main.py                                       # CT denoising / artifact reduction
  Fast-DDPM/fast_ddpm_main.py                            # MRI SR, CT denoise, translation
  SNRAware/src/snraware/projects/mri/denoising/run_inference.py  # MRI denoising

external_weights/
  SNRAware/{small,medium,large}/*.pts + *.yaml           # ✅ present
  FastDDPM/ckpt_{PMUB,LDFDCT,BRATS}.pth                  # ✅ present
  CoreDiff/<release checkpoints>                         # ❌ must be downloaded
```

Paths can be overridden with env vars: `CHATCLINIC_BACKENDS_DIR`,
`CHATCLINIC_WEIGHTS_DIR`, `CHATCLINIC_RESTORATION_CACHE`.

## Status matrix

| Backend | Weights | Adapter (this repo) | Ready? |
|---------|---------|--------------------|--------|
| SNRAware | ✅ downloaded | ✅ subprocess via `uv`, magnitude→complex synth | **Ready on 1× RTX 3090** once backend `uv sync` has been run in `external_backends/SNRAware/` |
| Fast-DDPM | ✅ downloaded | ⚠️ classical fallback only — backend expects dataset-structured folders (`data/PMUB-test/` etc.) rather than single-slice input | Needs slice→dataset shim (see **Extending Fast-DDPM** below) |
| CoreDiff | ❌ | ⚠️ classical fallback only | Needs weights + dataset-structured loader |

The adapter layer is intentionally conservative — when any precondition is
missing, the plugin still returns a valid Studio card and records why in
`artifacts.restoration.notes`. This keeps the system usable while backends are
being wired up.

## GPU placement (up to 4× RTX 3090, 24 GB each)

| Task | Expected VRAM | Notes |
|------|---------------|-------|
| SNRAware small | ~4 GB | batch 1 slice fits easily |
| SNRAware medium | ~8 GB | comfortable |
| SNRAware large | ~14 GB | one 3090 |
| Fast-DDPM sampling (10 steps) | ~6-10 GB | one 3090 |
| CoreDiff sampling | ~8-12 GB | one 3090 |

Pin each backend to its own GPU with `CUDA_VISIBLE_DEVICES=N uv run …`.
Running the four backends in parallel across four 3090s is well within budget.

## One-time setup per backend

### SNRAware
```bash
cd external_backends/SNRAware
uv sync                                  # installs backend deps into its own venv
# weights are already at external_weights/SNRAware/{small,medium,large}/
```

### Fast-DDPM
```bash
cd external_backends/Fast-DDPM
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt         # (create from paper's listed deps)
# weights are already at external_weights/FastDDPM/ckpt_{PMUB,LDFDCT,BRATS}.pth
```

### CoreDiff
```bash
cd external_backends/CoreDiff
# Download the AAPM-Mayo 2016 checkpoints linked in external_backends/CoreDiff/README.md
# place into external_weights/CoreDiff/
```

## Extending Fast-DDPM beyond the classical fallback

Fast-DDPM's `fast_ddpm_main.py --sample` iterates over a test-split folder
under `data/` rather than a single slice. To wire the adapter:

1. Stage the incoming slice into a temporary directory that mimics the
   `data/PMUB-test/<case>/<slice>.npy` layout the backend expects.
2. Point the config's `data.dataroot` at that directory (copy
   `configs/PMUB_linear.yml` to the cache and edit).
3. Invoke:
   ```bash
   python fast_ddpm_main.py \
     --config <cached-cfg>.yml \
     --dataset PMUB --sample --fid \
     --scheduler_type uniform --timesteps 10 \
     --exp <cache>/run --doc run \
     --ckpt external_weights/FastDDPM/ckpt_PMUB.pth
   ```
4. Collect the resulting sample from `<cache>/run/image_samples/` and load it
   back into the adapter.

The same pattern applies to LDFDCT (CT denoising) and BRATS (translation) with
the respective config + dataset tag. See `adapters.run_fastddpm` for where to
drop this in — the current fallback path is isolated in one function.

## Extending CoreDiff

CoreDiff's training/testing scripts (`test_mayo2016.sh` / `test.sh`) expect a
paired low-dose / full-dose DICOM directory structure. For @tool use, the
adapter should:

1. Wrap the incoming slice as a minimal paired directory (duplicating the
   slice for "full-dose" side — unused at inference time).
2. Run `python main.py --mode test --resume <ckpt> --data_root <tmp_dir>`.
3. Read the denoised volume from the CoreDiff output directory.

## Testing the plugins (without weights)

```bash
cd chatclinic-multimodal
python3 -c "
from plugins.mri_denoise_tool.logic import execute
print(execute({'image_path': 'path/to/any.png'})['analysis']['draft_answer'])
"
```

Each plugin will run its classical fallback and produce a before/after
comparison card so the UI flow is exercised end-to-end.
