# Skill Patch Proposal

## Tool
- `lcr_mpi_tool`

## Purpose
- Generates high-resolution `.npy` reconstructions from low-resolution MPI (Magnetic Particle Imaging) data by running three deep learning-based image restoration models: GAN, I2SB, and RDDM.

## When to use
- When the user explicitly requests image restoration mentioning the tool or specific models (e.g., "Use lcr_mpi_tool", "Restore the image with GAN", "Reconstruct using I2SB").
- When medical/MPI images in `.png` format are provided as input data.
- **[CRITICAL]** When using the I2SB model, the original image must be paired 1:1 with a condition image containing `_cond` or `cond_` in its filename.
- When a specific `seed` value is requested for reproducible inference.

## When not to use
- When the input data is not an MPI `.png` file (e.g., other modalities like CT, MRI, X-ray).
- When I2SB restoration is requested, but there is no matching condition image (`_cond`) among the uploaded files.
- When the `GAN/`, `I2SB/`, or `RDDM/` folders, or their respective `best_model.pth` weight files are missing.
- When the user requests tasks unrelated to image restoration, such as simple statistical analysis, report generation, or DICOM metadata review.

## Modality
- medical-image

## Recommended stage
- post-intake

## Depends on
- none

## Approval policy
- approval required

## Produces
- `outputs/*/results_<timestamp>/*.npy` — Restored data arrays corresponding to the number of input images.
- `outputs/*/results_<timestamp>/summary_grid.png` — A single summary grid image that automatically arranges all restoration results into dynamic rows/columns for easy visual comparison.
