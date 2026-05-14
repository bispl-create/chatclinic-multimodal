# Skill Patch Proposal

## Tool
- `pancreas_tumor_seg`

## Purpose
- `pancreas_tumor_seg`: Pancreas and tumor CT segmentation. Uses MONAI DiNTS to produce a 3-class mask (background / pancreas / tumor) from a 3D abdominal CT.

## When to use
- `pancreas_tumor_seg` (`@seg_pancreas`): Use when the user has uploaded an abdominal CT volume in NIfTI format and explicitly requests pancreas and pancreatic-tumor segmentation.

## When not to use
- Do not use when the user is only asking for general image metadata inspection or visualization. In that case, use `nifti_review_tool` instead.
- Do not use when the modality does not match. For example, running `@seg_pancreas` (CT-only) on an MRI volume will produce invalid results.
- Do not execute the tool when the user is asking only a conceptual question about segmentation; answer from general knowledge instead.
- Do not use when no source file has been uploaded.

## Source type
- nifti

## Modality
- 3D CT (NIfTI)

## Recommended stage
- on-demand

## Depends on
- `nifti_review_tool` (runs automatically on upload to provide NIfTI metadata)

## Runtime
- host compatible: gpu
- supported accelerators: gpu
- preferred accelerator: gpu
- requires gpu: yes
- cpu fallback allowed: no
- estimated runtime: 10 sec

## Approval policy
- approval required

## Produces
- `segmentation_result` (pancreas/tumor 3-class mask NIfTI)

## Example routing note
- On NIfTI upload, `nifti_review_tool` runs automatically to populate file path, shape, and voxel information; route to this tool only when the user explicitly invokes `@seg_pancreas` (abdominal CT).
- Because NIfTI metadata alone cannot distinguish MRI from CT, routing is based on user intent (the alias entered), and the `source_types` field provides a minimal file-format safeguard.