# Skill Patch Proposal

## Tool
- `brain_tumor_seg`

## Purpose
- `brain_tumor_seg`: BraTS-based 3D MRI brain tumor segmentation. Uses MONAI SegResNet to produce ET, TC, WT masks from a 4-channel BraTS MRI volume.

## When to use
- `brain_tumor_seg` (`@seg_brain`): Use when the user has uploaded a BraTS MRI volume in NIfTI format and explicitly requests brain tumor segmentation. Requires 4-channel (T1, T1ce, T2, FLAIR) MRI data.

## When not to use
- Do not use when the user is only asking for general image metadata inspection or visualization. In that case, use `nifti_review_tool` instead.
- Do not use when the modality does not match. For example, running `@seg_brain` (MRI-only) on a CT volume will produce invalid results.
- Do not execute the tool when the user is asking only a conceptual question about segmentation; answer from general knowledge instead.
- Do not use when no source file has been uploaded.

## Source type
- nifti

## Modality
- 3D MRI (NIfTI, 4-channel BraTS)

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
- estimated runtime: 3 sec

## Approval policy
- approval required

## Produces
- `segmentation_result` (ET/TC/WT mask NIfTI)

## Example routing note
- On NIfTI upload, `nifti_review_tool` runs automatically to populate file path, shape, and voxel information; route to this tool only when the user explicitly invokes `@seg_brain` (brain MRI).
- Because NIfTI metadata alone cannot distinguish MRI from CT, routing is based on user intent (the alias entered), and the `source_types` field provides a minimal file-format safeguard.