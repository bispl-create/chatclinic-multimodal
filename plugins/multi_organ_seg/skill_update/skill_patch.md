# Skill Patch Proposal

## Tool
- `multi_organ_seg`

## Purpose
- `multi_organ_seg`: AMOS CT multi-organ segmentation. Uses a MaskSAM-based nnU-Net v2 to produce a 16-class mask (including background, 15 organs) from a 3D CT volume.

## When to use
- `multi_organ_seg` (`@seg_organ`): Use when the user has uploaded an abdominal CT volume in NIfTI format and explicitly requests multi-organ segmentation. Segments 15 organs (liver, kidney, spleen, pancreas, etc.) together with background in a single 16-class output.

## When not to use
- Do not use when the user is only asking for general image metadata inspection or visualization. In that case, use `nifti_review_tool` instead.
- Do not use when the modality does not match. For example, running `@seg_organ` (CT-only) on an MRI volume will produce invalid results.
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
- estimated runtime: about 2 min 30 sec

## Approval policy
- approval required

## Produces
- `segmentation_result` (16-class mask NIfTI; background + 15 organs)

## Example routing note
- On NIfTI upload, `nifti_review_tool` runs automatically to populate file path, shape, and voxel information; route to this tool only when the user explicitly invokes `@seg_organ` (abdominal CT).
- Because NIfTI metadata alone cannot distinguish MRI from CT, routing is based on user intent (the alias entered), and the `source_types` field provides a minimal file-format safeguard.