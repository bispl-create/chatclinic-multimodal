# Skill Patch Proposal

## Tool
- `lung_seg`

## Purpose
- `lung_seg`: Chest X-ray lung segmentation. Uses torchxrayvision PSPNet to produce a lung-region mask from a 2D CXR image.

## When to use
- `lung_seg` (`@seg_lung`): Use when the user has uploaded a chest X-ray image in PNG/JPG format and explicitly requests lung-region segmentation.

## When not to use
- Do not use when the user is only asking for general image metadata inspection or visualization. In that case, use `image_review_tool` instead.
- Do not use when the modality does not match.
- Do not execute the tool when the user is asking only a conceptual question about segmentation; answer from general knowledge instead.
- Do not use when no source file has been uploaded.

## Source type
- image

## Modality
- 2D X-ray (PNG/JPG)

## Recommended stage
- on-demand

## Depends on
- `image_review_tool` (runs automatically on upload to provide image metadata)

## Runtime
- host compatible: gpu
- supported accelerators: gpu
- preferred accelerator: gpu
- requires gpu: yes
- cpu fallback allowed: no
- estimated runtime: 15 sec

## Approval policy
- approval required

## Produces
- `segmentation_result` (lung-region mask PNG)

## Example routing note
- On PNG/JPG upload, `image_review_tool` runs automatically to populate image metadata; route to this tool only when `@seg_lung` is invoked in a CXR clinical context.
- Routing is based on user intent (the alias entered), and the `source_types` field provides a minimal file-format safeguard.