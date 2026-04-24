# Skill Rationale

## Why this tool should exist
- ChatClinic currently accepts a variety of medical data inputs (VCF, DICOM, NIfTI, images, etc.), but it does not provide a segmentation capability for extracting structures from uploaded medical images. Even though `nifti_review_tool` provides file metadata and previews, the next step — lesion/organ extraction — still requires running an external tool separately. In clinical research and educational settings, segmentation is the prerequisite for almost every downstream analysis such as volumetric measurement, lesion quantification, and treatment-response evaluation, so a segmentation tool is necessary.
- `spleen_seg` (MONAI UNet, spleen_ct bundle): Produces a 2-class spleen mask from a 3D abdominal CT, useful for single-organ analyses such as spleen volume measurement.
- MONAI, used in this patch, is a PyTorch-based open-source medical-imaging deep-learning framework led by NVIDIA. It distributes pretrained segmentation models as reproducible bundles.

## Why the orchestrator should call it
- When a user uploads a medical image and explicitly requests segmentation (`@seg_spleen`), the orchestrator matches the keyword against the `aliases` defined in tool.json and invokes the corresponding tool. At the same time, tool.json's `source_types` field causes the system to automatically block the call if the current source type does not match, preventing the tool from being executed on the wrong modality. Image segmentation is a computational step that cannot be replaced by simple metadata responses or free-form LLM narration, so mediating the tool call through the orchestrator is the only way to obtain reproducible, quantitative results.
- This tool is registered in `trigger_keywords` with a unique alias (`seg_spleen`), ensuring 1:1 routing without collisions where the same keyword matches multiple tools. The user only needs to pick an alias that matches the modality/target of the uploaded image; the source_type check is handled by the system through tool.json declarations.
  - Abdominal CT: `@seg_spleen` (`source_types: ["nifti"]`)
- The `source_types` and `routing.trigger_keywords` fields in each tool.json serve as the orchestrator's routing basis, so the orchestrator can pick the right tool purely from manifest information without opening the file itself.
- This tool runs a deterministic inference pipeline and returns an actual mask file as an artifact. Routing a segmentation request to this tool therefore guarantees the orchestrator delivers results in a trustworthy form for clinical/research contexts.

## Why approval is or is not required
- approval required. Execution after explicit user approval is required for the following three reasons.
  - The tool performs GPU-based 3D deep-learning inference and is resource-intensive.
  - Segmentation masks can feed into downstream clinical decisions such as volume measurement or lesion quantification. For that reason, it is safer to have the user confirm the target and parameters once more before execution.
  - NIfTI file metadata alone cannot definitively distinguish MRI from CT. An incorrect combination can produce meaningless results, so the approval step provides an additional check that the input and tool match.

## What educational value it adds
- Within the same ChatClinic interface, learners can see how the 3D CT modality and the UNet architecture are applied to a specific target. `@seg_spleen` demonstrates a 2-class spleen output mask structure.
- The ChatClinic plugin pattern of `tool.json` + `logic.py` + skill patch can be practiced directly, making it reusable as a template when adding new tools later.
- Experiencing how segmentation results flow through the `segmentation_result` slot and render in Studio helps the learner understand the end-to-end medical-AI workflow from upload → tool invocation → result interpretation.