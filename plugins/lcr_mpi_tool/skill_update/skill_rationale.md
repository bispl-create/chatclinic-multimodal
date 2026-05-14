# Skill Patch Rationale

## Tool
- `lcr_mpi_tool`

## Summary

This document explains why `lcr_mpi_tool` should be integrated into the ChatClinic master Skill, and the keyword-based model selection strategy it uses.

---

## Why this tool should be added to the master Skill

MPI (Magnetic Particle Imaging) is an emerging medical imaging modality that requires specialized deep learning-based reconstruction to overcome its inherently low spatial resolution. Existing ChatClinic tools do not cover MPI image restoration. This plugin fills that gap by providing three distinct PyTorch models — GAN, I2SB, and RDDM.

Adding this tool allows ChatClinic to respond meaningfully when a user requests MPI image quality improvement.

---

## Model Selection Rationale

To ensure stable and predictable execution, the underlying script (run.py) determines which model to run strictly based on the presence of specific keywords ("gan", "i2sb", "rddm") in the user's prompt. It simply needs to pass the user's prompt, and the tool will execute the model explicitly requested by the user.

### RDDM (Residual Denoising Diffusion Model)
Trigger: The user explicitly mentions "RDDM"

Characteristics: Diffusion-based generative model. Excellent for recovering fine structural details, though it has a longer inference time.

### GAN (Generative Adversarial Network)

Trigger: The user explicitly mentions "GAN"

Characteristics: Fast, single feed-forward pass. Great for quick throughput, though it may produce slightly smoother outputs compared to diffusion.

### I2SB (Image-to-Image Schrödinger Bridge)

Trigger: The user explicitly mentions "I2SB"

Characteristics: A balanced option between speed and detail.

Constraint: Running I2SB requires a matching condition image file (filename containing _cond or cond_) uploaded alongside the input image.

### Default behavior (`"model": "all"`)

When the user does not specify a model, all three models run in sequence (GAN → I2SB → RDDM), producing a comprehensive set of results for comparison.

---

## Orchestrator Routing Suggestion

The AI should route requests to this tool based on direct keyword matching.

| User utterance | Execution Behavior | Prerequisites |
|---|---|---|
| "Restore the MPI image" | `"all"` (default) | Requires a matching `_cond` file for I2SB to succeed |
| "Run it with RDDM" | `"rddm"` | Single image is fine |
| "Run it with GAN" | `"gan"` | Single image is fine |
| "Run it with I2SB" | `"i2sb"` | **MUST include a `_cond` file** |

---

## Conclusion

The `lcr_mpi_tool` is the only ChatClinic plugin that handles MPI image restoration. Its three-model architecture provides flexibility for a wide range of clinical and research scenarios, grounded in the quality-speed tradeoff that is well-established in the deep learning image reconstruction literature. We recommend adding the `When to use` and routing rules from `skill_patch.md` to the master Skill.
