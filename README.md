# ChatClinic Multimodal

ChatClinic is a multimodal clinical workspace that combines deterministic tool execution, Studio card rendering, and chat-based reasoning across genomics, medical imaging, tabular data, text, and FHIR.

## What the System Does

1. You upload a source file (VCF, DICOM, image, NIfTI, raw sequencing, spreadsheet, text, FHIR, or summary stats).
2. The backend auto-detects source type and runs the bootstrap review tool for that source.
3. The result is normalized into a typed analysis payload and rendered as Studio cards.
4. Chat can then:
   - Run explicit tool commands like `@ct_denoise` or `@qqman`.
   - Interpret natural-language requests and map them to tools when confidence is high.
   - Answer in grounded mode with `$studio`, `$current analysis`, `$current card`, or `$grounded`.

## Core Architecture

- Backend API: `app/main.py`
- Data models: `app/models.py`
- Chat orchestration and NL routing: `app/services/chat.py`
- Tool discovery and execution: `app/services/tool_runner.py`
- Upload persistence and bootstrap analysis: `app/services/source_bootstrap.py`
- Frontend state and orchestration: `webapp/app/page.tsx`
- Plugins and tool metadata: `plugins/*/tool.json`, `plugins/*/logic.py`

## Lazy Loading Strategy (Does Not Load All Models at Startup)

The system is intentionally designed to avoid loading every model during service startup.

### 1) Tool metadata is cached, not model weights

In `app/services/tool_runner.py`, `load_tool_manifests()` uses `@lru_cache(maxsize=1)`.
This means tool metadata is scanned once, cached, and reused, without loading heavy backend models.

### 2) Plugin code is loaded only when a tool is invoked

`run_tool()` resolves a tool manifest, then loads and executes only that tool entrypoint (`plugins.<tool>.logic:execute`).
Unused tools are not imported into runtime.

### 3) Restoration backends load on first use, then stay cached

The medical restoration stack in `plugins/medical_restoration_common` is lazy by design:

- `adapters.py` imports heavy runners inside execution paths (for example inside `run_fastddpm`, `run_corediff`, `run_xray_denoise`).
- `fastddpm_runner.py` keeps model instances in `_MODEL_CACHE` keyed by task.
- `corediff_runner.py` keeps the diffusion model in `_CACHE`.
- `sharpxr_runner.py` and `torchxrayvision_runner.py` keep loaded models in in-memory cache dictionaries.

Effect:
- First request to a backend pays model load cost.
- Later requests reuse already-loaded models.
- Startup stays fast and resilient even if some weights are missing.

### 4) Graceful fallback when weights or backend runtime are missing

`plugins/medical_restoration_common/adapters.py` falls back to classical processing if model execution cannot run.
Studio still receives a usable artifact, and notes explicitly describe what path was used.

## Natural Language Prompt Handling

Natural language handling is implemented in `app/services/chat.py`.

### A) Explicit command path

If user input starts with `@tool`, the system parses it with `_parse_at_tool_request()` and routes to:
- Direct endpoint execution (when configured), or
- Generic plugin entrypoint execution.

### B) Implicit NL-to-tool path for imaging restoration

If the message does not start with `@`, `_parse_nl_tool_request()` may infer a tool alias from language cues.
Examples of cues include denoise, artifact reduction, super-resolution, translation, plus modality hints (CT/MRI/X-ray).

The flow is:

1. `_infer_restoration_alias_from_nl(question, source_type)`
2. `_infer_tool_remainder_from_nl(alias, question)`
3. Build inferred tool request
4. Execute as if it were `@alias`

When inferred, response text is annotated to show which tool alias was interpreted.

### C) Grounded vs general answering

If no tool command is resolved, chat calls OpenAI:
- Grounded mode if `$studio`-style trigger exists.
- General mode otherwise.

For grounded responses, compact source-specific context is built and passed with stricter system instructions.

### D) Multimodal merged context

`answer_multimodal_chat()` merges active contexts (VCF/raw_qc/summary_stats/text/spreadsheet/dicom/image/nifti/fhir) and calls one multimodal response path.
For tool execution inside multimodal chat, a primary source is selected and converted into a single-source payload for deterministic routing.

## Multi-Image Upload Handling and Active Image Selection

The system previously could become ambiguous when multiple images were uploaded close together, because one late response could override active image state.

The frontend flow in `webapp/app/page.tsx` now addresses this:

1. Every uploaded source is kept in `uploadedSources` (no replacement by source type).
2. Each source entry stores identity and resolved source path.
3. Source list entries are selectable, so user can explicitly choose active source.
4. Image analyses are cached by `source_image_path` (`imageAnalysesByPath`).
5. A stale-upload guard (`latestImageUploadRequestRef`) prevents older image requests from overriding newer active state.

Result:
- If multiple images are uploaded sequentially or near-concurrently, the app can keep each source entry.
- The active image for testing is explicit and switchable.
- Out-of-order upload completions no longer hijack active image selection.

## Supported Sources

- VCF: `.vcf`, `.vcf.gz`
- Raw QC: `.fastq`, `.fastq.gz`, `.fq`, `.fq.gz`, `.bam`, `.sam`, `.cram`
- Summary stats: `.tsv`, `.csv`, `.txt`, and gz variants
- Text: `.md`, `.markdown`, `.text`, `.note`, `.log`
- Spreadsheet: `.xlsx`, `.xlsm`
- DICOM: `.dcm`, `.dicom`
- Image: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp`
- NIfTI: `.nii`, `.nii.gz`
- FHIR bundle: `.fhir.json`, `.fhir.xml`, `.ndjson`, selected `.json`/`.xml` naming patterns

## Quick Start

### 1) Environment bootstrap

```bash
bash bootstrap.sh
```

### 2) Run backend

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

### 3) Run frontend

```bash
npm install
npm --workspace webapp run dev
```

Open: `http://127.0.0.1:3000`

## Useful Chat Patterns

- Grounded explanation: `$studio summarize this result`
- Explicit imaging tool: `@ct_denoise backend=corediff`
- NL imaging request: `Please denoise this CT image`
- Tool help: `@mri_denoise help`

## Developer Notes

- Add a new source type: update `app/services/source_registry.py`, bootstrap manifest, API models/endpoints, frontend upload + renderer wiring.
- Add a new tool: create `plugins/<tool>/tool.json` and `plugins/<tool>/logic.py`, then expose UI renderer if needed.
- Keep direct-chat metadata (`direct_chat`) aligned with renderer and result slots for reliable Studio updates.

## License

Copyright 2026. BISPL@KAIST AI, All rights reserved.

All integratation code and research for this restoration topic project done entirely by Tien Tran contact if you have any problem