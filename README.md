# ChatClinic

Multimodal clinical workspace with auto-detected source types, deterministic tool execution, Studio cards, and grounded chat.

![ChatClinic workspace preview](docs/chatclinic-ui-preview.png)

## Overview

ChatClinic is a three-panel workspace: **Sources**, **Chat**, and **Studio**.

1. Upload a source file — the type is auto-detected.
2. Bootstrap tools run automatically on upload.
3. Review outputs in Studio cards.
4. Ask questions in Chat — use `$studio` to ground answers in current results.
5. Run additional tools with `@toolname` commands.
6. Type `@help` for the full tool guide.

## Supported Source Types

| Source Type | Formats | Auto Tools |
|-------------|---------|------------|
| **DICOM** | `.dcm`, `.dicom` | DICOM Review (metadata, series summary, preview) |
| **Image** | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp` | Image Review (metadata, EXIF, thumbnail) |
| **NIfTI** | `.nii`, `.nii.gz` | NIfTI Review (shape, voxel dims, orientation, 3D Niivue viewer) |
| **FHIR Bundle** | `.fhir.json`, `.fhir.xml`, `.ndjson` | FHIR Browser (patient, medications, labs, care team) |
| **Excel Workbook** | `.xlsx`, `.xls` | Cohort Browser (sheets, schema, missingness) |
| **Text / Markdown** | `.txt`, `.md`, `.markdown` | Text Review (preview, grounded summary) |
| **VCF** | `.vcf`, `.vcf.gz` | QC Summary, Annotation, ClinVar, VEP, Candidate Ranking, ROH, CADD/REVEL, Grounded Summary |
| **Raw Sequencing** | `.fastq`, `.fastq.gz`, `.fq`, `.fq.gz`, `.bam`, `.sam`, `.cram` | FastQC Review |
| **Summary Statistics** | `.tsv`, `.csv`, `.txt`, `.gz` (GWAS) | Summary Stats Review (column detection, schema mapping) |

## Tool Commands

### Direct `@tool` commands

| Command | Purpose | Source |
|---------|---------|--------|
| `@liftover [target=hg38]` | Convert genome build (hg19 ↔ hg38) | VCF |
| `@snpeff [genome=...]` | Run local SnpEff variant annotation | VCF |
| `@plink [mode=qc\|score]` | PLINK 2 QC or PRS scoring | VCF |
| `@ldblockshow chr:start:end` | LD heatmap for a genomic region | VCF |
| `@samtools [mode=qc]` | samtools flagstat / stats / idxstats | FASTQ/BAM/SAM |
| `@qqman` | Manhattan plot + QQ plot | Summary statistics |
| `@prs_prep` | Build check, harmonization, score-file preparation | Summary statistics |
| `@unsb [num_timesteps=5 tau=0.01]` | Run local UNSB enhancement and attach the enhanced preview | Image |

Each tool supports `@toolname help` for detailed options.

### Studio grounding

| Prefix | Effect |
|--------|--------|
| `$studio` | Interpret the active Studio cards |
| `$current analysis` | Summarize the current analysis artifacts |
| `$current card` | Explain the currently open card |
| `$grounded` | Answer from tool-derived state only |

Without a grounding prefix, questions are answered as general knowledge.

### Other commands

| Command | Effect |
|---------|--------|
| `@help` | Show full tool guide (loaded from SKILL.md) |

## Architecture

```
Sources (left)          Chat (center)           Studio (right)
┌──────────────┐   ┌────────────────────┐   ┌──────────────────┐
│ Auto-detect  │   │ General + grounded │   │ Card grid        │
│ Multi-source │   │ @tool execution    │   │ Tool outputs     │
│ Upload       │   │ $studio grounding  │   │ Interactive view  │
└──────────────┘   └────────────────────┘   └──────────────────┘
```

### Key directories

| Path | Purpose |
|------|---------|
| `app/main.py` | FastAPI endpoints |
| `app/models.py` | Request/response schemas |
| `app/services/` | Chat, workflows, bootstrap, source registry, tool runner |
| `plugins/` | Tool plugins (`tool.json` + `logic.py`) |
| `skills/chatgenome-orchestrator/` | SKILL.md policy, bootstrap manifests |
| `webapp/app/` | Next.js frontend (page.tsx, renderers, CSS) |

### Plugin structure

```
plugins/<tool_name>/
  tool.json       — metadata, aliases, help, workflow_binding
  logic.py        — Python implementation (entrypoint)
  run.py          — optional CLI wrapper (--input/--output)
```

## Adding a New Tool

### Step 1. Create the plugin

```
plugins/my_tool/
  tool.json
  logic.py
```

### Step 2. Write `tool.json`

```json
{
  "name": "my_tool",
  "description": "What the tool does.",
  "task": "my-task",
  "modality": "genomics",
  "source": "plugin",
  "aliases": ["mytool"],
  "help": {
    "summary": "Short user-facing summary.",
    "options": [],
    "examples": ["@mytool help", "@mytool"]
  }
}
```

### Step 3. Write `logic.py`

```python
def run(payload: dict) -> dict:
    # deterministic work here
    return {"tool": "my_tool", "summary": "Done.", "artifacts": {}}
```

### Step 4. Wire the backend

- Add endpoint in `app/main.py` if the tool is user-facing
- Add request/response models in `app/models.py`
- Tool routing via registry is often automatic

### Step 5. Add Studio rendering

- Add renderer entry in `webapp/app/components/studioRenderers.tsx`
- Add card component in `customStudioRenderers.tsx` or `genericStudioRenderers.tsx`

### Step 6. Update SKILL.md

- Add tool to the `## Help message` section
- Add tool to the orchestrator rules

## Adding a New Source Type

1. Register in `app/services/source_registry.py` (suffixes, labels, initial tools)
2. Create bootstrap manifest in `skills/chatgenome-orchestrator/bootstrap/`
3. Add upload + chat endpoints in `app/main.py`
4. Add response models in `app/models.py`
5. Add frontend detection, state, and handlers in `webapp/app/page.tsx`
6. Add Studio renderer component

## Quick Start

### Setup

```bash
git clone https://github.com/bispl-create/chatclinic-multimodal.git
cd chatclinic-multimodal
conda env create -f environment.yml
conda activate chatclinic
cp .env.example .env
```

This environment targets Python 3.10, PyTorch 2.5.1, TorchVision 0.20.1, and CUDA 12.1 via the `pytorch` and `nvidia` conda channels.

Set your API key in `.env`:

```bash
OPENAI_API_KEY=sk-...
```

### Install frontend

```bash
cd webapp
npm install
```

### Run

Backend:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Frontend:

```bash
cd webapp
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Documentation

- [docs/DEVELOPER_MANUAL.md](docs/DEVELOPER_MANUAL.md) — Full developer guide
- [docs/TOOL_PLUGIN_GUIDE.md](docs/TOOL_PLUGIN_GUIDE.md) — Tool plugin reference
- [docs/FRONTEND_RENDERER_INVENTORY.md](docs/FRONTEND_RENDERER_INVENTORY.md) — Studio renderer inventory
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contributor setup and checklist

## Design Principle

1. Use deterministic tools to establish facts.
2. Render outputs in Studio cards.
3. Use `$studio` only when you want the model to explain grounded results.

This keeps execution, evidence, and explanation clearly separated.

## License

Copyright 2026. BISPL@KAIST AI, All rights reserved.
