# ChatGenome Developer Manual

This manual is for collaborators who want to add workflows, tools, metadata, or Studio renderers to ChatGenome.

## 1. Core Concepts

ChatGenome separates four concepts:

### `@mode`

`@mode` selects session purpose and shapes the input UI.

Current modes:
- `@mode prs`
- `@mode vcf_analysis`
- `@mode raw_sequence`

What a mode controls:
- which source slots appear
- which workflows are compatible
- which direct tools make sense
- how uploads are interpreted

### `@skill`

`@skill` runs a multi-step workflow.

Current examples:
- `@skill representative_vcf_review`
- `@skill raw_qc_review`
- `@skill summary_stats_review`
- `@skill prs_prep`

What a workflow controls:
- ordered execution steps
- workflow-specific guidance
- the Studio view that should open

### `@toolname`

`@toolname` runs one deterministic tool.

Examples:
- `@liftover`
- `@samtools`
- `@plink`
- `@qqman`

What a direct tool controls:
- one execution step
- one result payload
- one or more Studio cards

### `$studio`

`$studio` tells chat to explain the current Studio state rather than answer as a general GPT assistant.

## 2. Repository Areas You Will Touch

### Skill policy

- [../skills/chatgenome-orchestrator/SKILL.md](../skills/chatgenome-orchestrator/SKILL.md)

Use this file to define:
- trigger policy
- workflow recommendation policy
- interpretation rules
- grounded-chat behavior

### Workflow definitions

- [../skills/chatgenome-orchestrator/workflows](../skills/chatgenome-orchestrator/workflows)

Each workflow is a JSON file such as:
- `representative_vcf_review.json`
- `raw_qc_review.json`
- `summary_stats_review.json`
- `prs_prep.json`

### Tool plugins

- [../plugins](../plugins)

Each plugin typically contains:

```text
plugins/<tool_folder>/
  tool.json
  run.py
```

### Backend runtime

Main files:
- [../app/main.py](../app/main.py)
- [../app/models.py](../app/models.py)
- [../app/services/chat.py](../app/services/chat.py)
- [../app/services/workflows.py](../app/services/workflows.py)

### Frontend

Main file:
- [../webapp/app/page.tsx](../webapp/app/page.tsx)

This file currently handles:
- mode selection
- source slots
- chat command parsing
- Studio card selection
- direct tool result rendering

## 3. How To Add A New Tool

### Step 1. Decide the user-facing alias

Pick the command users should type.

Examples:
- `@clinvar`
- `@gnomad`
- `@vep`

Do not expose only the internal folder name if a simpler alias is better.

### Step 2. Create the plugin folder

Create:

```text
plugins/<tool_folder>/
  tool.json
  run.py
```

### Step 3. Write `tool.json`

At minimum, include:

```json
{
  "name": "example_execution_tool",
  "description": "What the tool does.",
  "task": "example-task",
  "modality": "genomics",
  "approval_required": false,
  "source": "plugin"
}
```

For direct `@tool help` support, also add a `help` section.

Recommended shape:

```json
{
  "name": "example_execution_tool",
  "description": "Run example deterministic processing.",
  "task": "example-task",
  "modality": "genomics",
  "approval_required": false,
  "source": "plugin",
  "aliases": ["example"],
  "help": {
    "summary": "Short user-facing summary.",
    "modes": [
      {
        "name": "run",
        "description": "Default execution mode."
      }
    ],
    "options": [
      {
        "name": "mode",
        "type": "string",
        "required": false,
        "default": "run",
        "description": "Execution mode."
      }
    ],
    "examples": [
      "@example help",
      "@example",
      "@example mode=run"
    ],
    "notes": [
      "Any environment or input constraints."
    ]
  }
}
```

Important rules:
- keep option names curated and stable
- do not dump an entire upstream CLI into `help`
- document only what ChatGenome actually supports

### Step 4. Write `run.py`

Your script must:

1. read JSON from `--input`
2. perform deterministic work
3. write JSON to `--output`
4. exit `0` on success

Minimal template:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))

    result = {
        "tool": "example_execution_tool",
        "summary": "Example tool finished successfully.",
        "artifacts": {},
    }

    Path(args.output).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
```

### Step 5. Define the result shape

Prefer structured JSON over free-form prose.

Typical fields:
- `tool`
- `summary`
- `warnings`
- `output_path`
- `log_path`
- `preview_rows`
- tool-specific artifacts

If a Studio card will need it, return it structurally.

### Step 6. Wire the backend

Depending on the tool, wire it in:
- direct `@tool` execution path
- workflow runner
- both

Typical places:
- `app/services/chat.py`
- `app/services/workflows.py`
- `app/main.py`

Recommended division of responsibility:
- `tool.json`: metadata and help facts
- `SKILL.md`: policy and recommendation
- backend: dispatch, execution, persistence

### Step 7. Add Studio rendering

If the result needs a card, update:
- [../webapp/app/page.tsx](../webapp/app/page.tsx)

There are usually three parts:
- card registration
- card content rendering
- direct-result fallback for pre-analysis runs

## 4. How To Add A New Workflow

### Step 1. Create a workflow JSON

Add a file under:
- `/skills/chatgenome-orchestrator/workflows/<name>.json`

Example:

```json
{
  "name": "representative_vcf_review",
  "description": "Run the default representative VCF review workflow.",
  "source_type": "vcf",
  "steps": [
    "vcf_qc_tool",
    "annotation_tool",
    "roh_analysis_tool",
    "candidate_ranking_tool",
    "clinvar_review_tool",
    "vep_consequence_tool",
    "grounded_summary_tool"
  ],
  "default_view": "summary"
}
```

### Step 2. Update the orchestrator skill

Document:
- when the workflow should be recommended
- which source type it expects
- what it should produce

Update:
- [../skills/chatgenome-orchestrator/SKILL.md](../skills/chatgenome-orchestrator/SKILL.md)

### Step 3. Wire `@skill`

Make sure:
- `@skill help` includes it
- `@skill <workflow> help` shows its steps and purpose
- `@skill <workflow>` can execute it

### Step 4. Make sure the result is visible

Most workflows should:
- update analysis state
- open an appropriate Studio view
- optionally add a grounded assistant summary

## 5. How To Add A New Mode

Modes are different from workflows.

Use a new `@mode` when the session needs:
- a different source layout
- different recommended workflows
- different semantics for uploads

Examples:
- PRS mode needs two input slots
- VCF analysis mode needs one VCF slot

When adding a new mode, update:
- command parsing in `webapp/app/page.tsx`
- source-slot UI
- mode help text
- any workflow compatibility checks

## 6. Metadata Preparation Rules

When contributing a new tool, prepare these layers:

### Skill policy

Put policy in:
- `skills/chatgenome-orchestrator/SKILL.md`

Examples:
- when to recommend the tool
- when not to recommend it
- what follow-up it should suggest

### Workflow definition

Put ordered step definitions in:
- `skills/chatgenome-orchestrator/workflows/*.json`

### Tool metadata

Put execution facts in:
- `plugins/<tool>/tool.json`

Examples:
- alias
- help summary
- options
- examples

### Runtime schema

Put structured request/response models in:
- `app/models.py`

Examples:
- request payload
- response payload
- preview row model

### Backend service

Put actual execution in:
- `app/services/*.py`

### Frontend mapping

Put Studio card wiring in:
- `webapp/app/page.tsx`

## 7. Recommended Contributor Checklist

Before opening a PR:

1. Validate Python syntax:

```bash
PYTHONPATH=/Users/jongcye/Documents/Codex/.vendor PYTHONPYCACHEPREFIX=/tmp python3 -m py_compile app/main.py app/models.py app/services/*.py
```

2. Build the frontend:

```bash
cd webapp
PATH=/Users/jongcye/Documents/Codex/.local/node/node-v22.22.1-darwin-arm64/bin:$PATH npm run build:local
```

3. Test `@tool help` if you added a direct tool.

4. Test `@skill help` if you added a workflow.

5. Confirm a Studio card appears if the tool produces a review artifact.

## 8. Current Contribution Pattern

The current repo is in a transition state:

- policy is increasingly moving to `SKILL.md`
- tool help is increasingly moving to `tool.json`
- workflow step ordering is increasingly moving to workflow JSON
- some backend dispatch is still explicit and not yet fully generic

When contributing, prefer:
- adding metadata
- keeping payloads structured
- minimizing new backend keyword heuristics

## 9. Good First Extensions

Current high-value additions:
- standalone `@clinvar`
- standalone `@gnomad`
- standalone `@vep`
- more generic `@tool` runner
- more generic `@skill` runner

These are better next steps than adding a very large new annotation stack first.
