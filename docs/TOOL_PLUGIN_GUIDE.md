# Tool Plugin Guide

This guide is the short version of the full developer manual.

For the complete architecture and contribution flow, read:
- [DEVELOPER_MANUAL.md](DEVELOPER_MANUAL.md)

## Minimal Tool Submission

Each new tool should live under:

```text
plugins/<tool_folder>/
  tool.json
  run.py
```

## Required Metadata

Your `tool.json` should include:

```json
{
  "name": "example_execution_tool",
  "description": "Short deterministic tool summary.",
  "task": "example-task",
  "modality": "genomics",
  "approval_required": false,
  "source": "plugin",
  "aliases": ["example"],
  "help": {
    "summary": "What the tool does.",
    "modes": [],
    "options": [],
    "examples": [
      "@example help",
      "@example"
    ],
    "notes": []
  }
}
```

Use `help` metadata for:
- `@toolname help`
- option documentation
- curated examples

## Required Runtime Contract

`run.py` is expected to support:

```bash
python run.py --input <input.json> --output <output.json>
```

The script should:

1. read JSON from `--input`
2. perform deterministic work
3. write JSON to `--output`
4. exit successfully on success

## Required Integration Steps

After adding the plugin files, make sure you also:

1. define the request/response shape in
   - [../app/models.py](../app/models.py)
2. add execution wiring in backend runtime
   - usually in [../app/main.py](../app/main.py),
   - [../app/services/chat.py](../app/services/chat.py), or
   - [../app/services/workflows.py](../app/services/workflows.py)
3. add Studio rendering in
   - [../webapp/app/page.tsx](../webapp/app/page.tsx)
4. update orchestrator policy in
   - [../skills/chatgenome-orchestrator/SKILL.md](../skills/chatgenome-orchestrator/SKILL.md)
   if the tool should be recommended or used in workflows

## Testing Checklist

Python syntax:

```bash
PYTHONPATH=/Users/jongcye/Documents/Codex/.vendor PYTHONPYCACHEPREFIX=/tmp python3 -m py_compile app/main.py app/models.py app/services/*.py plugins/<tool_folder>/run.py
```

Frontend build:

```bash
cd webapp
PATH=/Users/jongcye/Documents/Codex/.local/node/node-v22.22.1-darwin-arm64/bin:$PATH npm run build:local
```

Runtime checks:
- `@toolname help`
- direct `@toolname`
- Studio card rendering

## Preferred Design Rule

Put:
- policy in `SKILL.md`
- workflow ordering in workflow JSON
- exact tool facts in `tool.json`
- runtime execution in backend services

Avoid introducing new keyword-based chat heuristics unless absolutely necessary.
