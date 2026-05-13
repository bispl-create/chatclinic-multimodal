# Contributing

Before making changes, read:

- [README.md](README.md)
- [docs/DEVELOPER_MANUAL.md](docs/DEVELOPER_MANUAL.md)
- [docs/TOOL_PLUGIN_GUIDE.md](docs/TOOL_PLUGIN_GUIDE.md)

## Local setup

1. Clone and create virtual environment:

```bash
git clone https://github.com/bispl-create/chatclinic-multimodal.git
cd chatclinic-multimodal
conda env create -f environment.yml
conda activate chatclinic
```

2. Install frontend dependencies:

```bash
cd webapp
npm install
```

3. Create local environment variables:

```bash
cp .env.example .env
```

## Run

Backend:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Frontend:

```bash
cd webapp
npm run dev
```

## Before opening a PR

Run Python syntax check:

```bash
python3 -m py_compile app/main.py app/models.py app/services/*.py
```

Run frontend build:

```bash
cd webapp
npm run build
```

## Project notes

- The UI is a 3-column **Sources / Chat / Studio** workspace.
- Source types are auto-detected on upload (no `@mode` required).
- Multiple sources can coexist in a single session.
- Studio-derived summaries are forwarded into Chat through `studio_context`.
- Keep secrets out of the repository. Use `.env` only for local development.
- Current explicit triggers:
  - `@toolname` — run a deterministic tool
  - `@help` — show tool guide
  - `$studio` / `$grounded` — ground chat in Studio state
