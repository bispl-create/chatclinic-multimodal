# ChatGenome

Interactive genomics review workspace for VCF-driven analysis, grounded summaries, Studio cards, and follow-up chat.

![ChatGenome workspace preview](docs/chatgenome-ui-preview.svg)

This MVP is a starting point for an OpenEvidence-like genomics assistant:

- upload a `VCF` or `VCF.gz`
- parse and summarize variants with `pysam`
- attach citation-ready evidence records
- propose next analysis steps with a rule-driven engine
- return a grounded response payload that a chat UI can render
- run a GPT-backed intake conversation before analysis starts
- combine backend grounded summaries with Studio-derived review context for the first answer
- keep IGV and annotation detail available below the summary

## Quick Start For Contributors

1. Clone the repository.
2. Create your local environment file from the template.
3. Add your own OpenAI API key to `.env`.
4. Install Python and frontend dependencies.
5. Start the API and frontend.

If you are adding a new analysis tool, use the collaborator guide:

- [TOOL_PLUGIN_GUIDE.md](docs/TOOL_PLUGIN_GUIDE.md)

```bash
git clone https://github.com/bispl-create/chatgenome.git
cd chatgenome
cp .env.example .env
```

Then edit `.env` and set your own key:

```bash
OPENAI_API_KEY=sk-...
OPENAI_WORKFLOW_MODEL=gpt-5-nano
OPENAI_MODEL=gpt-5-mini
```

Notes:

- keep `.env` local only; it is excluded from git
- do not commit your personal API key
- if you do not set `OPENAI_API_KEY`, the app still runs in deterministic fallback mode

## What This Prototype Does

- accepts a local VCF upload through FastAPI
- summarizes samples, contigs, variant classes, genotype counts, and example variants
- runs a queue-analysis conversation that asks for annotation scope and range in chat
- uses `gpt-5-nano` for intake parsing when `OPENAI_API_KEY` is configured
- uses `gpt-5-mini` for grounded explanation of summary and annotation meaning
- produces a structured "analysis brief" with:
  - `facts`
  - `references`
  - `recommendations`
  - `ui_cards`
- keeps the evidence layer separate from the language model layer
- shows a 3-column `Sources / Chat / Studio` workspace

## What It Does Not Yet Do

- no full local clinical annotation stack such as VEP CLI, ANNOVAR, or production-scale database mirrors
- no persistent database
- no authentication or PHI controls
- no ACMG classifier
- can attach live Europe PMC / PubMed-backed literature references when network access is available

## Why The Separation Matters

For a medical or bioinformatics workflow, the model should not infer variant meaning directly from raw VCF rows. The safer pattern is:

1. parse the VCF
2. annotate with deterministic tools and curated databases
3. rank evidence
4. let the LLM explain only the grounded evidence

## Project Layout

```text
bioinformatics_vcf_evidence_mvp/
  README.md
  architecture.md
  requirements.txt
  sample.env
  .env.example
  CONTRIBUTING.md
  HANDOFF.md
  app/
    main.py
    models.py
    services/
      annotation.py
      recommendation.py
      references.py
      vcf_summary.py
  frontend/
    index.html
    styles.css
    app.js
```

## Run

Install dependencies:

```bash
python3 -m pip install --target /Users/jongcye/Documents/Codex/.vendor -r /Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/requirements.txt
PATH=/Users/jongcye/Documents/Codex/.local/node-v22.14.0-darwin-arm64/bin:$PATH npm install
```

Configure environment:

```bash
cd /Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp
cp .env.example .env
```

Then edit `.env` and set:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_WORKFLOW_MODEL=gpt-5-nano
OPENAI_MODEL=gpt-5-mini
```

The FastAPI app auto-loads `.env` from the project root at startup.

Start the API:

```bash
cd /Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp
PYTHONPATH=/Users/jongcye/Documents/Codex/.vendor uvicorn app.main:app --reload
```

Then open:

- `GET /health`
- `POST /api/v1/analysis/upload`
- `POST /api/v1/analysis/from-path`
- `POST /api/v1/analysis/from-path/async`
- `GET /api/v1/analysis/jobs/{job_id}`
- `POST /api/v1/chat/analysis`
- `POST /api/v1/workflow/start`
- `POST /api/v1/workflow/reply`

Start the Next.js frontend:

```bash
cd /Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp
PATH=/Users/jongcye/Documents/Codex/.local/node-v22.14.0-darwin-arm64/bin:$PATH npm run dev:webapp
```

Then open [http://127.0.0.1:3000](http://127.0.0.1:3000).

## Primary UI Flow

The main screen is a 3-column workspace:

1. `Sources`
   Attach a VCF and monitor run status.
2. `Chat`
   Answer intake prompts, read grounded summaries, and continue analysis chat.
3. `Studio`
   Open QC, filtering, annotation, ROH/recessive, IGV, and references views.

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analysis/from-path \
  -H 'Content-Type: application/json' \
  -d '{"vcf_path":"/Users/jongcye/Documents/Codex/roh.1.vcf.gz"}'
```

Async request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analysis/from-path/async \
  -H 'Content-Type: application/json' \
  -d '{"vcf_path":"/Users/jongcye/Documents/Codex/roh.1.vcf.gz"}'
```

Then poll:

```bash
curl http://127.0.0.1:8000/api/v1/analysis/jobs/<job-id>
```

Chat request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/analysis \
  -H 'Content-Type: application/json' \
  -d '{"question":"Summarize the key findings","analysis":{...},"history":[]}'
```

## Frontend Tooling

- `npm run check:static-frontend` checks the static frontend JavaScript
- `npm run build:webapp` builds the Next.js frontend
- `npm run dev:webapp` starts the Next.js frontend

## Handoff

For another developer picking this up:

- start with [HANDOFF.md](/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/HANDOFF.md)
- use [CONTRIBUTING.md](/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/CONTRIBUTING.md) for setup and checks
- keep `.env` local and use `.env.example` as the template
- install the Codex skill at `skills/chatgenome-dev/SKILL.md` if they want repo-specific Codex guidance

## GPT Workflow And Analysis Chat

- `POST /api/v1/workflow/start` begins the queue-analysis conversation after a VCF is attached
- `POST /api/v1/workflow/reply` interprets the user's natural-language option reply
- `POST /api/v1/chat/analysis` answers questions about the grounded summary and annotations
- if `OPENAI_API_KEY` is configured:
  - workflow intake uses `OPENAI_WORKFLOW_MODEL` and defaults to `gpt-5-nano`
  - grounded explanation uses `OPENAI_MODEL` and defaults to `gpt-5-mini`
- without `OPENAI_API_KEY`, both routes fall back to deterministic local behavior

## Near-Term Extensions

1. Add VEP or snpEff consequence annotation.
2. Add ClinVar and gnomAD lookup services.
3. Add PubMed/Europe PMC retrieval and evidence ranking.
4. Add pgvector-backed retrieval for prior analyses and literature snippets.
5. Replace the static frontend with a Next.js app when you want routing, auth, and richer state handling.

## Notes On Literature Retrieval

- the MVP uses Europe PMC search as the live retrieval entrypoint
- PubMed review queries are also used for gene-condition review retrieval
- when a PMID is available, cards link directly to PubMed
- if the network is unavailable, the API still returns the static foundational references
- external HTTP responses are cached on disk to reduce repeated latency
- next step: add better query generation from annotated genes, rsIDs, and diseases instead of file-level heuristics
