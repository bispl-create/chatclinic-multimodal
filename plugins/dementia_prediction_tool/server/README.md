# Dementia-R1 Inference Server

Standalone FastAPI server that hosts the fine-tuned Qwen2 (Dementia-R1) model
on a GPU host and exposes a `POST /predict` endpoint. The
`dementia_prediction_tool` plugin (at `plugins/dementia_prediction_tool/logic.py`)
calls this server over HTTP — locally when co-hosted, via an SSH tunnel when
remote.

This directory is **not executed by the ChatClinic-Multimodal webapp itself.**
It is deployed separately onto a GPU machine.

---

## Architecture

```
┌──────────────────────────────┐           ┌─────────────────────────────┐
│  Client (Mac / CPU host)     │   HTTP    │  GPU host                   │
│  chatclinic-multimodal        │ ───────▶ │  api_server.py (port 8020)  │
│  └─ plugins/                  │           │  └─ vLLM + Qwen2 (Dementia) │
│     dementia_prediction_tool/ │           │     resident in VRAM        │
│     └─ logic.py (HTTP client) │ ◀─────── │     ~6 s per request         │
└──────────────────────────────┘   JSON    └─────────────────────────────┘
        (no GPU required)                        (≥ 22 GB VRAM)
```

---

## Prerequisites

- Linux host with NVIDIA GPU and **≥ 22 GB VRAM** (model uses ~22 GB at inference)
- CUDA-compatible PyTorch install
- Python 3.10+
- Dementia-R1 model checkpoint (~14 GB on disk) — see [Model weights](#model-weights) below

---

## Setup

### 1. Install dependencies

```bash
cd plugins/dementia_prediction_tool/server
python -m venv .venv            # or use conda
source .venv/bin/activate
pip install -r requirements.txt
```

> `vllm` pulls in a specific PyTorch build. If `pip install vllm` complains
> about CUDA, install a matching `torch` first per
> https://pytorch.org/get-started/locally/.

### 2. Model weights

The fine-tuned checkpoint is **not** stored in this repository (≈ 14 GB).
Obtain it from the project maintainers and set its path via the
`DEMENTIA_MODEL_PATH` environment variable. The checkpoint directory should
contain `config.json`, `tokenizer.json`, `generation_config.json`, and the
safetensors shards.

```bash
export DEMENTIA_MODEL_PATH=/abs/path/to/Dementia-R1/checkpoint-4600
```

If you have access to the `bispl-server-muzi` host, the checkpoint is already
available at
`/mnt/data1/choonghan/Digital_Bio/Asan/Dementia-R1/checkpoint-4600`.

### 3. Start the API server

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python api_server.py > /tmp/dementia_api.log 2>&1 &
```

The server loads the model eagerly (~30 s). When ready it prints
`[API] Model loaded successfully.` to the log. Verify:

```bash
curl http://127.0.0.1:8020/health
# {"status":"ok","model_loaded":true}
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEMENTIA_API_PORT` | `8020` | Port the FastAPI server binds to |
| `DEMENTIA_MODEL_PATH` | *(muzi-local relative path)* | **Required on a fresh host** — absolute path to the `checkpoint-4600` directory |
| `DEMENTIA_GPU_DEVICE` | `0` | `CUDA_VISIBLE_DEVICES` value (applied before CUDA init) |
| `DEMENTIA_MAX_TOKENS` | `1024` | Max new tokens generated per request |
| `DEMENTIA_MAX_MODEL_LEN` | `4096` | Model context length |
| `DEMENTIA_TP_SIZE` | `1` | vLLM tensor-parallel size |
| `DEMENTIA_GPU_UTIL` | `0.90` | vLLM `gpu_memory_utilization` fraction |

---

## API

### `GET /health`

Returns `{"status": "ok", "model_loaded": bool}`. `model_loaded` is `true`
only after the first successful model load (automatic at server start).

### `POST /predict`

Request body:

```json
{ "clinical_notes": ["<longitudinal clinical note text>", "..."] }
```

Response:

```json
{
  "predictions": [
    {
      "patient_index": 0,
      "prediction": 1,
      "prediction_label": "likely dementia",
      "probability_score": 0.9987,
      "reasoning": "<chain-of-thought>"
    }
  ],
  "summary": "## Dementia Prediction Results …",
  "model": "Dementia-R1",
  "elapsed_sec": 6.1
}
```

---

## Using the server from the webapp

The plugin at `plugins/dementia_prediction_tool/logic.py` reads
`DEMENTIA_API_URL` (default `http://127.0.0.1:8020`). Point it at the server
wherever it lives.

- **Co-hosted** (webapp and GPU on the same machine): leave the default.
- **Remote GPU** (typical — webapp on a Mac / laptop, GPU on a server): open
  an SSH tunnel so `localhost:8020` on the client forwards to the GPU host.

```bash
ssh -L 8020:127.0.0.1:8020 <gpu-host>
```

Keep the tunnel open for the duration of the session.

---

## Operations

### Restart the server

```bash
pkill -f api_server.py
pkill -f EngineCore          # in case vLLM left workers behind
# wait for GPU memory to free
export CUDA_VISIBLE_DEVICES=0
nohup python api_server.py > /tmp/dementia_api.log 2>&1 &
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` on `/health` | Server not running | Start it with the command above |
| Server starts but `/health` returns `model_loaded: false` | Model still loading | Wait ~30 s and retry |
| `FileNotFoundError: Model not found` | `DEMENTIA_MODEL_PATH` unset or wrong | Export the correct absolute path and restart |
| `CUDA out of memory` at startup | Another process on the GPU | `nvidia-smi`, clear residual processes, then retry |
| Same prediction for every input | Client not sending clinical-note text (may be hitting a different route) | Confirm the webapp patches (`.txt` → text source, generic text-source tool handler) are deployed |

### Logs

```bash
tail -f /tmp/dementia_api.log
```

---

## Files in this directory

| File | Purpose |
|---|---|
| `api_server.py` | FastAPI app; loads the model once at startup and serves `/predict` |
| `run.py` | Core inference primitives (prompt build, label extraction, probability) used by `api_server.py`; also has a standalone CLI mode (`python run.py --input in.json --output out.json`) |
| `requirements.txt` | Minimum Python dependencies |
| `README.md` | This file |

## Model card summary

- Architecture: Qwen2ForCausalLM (~7B parameters)
- Checkpoint: `Dementia-R1/checkpoint-4600`
- Training data: Asan Medical Center longitudinal clinical notes with GDS labels
- Output: binary prediction (0/1), probability, chain-of-thought reasoning
- Serving: vLLM with continuous batching

> **Clinical note.** Predictions are for research only and must not replace
> clinical judgement.
