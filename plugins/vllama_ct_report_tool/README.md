# MS-VLM Brain CT Report Tool

`ms_vlm_ct_report_tool` adds MS-VLM brain CT report generation to ChatClinic for active 3D NIfTI sources. Users can run it from chat with `@ctreport` or `@ms-vlm`, or call the plugin runner directly with a JSON payload.

This README is intended to ship with the pull request. Large sample volumes and model checkpoints are intentionally not committed; download them from the Google Drive link shared with the PR and place them in the paths described below.

## Included In Git

- `tool.json`: ChatClinic tool manifest, aliases, runtime metadata, and help text.
- `logic.py`: model loading, NIfTI preprocessing, inference, and response formatting.
- `run.py`: CLI wrapper for local/plugin execution.
- `model/chatmedi_img/`: bundled checkpoint-compatible CT report runtime.
- `model/vllama/`: bundled MS-VLM modules needed by the local runtime.
- `model/eval_configs/brainct_stage2.yaml`: default inference config.
- `model/prompts/`: prompt templates and prompt candidates.
- `model/weights/README.md` and `model/samples/README.md`: placeholders for external artifacts.

## Not Included In Git

The PR should not include files under these directories:

```text
vllama_ct_report_tool/samples/
vllama_ct_report_tool/weights/
plugins/vllama_ct_report_tool/model/samples/
plugins/vllama_ct_report_tool/model/weights/
```

The plugin expects the model artifacts to be restored locally under:

```text
plugins/vllama_ct_report_tool/model/weights/
```

Default expected filenames:

```text
brainct_ckpt_epoch6.pth  # MS-VLM brain CT report-generation checkpoint
checkpoint0009.pth       # DINO/vision encoder checkpoint
```

Optional sample NIfTI volumes can be placed under:

```text
plugins/vllama_ct_report_tool/model/samples/
```

## Environment

Run the backend from the repository root:

```bash
cd chatclinic-multimodal
export PYTHONPATH="$PWD:$PYTHONPATH"
```

The tool requires CUDA. The Python environment must be able to import ChatClinic plus the MS-VLM runtime dependencies, including:

```text
torch
torchvision
transformers
omegaconf
peft
einops
einops_exts
open_clip_torch
kornia
iopath
nibabel
pandas
h5py
Pillow
```

## Restore External Artifacts

After downloading the artifact bundle from the Google Drive link, copy the checkpoints into the plugin-local weights directory:

```bash
mkdir -p plugins/vllama_ct_report_tool/model/weights

cp /path/to/brainct_ckpt_epoch6.pth \
  plugins/vllama_ct_report_tool/model/weights/brainct_ckpt_epoch6.pth

cp /path/to/checkpoint0009.pth \
  plugins/vllama_ct_report_tool/model/weights/checkpoint0009.pth
```

To add a local test volume:

```bash
mkdir -p plugins/vllama_ct_report_tool/model/samples
cp /path/to/brainct_example.nii.gz \
  plugins/vllama_ct_report_tool/model/samples/brainct_example.nii.gz
```

These artifact paths are ignored by git.

## Optional Configuration

The bundled config is used by default:

```text
plugins/vllama_ct_report_tool/model/eval_configs/brainct_stage2.yaml
```

You can override paths with environment variables:

```bash
export MS_VLM_CT_CFG_PATH=plugins/vllama_ct_report_tool/model/eval_configs/brainct_stage2.yaml
export MS_VLM_CT_CHECKPOINT=plugins/vllama_ct_report_tool/model/weights/brainct_ckpt_epoch6.pth
export MS_VLM_CT_VIT_PATH=plugins/vllama_ct_report_tool/model/weights/checkpoint0009.pth
```

For lower-memory loading:

```bash
export MS_VLM_CT_LOW_RESOURCE=1
```

Checkpoint overrides must resolve inside `plugins/vllama_ct_report_tool/model/weights/`.

## Start ChatClinic

Start the backend:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Start the web app in another terminal if needed:

```bash
cd webapp
npm install
npm run dev
```

Upload a `.nii` or `.nii.gz` brain CT volume through the UI, or register a local NIfTI by path:

```bash
curl -s http://127.0.0.1:8001/api/v1/source/from-path \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "plugins/vllama_ct_report_tool/model/samples/brainct_example.nii.gz",
    "source_type": "nifti",
    "file_name": "brainct_example.nii.gz"
  }'
```

## Run From Chat

With a NIfTI source active, run:

```text
@ctreport
```

Common options:

```text
@ctreport gpu_id=0 max_new_tokens=220 temperature=0.4
@ctreport checkpoint_path=brainct_ckpt_epoch6.pth vit_path=checkpoint0009.pth
@ms-vlm
```

The first call loads the model and will be slower. Subsequent calls reuse the loaded model in the same backend process.

The generated report is returned in chat and stored on the active NIfTI analysis under:

```text
artifacts.ms_vlm_ct_report
```

The Studio view uses the `ct_report` renderer.

## Run From The Plugin CLI

Create a payload:

```json
{
  "nifti_path": "plugins/vllama_ct_report_tool/model/samples/brainct_example.nii.gz",
  "file_name": "brainct_example.nii.gz",
  "gpu_id": 0,
  "max_new_tokens": 200,
  "temperature": 0.5
}
```

Run:

```bash
python -m plugins.vllama_ct_report_tool.run \
  --input /path/to/payload.json \
  --output /path/to/ct_report_result.json
```

## Troubleshooting

- `MS-VLM checkpoint does not exist`: copy `brainct_ckpt_epoch6.pth` into `plugins/vllama_ct_report_tool/model/weights/`.
- `MS-VLM vision checkpoint does not exist`: copy `checkpoint0009.pth` into `plugins/vllama_ct_report_tool/model/weights/`.
- `requires CUDA`: run the backend in a CUDA-enabled environment and pass a valid `gpu_id`.
- Import errors from `transformers`, `peft`, `open_clip`, `kornia`, or `nibabel`: install the missing dependency in the backend environment.
- Empty report text: inspect `raw_output` in the CLI/API result and try a lower `temperature` or a different `prompt_index`.

This tool generates research support text from brain CT volumes. It is not a standalone clinical diagnostic system.
