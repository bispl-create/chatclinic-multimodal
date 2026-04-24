# Pancreas + Tumor Segmentation (`pancreas_tumor_seg`)

**Model file**: `tools/pancreas_tumor_seg/weights/pancreas_ct_dints_segmentation/models/pancreas_tumor_seg.pt`

Segments the pancreas and tumor from 3D abdominal CT using MONAI Bundle's DiNTS (NAS-based architecture).

## Dependencies

`torch`, `torchvision`, `monai==1.3.2`, `nibabel==5.4.2`, `numpy==1.26.4`, `omegaconf==2.3.0`, `scipy==1.13.1`

```bash
pip install -r requirements.txt
```

> `torch` and `torchvision` are not included in `requirements.txt` — install them separately with the appropriate CUDA index.

## Input

- **Format**: Single-channel 3D CT NIfTI file (`.nii` / `.nii.gz`)
- **Payload key**: `nifti_path`

## Output

- **Format**: 3-class NIfTI segmentation mask
- **Classes**:
  - Background
  - Pancreas
  - Tumor
- **Location**: `results/<timestamp>/`

## Model Weights Placement

Place weights under the `weights/` directory. Run `setup_bundles.py --bundles-only` from the project root to download automatically, or arrange files manually:

```
weights/pancreas_ct_dints_segmentation/
├── configs/inference.yaml
└── models/pancreas_tumor_seg.pt
```

## Dataset Placement

```
dataset/Task07_Pancreas/
├── imagesTr/
└── labelsTr/
```

## Usage

```python
from tools.pancreas_tumor_seg.logic import execute

result = execute({
    "nifti_path": "tools/pancreas_tumor_seg/dataset/Task07_Pancreas/imagesTr/pancreas_001.nii.gz",
})

print(result["summary"])
print(result["artifacts"]["mask_path"])
```

### Payload Fields

| Key | Type | Description |
|---|---|---|
| `nifti_path` | `str` | Path to a single-channel 3D CT NIfTI file |
| `device` | `str` | `"gpu"` (default) or `"cpu"` |
| `output_dir` | `str` | Override output directory |
| `overrides` | `list[str]` | OmegaConf key=value overrides |
| `file_name` | `str` | Display name for results |

## Configuration

Default settings in `config.yaml`:

| Key | Value |
|---|---|
| `roi_size` | `[96, 96, 96]` |
| `sw_batch_size` | `1` |
| `overlap` | `0.5` |

Override via payload:

```python
execute({
    "nifti_path": "...",
    "overrides": ["tool.sw_batch_size=2", "tool.overlap=0.25"],
})
```

## Limitations

- Designed for abdominal CT only
- Fixed ROI size `[96, 96, 96]`
- Relatively slow inference due to DiNTS architecture
