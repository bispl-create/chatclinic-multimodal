# Brain Tumor Segmentation (`brain_tumor_seg`)

**Model file**: `tools/brain_tumor_seg/weights/brats_mri_segmentation/models/brain_tumor_seg.pt`

Segments brain tumor regions (ET, TC, WT) from 3D MRI using MONAI Bundle's SegResNet.

## Dependencies

`torch`, `torchvision`, `monai==1.3.2`, `nibabel==5.4.2`, `numpy==1.26.4`, `omegaconf==2.3.0`, `scipy==1.13.1`

```bash
pip install -r requirements.txt
```

> `torch` and `torchvision` are not included in `requirements.txt` — install them separately with the appropriate CUDA index.

## Input

- **Format**: 4-channel 3D MRI NIfTI file (`.nii` / `.nii.gz`)
- **Channels**: T1, T1ce, T2, FLAIR
- **Payload key**: `nifti_path`

## Output

- **Format**: 3-class NIfTI segmentation mask
- **Classes**:
  - ET (Enhancing Tumor)
  - TC (Tumor Core)
  - WT (Whole Tumor)
- **Location**: `results/<timestamp>/`

## Model Weights Placement

Place weights under the `weights/` directory. Run `setup_bundles.py --bundles-only` from the project root to download automatically, or arrange files manually:

```
weights/brats_mri_segmentation/
├── configs/inference.json
└── models/brain_tumor_seg.pt
```

## Dataset Placement

```
dataset/Task01_BrainTumour/
├── imagesTr/
└── labelsTr/
```

## Usage

```python
from tools.brain_tumor_seg.logic import execute

result = execute({
    "nifti_path": "tools/brain_tumor_seg/dataset/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz",
})

print(result["summary"])
print(result["artifacts"]["mask_path"])
```

### Payload Fields

| Key | Type | Description |
|---|---|---|
| `nifti_path` | `str` | Path to a 4-channel 3D MRI NIfTI file |
| `device` | `str` | `"gpu"` (default) or `"cpu"` |
| `output_dir` | `str` | Override output directory |
| `overrides` | `list[str]` | OmegaConf key=value overrides |
| `file_name` | `str` | Display name for results |

## Configuration

Default settings in `config.yaml`:

| Key | Value |
|---|---|
| `roi_size` | `[224, 224, 144]` |
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

- Only supports 4-channel MRI input (single-sequence not accepted)
- Fixed ROI size `[224, 224, 144]` — requires ~8 GB VRAM or more
- Only validated on MSD Task01_BrainTumour format
