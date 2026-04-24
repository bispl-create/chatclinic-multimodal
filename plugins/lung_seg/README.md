# Lung Segmentation (`lung_seg`)

**Model file**: `tools/lung_seg/weights/cxr/pspnet_chestxray_best_model_4.pth`

Segments lung regions from 2D chest X-ray (CXR) images using torchxrayvision's PSPNet.

## Dependencies

`torch`, `torchvision`, `torchxrayvision==1.4.0`, `scikit-image==0.23.2`, `numpy==1.26.4`, `omegaconf==2.3.0`, `scipy==1.13.1`

```bash
pip install -r requirements.txt
```

> `torch` and `torchvision` are not included in `requirements.txt` — install them separately with the appropriate CUDA index.

## Input

- **Format**: 2D CXR image (`.png` / `.jpg`)
- **Payload key**: `image_path`

## Output

- **Format**: Binary lung mask PNG
- **Classes**: Left Lung + Right Lung (extracted from 14-class output, indices [4, 5])
- **Location**: `results/<timestamp>/`

## Model Weights Placement

Weights are stored under the `weights/cxr/` directory:

```
weights/cxr/
└── pspnet_chestxray_best_model_4.pth
```

> If the weights file is missing, torchxrayvision will automatically download it on first run (internet connection required).

## Dataset Placement

Place CXR images directly in the `dataset/` folder (flat directory):

```
dataset/MCUCXR_0001_0.png
dataset/MCUCXR_0002_0.png
...
```

## Usage

```python
from tools.lung_seg.logic import execute

result = execute({
    "image_path": "tools/lung_seg/dataset/MCUCXR_0001_0.png",
})

print(result["summary"])
print(result["artifacts"]["mask_path"])
```

### Payload Fields

| Key | Type | Description |
|---|---|---|
| `image_path` | `str` | Path to a 2D CXR image (`.png` / `.jpg`) |
| `device` | `str` | `"gpu"` (default) or `"cpu"` |
| `output_dir` | `str` | Override output directory |
| `overrides` | `list[str]` | OmegaConf key=value overrides |
| `file_name` | `str` | Display name for results |

## Configuration

Default settings in `config.yaml`:

| Key | Value |
|---|---|
| `input_size` | `512` |
| `num_classes` | `14` |
| `lung_indices` | `[4, 5]` |

## Limitations

- 2D CXR only (not compatible with CT or MRI)
- Input is resized to 512x512
- Weights are auto-downloaded on first run if missing (internet required)
