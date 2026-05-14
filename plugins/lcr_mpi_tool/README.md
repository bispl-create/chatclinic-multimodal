# LCR_MPI_Tool Plugin

ChatClinic plugin package. 
It runs inference for three PyTorch image reconstruction models (**GAN**, **I2SB**, **RDDM**) on the LCR-MPI challenge data.

---

## Package Structure

```text
lcr_mpi_tool/
├── tool.json        # ChatClinic plugin specification file
├── logic.py         # Direct invocation entry point for ChatClinic platform
├── run.py           # CLI local test and execution script
├── sample_images/   # Sample images for testing (Input/Condition)
├── requirements.txt # Dependency libraries list
└── README.md        # This file
```

## Installation and Integration

1. Plugin Folder Placement: Place the `lcr_mpi_tool` folder under the `chatclinic-main/plugins/` directory of the main ChatClinic platform.
   - Example of correct path: `chatclinic-main/plugins/lcr_mpi_tool/tool.json`
2. The `skill_update` folder included inside contains guide documents for integrating with the platform's `SKILL.md`.

> Prerequisites: 
1. Please ensure that the `inference.py` and `best_model.pth` files exist correctly in the `GAN/`, `I2SB/`, and `RDDM/` folders inside this project root (`lcr_mpi_tool/`).

2. When uploading images to ChatClinic, to run the I2SB model, you must upload the original image (Input) and the condition image (Condition) paired 1:1. The condition image filename must include `_cond` or `cond_` so the plugin can correctly match the two files.

Correct filename example:
- Input file: `test_001.png`
- Condition file: `test_001_cond.png` (or `cond_test_001.png`)

> Tip: Sample images are provided in the included `sample_images/` folder to check if it works correctly. You can use these images for testing during platform integration.

*Note: If you run only the GAN or RDDM models, it will work normally by uploading just the Input file (`test_001.png`) without a Condition file.*

## How to Run

### 1. Run on ChatClinic Platform
Upload the image to be reconstructed (including the Condition file if necessary) to the platform chat window, and ask the AI agent by specifying the tool name and model name to run it automatically.

> Prompt example:
> "Restore the uploaded file with the gan model using the lcr_mpi_tool."

### 2. Run Locally
Internally, the platform runs the plugin using the command below. You can use the same command when modifying the code and testing directly in your terminal.

```bash
python3 run.py --input input.json --output output.json
```
---

## `input.json` Format

```json
{
  "model": "all",
  "seed": 42
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | X | `"all"` | Select model to run |
| `seed` | integer | X | `null` | Random seed for reproducibility |

### `model` Options

| Value | Description |
|-------|-------------|
| `"all"` | Run GAN → I2SB → RDDM sequentially |
| `"gan"` | Run GAN model only |
| `"i2sb"` | Run I2SB model only |
| `"rddm"` | Run RDDM model only |

### Input Examples

```json
// I2SB only, no seed
{ "model": "i2sb" }

// GAN only, fixed seed
{ "model": "gan", "seed": 123 }

// All models, fixed seed
{ "model": "all", "seed": 42 }
```

---

## `output.json` Format

```json
{
  "success_count": 3,
  "total": 3,
  "results": {
    "gan":  { "success": true,  "message": "Success" },
    "i2sb": { "success": true,  "message": "Success" },
    "rddm": { "success": true,  "message": "Success" }
  },
  "status": "success"
}
```

| Field | Description |
|-------|-------------|
| `success_count` | Number of successful models |
| `total` | Number of requested models |
| `results` | Success status and message per model |
| `status` | `"success"` or `"partial_failure"` |

---

### Generated Files
If inference is successful, the following files are created in the `outputs/{model_name}/results_{timestamp}` folder for each model:
- `.npy files` (as many as inputs): Precise reconstruction data arrays for each input
- `summary_grid.png` (1 image): An integrated summary image resized automatically to compare all reconstruction results at a glance

---

## Dependencies

```bash
pip install -r requirements.txt
```

Main libraries: `torch`, `torchvision`, `numpy`, `scipy`, `h5py`, `opencv-python`, `Pillow`, `matplotlib`, `einops`, `tqdm`

---

## Team Information

- Team: Legend_intern
- Challenge: Low Concentration Reconstruction Challenge in Magnetic Particle Imaging (LCR-MPI) 
- Models: GAN · I2SB · RDDM