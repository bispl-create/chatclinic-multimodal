"""
CXR Lung Segmentation inference module.

Uses TorchXRayVision's PSPNet (chestx_det) to produce lung segmentation masks
from chest X-ray images.
"""

import glob
import os
import time
import logging

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from skimage import io as skimage_io
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run_inference(cfg: DictConfig) -> dict:
    """Run CXR lung segmentation inference.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with keys described in the module-level docstring.

    Returns
    -------
    dict
        {"mask_path": str, "num_classes": int, "elapsed_sec": float}
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # Gather inputs: process every image in the cxr samples directory,
    # or fall back to the single cfg.paths.cxr_image_path.
    # ------------------------------------------------------------------
    input_path = cfg.paths.cxr_image_path
    samples_dir = os.path.dirname(input_path)
    input_paths = []
    if os.path.isdir(samples_dir):
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            input_paths.extend(glob.glob(os.path.join(samples_dir, ext)))
    input_paths = sorted(set(input_paths))
    if not input_paths:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(
                f"No CXR images found. Checked dir '{samples_dir}' and "
                f"file '{input_path}'."
            )
        input_paths = [input_path]

    log.info("Found %d CXR image(s) to process.", len(input_paths))

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    device = torch.device(cfg.device_name)

    # ------------------------------------------------------------------
    # Load model (once)
    # ------------------------------------------------------------------
    log.info("Loading PSPNet model (chestx_det) ...")
    cxr_cache = os.path.join(cfg.paths.weights_dir, "cxr")
    os.makedirs(cxr_cache, exist_ok=True)
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=cxr_cache)
    model = model.to(device)
    model.eval()

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(cfg.tool.input_size),
    ])
    lung_indices = list(cfg.tool.lung_indices)

    mask_paths = []
    last = {"mask_path": "", "image_path": ""}

    for idx, path in enumerate(input_paths, start=1):
        log.info("[%d/%d] Processing %s", idx, len(input_paths), path)

        img = skimage_io.imread(path)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.mean(2)
        img = img[None, ...]
        img = transform(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)

        lung_masks = output[0, lung_indices, :, :]
        combined_mask = (lung_masks.max(dim=0).values > 0.5).cpu().numpy().astype(np.uint8)
        combined_mask *= 255

        base = os.path.splitext(os.path.basename(path))[0]
        mask_path = os.path.join(cfg.paths.output_dir, f"{base}_lung_mask.png")
        skimage_io.imsave(mask_path, combined_mask, check_contrast=False)
        log.info("  Saved lung mask to: %s", mask_path)
        mask_paths.append(mask_path)

        if cfg.get("save_output", False):
            full_pred = (output[0].cpu().numpy() > 0.5).astype(np.uint8)
            np.save(
                os.path.join(cfg.paths.output_dir, f"{base}_full_prediction.npy"),
                full_pred,
            )

        last = {"mask_path": mask_path, "image_path": path}

    elapsed = time.time() - start_time
    log.info(
        "Inference completed for %d image(s) in %.2f seconds.",
        len(input_paths),
        elapsed,
    )

    return {
        "mask_path": last["mask_path"],
        "mask_paths": mask_paths,
        "num_samples": len(input_paths),
        "num_classes": int(cfg.tool.num_classes),
        "elapsed_sec": round(elapsed, 4),
        "image_path": last["image_path"],
        "label_path": "",
    }
