"""
Brain Tumor Segmentation inference module.

Uses MONAI's SegResNet (brats_mri_segmentation bundle) to produce
multi-class tumor segmentation masks from 4-channel BraTS MRI volumes.

BraTS label mapping (handled by ConvertToMultiChannelBasedOnBratsClassesd):
    ET (Enhancing Tumor)  : original label 4
    TC (Tumor Core)       : original labels 1 + 4
    WT (Whole Tumor)      : original labels 1 + 2 + 4
"""

import os
import glob
import time
import logging

import torch
import numpy as np
import nibabel as nib
from omegaconf import DictConfig

from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    NormalizeIntensityd,
)

log = logging.getLogger(__name__)


def _find_sample_data(msd_data_dir: str, dataset_task: str) -> list:
    """Locate all BraTS subjects and return a list of sample dicts.

    Handles two common layouts:
    1. Single 4-D file per subject  (e.g., BRATS_001.nii.gz)
    2. Separate per-modality files  (e.g., BRATS_001_t1.nii.gz, ..._flair.nii.gz)

    Returns
    -------
    list of dict
        [{"image": str or list[str], "label": str, "multi_file": bool}, ...]
    """
    images_dir = os.path.join(msd_data_dir, dataset_task, "images")
    labels_dir = os.path.join(msd_data_dir, dataset_task, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Image directory not found: {images_dir}\n"
            "Please download the MSD BraTS dataset first.\n"
            "  python setup_bundles.py          (to download bundle + data)\n"
            "  Expected layout: {msd_data_dir}/{dataset_task}/images/"
        )

    nifti_files = sorted(
        glob.glob(os.path.join(images_dir, "*.nii.gz"))
        + glob.glob(os.path.join(images_dir, "*.nii"))
    )
    if not nifti_files:
        raise FileNotFoundError(
            f"No NIfTI files found in {images_dir}. "
            "Please verify the MSD BraTS data was downloaded correctly."
        )

    modality_suffixes = ["_t1.", "_t1ce.", "_t2.", "_flair."]
    first_base = os.path.basename(nifti_files[0]).lower()
    is_multi_file = any(sfx in first_base for sfx in modality_suffixes)

    samples = []
    if is_multi_file:
        # Group files by subject prefix
        seen_prefixes = set()
        for f in nifti_files:
            base = os.path.basename(f)
            base_lower = base.lower()
            prefix = None
            for sfx in modality_suffixes:
                idx = base_lower.find(sfx)
                if idx != -1:
                    prefix = base[:idx]
                    break
            if prefix is None or prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)

            modality_paths = []
            ok = True
            for mod in modality_suffixes:
                matches = glob.glob(os.path.join(images_dir, f"{prefix}{mod}*"))
                if not matches:
                    matches = glob.glob(os.path.join(images_dir, f"{prefix.lower()}{mod}*"))
                if not matches:
                    log.warning("Subject %s missing modality %s — skipping.", prefix, mod)
                    ok = False
                    break
                modality_paths.append(sorted(matches)[0])
            if not ok:
                continue

            label_path = _find_label(labels_dir, prefix)
            samples.append({"image": modality_paths, "label": label_path, "multi_file": True})
    else:
        for f in nifti_files:
            subject_id = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
            label_path = _find_label(labels_dir, subject_id)
            samples.append({"image": f, "label": label_path, "multi_file": False})

    return samples


def _find_label(labels_dir: str, subject_id: str) -> str:
    """Find the label file for a given subject ID."""
    if not os.path.isdir(labels_dir):
        log.warning("Label directory not found: %s — skipping label loading.", labels_dir)
        return ""

    candidates = (
        glob.glob(os.path.join(labels_dir, f"{subject_id}.nii.gz"))
        + glob.glob(os.path.join(labels_dir, f"{subject_id}.nii"))
        + glob.glob(os.path.join(labels_dir, f"{subject_id}*"))
    )
    if candidates:
        return sorted(candidates)[0]

    log.warning("No label file found for subject '%s' in %s.", subject_id, labels_dir)
    return ""


def _load_weights(model: torch.nn.Module, bundle_dir: str, bundle_name: str) -> None:
    """Load pretrained weights from the downloaded bundle."""
    weights_path = os.path.join(bundle_dir, bundle_name, "models", "brain_tumor_seg.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Please download the MONAI bundle first:\n"
            "  python setup_bundles.py"
        )
    log.info("Loading model weights from: %s", weights_path)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)


def run_inference(cfg: DictConfig) -> dict:
    """Run Brain Tumor Segmentation inference.

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

    # cuDNN's autotuner (benchmark=True) selects a conv3d kernel that crashes
    # on RTX 40-series for SegResNet's 224x224x144 shape. Force deterministic.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    device = torch.device(cfg.device_name)

    # ------------------------------------------------------------------
    # 1. Build model
    # ------------------------------------------------------------------
    log.info("Creating SegResNet model ...")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=cfg.tool.input_channels,   # 4
        out_channels=cfg.tool.num_classes,      # 3
        dropout_prob=0.2,
    )

    # ------------------------------------------------------------------
    # 2. Load pretrained weights
    # ------------------------------------------------------------------
    _load_weights(model, cfg.paths.bundle_dir, cfg.tool.bundle_name)
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Find input data (uploaded file or all MSD samples)
    # ------------------------------------------------------------------
    input_image = cfg.paths.get("input_image", None)
    if input_image and os.path.isfile(input_image):
        log.info("Using uploaded image: %s", input_image)
        samples = [{"image": input_image, "label": "", "multi_file": False}]
    else:
        samples = _find_sample_data(cfg.paths.msd_data_dir, cfg.tool.dataset_task)
        log.info("Found %d MSD subject(s) to process.", len(samples))

    # ------------------------------------------------------------------
    # 4. Shared inferer
    # ------------------------------------------------------------------
    roi_size = list(cfg.tool.roi_size)
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=cfg.tool.sw_batch_size,
        overlap=cfg.tool.overlap,
    )

    mask_paths = []
    last = {"mask_path": "", "image_path": "", "label_path": ""}

    for s_idx, sample in enumerate(samples, start=1):
        log.info("[%d/%d] Processing %s", s_idx, len(samples), sample["image"])

        if sample["multi_file"]:
            log.info("  Loading %d modality files and stacking ...", len(sample["image"]))
            volumes = []
            affine = None
            for mod_path in sample["image"]:
                nii = nib.load(mod_path)
                if affine is None:
                    affine = nii.affine
                volumes.append(nii.get_fdata(dtype=np.float32))
            stacked = np.stack(volumes, axis=0)
            img_tensor = torch.from_numpy(stacked).unsqueeze(0)

            for ch in range(img_tensor.shape[1]):
                ch_data = img_tensor[0, ch]
                nonzero_mask = ch_data != 0
                if nonzero_mask.any():
                    mean = ch_data[nonzero_mask].mean()
                    std = ch_data[nonzero_mask].std()
                    if std > 0:
                        img_tensor[0, ch] = torch.where(
                            nonzero_mask,
                            (ch_data - mean) / std,
                            ch_data,
                        )
        else:
            data_dict = {"image": sample["image"]}
            has_label = sample["label"] and os.path.isfile(sample["label"])
            if has_label:
                data_dict["label"] = sample["label"]
                transforms = Compose([
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ])
            else:
                transforms = Compose([
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ])

            data_dict = transforms(data_dict)
            img_tensor = data_dict["image"].unsqueeze(0)
            affine = (
                data_dict["image"].meta.get("affine", np.eye(4))
                if hasattr(data_dict["image"], "meta")
                else np.eye(4)
            )

        log.info("  Input tensor shape: %s", list(img_tensor.shape))
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = inferer(img_tensor, model)

        output_sigmoid = torch.sigmoid(output)
        seg_mask = (output_sigmoid > 0.5).float()
        seg_np = seg_mask[0].cpu().numpy().astype(np.uint8)
        log.info("  Segmentation mask shape: %s", list(seg_np.shape))
        for i, target in enumerate(cfg.tool.targets):
            voxel_count = int(seg_np[i].sum())
            log.info("    %s: %d positive voxels", target, voxel_count)

        if affine is None:
            affine = np.eye(4)
        if isinstance(affine, torch.Tensor):
            affine = affine.cpu().numpy()
        affine = np.array(affine, dtype=np.float64)

        seg_nifti_data = np.transpose(seg_np, (1, 2, 3, 0))

        # Derive a per-sample basename
        if sample["multi_file"]:
            first_mod = os.path.basename(sample["image"][0])
            base_no_ext = os.path.splitext(os.path.splitext(first_mod)[0])[0]
            for sfx in ("_t1", "_t1ce", "_t2", "_flair"):
                if base_no_ext.lower().endswith(sfx):
                    base_no_ext = base_no_ext[: -len(sfx)]
                    break
            sample_basename = base_no_ext
        else:
            sample_basename = os.path.splitext(
                os.path.splitext(os.path.basename(sample["image"]))[0]
            )[0]

        mask_path = os.path.join(
            cfg.paths.output_dir, f"{sample_basename}_brain_tumor_seg.nii.gz"
        )
        nib.save(nib.Nifti1Image(seg_nifti_data, affine), mask_path)
        log.info("  Saved 3-channel segmentation mask to: %s", mask_path)
        mask_paths.append(mask_path)

        last = {
            "mask_path": mask_path,
            "image_path": sample["image"],
            "label_path": sample.get("label", ""),
        }

    elapsed = time.time() - start_time
    log.info(
        "Inference completed for %d sample(s) in %.2f seconds.",
        len(samples),
        elapsed,
    )

    return {
        "mask_path": last["mask_path"],
        "mask_paths": mask_paths,
        "num_samples": len(samples),
        "num_classes": int(cfg.tool.num_classes),
        "elapsed_sec": round(elapsed, 4),
        "image_path": last["image_path"],
        "label_path": last["label_path"],
    }
