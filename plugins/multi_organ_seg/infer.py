"""nnU-Net v2 inference wrapper for AMOS CT multi-organ segmentation.

Runs a trained nnU-Net model (default: ``Dataset052_AMOS22_OnlyCT`` /
``MaskSAM_AMOS`` / ``3d_fullres`` / fold 2) on a single CT volume.

The nnUNet library is vendored under ``tools/multi_organ_seg/vendor/`` and
inserted into ``sys.path`` at import time so no global install is needed.
Required env vars (``nnUNet_raw``, ``nnUNet_preprocessed``, ``nnUNet_results``)
are populated from ``cfg.paths`` before nnUNet is imported.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import time

from omegaconf import DictConfig

log = logging.getLogger(__name__)

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)


def _pick_inputs(cfg: DictConfig) -> list:
    """Return a list of input volume paths for inference."""
    return [cfg.tool.input_image]


def run_inference(cfg: DictConfig) -> dict:
    start = time.time()

    # nnUNet reads these from the environment at import / predict time.
    os.environ["nnUNet_raw"] = os.path.abspath(cfg.paths.nnunet_raw)
    os.environ["nnUNet_preprocessed"] = os.path.abspath(cfg.paths.nnunet_preprocessed)
    os.environ["nnUNet_results"] = os.path.abspath(cfg.paths.nnunet_results)
    # Vendored SAM model reads this to locate sam_vit_h_4b8939.pth
    os.environ["SAM_CHECKPOINT"] = os.path.abspath(cfg.paths.sam_checkpoint)
    if not os.path.isfile(os.environ["SAM_CHECKPOINT"]):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {os.environ['SAM_CHECKPOINT']}. "
            "Place sam_vit_h_4b8939.pth there or override paths.sam_checkpoint."
        )

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_dir = os.path.join(
        cfg.paths.nnunet_results,
        cfg.tool.dataset_name,
        f"{cfg.tool.trainer}__{cfg.tool.plans}__{cfg.tool.configuration}",
    )
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Trained nnUNet model folder not found: {model_dir}. "
            "Ensure nnunet_results/<dataset>/<trainer>__<plans>__<cfg>/ exists."
        )

    fold = int(cfg.tool.fold)
    checkpoint = cfg.tool.get("checkpoint_name", "checkpoint_final.pth")

    input_paths = _pick_inputs(cfg)
    log.info("Found %d input volume(s).", len(input_paths))

    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    device = torch.device(cfg.device_name if torch.cuda.is_available() else "cpu")
    log.info("Loading nnUNet model from: %s (fold=%d)", model_dir, fold)

    predictor = nnUNetPredictor(
        tile_step_size=float(cfg.tool.get("tile_step_size", 0.5)),
        use_gaussian=True,
        use_mirroring=bool(cfg.tool.get("use_mirroring", True)),
        perform_everything_on_device=bool(cfg.tool.get("perform_everything_on_device", False)),
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(fold,),
        checkpoint_name=checkpoint,
    )

    # nnUNet expects list[list[str]] -- one inner list per case with all modality files.
    log.info("Running nnUNet inference on %d case(s) ...", len(input_paths))
    predictor.predict_from_files(
        [[p] for p in input_paths],
        cfg.paths.output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=2,
    )

    mask_paths = []
    for p in input_paths:
        base = os.path.basename(p)
        for suffix in ("_0000.nii.gz", ".nii.gz"):
            if base.endswith(suffix):
                case_id = base[: -len(suffix)]
                break
        else:
            case_id = os.path.splitext(os.path.splitext(base)[0])[0]
        mp = os.path.join(cfg.paths.output_dir, f"{case_id}.nii.gz")
        if os.path.isfile(mp):
            mask_paths.append(mp)

    if not mask_paths:
        mask_paths = sorted(glob.glob(os.path.join(cfg.paths.output_dir, "*.nii.gz")))

    last_mask = mask_paths[-1] if mask_paths else ""
    last_input = input_paths[-1] if input_paths else ""

    elapsed = time.time() - start
    log.info(
        "nnUNet inference done for %d case(s) in %.1fs -> last: %s",
        len(input_paths),
        elapsed,
        last_mask,
    )

    return {
        "mask_path": last_mask,
        "mask_paths": mask_paths,
        "num_samples": len(input_paths),
        "num_classes": int(cfg.tool.num_classes),
        "elapsed_sec": round(elapsed, 4),
        "image_path": last_input,
        "label_path": "",
    }
