from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model definition (ConvNeXt U-Net)
# ---------------------------------------------------------------------------

try:
    import timm
except Exception as e:
    raise ImportError("timm is required for ConvNeXt. Please `pip install timm`.") from e


class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_chans: int = 1, model_name: str = "convnext_nano", pretrained: bool = True):
        super().__init__()
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        feat_channels = self.feature_extractor.feature_info.channels()
        self.c1_ch, self.c2_ch, self.c3_ch, self.c4_ch = feat_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, self.c1_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.c1_ch),
            nn.ReLU(inplace=True),
        )
        self.out_channels = (self.c1_ch, self.c1_ch, self.c2_ch, self.c3_ch, self.c4_ch)

    def forward(self, x: torch.Tensor):
        c0 = self.stem(x)
        c1, c2, c3, c4 = self.feature_extractor(x)
        return c0, c1, c2, c3, c4

    def freeze_stages(self, stages: List[int], freeze_stem: bool = True):
        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad = False
        for name, param in self.feature_extractor.named_parameters():
            lname = name.lower()
            for s in stages:
                if f"stages.{s}" in lname or f"downsample_layers.{s}" in lname:
                    param.requires_grad = False
                    break

    def unfreeze_all(self):
        for p in self.stem.parameters():
            p.requires_grad = True
        for p in self.feature_extractor.parameters():
            p.requires_grad = True


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    def __init__(self, c0_ch, c1_ch, c2_ch, c3_ch, c4_ch, num_seg_classes=3):
        super().__init__()
        out0_ch = max(32, c0_ch // 2)
        self.up_43 = UpBlock(in_ch=c4_ch, skip_ch=c3_ch, out_ch=c3_ch)
        self.up_32 = UpBlock(in_ch=c3_ch, skip_ch=c2_ch, out_ch=c2_ch)
        self.up_21 = UpBlock(in_ch=c2_ch, skip_ch=c1_ch, out_ch=c1_ch)
        self.up_10 = UpBlock(in_ch=c1_ch, skip_ch=c0_ch, out_ch=out0_ch)
        self.final_head = nn.Conv2d(out0_ch, num_seg_classes, kernel_size=1)

    def forward(self, c0, c1, c2, c3, c4):
        d3 = self.up_43(c4, c3)
        d2 = self.up_32(d3, c2)
        d1 = self.up_21(d2, c1)
        d0 = self.up_10(d1, c0)
        return self.final_head(d0)


class MorphologicalFeatureExtractor(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.plaque_class = 1
        self.vessel_class = 2
        self.out_dim = 10

    def forward(self, seg_logits: torch.Tensor, seg_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if seg_probs is None:
            seg_probs = F.softmax(seg_logits, dim=1)

        batch_size = seg_logits.shape[0]
        device = seg_logits.device
        features_list: List[List[float]] = []

        seg_pred = torch.argmax(seg_logits, dim=1)
        plaque_mask = (seg_pred == self.plaque_class).float()
        vessel_mask = (seg_pred == self.vessel_class).float()
        foreground_mask = (seg_pred > 0).float()

        for b in range(batch_size):
            plaque_pixels = plaque_mask[b]
            vessel_pixels = vessel_mask[b]
            fg_pixels = foreground_mask[b]

            plaque_area = plaque_pixels.sum().item()
            vessel_area = vessel_pixels.sum().item()
            plaque_ratio = plaque_area / (vessel_area + 1e-6)

            fg_area = fg_pixels.sum().item()
            plaque_ratio_fg = plaque_area / (fg_area + 1e-6)

            plaque_conf = seg_probs[b, self.plaque_class] * plaque_mask[b]
            plaque_conf_weighted = plaque_conf.sum().item()

            if plaque_area > 0:
                plaque_coords = torch.nonzero(plaque_pixels, as_tuple=True)
                if len(plaque_coords[0]) > 0:
                    height_range = (plaque_coords[0].max() - plaque_coords[0].min()).float().item()
                    width_range = (plaque_coords[1].max() - plaque_coords[1].min()).float().item()
                    plaque_perimeter_approx = 2 * (height_range + width_range)
                else:
                    height_range = width_range = plaque_perimeter_approx = 0.0
            else:
                height_range = width_range = plaque_perimeter_approx = 0.0

            vessel_conf_weighted = (seg_probs[b, self.vessel_class] * vessel_mask[b]).sum().item()
            mean_plaque_conf = (
                (seg_probs[b, self.plaque_class] * plaque_mask[b]).sum().item() / (plaque_area + 1e-6)
                if plaque_area > 0 else 0.0
            )
            fg_ratio = fg_area / (seg_logits.shape[2] * seg_logits.shape[3])

            features_list.append([
                plaque_ratio,
                plaque_ratio_fg,
                plaque_conf_weighted,
                height_range / seg_logits.shape[2],
                width_range / seg_logits.shape[3],
                plaque_perimeter_approx / (seg_logits.shape[2] + seg_logits.shape[3]),
                vessel_conf_weighted,
                mean_plaque_conf,
                fg_ratio,
                float(plaque_area) / (seg_logits.shape[2] * seg_logits.shape[3]),
            ])

        return torch.tensor(features_list, dtype=torch.float32, device=device)


class ConvNeXtUNet(nn.Module):
    """ConvNeXt + U-Net (two-view segmentation) + bottleneck + morphological classification."""

    def __init__(
        self,
        in_chans: int = 1,
        num_seg_classes: int = 3,
        cls_class_num: int = 1,
        pretrained_encoder: bool = True,
        convnext_model: str = "convnext_nano",
        cls_hidden: int = 512,
        cls_dropout: float = 0.3,
        plaque_gate_alpha: float = 1.0,
        morph_proj_dim: int = 64,
    ):
        super().__init__()
        self.encoder = ConvNeXtEncoder(in_chans=in_chans, model_name=convnext_model, pretrained=pretrained_encoder)
        c0_ch, c1_ch, c2_ch, c3_ch, c4_ch = self.encoder.out_channels

        self.decoder_long = UNetDecoder(c0_ch, c1_ch, c2_ch, c3_ch, c4_ch, num_seg_classes=num_seg_classes)
        self.decoder_trans = UNetDecoder(c0_ch, c1_ch, c2_ch, c3_ch, c4_ch, num_seg_classes=num_seg_classes)

        self.morph_extractor = MorphologicalFeatureExtractor(num_classes=num_seg_classes)
        num_morph_features = self.morph_extractor.out_dim
        self.morph_proj = nn.Sequential(
            nn.LayerNorm(num_morph_features),
            nn.Linear(num_morph_features, morph_proj_dim),
            nn.ReLU(inplace=True),
        )

        bottleneck_dim = c4_ch
        cls_input_dim = bottleneck_dim * 2 + morph_proj_dim * 2
        self.cls_head = nn.Sequential(
            nn.Linear(cls_input_dim, cls_hidden),
            nn.LayerNorm(cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, cls_hidden // 2),
            nn.LayerNorm(cls_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden // 2, cls_class_num),
        )

        self.plaque_class = 1
        self.plaque_gate_alpha = plaque_gate_alpha

    def _mask_weighted_pool(self, feat_map: torch.Tensor, seg_logits: torch.Tensor) -> torch.Tensor:
        seg_probs = F.softmax(seg_logits.detach(), dim=1)
        p_plaque = seg_probs[:, self.plaque_class: self.plaque_class + 1]
        p_plaque = F.interpolate(p_plaque, size=feat_map.shape[2:], mode="bilinear", align_corners=False)
        weights = 1.0 + self.plaque_gate_alpha * p_plaque
        weighted = feat_map * weights
        sum_feat = weighted.sum(dim=(2, 3))
        sum_w = weights.sum(dim=(2, 3)).clamp_min(1e-6)
        return sum_feat / sum_w

    def forward(
        self,
        x_long: torch.Tensor,
        x_trans: Optional[torch.Tensor] = None,
        return_cls: bool = False,
        return_moe_aux: bool = False,
        router_temp: float = 1.0,
        router_noise_std: float = 0.0,
        moe_lb_lambda: float = 0.0,
    ):
        _ = (return_moe_aux, router_temp, router_noise_std, moe_lb_lambda)
        H, W = x_long.shape[2], x_long.shape[3]

        c0_l, c1_l, c2_l, c3_l, c4_l = self.encoder(x_long)
        seg_long = self.decoder_long(c0_l, c1_l, c2_l, c3_l, c4_l)
        seg_long = F.interpolate(seg_long, size=(H, W), mode="bilinear", align_corners=False)

        if x_trans is None:
            if return_cls:
                seg_long_d = seg_long.detach()
                probs_long = F.softmax(seg_long_d, dim=1)
                emb_long = self._mask_weighted_pool(c4_l, seg_long_d)
                feat_morph = self.morph_proj(self.morph_extractor(seg_long_d, probs_long))
                feat_cls = torch.cat([emb_long, emb_long, feat_morph, feat_morph], dim=1)
                return seg_long, self.cls_head(feat_cls)
            return seg_long

        c0_t, c1_t, c2_t, c3_t, c4_t = self.encoder(x_trans)
        seg_trans = self.decoder_trans(c0_t, c1_t, c2_t, c3_t, c4_t)
        seg_trans = F.interpolate(seg_trans, size=(H, W), mode="bilinear", align_corners=False)

        if return_cls:
            seg_long_d = seg_long.detach()
            seg_trans_d = seg_trans.detach()
            emb_long = self._mask_weighted_pool(c4_l, seg_long_d)
            emb_trans = self._mask_weighted_pool(c4_t, seg_trans_d)
            feat_morph_long = self.morph_proj(self.morph_extractor(seg_long_d, F.softmax(seg_long_d, dim=1)))
            feat_morph_trans = self.morph_proj(self.morph_extractor(seg_trans_d, F.softmax(seg_trans_d, dim=1)))
            feat_cls = torch.cat([emb_long, emb_trans, feat_morph_long, feat_morph_trans], dim=1)
            return seg_long, seg_trans, self.cls_head(feat_cls)

        return seg_long, seg_trans


class SubmissionModel(nn.Module):
    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core = core_model

    def forward(self, x_long: torch.Tensor, x_trans: torch.Tensor):
        outputs = self.core(x_long, x_trans, return_cls=True)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise RuntimeError("Model must return (seg_long, seg_trans, cls_logits).")
        return outputs


def _build_model() -> SubmissionModel:
    core = ConvNeXtUNet(
        in_chans=1,
        num_seg_classes=3,
        cls_class_num=1,
        pretrained_encoder=False,
        convnext_model="convnext_nano",
        cls_hidden=512,
        cls_dropout=0.3,
        plaque_gate_alpha=1.0,
        morph_proj_dim=64,
    )
    return SubmissionModel(core)


def _load_weights(model: nn.Module, weights_path: str, map_location="cpu") -> nn.Module:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights_path not found: {weights_path}")
    try:
        checkpoint = torch.load(weights_path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=map_location)
    except Exception:
        checkpoint = torch.load(weights_path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "net", "network"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    target = model.core if hasattr(model, "core") else model
    current = target.state_dict()
    filtered = {k: v for k, v in checkpoint.items() if k in current and hasattr(v, "shape") and v.shape == current[k].shape}
    target.load_state_dict(filtered, strict=False)
    return model


# ---------------------------------------------------------------------------
# Plugin helpers
# ---------------------------------------------------------------------------

_MASK_COLORS = {
    0: (0, 0, 0),        # background — black
    1: (220, 50, 50),    # plaque — red
    2: (50, 100, 220),   # vessel — blue
}

_DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pth"


def _mask_to_data_url(mask) -> str:
    import numpy as np
    from PIL import Image

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in _MASK_COLORS.items():
        rgb[mask == label] = color
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _resolve_source_path(payload: dict) -> str:
    if payload.get("source_path"):
        return payload["source_path"]

    artifacts = payload.get("analysis_artifacts") or {}
    meta_path = (artifacts.get("_meta") or {}).get("source_path")
    if meta_path and Path(meta_path).exists():
        return meta_path

    file_name = (
        (artifacts.get("_meta") or {}).get("file_name")
        or (artifacts.get("segmentation_masks") or {}).get("file_name")
        or (payload.get("analysis_source") or {}).get("file_name")
    )
    uploads_dir = Path(__file__).resolve().parents[2] / "runtime_uploads"
    if uploads_dir.exists():
        candidates = [
            p for p in uploads_dir.rglob("*")
            if p.suffix.lower() in {".h5", ".hdf5"}
            and (file_name is None or p.name == file_name)
        ]
        if candidates:
            return str(max(candidates, key=lambda p: p.stat().st_mtime))

    raise ValueError("source_path not found in payload. Re-upload the .h5 file to retry.")


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def _run(payload: dict) -> dict:
    import numpy as np
    import h5py

    source_path = _resolve_source_path(payload)
    model_path = payload.get("model_path") or os.environ.get("MODEL_PATH") or str(_DEFAULT_WEIGHTS)
    cls_threshold = float(payload.get("cls_threshold", 0.8))
    resize_target = int(payload.get("resize_target", 512))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _build_model()
    model = _load_weights(model, model_path, map_location=device)
    model.to(device).eval()

    with h5py.File(source_path, "r") as f:
        long_img = f["long_img"][()]
        trans_img = f["trans_img"][()]

    def preprocess(img):
        if img.ndim == 3:
            img = img[..., 0]
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        orig_shape = img.shape
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(resize_target, resize_target), mode="bilinear", align_corners=False)
        return t.to(device), orig_shape

    long_t, orig_shape_l = preprocess(long_img)
    trans_t, orig_shape_t = preprocess(trans_img)

    with torch.no_grad():
        seg_long, seg_trans, cls_logit = model(long_t, trans_t)

    def postprocess_seg(logits, orig_shape):
        logits = F.interpolate(logits, size=orig_shape, mode="bilinear", align_corners=False)
        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    long_mask = postprocess_seg(seg_long, orig_shape_l)
    trans_mask = postprocess_seg(seg_trans, orig_shape_t)
    cls_score = float(torch.sigmoid(cls_logit).item())
    cls_pred = int(cls_score >= cls_threshold)

    file_name = payload.get("file_name", Path(source_path).name)
    label = "High-risk (RADS 3-4)" if cls_pred else "Low-risk (RADS 2)"

    return {
        "source": {
            "file_name": file_name,
            "file_type": payload.get("suffix", "h5"),
            "modality": "medical-image",
            "size_bytes": Path(source_path).stat().st_size,
            "status": "parsed",
        },
        "grounded_summary": (
            f"Carotid plaque analysis completed for **{file_name}**. "
            f"Vulnerability classification: **{label}** "
            f"(probability {cls_score:.3f}). "
            "Segmentation masks are available in the Studio panel (0 = background, 1 = plaque, 2 = vessel)."
        ),
        "studio_cards": [
            {"id": "segmentation_masks", "title": "Segmentation Masks", "subtitle": f"{file_name} — longitudinal & transverse", "base_id": "segmentation_masks"},
            {"id": "classification", "title": "Vulnerability Classification", "subtitle": label, "base_id": "classification"},
        ],
        "artifacts": {
            "_meta": {"source_path": source_path, "file_name": file_name},
            "segmentation_masks": {
                "file_name": file_name,
                "classes": {"0": "background", "1": "plaque", "2": "vessel"},
                "longitudinal": {
                    "image_data_url": _mask_to_data_url(long_mask),
                    "shape": list(long_mask.shape),
                    "unique_labels": [int(v) for v in sorted(set(long_mask.flatten().tolist()))],
                },
                "transverse": {
                    "image_data_url": _mask_to_data_url(trans_mask),
                    "shape": list(trans_mask.shape),
                    "unique_labels": [int(v) for v in sorted(set(trans_mask.flatten().tolist()))],
                },
            },
            "classification": {
                "file_name": file_name,
                "cls": cls_pred,
                "label": label.lower(),
                "probability": round(cls_score, 4),
            },
        },
        "sources": [],
        "used_tools": ["carotid_plaque_analysis_tool"],
        "summary": f"Carotid plaque analysis completed for {file_name}. Classification: {label} (probability {cls_score:.3f}).",
        "provenance": {"tool_version": "1.0.0", "received_keys": sorted(payload.keys())},
    }


def execute(payload: dict) -> dict:
    return _run(payload)
