from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import io
import cv2
from pathlib import Path

from plugins.cxr_classification_tool.model.convnext_v2 import (
    ConvNeXtV2_newPCAM,
    ConvNeXt_CaiT_Hybrid,
    SwinMultiLabel,
)

def load_models(device):
    num_classes = 30

    repo_root = Path(__file__).resolve().parents[3]
    weights_dir = repo_root / "ckpt_and_file" / "cxr_classification_tool"

    pcam_path = weights_dir / "PCAM_AP.pth"
    cait_path = weights_dir / "CAIT_AP.pth"
    swin_path = weights_dir / "SWIN_AP.pth"

    model_pcam = ConvNeXtV2_newPCAM(pretrained=False)
    ckpt = torch.load(pcam_path, map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    missing, unexpected = model_pcam.load_state_dict(ckpt, strict=False)

    model_cait = ConvNeXt_CaiT_Hybrid(backbone_path=pcam_path)
    state = torch.load(cait_path, map_location=device)
    if "model" in state:
        state = state["model"]
    missing, unexpected = model_cait.load_state_dict(state, strict=False)

    model_swin = SwinMultiLabel(
        model_name="swinv2_tiny_window16_256.ms_in1k",
        num_classes=num_classes,
        pretrained=False,
    )
    ckpt = torch.load(swin_path, map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    missing, unexpected = model_swin.load_state_dict(ckpt, strict=False)

    model_pcam.to(device).eval()
    model_cait.to(device).eval()
    model_swin.to(device).eval()

    return model_pcam, model_cait, model_swin

def preprocess(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_array is None:
        raise ValueError(f"Image load failed: {image_path}")

    h, w = img_array.shape[:2]
    target_h, target_w = 512, 512

    if (w, h) != (target_w, target_h):
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        img_array = cv2.copyMakeBorder(
            img_array,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=0,
        )

    if img_array.dtype == np.uint16:
        img_array = img_array.astype(np.float32) / 65535.0
    else:
        img_array = img_array.astype(np.float32) / 255.0

    img_array = (img_array - 0.5) / (1 / 2048.0)

    image = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).float()

    return image

def infer(models, image, device):
    model_pcam, model_cait, model_swin = models

    image = image.to(device)
    image_flip = torch.flip(image, dims=[-1])

    with torch.no_grad():
        # PCAM
        p1 = model_pcam(image)
        if isinstance(p1, tuple): p1 = p1[0]
        p1_flip = model_pcam(image_flip)
        if isinstance(p1_flip, tuple): p1_flip = p1_flip[0]
        p1 = (p1 + p1_flip) / 2

        # CAIT
        p2 = model_cait(image)
        p2_flip = model_cait(image_flip)
        p2 = (p2 + p2_flip) / 2

        # SWIN
        p3 = model_swin(image)
        p3_flip = model_swin(image_flip)
        p3 = (p3 + p3_flip) / 2

        final = 0.4 * p1 + 0.2 * p2 + 0.4 * p3
        final = torch.sigmoid(final)

    return final.cpu().numpy().tolist()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    execution_context = payload.get("execution_context") or {}
    files = payload.get("files") or []
    if not files:
        raise ValueError("cxr_classification_tool requires one or more files")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = load_models(device)

    results = []

    for item in files:
        raw = base64.b64decode(item["raw_base64"])
        image = preprocess(raw)
        pred = infer(models, image, device)
        results.append(
            {
                "file_name": item.get("file_name"),
                "prediction": pred
            }
        )

    output = {
        "results": results,
        "used_tools": ["cxr_classification_tool"]
    }

    Path(args.output).write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
