from __future__ import annotations

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
import clip
from pathlib import Path

from plugins.cxr_zeroshot_tool.model.zero_shot_inference import load_clip


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "ckpt_and_file" / "cxr_zeroshot_tool" / "checkpoint_ema_epoch_4.pt"

UNSEEN_PROMPTS = {
    "Scoliosis": {
        "positive": [
            "Findings suggesting Scoliosis, lateral curvature of the thoracic spine, Abnormal curvature of the spine on chest radiograph",
            "Findings suggesting Scoliosis",
        ],
        "negative": ["No findings suggesting Scoliosis"],
    },
    "Osteopenia": {
        "positive": [
            "Findings suggesting Osteopenia, diffuse decreased bone density",
            "Findings suggesting Osteopenia",
        ],
        "negative": [
            "No findings suggesting Osteopenia",
            "Normal bone density with sharp cortical margins",
            "Well-mineralized ribs and clavicles",
        ],
    },
    "Bulla": {
        "positive": [
            "Findings suggesting Bulla, thin-walled air-filled space and focal hyperlucency",
            "Findings suggesting Bulla",
        ],
        "negative": [
            "No findings suggesting Bulla",
            "Normal lung markings without focal lucency",
        ],
    },
    "Infarction": {
        "positive": [
            "Findings suggesting Infarction, wedge-shaped pleural-based opacity",
            "Findings suggesting Infarction",
        ],
        "negative": [
            "No findings suggesting Infarction",
            "Diffuse airspace consolidation",
            "Lobar homogeneous opacity",
        ],
    },
    "Adenopathy": {
        "positive": [
            "Findings suggesting Adenopathy, enlargement of hilar or mediastinal lymph nodes",
            "Findings suggesting Adenopathy",
        ],
        "negative": ["No findings suggesting Adenopathy"],
    },
    "goiter": {
        "positive": [
            "Findings suggesting goiter, enlarged thyroid, widened superior mediastinum",
        ],
        "negative": ["No findings suggesting goiter"],
    },
}


MODEL = load_clip(
    model_path=MODEL_PATH,
    pretrained=True,
    context_length=77,
    device=DEVICE,
)
print("[CXR-ZEROSHOT] Model loaded.")


TRANSFORM = Compose([
    Normalize(
        (101.48761, 101.48761, 101.48761),
        (83.43944, 83.43944, 83.43944),
    ),
    Resize(224, interpolation=InterpolationMode.BICUBIC),
])


def preprocess_single_image(image_path: str) -> torch.Tensor:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Task2 inference 방식 재현
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
    else:
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)

    img = torch.from_numpy(img).float()

    img = TRANSFORM(img)
    img = img.unsqueeze(0)

    return img

def compute_text_weights(model, prompts_dict, context_length=77):
    class_weights = {}

    with torch.no_grad():
        for label, texts in prompts_dict.items():
            pos_tokens = clip.tokenize(
                texts["positive"],
                context_length=context_length,
            ).to(DEVICE)

            neg_tokens = clip.tokenize(
                texts["negative"],
                context_length=context_length,
            ).to(DEVICE)

            pos_feats = model.encode_text(pos_tokens)
            pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
            pos_proto = pos_feats.mean(dim=0)
            pos_proto = pos_proto / pos_proto.norm()

            neg_feats = model.encode_text(neg_tokens)
            neg_feats = neg_feats / neg_feats.norm(dim=-1, keepdim=True)
            neg_proto = neg_feats.mean(dim=0)
            neg_proto = neg_proto / neg_proto.norm()

            class_weights[label] = torch.stack([pos_proto, neg_proto])

    return class_weights


TEXT_WEIGHTS = compute_text_weights(MODEL, UNSEEN_PROMPTS)


def infer_single(model, image_tensor):
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        results = []

        for label, weights in TEXT_WEIGHTS.items():
            logits = image_features @ weights.t()
            probs = F.softmax(logits, dim=-1)
            pos_prob = float(probs[0, 0].cpu())

            results.append({
                "name": label,
                "probability": pos_prob,
            })

    return sorted(results, key=lambda x: x["probability"], reverse=True)


def run(payload: dict) -> dict:
    image_path = payload.get("image_path")

    if not image_path:
        return {
            "tool": "cxr_zeroshot_tool",
            "summary": "No image path provided.",
            "labels": [],
            "artifacts": {},
            "error": "Missing image_path",
        }

    try:
        image = preprocess_single_image(image_path)
        results = infer_single(MODEL, image)

        return {
            "tool": "cxr_zeroshot_tool",
            "summary": "CXR-LT zero-shot classification completed.",
            "prediction_type": "zero_shot_classification",
            "num_classes": len(UNSEEN_PROMPTS),
            "labels": results,
            "artifacts": {},
        }

    except Exception as e:
        return {
            "tool": "cxr_zeroshot_tool",
            "summary": f"CXR zero-shot classification failed: {str(e)}",
            "labels": [],
            "artifacts": {},
            "error": str(e),
        }


def execute(payload: dict) -> dict:
    return run(payload)
