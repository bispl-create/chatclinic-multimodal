from __future__ import annotations

import torch
from plugins.cxr_classification_tool.model.run import load_models, preprocess, infer


LABEL_NAMES = [
    "Normal",
    "Aortic Elongation",
    "Cardiomegaly",
    "Pleural Effusion",
    "Nodule",
    "Atelectasis",
    "Pleural Thickening",
    "Aortic Atheromatosis",
    "Support Devices",
    "Alveolar Pattern",
    "Fracture",
    "Hernia",
    "Emphysema",
    "Azygos Lobe",
    "Hydropneumothorax",
    "Kyphosis",
    "Mass",
    "Pneumothorax",
    "Subcutaneous Emphysema",
    "Pneumoperitoneo",
    "Vascular Hilar Enlargement",
    "Vertebral Degenerative Changes",
    "Hyperinflated Lung",
    "Interstitial Pattern",
    "Central Venous Catheter",
    "Hypoexpansion",
    "Bronchiectasis",
    "Hemidiaphragm Elevation",
    "Sternotomy",
    "Calcified Densities",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = load_models(DEVICE)
print("[CXR] Models loaded.")


def run(payload: dict) -> dict:
    image_path = payload.get("image_path")

    if not image_path:
        return {
            "tool": "cxr_classification_tool",
            "summary": "No image path provided.",
            "labels": [],
            "artifacts": {},
            "error": "Missing image_path",
        }

    try:
        image = preprocess(image_path)
        pred = infer(MODELS, image, DEVICE)[0]

        results = [
            {
                "name": name,
                "probability": float(prob),
            }
            for name, prob in zip(LABEL_NAMES, pred)
        ]

        results = sorted(results, key=lambda x: x["probability"], reverse=True)

        return {
            "tool": "cxr_classification_tool",
            "summary": "CXR-LT multi-label classification completed.",
            "prediction_type": "multi_label_classification",
            "num_classes": len(LABEL_NAMES),
            "labels": results,
            "artifacts": {},
        }

    except Exception as e:
        return {
            "tool": "cxr_classification_tool",
            "summary": f"CXR classification failed: {str(e)}",
            "labels": [],
            "artifacts": {},
            "error": str(e),
        }


def execute(payload: dict) -> dict:
    return run(payload)