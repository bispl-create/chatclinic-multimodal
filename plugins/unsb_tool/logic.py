from __future__ import annotations

import base64
import importlib.util
import io
import os
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from app.models import ImageSourceResponse
from app.services.tool_runner import discover_tools
from plugins.image_review_tool.logic import analyze_image_source


ROOT_DIR = Path(__file__).resolve().parents[2]
UNSB_ROOT = Path(os.getenv("UNSB_ROOT", str(ROOT_DIR.parent / "UNSB"))).expanduser().resolve()
DEFAULT_CKPT_DIR = Path(os.getenv("UNSB_CKPT_DIR", str(UNSB_ROOT / "ckpt"))).expanduser().resolve()
DEFAULT_EPOCH = os.getenv("UNSB_DEFAULT_EPOCH", "iter_65000")
OUTPUT_ROOT = ROOT_DIR / "uploads" / "unsb"
THUMBNAIL_MAX_PX = 512


def _safe_int(value: object, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return default


def _safe_float(value: object, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return default


def _safe_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _safe_optional_int(value: object) -> int | None:
    if value in (None, "", "none", "null"):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return None


def _build_data_url(image_path: Path) -> str | None:
    try:
        with Image.open(image_path) as img:
            thumb = img.copy()
            thumb.thumbnail((THUMBNAIL_MAX_PX, THUMBNAIL_MAX_PX), Image.LANCZOS)
            if thumb.mode in ("RGBA", "LA", "PA"):
                pass
            elif thumb.mode not in ("RGB", "L"):
                thumb = thumb.convert("RGB")
            buf = io.BytesIO()
            thumb.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_unsb_module() -> Any:
    inference_path = UNSB_ROOT / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError(f"UNSB inference script not found: {inference_path}")
    if str(UNSB_ROOT) not in sys.path:
        sys.path.insert(0, str(UNSB_ROOT))
    spec = importlib.util.spec_from_file_location("chatclinic_unsb_inference", inference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load UNSB inference module from {inference_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("chatclinic_unsb_inference", module)
    spec.loader.exec_module(module)
    return module


def _resolve_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


@lru_cache(maxsize=8)
def _load_generator_bundle(ckpt_dir: str, epoch: str, gpu: int) -> dict[str, Any]:
    module = _load_unsb_module()
    args = SimpleNamespace(num_timesteps=5, tau=0.01, crop_size=256)
    opt = module.make_default_opt(args)
    device = _resolve_device(gpu)
    opt.gpu_ids = [gpu] if device.type == "cuda" else []
    net_g = module.load_generator(ckpt_dir, epoch, opt)
    net_g = net_g.to(device)
    net_g.eval()
    return {
        "module": module,
        "net_g": net_g,
        "device": device,
        "ngf": opt.ngf,
    }


def _safe_output_stem(path: Path, output_prefix: str | None = None) -> str:
    raw = (output_prefix or path.stem).strip()
    if not raw:
        raw = path.stem or "image"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def run_unsb_enhancement(
    image_path: str,
    *,
    ckpt_dir: str,
    epoch: str,
    num_timesteps: int,
    tau: float,
    crop_size: int,
    deterministic: bool,
    gpu: int,
    save_channel: int | None,
    output_prefix: str | None,
) -> dict[str, Any]:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image source not found: {path}")

    ckpt_path = Path(ckpt_dir).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"UNSB checkpoint directory not found: {ckpt_path}")

    bundle = _load_generator_bundle(str(ckpt_path), epoch, gpu)
    module = bundle["module"]
    net_g = bundle["net_g"]
    device = bundle["device"]
    ngf = int(bundle["ngf"])

    tensor = module.load_image(str(path), crop_size)
    enhanced = module.enhance_single(
        net_g,
        tensor.to(device),
        num_timesteps=num_timesteps,
        tau=tau,
        ngf=ngf,
        device=device,
        deterministic=deterministic,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_name = f"{_safe_output_stem(path, output_prefix)}_{uuid.uuid4().hex[:8]}_enhanced.png"
    output_path = OUTPUT_ROOT / output_name
    module.save_image(enhanced, str(output_path), save_channel=save_channel)

    with Image.open(output_path) as out_img:
        output_width, output_height = out_img.size

    return {
        "input_image_path": str(path),
        "output_image_path": str(output_path),
        "output_file_name": output_path.name,
        "preview_data_url": _build_data_url(output_path),
        "device": str(device),
        "ckpt_dir": str(ckpt_path),
        "epoch": epoch,
        "num_timesteps": num_timesteps,
        "tau": tau,
        "crop_size": crop_size,
        "deterministic": deterministic,
        "gpu": gpu,
        "save_channel": save_channel,
        "output_width": output_width,
        "output_height": output_height,
    }


def enrich_image_analysis(base_analysis: ImageSourceResponse, enhancement: dict[str, Any]) -> ImageSourceResponse:
    artifacts = dict(base_analysis.artifacts or {})
    artifacts["unsb_enhancement"] = enhancement

    warnings = list(base_analysis.warnings or [])
    if not enhancement.get("preview_data_url"):
        warning = "UNSB output preview generation failed."
        if warning not in warnings:
            warnings.append(warning)

    used_tools = list(base_analysis.used_tools or [])
    if "unsb_tool" not in used_tools:
        used_tools.append("unsb_tool")

    studio_cards = list(base_analysis.studio_cards or [])
    if not any(str(card.get("id") or "").strip() == "image_review" for card in studio_cards):
        studio_cards.append(
            {
                "id": "image_review",
                "title": "Image Review",
                "subtitle": "Metadata, EXIF, thumbnail, and UNSB enhancement preview",
            }
        )

    draft_answer = (
        f"UNSB enhancement is ready for `{base_analysis.file_name}`.\n\n"
        f"- Output: `{enhancement['output_image_path']}`\n"
        f"- Device: {enhancement['device']}\n"
        f"- Checkpoint: `{enhancement['epoch']}`\n"
        f"- Steps: {enhancement['num_timesteps']}\n"
        f"- Tau: {enhancement['tau']}\n\n"
        "The Image Review card now includes the enhanced preview."
    )

    return base_analysis.model_copy(
        update={
            "artifacts": artifacts,
            "warnings": warnings,
            "draft_answer": draft_answer,
            "used_tools": used_tools,
            "tool_registry": base_analysis.tool_registry or discover_tools(),
            "studio_cards": studio_cards,
            "source_type": "image",
            "result_kind": "image_analysis",
            "requested_view": "image_review",
            "studio": {"renderer": "image_review"},
        }
    )


def execute(payload: dict[str, object]) -> dict[str, object]:
    image_path = str(payload.get("image_path") or payload.get("source_image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")

    file_name = str(payload.get("file_name") or Path(image_path).name).strip() or Path(image_path).name
    analysis_payload = payload.get("analysis")

    if isinstance(analysis_payload, dict):
        base_analysis = ImageSourceResponse(**analysis_payload)
    else:
        base_analysis = analyze_image_source(image_path, file_name=file_name)
        base_analysis.tool_registry = discover_tools()

    enhancement = run_unsb_enhancement(
        image_path=image_path,
        ckpt_dir=str(payload.get("ckpt_dir") or DEFAULT_CKPT_DIR),
        epoch=str(payload.get("epoch") or DEFAULT_EPOCH),
        num_timesteps=_safe_int(payload.get("num_timesteps"), 5),
        tau=_safe_float(payload.get("tau"), 0.01),
        crop_size=_safe_int(payload.get("crop_size"), 256),
        deterministic=_safe_bool(payload.get("deterministic"), False),
        gpu=_safe_int(payload.get("gpu"), 0),
        save_channel=_safe_optional_int(payload.get("save_channel")),
        output_prefix=str(payload.get("output_prefix") or "").strip() or None,
    )
    updated_analysis = enrich_image_analysis(base_analysis, enhancement)

    return {
        "tool": "unsb_tool",
        "summary": updated_analysis.draft_answer,
        "analysis": updated_analysis.model_dump(),
        "unsb_enhancement": enhancement,
        "result_kind": "image_analysis",
        "requested_view": "image_review",
        "studio": {"renderer": "image_review"},
    }
