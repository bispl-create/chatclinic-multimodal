from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from app.models import MsVlmCtReportRequest, MsVlmCtReportResponse


TOOL_NAME = "ms_vlm_ct_report_tool"
PLUGIN_DIR = Path(__file__).resolve().parent
MODEL_DIR = PLUGIN_DIR / "model"
BUNDLED_MODEL_PACKAGE_DIR = MODEL_DIR / "vllama"
BUNDLED_CHATMEDI_PACKAGE_DIR = MODEL_DIR / "chatmedi_img"
DEFAULT_CFG_PATH = MODEL_DIR / "eval_configs" / "brainct_stage2.yaml"
DEFAULT_PROMPT_PATH = MODEL_DIR / "prompts" / "alignment_brain.txt"
WEIGHTS_DIR = MODEL_DIR / "weights"
DEFAULT_REPORT_CHECKPOINT = WEIGHTS_DIR / "brainct_ckpt_epoch6.pth"
DEFAULT_VISION_CHECKPOINT = WEIGHTS_DIR / "checkpoint0009.pth"
ARCH_ALIASES = {
    "ms_vlm_stage2_opt_mt": "vllama_stage2_opt_plusvqa",
    "vllama_stage2_opt_mt": "vllama_stage2_opt_plusvqa",
}
DEFAULT_BRAIN_CT_PROMPTS = [
    "<Img><ImageHere></Img> This is a brain CT scan showing abnormalities. Describe all the abnormalities that are depicted in the brain CT scan.",
    "<Img><ImageHere></Img> Diagnose this brain CT scan with any abnormalities.",
    "<Img><ImageHere></Img> Provide a detailed description of this brain CT scan.",
    "<Img><ImageHere></Img> Could you describe what this brain CT scan shows?",
    "<Img><ImageHere></Img> Describe any abnormalities shown in this brain CT scan.",
]


_MODEL_CACHE: dict[tuple[str, str | None, tuple[str, ...], int, str | None], tuple[Any, Any, list[str]]] = {}
_MODEL_LOCK = threading.Lock()


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return None


def _bool_env(*names: str) -> bool | None:
    value = _env_value(*names)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cfg_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _none_if_string_none(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ensure_inside_weights_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    weights_root = WEIGHTS_DIR.resolve()
    if resolved != weights_root and weights_root not in resolved.parents:
        raise ValueError(f"{label} must be inside {weights_root}; got {resolved}.")
    return resolved


def _resolve_path(value: str | None, *, base: Path | None = None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = base or Path.cwd()
    return (root / path).resolve()


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict))
    ):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _ensure_model_runtime_importable():
    if not BUNDLED_CHATMEDI_PACKAGE_DIR.exists():
        raise FileNotFoundError(
            f"Bundled CT report runtime not found: {BUNDLED_CHATMEDI_PACKAGE_DIR}. "
            "The plugin must include its runtime code under model/chatmedi_img."
        )
    model_root = MODEL_DIR.resolve()
    model_root_text = str(model_root)
    if model_root_text in sys.path:
        sys.path.remove(model_root_text)
    sys.path.insert(0, model_root_text)

    existing = sys.modules.get("chatmedi_img")
    existing_file = Path(str(getattr(existing, "__file__", "") or "")).expanduser()
    if existing is not None and (
        not existing_file
        or not _is_relative_to(existing_file.resolve(), BUNDLED_CHATMEDI_PACKAGE_DIR.resolve())
    ):
        for module_name in list(sys.modules):
            if module_name == "chatmedi_img" or module_name.startswith("chatmedi_img."):
                del sys.modules[module_name]

    from chatmedi_img.vol_model import vllamastage2

    return vllamastage2


def _load_model_config(
    cfg_path: Path,
    *,
    checkpoint_path: Path | None,
    vit_path: Path | None,
    gpu_id: int,
) -> tuple[Any, Any, str, list[str], list[str]]:
    from omegaconf import OmegaConf

    model_cls = _ensure_model_runtime_importable()
    cfg = OmegaConf.load(str(cfg_path))
    model_section = cfg.get("model")
    if model_section is None:
        raise ValueError(f"Missing `model` section in MS-VLM config: {cfg_path}")

    warnings: list[str] = []
    requested_arch = str(_env_value("MS_VLM_CT_ARCH") or model_section.get("arch") or "").strip()
    if requested_arch and requested_arch not in ARCH_ALIASES and requested_arch != "vllamastage2":
        warnings.append(f"Using bundled `chatmedi_img.vllamastage2` runtime; config requested `{requested_arch}`.")

    model_cfg = OmegaConf.create(OmegaConf.to_container(model_section, resolve=True))
    model_cfg.arch = "vllamastage2"
    model_cfg.device_8bit = int(gpu_id)
    model_cfg.evaluate = True

    low_resource = _bool_env("MS_VLM_CT_LOW_RESOURCE")
    if low_resource is not None:
        model_cfg.low_resource = low_resource

    env_checkpoint = _resolve_path(_env_value("MS_VLM_CT_CHECKPOINT"), base=WEIGHTS_DIR)
    configured_checkpoint_path = checkpoint_path or env_checkpoint or DEFAULT_REPORT_CHECKPOINT
    configured_checkpoint_path = _ensure_inside_weights_dir(configured_checkpoint_path, "MS-VLM report checkpoint")
    model_cfg.ckpt = str(configured_checkpoint_path)

    env_vit_path = _resolve_path(_env_value("MS_VLM_CT_VIT_PATH"), base=WEIGHTS_DIR)
    configured_vit_path = vit_path or env_vit_path or DEFAULT_VISION_CHECKPOINT
    configured_vit_path = _ensure_inside_weights_dir(configured_vit_path, "MS-VLM vision checkpoint")
    model_cfg.vit_path = [str(configured_vit_path)]

    prompt_path = _resolve_path(str(model_cfg.get("prompt_path") or ""), base=MODEL_DIR)
    if prompt_path is None or not prompt_path.exists():
        prompt_path = DEFAULT_PROMPT_PATH
    if prompt_path.exists():
        model_cfg.prompt_path = str(prompt_path)

    configured_checkpoint = str(model_cfg.get("ckpt") or "").strip()
    if configured_checkpoint and not Path(configured_checkpoint).expanduser().exists():
        raise FileNotFoundError(
            f"MS-VLM checkpoint does not exist: {configured_checkpoint}. "
            f"Place the report checkpoint under {WEIGHTS_DIR}."
        )

    configured_vit_paths = []
    for item in _string_list(model_cfg.get("vit_path")):
        path = _resolve_path(item, base=MODEL_DIR)
        if path is not None:
            path = _ensure_inside_weights_dir(path, "MS-VLM vision checkpoint")
            configured_vit_paths.append(str(path))
    if configured_vit_paths:
        model_cfg.vit_path = configured_vit_paths
    for item in configured_vit_paths:
        if not Path(item).exists():
            raise FileNotFoundError(
                f"MS-VLM vision checkpoint does not exist: {item}. "
                f"Place the vision checkpoint under {WEIGHTS_DIR}."
            )

    return model_cls, model_cfg, str(configured_checkpoint or ""), configured_vit_paths, warnings


def _load_checkpoint_state(model: Any, checkpoint_path: str, warnings: list[str]) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    try:
        model.load_state_dict(state, strict=False)
        return
    except RuntimeError as exc:
        current_state = model.state_dict()
        filtered_state = {}
        skipped: list[str] = []
        for key, tensor in state.items():
            current_tensor = current_state.get(key)
            if current_tensor is None or getattr(current_tensor, "shape", None) != getattr(tensor, "shape", None):
                skipped.append(key)
                continue
            filtered_state[key] = tensor
        if not filtered_state:
            raise exc
        model.load_state_dict(filtered_state, strict=False)
        preview = ", ".join(skipped[:5])
        suffix = "..." if len(skipped) > 5 else ""
        warnings.append(f"Skipped {len(skipped)} checkpoint tensor(s) with incompatible shapes: {preview}{suffix}")


def _build_model(model_cls: Any, model_cfg: Any, checkpoint: str, vit_paths: list[str], gpu_id: int, warnings: list[str]) -> Any:
    vit_path = vit_paths[0] if vit_paths else None
    lora_targets = _string_list(model_cfg.get("lora_target_modules")) or ["q_proj", "k_proj", "v_proj", "o_proj"]
    heads_to_use = [int(item) for item in (_string_list(model_cfg.get("heads_to_use")) or ["1", "4"])]

    model = model_cls(
        vit_model=str(model_cfg.get("vit_model") or "vit_base"),
        freeze_vit=_cfg_bool(model_cfg.get("freeze_vit"), True),
        freeze_zformer=_cfg_bool(model_cfg.get("freeze_zformer"), True),
        freeze_perceiver=_cfg_bool(model_cfg.get("freeze_perceiver"), False),
        num_query_token=int(model_cfg.get("num_query_token") or 32),
        llama_model=str(model_cfg.get("llama_model") or "facebook/opt-1.3b"),
        low_resource=_cfg_bool(model_cfg.get("low_resource"), False),
        device_8bit=int(gpu_id),
        vit_path=vit_path,
        lora_r=int(model_cfg.get("lora_r") or 64),
        lora_target_modules=lora_targets,
        lora_alpha=int(model_cfg.get("lora_alpha") or 128),
        lora_dropout=float(model_cfg.get("lora_dropout") or 0.05),
        z_path=_none_if_string_none(model_cfg.get("z_path")),
        heads_to_use=heads_to_use,
        depth=int(model_cfg.get("depth") or 6),
        big_bird=_cfg_bool(model_cfg.get("big_bird"), False),
        evaluate=True,
        ve_type=_none_if_string_none(model_cfg.get("ve_type")),
    )
    if checkpoint:
        _load_checkpoint_state(model, checkpoint, warnings)
    return model


def _get_model(request: MsVlmCtReportRequest) -> tuple[Any, Any, list[str]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MS-VLM CT report generation requires CUDA because the model code loads modules on cuda:<gpu_id>.")
    if request.gpu_id < 0 or request.gpu_id >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid gpu_id={request.gpu_id}; available CUDA devices: {torch.cuda.device_count()}.")

    cfg_path = _resolve_path(
        request.cfg_path or _env_value("MS_VLM_CT_CFG_PATH"),
        base=MODEL_DIR,
    ) or DEFAULT_CFG_PATH
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"MS-VLM config not found: {cfg_path}")
    checkpoint_path = _resolve_path(request.checkpoint_path, base=WEIGHTS_DIR)
    vit_path = _resolve_path(request.vit_path, base=WEIGHTS_DIR)
    effective_checkpoint_path = checkpoint_path or _resolve_path(_env_value("MS_VLM_CT_CHECKPOINT"), base=WEIGHTS_DIR) or DEFAULT_REPORT_CHECKPOINT
    effective_checkpoint_path = _ensure_inside_weights_dir(effective_checkpoint_path, "MS-VLM report checkpoint")
    effective_vit_path = vit_path or _resolve_path(_env_value("MS_VLM_CT_VIT_PATH"), base=WEIGHTS_DIR) or DEFAULT_VISION_CHECKPOINT
    effective_vit_path = _ensure_inside_weights_dir(effective_vit_path, "MS-VLM vision checkpoint")
    cache_key = (
        str(cfg_path),
        str(effective_checkpoint_path) if effective_checkpoint_path else None,
        tuple([str(effective_vit_path)] if effective_vit_path else []),
        int(request.gpu_id),
        _env_value("MS_VLM_CT_LOW_RESOURCE"),
    )

    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        model_cls, model_cfg, checkpoint, vit_paths, warnings = _load_model_config(
            cfg_path,
            checkpoint_path=checkpoint_path,
            vit_path=vit_path,
            gpu_id=request.gpu_id,
        )
        model = _build_model(model_cls, model_cfg, checkpoint, vit_paths, request.gpu_id, warnings).to(f"cuda:{request.gpu_id}")
        model.eval()
        metadata = {
            "cfg_path": str(cfg_path),
            "checkpoint_path": checkpoint,
            "vit_path": vit_paths,
        }
        _MODEL_CACHE[cache_key] = (model, metadata, warnings)
        return _MODEL_CACHE[cache_key]


def _select_slice_axis(data: Any, slice_axis: object | None) -> int:
    if slice_axis is not None and str(slice_axis).strip().lower() not in {"", "auto"}:
        axis = int(slice_axis)
        if axis not in {0, 1, 2}:
            raise ValueError("`slice_axis` must be 0, 1, 2, or 'auto'.")
        return axis

    shape = tuple(int(dim) for dim in data.shape[:3])
    min_dim = min(shape)
    if shape.count(min_dim) == 1 and min_dim <= 256:
        return shape.index(min_dim)
    return 0


def _load_brain_ct_volume(nifti_path: str, *, slice_axis: object | None = None):
    import nibabel as nib
    import numpy as np
    import torch
    import torch.nn.functional as torch_f
    from torchvision.transforms import CenterCrop

    path = Path(nifti_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"NIfTI volume not found: {path}")

    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim >= 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {tuple(data.shape)} from {path}.")

    axis = _select_slice_axis(data, slice_axis)
    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    tensor = torch.from_numpy(np.ascontiguousarray(data)).to(torch.float32)
    tensor = CenterCrop((512, 512))(tensor)

    depth = int(tensor.shape[0])
    center = depth // 2
    if depth < 32:
        tensor = torch_f.pad(tensor, [0, 0, 0, 0, 0, 32 - depth], mode="constant", value=0)
    else:
        tensor = tensor[center - 16 : center + 16, :, :]

    tensor = tensor.unsqueeze(1)
    volume = torch.cat([tensor, tensor, tensor], dim=1).unsqueeze(0)
    attention_mask = torch.ones(volume.shape[1], dtype=torch.long).unsqueeze(0)
    return volume, attention_mask, {"original_shape": list(image.shape), "slice_axis": axis, "model_shape": list(volume.shape)}


def _read_prompt_candidates(model_metadata: dict[str, Any]) -> list[str]:
    cfg_path = Path(str(model_metadata.get("cfg_path") or ""))
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(cfg_path))
        prompt_path = _resolve_path(str(cfg.model.get("prompt_path") or ""), base=MODEL_DIR)
        if prompt_path and prompt_path.exists():
            lines = [
                line.strip()
                for line in prompt_path.read_text(encoding="utf-8").splitlines()
                if "<ImageHere>" in line
            ]
            if lines:
                return lines
    except Exception:
        pass
    try:
        if DEFAULT_PROMPT_PATH.exists():
            lines = [
                line.strip()
                for line in DEFAULT_PROMPT_PATH.read_text(encoding="utf-8").splitlines()
                if "<ImageHere>" in line
            ]
            if lines:
                return lines
    except Exception:
        pass
    return DEFAULT_BRAIN_CT_PROMPTS


def _format_prompt(request: MsVlmCtReportRequest, model_metadata: dict[str, Any]) -> str:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(model_metadata["cfg_path"]))
    template = str(cfg.model.get("prompt_template") or "").strip()

    prompt = (request.prompt or "").strip()
    if not prompt:
        candidates = _read_prompt_candidates(model_metadata)
        prompt = candidates[request.prompt_index % len(candidates)]
    elif "<ImageHere>" not in prompt:
        prompt = f"<Img><ImageHere></Img> {prompt}"

    if "{}" in template and not bool(getattr(request, "formatted_prompt", False)):
        return template.format(prompt)
    return prompt


def _stopping_criteria(device: Any):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self, stops: list[Any]):
            super().__init__()
            self.stops = stops

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            for stop in self.stops:
                if len(input_ids[0]) >= len(stop) and torch.all((stop == input_ids[0][-len(stop) :])).item():
                    return True
            return False

    stop_words_ids = [torch.tensor([835], device=device), torch.tensor([2277, 29937], device=device)]
    return StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


def _decode_generated_text(model: Any, output_token: Any) -> str:
    while len(output_token) > 0 and int(output_token[0]) in {0, 1}:
        output_token = output_token[1:]
    try:
        raw = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    except TypeError:
        raw = model.llama_tokenizer.decode(output_token, skip_special_tokens=False)
    return str(raw)


def _clean_report_text(raw: str) -> str:
    text = raw.strip()
    for marker in ("ASSISTANT:", "<|im_start|>assistant", "<|im_start|> assistant", "assistant\n"):
        if marker in text:
            text = text.split(marker)[-1].strip()
    for stop in ("<|im_end|>", "</s>", "<|endoftext|>"):
        if stop in text:
            parts = [part.strip() for part in text.split(stop) if part.strip()]
            text = parts[0] if parts else ""
    return text.strip()


def run_ms_vlm_ct_report(request: MsVlmCtReportRequest, *, raw_payload: dict[str, object] | None = None) -> MsVlmCtReportResponse:
    import torch

    model, model_metadata, load_warnings = _get_model(request)
    device = getattr(model, "llama_model_device", torch.device(f"cuda:{request.gpu_id}"))
    volume, attention_mask, volume_metadata = _load_brain_ct_volume(
        request.nifti_path,
        slice_axis=(raw_payload or {}).get("slice_axis"),
    )
    volume = volume.to(device)
    attention_mask = attention_mask.to(device)
    prompt = _format_prompt(request, model_metadata)

    generation_kwargs = {
        "max_new_tokens": int(request.max_new_tokens),
        "num_beams": int(request.num_beams),
        "do_sample": bool(request.do_sample),
        "min_length": int(request.min_length),
        "top_p": float(request.top_p),
        "repetition_penalty": float(request.repetition_penalty),
        "length_penalty": float(request.length_penalty),
        "temperature": float(request.temperature),
        "stopping_criteria": _stopping_criteria(device),
    }

    start_time = time.time()
    with torch.inference_mode():
        try:
            image_embeds, atts_img = model.encode_img(volume, attention_mask, ["brainCT"])
        except TypeError:
            image_embeds, atts_img = model.encode_img(volume, attention_mask)
        input_embeds, wrapped_attention = model.prompt_wrap(image_embeds, atts_img, prompt)
        if bool((raw_payload or {}).get("use_attention_mask", True)):
            generation_kwargs["attention_mask"] = wrapped_attention
        outputs = model.llama_model.generate(inputs_embeds=input_embeds, **generation_kwargs)
    inference_time = time.time() - start_time

    raw_output = _decode_generated_text(model, outputs[0])
    report = _clean_report_text(raw_output)
    warnings = list(load_warnings)
    if not report:
        warnings.append("MS-VLM returned an empty report after output cleanup; inspect raw_output.")

    return MsVlmCtReportResponse(
        source_nifti_path=str(Path(request.nifti_path).expanduser().resolve()),
        file_name=request.file_name or Path(request.nifti_path).name,
        report=report,
        raw_output=raw_output,
        prompt=prompt,
        cfg_path=str(model_metadata["cfg_path"]),
        checkpoint_path=str(model_metadata.get("checkpoint_path") or "") or None,
        vit_path=list(model_metadata.get("vit_path") or []),
        generation={
            **{key: value for key, value in generation_kwargs.items() if key not in {"stopping_criteria", "attention_mask"}},
            **volume_metadata,
        },
        inference_time_seconds=round(inference_time, 3),
        warnings=warnings,
    )


def execute(payload: dict[str, object]) -> dict[str, object]:
    request = MsVlmCtReportRequest(**payload)
    result = run_ms_vlm_ct_report(request, raw_payload=payload)
    result_payload = result.model_dump()
    return {
        "ct_report": result_payload,
        "result_kind": "ct_report_result",
        "requested_view": "ct_report",
        "studio": {"renderer": "ct_report"},
        "draft_answer": (
            f"MS-VLM generated a brain CT report for `{result.file_name}`.\n\n"
            f"{result.report}\n\n"
            f"Inference time: {result.inference_time_seconds:.2f} seconds."
        ),
    }
