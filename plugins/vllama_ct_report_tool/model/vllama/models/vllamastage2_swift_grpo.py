# msw_vls2_optvqa.py
import os, re, glob, json, torch
import numpy as np
from PIL import Image
import nibabel as nib
from typing import List, Dict, Any
from torch import nn

# === import your model class ===
from your_pkg_or_path.vllamastage2_file import vllamastage2optvqa  # <-- adjust

# === Swift registry bits ===
from swift.llm.utils.registry import register_model, ModelMeta, register_template, TemplateMeta
from swift.llm.utils.template import Template
from swift.plugin import orms, ORM

# ---------- 1) A thin wrapper that can GENERATE ----------
class Stage2SwiftWrapper(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        # Important: use evaluate=False -> device uses dist.get_rank() (your class logic)
        cfg.setdefault("evaluate", False)
        self.core = vllamastage2optvqa(**cfg)
        self.tok = self.core.llama_tokenizer
        self.generation_defaults = dict(
            max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95,
            eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id,
            return_dict_in_generate=True
        )

    @torch.no_grad()
    def _prompt_str(self, user_text: str) -> str:
        # Qwen-style chat header + <ImageHere>
        return (
            "<|im_start|>user\n"
            "<ImageHere>\n" + user_text.strip() + "\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @torch.no_grad()
    def _build_embeds(self, volume, attn_mask, modality: List[str], prompt: str):
        # 1) image -> image token embeddings
        img_embeds, atts_img = self.core.encode_img(volume, attn_mask, modality)
        # 2) place <ImageHere> into text stream
        if "<ImageHere>" not in prompt:
            raise ValueError("Prompt must contain <ImageHere>.")
        p_before, p_after = prompt.split("<ImageHere>")
        tok = self.tok
        embed = self.core.llama_model.model.model.embed_tokens

        pbt = tok(p_before, return_tensors="pt", add_special_tokens=True).to(img_embeds.device)
        pat = tok(p_after,  return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        pb_emb = embed(pbt.input_ids).expand(img_embeds.size(0), -1, -1)
        pa_emb = embed(pat.input_ids).expand(img_embeds.size(0), -1, -1)

        inputs_embeds = torch.cat([pb_emb, img_embeds, pa_emb], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        return inputs_embeds, attention_mask

    @torch.no_grad()}
    def generate_from_batch(self, batch: Dict[str, Any], **gen_kwargs) -> List[str]:
        # batch keys come from our Template + Processor below
        vols   = batch["volume"]      # [B, D, 3, H, W]
        masks  = batch["attn_mask"]   # [B, D]
        mods   = batch["modality"]    # List[str] len B
        prompts= batch["prompt"]      # List[str] len B
        outs = []
        for i in range(vols.shape[0]):
            p = prompts[i]
            prompt = p if "<ImageHere>" in p else self._prompt_str(p)
            inp, attn = self._build_embeds(vols[i:i+1].to(self.core.llama_model_device),
                                           masks[i:i+1].to(self.core.llama_model_device),
                                           [mods[i]], prompt)
            cfg = dict(self.generation_defaults); cfg.update(gen_kwargs)
            gen = self.core.llama_model.generate(inputs_embeds=inp, attention_mask=attn, **cfg)
            text = self.tok.decode(gen.sequences[0], skip_special_tokens=True)
            outs.append(text)
        return outs

# ---------- 2) Processor: load volumes and build attn_mask ----------
class Stage2VolumeProcessor:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def _load_volume(self, path: str) -> np.ndarray:
        if path.endswith((".nii", ".nii.gz")):
            v = nib.load(path).get_fdata()
            if v.ndim == 3 and v.shape[-1] != min(v.shape):  # [H,W,D] -> [D,H,W]
                v = np.moveaxis(v, -1, 0)
        elif path.endswith(".npy"):
            v = np.load(path)
            if v.ndim == 4: v = v[..., 0]
        else:
            # folder of PNGs
            ss = sorted(glob.glob(os.path.join(path, "*.png")))
            v = np.stack([np.array(Image.open(s).convert("L")) for s in ss], 0)
        return v.astype(np.float32)  # [D,H,W]

    def __call__(self, images: List[str], modality: List[str], return_tensors="pt"):
        vols, masks = [], []
        for p in images:
            v = self._load_volume(p)                     # [D,H,W]
            v = (v - v.mean()) / (v.std() + 1e-6)        # per-volume norm
            v = np.repeat(v[:, None, :, :], 3, axis=1)   # [D,3,H,W] for your ViT
            D = v.shape[0]
            vols.append(v); masks.append(np.ones((D,), np.int64))
        # pad batch by max depth
        maxD = max(v.shape[0] for v in vols)
        def padV(x): return np.pad(x, ((0,maxD-x.shape[0]),(0,0),(0,0),(0,0)), mode="constant")
        def padM(m): return np.pad(m, ((0,maxD-m.shape[0])), mode="constant")
        V = np.stack([padV(v) for v in vols], 0)  # [B,D,3,H,W]
        M = np.stack([padM(m) for m in masks], 0) # [B,D]
        if return_tensors == "pt":
            V = torch.from_numpy(V); M = torch.from_numpy(M)
        return {"volume": V, "attn_mask": M, "modality": modality}

# ---------- 3) Template: dataset row -> prompt/images/modality ----------
def _messages_to_prompt(messages: List[Dict[str,str]]) -> str:
    # Just grab user's last turn; you can expand later
    user = [m for m in messages if m["role"] == "user"][-1]["content"]
    return user  # wrapper will inject Qwen chat header and <ImageHere> if missing

class VLS2Template(Template):
    def _encode(self, item, tokenizer, processor, **kw):
        # Expect standard keys: messages (required), images (list), modality (str)
        prompt = _messages_to_prompt(item["messages"])
        enc = {"prompt": prompt, "images": item.get("images", []), "modality": [item.get("modality", "brainCT")]}
        return enc

register_template(TemplateMeta(
    template_type="vls2_optvqa_chat",
    prefix="",
    prompt="{{QUERY}}",
    chat_sep="\n",
))

# ---------- 4) Register model with Swift ----------
def get_model_fn(load=None, torch_dtype=None, **model_kwargs):
    # Pass through your constructor knobs from --model_kwargs
    model = Stage2SwiftWrapper(**model_kwargs)
    tok = model.tok
    proc = Stage2VolumeProcessor(image_size=224)
    return model, tok, proc

register_model(ModelMeta(
    model_type="vls2_optvqa_custom",
    model_groups=["/local/path/to/anything"],  # not used; required by registry
    template="vls2_optvqa_chat",
    get_function=get_model_fn,
    model_arch="vlm",
    is_multimodal=True,
))

# ---------- 5) A simple regex reward (edit to your task) ----------
@orms.register("tstage_regex")
class TStageRegex(ORM):
    def __call__(self, completions: List[str], solution: str = None, **kw):
        sol = (solution or "").strip().upper().lstrip("T")
        out = []
        for c in completions:
            m = re.search(r"\bT([0-4][AB]?)\b", c.upper())
            pred = m.group(1) if m else None
            out.append(1.0 if (pred and pred == sol) else 0.0)
        return out
