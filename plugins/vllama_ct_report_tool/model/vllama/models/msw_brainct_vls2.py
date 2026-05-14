# msw_brainct_vls2.py
import os, re, json, glob
import numpy as np
import nibabel as nib
from typing import List, Dict, Any
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

# === your model ===
#from vllama.models.vllamastage2_opt_plusvqa import vllamastage2optvqa  # <-- adjust import path
from .vllama_stage2_opt_plusvqa import vllamastage2optvqa
# === swift registries ===
from swift.llm.utils.registry import register_model, ModelMeta, register_template, TemplateMeta
from swift.llm.utils.template import Template
from swift.plugin import orms, ORM


# ---------------- 1) Wrapper (generate with inputs_embeds) ----------------
class Stage2SwiftWrapper(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        cfg.setdefault("evaluate", False)  # use dist.get_rank() devices
        self.core = vllamastage2optvqa(**cfg)
        self.tok = self.core.llama_tokenizer
        self.generation_defaults = dict(
            max_new_tokens=384, do_sample=True, temperature=0.7, top_p=0.95,
            eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id,
            return_dict_in_generate=True
        )

    @torch.no_grad()
    def _qwen_prompt(self, user_text: str) -> str:
        return (
            "<|im_start|>user\n"
            "<ImageHere>\n" + user_text.strip() + "\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @torch.no_grad()
    def _build(self, volume, attn_mask, modality: List[str], prompt: str):
        img_embeds, atts_img = self.core.encode_img(volume, attn_mask, modality)
        if "<ImageHere>" not in prompt:
            raise ValueError("Prompt must contain <ImageHere>.")
        p_before, p_after = prompt.split("<ImageHere>")
        tok = self.tok
        emb = self.core.llama_model.model.model.embed_tokens

        pb = tok(p_before, return_tensors="pt", add_special_tokens=True).to(img_embeds.device)
        pa = tok(p_after,  return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        pb_emb = emb(pb.input_ids).expand(img_embeds.size(0), -1, -1)
        pa_emb = emb(pa.input_ids).expand(img_embeds.size(0), -1, -1)

        inputs_embeds = torch.cat([pb_emb, img_embeds, pa_emb], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        return inputs_embeds, attention_mask

    @torch.no_grad()
    def generate_from_batch(self, batch: Dict[str, Any], **gen_kwargs) -> List[str]:
        V, M, mods, prompts = batch["volume"], batch["attn_mask"], batch["modality"], batch["prompt"]
        outs = []
        for i in range(V.shape[0]):
            p = prompts[i]
            if "<ImageHere>" not in p:
                p = self._qwen_prompt(p)  # add Qwen chat header + <ImageHere>
            inp, attn = self._build(
                V[i:i+1].to(self.core.llama_model_device),
                M[i:i+1].to(self.core.llama_model_device),
                [mods[i]], p)
            cfg = dict(self.generation_defaults); cfg.update(gen_kwargs)
            gen = self.core.llama_model.generate(inputs_embeds=inp, attention_mask=attn, **cfg)
            outs.append(self.tok.decode(gen.sequences[0], skip_special_tokens=True))
        return outs

# ---------------- 2) Processor (mirror your dataset transforms) ----------------
class CenterCrop512:
    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        # vol: [D, H, W]
        D, H, W = vol.shape
        th, tw = 512, 512
        y0 = max(0, (H - th) // 2); x0 = max(0, (W - tw) // 2)
        y1 = min(H, y0 + th); x1 = min(W, x0 + tw)
        cropped = vol[:, y0:y1, x0:x1]
        # pad if needed
        pad_h = 512 - cropped.shape[1]; pad_w = 512 - cropped.shape[2]
        if pad_h > 0 or pad_w > 0:
            cropped = F.pad(cropped, (0, max(0,pad_w), 0, max(0,pad_h)), value=0.0)
        return cropped

class BrainCTProcessor:
    """
    Loads NIfTI, center-crops to 512x512, keeps center 32 slices,
    replicates to 3 channels => [D=32, 3, 512, 512], builds attn_mask of ones.
    """
    def __init__(self): self.cc = CenterCrop512()

    def _load(self, path: str) -> torch.Tensor:
        v = nib.load(path).get_fdata().astype(np.float32)  # [H,W,D] or [D,H,W]
        if v.shape[0] != 32 and v.ndim == 3 and v.shape[-1] != min(v.shape):
            v = np.moveaxis(v, -1, 0)  # -> [D,H,W]
        t = torch.from_numpy(v)  # [D,H,W]
        # center 32 slices
        D = t.shape[0]; center = D // 2
        if D < 32:
            pad = 32 - D
            t = F.pad(t, (0,0,0,0,0,pad), value=0.0)
        else:
            t = t[center-16:center+16, :, :]
        t = self.cc(t)  # [32,512,512]
        # replicate to 3 channels
        t = t.unsqueeze(1).repeat(1, 3, 1, 1)  # [32,3,512,512]
        return t

    def __call__(self, images: List[str], modality: List[str], return_tensors="pt"):
        vols, masks = [], []
        for p in images:
            v = self._load(p)                    # [32,3,512,512]
            vols.append(v); masks.append(torch.ones((v.shape[0],), dtype=torch.long))
        V = torch.stack(vols, 0)   # [B,32,3,512,512]
        M = torch.stack(masks, 0)  # [B,32]
        return {"volume": V, "attn_mask": M, "modality": modality}

# ---------------- 3) Template (use gold report for reward) ----------------
def _build_prompt(user_txt: str) -> str:
    # user asks for a report; <ImageHere> will be placed by wrapper if missing
    return f"Generate a detailed radiology report for this brain CT scan.\n{user_txt}"

class BrainCTTemplate(Template):
    def _encode(self, item, tokenizer, processor, **kw):
        # item needs: image_path, report, modality
        prompt = _build_prompt("")  # keep a stable instruction; no leakage of gold
        enc = {
            "prompt": prompt,
            "images": [item["image_path"]],
            "modality": [item.get("modality", "brainCT")],
            "reference": item["report"],  # for reward
        }
        return enc

register_template(TemplateMeta(
    template_type='custom',
    prefix = ['<extra_id_0>System\n{{SYSTEM}}\n'],
    prompt='<extra_id_1>User\n{{QUERY}}\n<extra_id_1>Assistant\n', chat_sep=["\n"],
))

# ---------------- 4) Model registration ----------------
def get_model(load=None, torch_dtype=None, **kw):
    model = Stage2SwiftWrapper(**kw)
    tok = model.tok
    proc = BrainCTProcessor()
    return model, tok, proc

register_model(ModelMeta(
    model_type="brainct_vls2_custom",
    model_groups=["/any/path"],  # not used
    template="brainct_vls2_chat",
    get_function=get_model,
    model_arch="vlm",
    is_multimodal=True,
))

# ---------------- 5) Reward: token-F1 vs. gold report ----------------
# ---- add to your msw_brainct_vls2.py (or your plugin) ----
import re
from swift.plugin import orms, ORM

# helpers
_THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.S|re.I)

def strip_think(s: str) -> str:
    return _THINK_RE.sub(" ", s)

def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", s.lower()).strip()

# 1) Format reward: exactly one <think>...</think>, non-empty, and non-empty report after it
@orms.register("format_think_report")
class FormatThinkReport(ORM):
    def __call__(self, completions, **kw):
        scores = []
        for c in completions:
            m = list(_THINK_RE.finditer(c))
            if len(m) != 1:
                scores.append(0.0); continue
            think_txt = m[0].group(1).strip()
            after = c[m[0].end():].strip()
            ok = (len(think_txt.split()) >= 8) and (len(after.split()) >= 8)
            # optional: require headings
            # ok = ok and ("impression" in after.lower() or "findings" in after.lower())
            scores.append(1.0 if ok else 0.0)
        return scores

# 2) Brain-CT clinical keyword F1 (CE-F1)
#    Provide a default set; optionally override via dataset column "ce_keywords"
DEFAULT_BRAIN_CT_KWS = [
    "intracerebral hemorrhage","ich","subarachnoid hemorrhage","sah",
    "intraventricular hemorrhage","ivh","subdural hemorrhage","sdh","epidural hematoma","edh",
    "infarct","ischemia","midline shift","edema","mass effect",
    "ventriculomegaly","hydrocephalus","skull fracture","contusion"
]

def kw_set(s: str, kws=None):
    s = norm_text(strip_think(s))
    tokens = set(s.split())
    out = set()
    for k in (kws or DEFAULT_BRAIN_CT_KWS):
        # simple multi-token match
        if all(w in tokens for w in norm_text(k).split()):
            out.add(k.lower())
    return out

# Add near top
_NEG_WORDS = {"no", "not", "absent", "without", "negative"}

def has_negation_before(s: str, kw: str, window: int = 3) -> bool:
    s = s.lower()
    toks = s.split()
    kw_toks = kw.lower().split()
    for i in range(len(toks) - len(kw_toks) + 1):
        if toks[i:i+len(kw_toks)] == kw_toks:
            context = toks[max(0, i - window):i]
            if any(tok in _NEG_WORDS for tok in context):
                return True
    return False

# Modify BrainCTCE inside your plugin:

@orms.register("brain_ct_ce_f1")
class BrainCTCE(ORM):
    def __call__(self, completions, reference=None, ce_keywords=None, **kw):
        kws = ce_keywords or DEFAULT_BRAIN_CT_KWS
        def extract(s):
            s = norm_text(strip_think(s))
            out = set()
            for k in kws:
                if all(w in s.split() for w in norm_text(k).split()):
                    if not has_negation_before(s, k):
                        out.add(k.lower())
            return out

        refset = extract(reference or "")
        out = []
        for c in completions:
            pset = extract(c)
            # compute F1
            if not pset and not refset:
                out.append(1.0)
                continue
            tp = len(pset & refset)
            prec = tp / (len(pset) + 1e-9)
            rec  = tp / (len(refset) + 1e-9)
            f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec+1e-9)
            out.append(float(f1))
        return out

# 3) Token-F1 over content OUTSIDE <think>
@orms.register("report_token_f1_no_think")
class ReportTokenF1NoThink(ORM):
    def __call__(self, completions, reference="", **kw):
        def toks(x): return [t for t in norm_text(strip_think(x)).split() if t]
        ref = set(toks(reference or ""))
        out = []
        for c in completions:
            p = set(toks(c))
            if not p and not ref: out.append(1.0); continue
            tp = len(p & ref)
            prec = tp / (len(p) + 1e-9)
            rec  = tp / (len(ref) + 1e-9)
            f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec+1e-9)
            out.append(float(f1))
        return out
