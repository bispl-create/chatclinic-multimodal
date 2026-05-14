"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
from omegaconf import OmegaConf

from vllama.common.registry import registry
from vllama.models.base_model import BaseModel
# from vllama.models.blip2 import Blip2Base
# from vllama.models.vllamaita_medblip import vllamaItaMedBLIP
# from vllama.models.vllamaita_zformer import vllamaItaZformer
# from vllama.models.vllamaita_zformer_single import vllamaItaZformerSingle
# from vllama.models.vllamaita_zformer_llava import vllamaItaZformerLLaVA
# from vllama.models.vllamaita_zformer_perceiver import vllamaItaZformerPer
# from vllama.models.vllama_mim import vllamaItaZformerMim
# from vllama.models.vllama_stage1 import vllamastage1
# from vllama.models.vllama_stage1_opt import vllamastage1opt
# from vllama.models.vllama_stage2_opt import vllamastage2opt
from vllama.models.vllama_stage2_opt_plusvqa import vllamastage2optvqa
# from vllama.models.vllama_stage1_falcon import vllamastage1Falcon
# from vllama.models.vllama_stage1_3d import vllamastage1threedim
# from vllama.models.vllama_stage2_falcon import vllamastage2Falcon
# from vllama.models.vllama_stage2 import vllamastage2
# from vllama.models.vllama_stage2_fsdp import vllamastage2FSDP
# from vllama.models.vllama_stage3 import vllamastage3
# from vllama.models.vllama_stage3_comb import vllamastage3Comb
# from vllama.models.vllama_stage3_vground import vllamastage3vground
# from vllama.models.vllamaita_zformer_llava_cls import vllamaItaZformerLLaVACls
# from vllama.models.vllamaita_zqformer_prealign_sam import vllamaItaSamZQformerPreAlign
# from vllama.models.vllamaita_zqformer_prealign import vllamaItaZQformerPreAlign
# from vllama.models.vllamaita_frozen import vllamaItaFrozen
# from vllama.models.vllamaita import vllamaIta
# from vllama.models.vllamaita import vllamaIta
# from vllama.models.mini_gpt4_orig import vllamaOrig
# from vllama.models.vllama import vllama
# from vllama.models.vllamaasclepius import vllamaAscl
# from vllama.processors.base_processor import BaseProcessor


__all__ = [
    "load_model",
    "BaseModel",
    "Blip2Base",
    "vllamaItaMedBLIP",
    "vllamaItaFrozen",
    "vllamaItaZformer",
    "vllamaItaZformerSingle",
    "vllamaItaZformerLLaVA",
    "vllamaItaZformerMim",
    "vllamaItaZformerPer",
    "vllamaItaZformerLLaVACls",
    "vllamaIta",
    "vllamaOrig",
    "vllama",
    "vllamastage1",
    "vllamastage1opt",
    "vllamastage2opt",
    "vllamastage2optvqa",
    "vllamastage1Falcon",
    "vllamastage1threedim",
    "vllamastage2Falcon",
    "vllamastage2",
    "vllamastage2FSDP",
    "vllamastage3",
    "vllamastage3Comb",
    "vllamastage3vground",
    "vllamaAscl",
    "vllamaItaSamZQformerPreAlign",
    "vllamaItaZQformerPreAlign"
]

def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    """
    Load supported models.

    To list all available models and types in registry:
    >>> from vllama.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.

    Returns:
        model (torch.nn.Module): model.
    """

    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)


def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.

        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """

    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else BaseProcessor()
        )

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")
    else:
        vis_train_cfg = None
        vis_eval_cfg = None

    vis_processors["train"] = _build_proc_from_cfg(vis_train_cfg)
    vis_processors["eval"] = _build_proc_from_cfg(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")
    else:
        txt_train_cfg = None
        txt_eval_cfg = None

    txt_processors["train"] = _build_proc_from_cfg(txt_train_cfg)
    txt_processors["eval"] = _build_proc_from_cfg(txt_eval_cfg)

    return vis_processors, txt_processors


def load_model_and_preprocess(name, model_type, is_eval=False, device="cpu"):
    """
    Load model and its related preprocessors.

    List all available models and types in registry:
    >>> from vllama.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".

    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    model_cls = registry.get_model_class(name)

    # load model
    model = model_cls.from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    # load preprocess
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vis_processors, txt_processors


class ModelZoo:
    """
    A utility class to create string representation of available model architectures and types.

    >>> from vllama.models import model_zoo
    >>> # list all available models
    >>> print(model_zoo)
    >>> # show total number of models
    >>> print(len(model_zoo))
    """

    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.PRETRAINED_MODEL_CONFIG_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
        return (
            "=" * 50
            + "\n"
            + f"{'Architectures':<30} {'Types'}\n"
            + "=" * 50
            + "\n"
            + "\n".join(
                [
                    f"{name:<30} {', '.join(types)}"
                    for name, types in self.model_zoo.items()
                ]
            )
        )

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()
