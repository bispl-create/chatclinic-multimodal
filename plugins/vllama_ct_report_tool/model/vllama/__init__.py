from __future__ import annotations

import os
import sys

from omegaconf import OmegaConf

from vllama.common.registry import registry


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))


def _register_path_once(name: str, path: str) -> None:
    existing = registry.get_path(name)
    if existing is None:
        registry.register_path(name, path)
    elif os.path.abspath(existing) != os.path.abspath(path):
        registry.mapping["paths"][name] = path


_register_path_once("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
_register_path_once("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
_register_path_once("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
