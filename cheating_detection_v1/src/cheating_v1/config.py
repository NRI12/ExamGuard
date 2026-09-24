"""Configuration: configs/default.yaml plus `--set key.sub=value` overrides.

Relative entries under `paths` resolve against the project root, i.e. the parent of the configs/ folder.
"""
import argparse
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path=None, overrides=()):
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    root = path.resolve().parent.parent
    cfg["paths"] = {k: str(Path(v) if Path(v).is_absolute() else root / v) for k, v in cfg["paths"].items()}
    cfg["root"] = str(root)
    if cfg.get("train", {}).get("workers") == "auto":
        cfg["train"]["workers"] = max(1, min(available_cpus() - 1, 11))
    return cfg


def available_cpus():
    """CPUs this process may use: the container CPU quota on Vast / Docker, else the affinity mask."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)


def config_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help=f"YAML config (default {DEFAULT_CONFIG.name})")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    return p
