from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return PROJECT_ROOT


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_dir(log_dir: str | Path, run_name: str) -> Path:
    base = Path(log_dir)
    if not base.is_absolute():
        base = project_root() / base
    return (base / run_name).resolve()


def write_config(run_dir: Path, args: Dict, filename: str = "config.json") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / filename).write_text(json.dumps(args, indent=2))


def apply_config(args, config_path: Optional[str]):
    if not config_path:
        return args
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    for key, value in data.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def cleanup_shared_memory() -> None:
    try:
        from env.pokemonred_puffer.pokemonred_puffer.environment import RedGymEnv

        RedGymEnv.env_id.close()
        RedGymEnv.env_id.unlink()
    except Exception:
        pass


def obs_to_dict(obs) -> Dict[str, np.ndarray]:
    if isinstance(obs, dict):
        return obs
    if isinstance(obs, np.ndarray):
        if obs.dtype.names:
            return {k: obs[k] for k in obs.dtype.names}
        return {"screen": obs}
    raise TypeError(f"Unsupported observation type: {type(obs)}")


def stack_frames(
    frame_stacks: List[Deque[np.ndarray]],
    target_hw: Optional[Tuple[int, int]] = None,
    min_hw: int = 36,
) -> np.ndarray:
    batches: List[np.ndarray] = []
    for frames in frame_stacks:
        arr = np.stack(list(frames), axis=0)

        while arr.ndim > 4 and (arr.shape[1] == 1 or arr.shape[-1] == 1):
            if arr.shape[1] == 1:
                arr = np.squeeze(arr, axis=1)
            elif arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            else:
                break

        if arr.ndim == 4 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))
        elif arr.ndim == 3:
            arr = arr[:, None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Unexpected frame shape {arr.shape}")

        _, _, h, w = arr.shape
        if target_hw:
            target_h, target_w = target_hw
            if h != target_h or w != target_w:
                stride_h = max(1, h // target_h)
                stride_w = max(1, w // target_w)
                arr = arr[:, :, ::stride_h, ::stride_w]
                arr = arr[:, :, :target_h, :target_w]
                h, w = arr.shape[2], arr.shape[3]
                pad_h = max(0, target_h - h)
                pad_w = max(0, target_w - w)
                if pad_h or pad_w:
                    arr = np.pad(arr, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")

        rep_h = (min_hw + h - 1) // h
        rep_w = (min_hw + w - 1) // w
        if rep_h > 1 or rep_w > 1:
            arr = np.repeat(np.repeat(arr, rep_h, axis=2), rep_w, axis=3)
        pad_h = max(0, min_hw - arr.shape[2])
        pad_w = max(0, min_hw - arr.shape[3])
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")

        t, c, h, w = arr.shape
        merged = arr.reshape(t * c, h, w).astype(np.float32)
        if merged.shape[1] < min_hw or merged.shape[2] < min_hw:
            pad_h = max(0, min_hw - merged.shape[1])
            pad_w = max(0, min_hw - merged.shape[2])
            merged = np.pad(merged, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        batches.append(merged)
    return np.stack(batches, axis=0)


__all__ = [
    "project_root",
    "setup_logging",
    "set_seed",
    "resolve_run_dir",
    "write_config",
    "apply_config",
    "cleanup_shared_memory",
    "obs_to_dict",
    "stack_frames",
]
