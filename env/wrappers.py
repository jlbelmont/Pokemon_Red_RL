from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from env.pokemonred_puffer.pokemonred_puffer.environment import RedGymEnv


class SafeRedEnv(RedGymEnv):
    """
    RedGymEnv wrapper that creates an init savestate if none exists.
    """

    def __init__(self, env_config: DictConfig):
        super().__init__(env_config)
        self.save_state_flag = bool(getattr(env_config, "save_state", False))
        self.save_state = self._save_emulator_state
        self._ensure_init_state()

    def _ensure_init_state(self) -> None:
        if not self.init_state_path.exists():
            self.init_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.init_state_path, "wb") as f:
                self.pyboy.save_state(f)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._ensure_init_state()
        return super().reset(seed=seed, options=options)

    def _save_emulator_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            self.pyboy.save_state(f)

    def seed(self, seed: Optional[int] = None) -> None:  # type: ignore[override]
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)


def stack_obs_list(obs_list: List[Dict]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    keys = obs_list[0].keys()
    for key in keys:
        vals = []
        for o in obs_list:
            arr = np.array(o[key])
            if arr.shape == ():
                arr = np.expand_dims(arr, 0)
            vals.append(arr)
        out[key] = np.stack(vals, axis=0)
    return out


class SimpleVectorEnv:
    """
    Minimal synchronous vector env that stacks dict observations.
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.single_action_space = self.envs[0].action_space

    def reset(self, seed: Optional[int] = None, options: Optional[list | dict] = None):
        obs_list, infos = [], []
        for idx, env in enumerate(self.envs):
            opt = None
            if isinstance(options, list):
                opt = options[idx] if idx < len(options) else None
            elif isinstance(options, dict):
                opt = options
            o, info = env.reset(seed=None if seed is None else seed + idx, options=opt)
            obs_list.append(o)
            infos.append(info)
        return stack_obs_list(obs_list), infos

    def step(self, actions):
        obs_list, rews, terms, truncs, infos = [], [], [], [], []
        for env, act in zip(self.envs, actions):
            o, r, t, tr, info = env.step(int(act))
            obs_list.append(o)
            rews.append(r)
            terms.append(t)
            truncs.append(tr)
            infos.append(info)
        return (
            stack_obs_list(obs_list),
            np.array(rews),
            np.array(terms, dtype=bool),
            np.array(truncs, dtype=bool),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()
