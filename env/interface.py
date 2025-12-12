from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from omegaconf import OmegaConf

from env.wrappers import SafeRedEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return PROJECT_ROOT


def _asset_paths() -> Tuple[Path, Path]:
    assets_root = PROJECT_ROOT / "env" / "assets"
    rom_path = assets_root / "red.gb"
    state_dir = assets_root / "pyboy_states"
    return rom_path, state_dir


def load_env_config(
    headless: bool = True,
    reduce_res: bool = False,
    two_bit: bool = False,
    use_rubinstein_reward: bool = True,
) -> Dict:
    cfg_path = PROJECT_ROOT / "env" / "pokemonred_puffer" / "config.yaml"
    config = OmegaConf.load(cfg_path)
    env_cfg = OmegaConf.to_container(config.env, resolve=True)
    env_cfg["headless"] = headless
    env_cfg["save_video"] = False
    env_cfg["reduce_res"] = reduce_res
    env_cfg["two_bit"] = two_bit
    env_cfg["use_rubinstein_reward"] = use_rubinstein_reward

    rom_path, state_dir = _asset_paths()
    env_cfg["gb_path"] = str(rom_path)
    env_cfg["state_dir"] = str(state_dir)

    init_state = env_cfg.get("init_state", "Bulbasaur")
    init_state_file = state_dir / f"{init_state}.state"
    if not init_state_file.exists():
        candidates = sorted(state_dir.glob("*.state"))
        if candidates:
            env_cfg["init_state"] = candidates[0].stem
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found at {rom_path}")
    return env_cfg


def make_env(env_cfg: Dict, seed: int):
    cfg = OmegaConf.create(env_cfg)
    env = SafeRedEnv(cfg)
    env.reset(seed=seed)
    return env


__all__ = ["load_env_config", "make_env", "project_root"]
