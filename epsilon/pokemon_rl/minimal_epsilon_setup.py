# All imports are local to epsilon/
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from typing import Iterable, Optional
from collections import Counter, defaultdict, deque, namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure local modules remain importable whether this file is executed as a script
# (python epsilon/pokemon_rl/minimal_epsilon_setup.py) or imported as part of the
# package (python -m epsilon.pokemon_rl.minimal_epsilon_setup).
_MODULE_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _MODULE_DIR.parent
for _path in (_MODULE_DIR, _PARENT_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from env_pokemon import PokemonRedEnv
from epsilon_env import EpsilonEnv
from map_features import extract_map_features
from rewards.battle_outcome import BattleOutcomeReward
from rewards.battle_damage_reward import BattleDamageReward
from rewards.badge_reward import BadgeReward
from rewards.story_flag_reward import StoryFlagReward
from rewards.champion_reward import ChampionReward
from rewards.item_collection import ItemCollectionReward
from rewards.pokedex_reward import PokedexReward
from rewards.trainer_tier_reward import TrainerBattleReward
from rewards.map_exploration import MapExplorationReward
from rewards.novelty import NoveltyReward
from rewards.learned_map_embedding import LearnedMapEmbeddingReward
from rewards.map_visit_reward import MapVisitReward
from rewards.exploration_frontier_reward import ExplorationFrontierReward
from rewards.quest_reward import QuestReward
from rewards.efficiency_penalty import EfficiencyPenalty
from rewards.safety_penalty import SafetyPenalty
from rewards.resource_reward import ResourceManagementReward
from rewards.latent_event_reward import LatentEventReward
from rewards.map_stay_penalty import MapStayPenalty
from rewards.exploration_progress_reward import ExplorationProgressReward
from intrinsic import VisitCounter, EpisodicLatentMemory, RNDModule, RunningRewardNormalizer
from simple_dqn import SimpleDQN, LowLevelDQNPolicy
from state_archive import StateArchive
from visualization import RouteMapVisualizer, MultiRouteMapVisualizer, GameplayGridVisualizer
from progress_tracking import (
    BADGE_NAMES,
    BayesProgressTracker,
    parse_progress_events,
    ProgressEvent,
)
from utils import purge_rom_save


Transition = namedtuple(
    "Transition",
    "obs map_feat goal_feat action reward discount next_obs next_map_feat next_goal_feat done",
)


def _coerce_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(part.strip(), 0) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    return [int(value)]


def _coerce_float_list(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        floats: list[float] = []
        for v in value:
            try:
                floats.append(float(v))
            except (TypeError, ValueError):
                continue
        return floats
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _coerce_float_tuple_list(value) -> list[tuple[int, float]]:
    if value is None:
        return []
    result: list[tuple[int, float]] = []
    if isinstance(value, str):
        # Expect JSON-style string: "count:reward,count:reward"
        if not value.strip():
            return []
        pairs = value.split(",")
        for pair in pairs:
            if ":" in pair:
                count_str, reward_str = pair.split(":", 1)
                try:
                    result.append((int(count_str.strip(), 0), float(reward_str.strip())))
                except ValueError:
                    continue
        return result
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                count = int(item[0])
                reward = float(item[1])
                result.append((count, reward))
            except (TypeError, ValueError):
                continue
    return result


def _coerce_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _normalise_reward_thresholds(entries) -> list[dict]:
    if not entries:
        return []
    normalised: list[dict] = []
    for idx, entry in enumerate(entries):
        if isinstance(entry, dict):
            try:
                threshold = float(entry.get("threshold"))
            except (TypeError, ValueError):
                continue
            if math.isnan(threshold):
                continue
            name = entry.get("name") or entry.get("component") or f"reward_threshold_{idx + 1}"
            source = (entry.get("source") or "").strip().lower()
            component = entry.get("component")
            if source not in {"episode_total", "reward_component", "intrinsic_component"}:
                source = "reward_component" if component else "episode_total"
            if source != "episode_total" and not component:
                continue
            normalised.append(
                {
                    "name": str(name),
                    "threshold": threshold,
                    "source": source,
                    "component": str(component) if component else None,
                }
            )
        else:
            try:
                threshold = float(entry)
            except (TypeError, ValueError):
                continue
            if math.isnan(threshold):
                continue
            normalised.append(
                {
                    "name": f"episode_total_ge_{threshold:g}",
                    "threshold": threshold,
                    "source": "episode_total",
                    "component": None,
                }
            )
    return normalised


def _expand_path_candidate(candidate: str | None, config_dir: str | None) -> list[str]:
    """Return absolute path candidates for a possibly relative entry."""
    if not candidate:
        return []
    expanded = os.path.expandvars(os.path.expanduser(str(candidate)))
    if os.path.isabs(expanded):
        return [os.path.normpath(expanded)]
    bases: list[str] = []
    if config_dir:
        bases.append(config_dir)
    bases.append(os.getcwd())
    bases.append(os.path.dirname(__file__))
    return [os.path.normpath(os.path.join(base, expanded)) for base in bases]


def _resolve_rom_path(
    primary: str | list[str] | tuple[str, ...] | None,
    fallbacks: list[str] | tuple[str, ...] | None,
    config_dir: str | None,
) -> str:
    """Pick the first ROM path that actually exists."""
    env_override = os.environ.get("POKEMON_ROM_PATH") or os.environ.get("POKEMON_ROM")
    candidates: list[str] = []

    def _extend(entry):
        if not entry:
            return
        if isinstance(entry, (list, tuple)):
            for item in entry:
                candidates.append(item)
        else:
            candidates.append(entry)

    _extend(env_override)
    _extend(primary)
    _extend(fallbacks or [])

    tried: list[str] = []
    for candidate in candidates:
        for resolved in _expand_path_candidate(candidate, config_dir):
            if resolved in tried:
                continue
            tried.append(resolved)
            if os.path.isfile(resolved):
                return resolved
    raise FileNotFoundError(
        f"Could not find a usable Pokemon ROM. Checked: {', '.join(tried) or 'no paths'}"
    )


def _resolve_path_like(path: str | None, base: str) -> str | None:
    """Return an absolute path using base when path is relative."""
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(base, path)


def _append_csv(path: str, header: str, row: str) -> None:
    """Append a CSV row, writing header once if the file is new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    need_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as fh:
        if need_header:
            fh.write(header.rstrip("\n") + "\n")
        fh.write(row.rstrip("\n") + "\n")


def _sanitize_map_stay_penalties(value) -> list[dict]:
    if not value:
        return []
    sanitized: list[dict] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        map_ids = _coerce_int_list(entry.get("map_ids") or entry.get("maps"))
        map_id = entry.get("map_id")
        if map_id is not None:
            try:
                map_id = int(map_id)
            except (TypeError, ValueError):
                map_id = None
        if not map_ids and map_id is None:
            continue
        interval = entry.get("interval", 200)
        try:
            interval = max(1, int(interval))
        except (TypeError, ValueError):
            interval = 200
        penalty = entry.get("penalty", -5.0)
        try:
            penalty = float(penalty)
        except (TypeError, ValueError):
            penalty = -5.0
        escalate_mode = entry.get("escalate_mode")
        escalate_rate = entry.get("escalate_rate", 1.0)
        max_penalty = entry.get("max_penalty")
        escalation = entry.get("escalation") if isinstance(entry.get("escalation"), dict) else {}
        if escalation:
            escalate_mode = escalation.get("mode", escalate_mode)
            if "rate" in escalation:
                escalate_rate = escalation.get("rate")
            if escalation.get("max") is not None:
                max_penalty = escalation.get("max")
        try:
            escalate_rate = float(escalate_rate)
        except (TypeError, ValueError):
            escalate_rate = 1.0
        if max_penalty is not None:
            try:
                max_penalty = float(max_penalty)
            except (TypeError, ValueError):
                max_penalty = None
        sanitized.append(
            {
                "name": entry.get("name"),
                "map_ids": map_ids or None,
                "map_id": map_id,
                "interval": interval,
                "penalty": penalty,
                "escalate_mode": escalate_mode,
                "escalate_rate": escalate_rate,
                "max_penalty": max_penalty,
            }
        )
    return sanitized


def _coerce_float_bounds(value, default: tuple[float, float]) -> tuple[float, float]:
    def _convert(obj):
        try:
            return float(obj)
        except (TypeError, ValueError):
            return None

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lo = _convert(value[0])
        hi = _convert(value[1])
    elif isinstance(value, str) and value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) >= 2:
            lo = _convert(parts[0])
            hi = _convert(parts[1])
        else:
            lo = hi = None
    else:
        lo = hi = None
    if lo is None or hi is None:
        lo, hi = default
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def _sanitize_curriculum_states(value, base_dir: str | None) -> list[dict]:
    if not value:
        return []
    sanitized: list[dict] = []
    warned_missing: set[str] = set()
    for entry in value:
        path = None
        label = None
        repeat = 1
        requires_badges = []
        requires_flags = []
        min_step = 0
        if entry is None:
            path = None
        elif isinstance(entry, str):
            path = entry.strip()
        elif isinstance(entry, dict):
            path = entry.get("path")
            label = entry.get("label")
            repeat = entry.get("episodes", entry.get("repeat", 1))
            requires_badges = _coerce_str_list(entry.get("requires_badges"))
            requires_flags = _coerce_str_list(entry.get("requires_story_flags"))
            min_step = int(entry.get("min_step", 0) or 0)
        else:
            continue
        if isinstance(path, str):
            trimmed = path.strip()
            if not trimmed or trimmed.lower() in {"boot", "power_on"}:
                path = None
            else:
                path = trimmed
                if base_dir and not os.path.isabs(path):
                    path = os.path.normpath(os.path.join(base_dir, path))
                else:
                    path = os.path.normpath(path)
        else:
            path = None
        try:
            repeat_int = int(repeat)
        except (TypeError, ValueError):
            repeat_int = 1
        repeat_int = max(1, repeat_int)
        if path and not os.path.exists(path) and path not in warned_missing:
            print(f"[curriculum] Warning: savestate '{path}' not found when configuring curriculum.")
            warned_missing.add(path)
        sanitized.append(
            {
                "path": path,
                "label": label.strip() if isinstance(label, str) and label.strip() else None,
                "episodes": repeat_int,
                "requires_badges": {b.lower() for b in requires_badges},
                "requires_story_flags": {f.lower() for f in requires_flags},
                "min_step": max(0, min_step),
            }
        )
    return sanitized


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "event"


def _capture_savestate(env, destination: str) -> bool:
    target_env = getattr(env, "env", env)
    saver = getattr(target_env, "save_state", None)
    if not callable(saver):
        print("[curriculum] Warning: environment does not support save_state; skipping capture.")
        return False
    try:
        ok = saver(destination)
    except Exception as exc:
        print(f"[curriculum] Failed to save state to {destination}: {exc}")
        return False
    return bool(ok)


def _prune_curriculum_dir(directory: str, keep_per_label: int = 5) -> None:
    """Remove older savestates per label to keep auto-curriculum tidy."""
    try:
        entries = []
        for filename in os.listdir(directory):
            if not filename.endswith(".state"):
                continue
            full_path = os.path.join(directory, filename)
            if not os.path.isfile(full_path):
                continue
            entries.append((filename, full_path, os.path.getmtime(full_path)))
        entries.sort(key=lambda item: item[2], reverse=True)
    except FileNotFoundError:
        return
    except OSError as exc:
        print(f"[curriculum] Warning: unable to scan {directory}: {exc}")
        return
    by_label: dict[str, list[str]] = defaultdict(list)
    for filename, path, _ in entries:
        label = filename.split("_ep", 1)[0]
        by_label[label].append(path)
    for label, files in by_label.items():
        for stale_path in files[keep_per_label:]:
            try:
                os.remove(stale_path)
                print(f"[curriculum] Pruned stale state {stale_path}")
            except OSError:
                continue


class RewardLikelihoodTracker:
    def __init__(self, configs: list[dict], path: str):
        self.configs = configs or []
        self.path = path
        self._stats = {
            cfg["name"]: {"success": 0.0, "total": 0.0, "config": cfg} for cfg in self.configs
        }

    @property
    def is_active(self) -> bool:
        return bool(self._stats)

    @staticmethod
    def _value_for_cfg(
        cfg: dict, episode_reward: float, reward_components: dict[str, float], intrinsic_components: dict[str, float]
    ) -> Optional[float]:
        source = cfg.get("source")
        if source == "episode_total":
            return float(episode_reward)
        component = cfg.get("component")
        if source == "reward_component":
            return float(reward_components.get(component, 0.0))
        if source == "intrinsic_component":
            return float(intrinsic_components.get(component, 0.0))
        return None

    def update(
        self,
        episode_reward: float,
        reward_components: dict[str, float],
        intrinsic_components: dict[str, float],
    ) -> None:
        if not self._stats:
            return
        for name, stat in self._stats.items():
            cfg = stat["config"]
            value = self._value_for_cfg(cfg, episode_reward, reward_components, intrinsic_components)
            if value is None:
                continue
            stat["total"] += 1.0
            if value >= cfg.get("threshold", 0.0):
                stat["success"] += 1.0

    def summarise(self) -> list[dict]:
        summary: list[dict] = []
        if not self._stats:
            return summary
        for name, stat in self._stats.items():
            cfg = stat["config"]
            successes = stat["success"]
            total = stat["total"]
            alpha = 1.0 + successes
            beta = 1.0 + max(0.0, total - successes)
            posterior_mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0
            try:
                dist = torch.distributions.Beta(
                    torch.tensor([alpha], dtype=torch.float64, device="cpu"),
                    torch.tensor([beta], dtype=torch.float64, device="cpu"),
                )
                try:
                    ci = (
                        float(dist.icdf(torch.tensor([0.025], dtype=torch.float64))),
                        float(dist.icdf(torch.tensor([0.975], dtype=torch.float64))),
                    )
                except NotImplementedError:
                    raise
            except Exception:
                samples = torch.distributions.Beta(
                    torch.tensor([alpha], dtype=torch.float64, device="cpu"),
                    torch.tensor([beta], dtype=torch.float64, device="cpu"),
                ).sample((20000,))
                ci = (
                    float(torch.quantile(samples, 0.025).item()),
                    float(torch.quantile(samples, 0.975).item()),
                )
            summary.append(
                {
                    "name": name,
                    "source": cfg.get("source"),
                    "component": cfg.get("component"),
                    "threshold": cfg.get("threshold"),
                    "trials": total,
                    "successes": successes,
                    "posterior_alpha": alpha,
                    "posterior_beta": beta,
                    "posterior_mean": posterior_mean,
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                }
            )
        return summary

    def save(self) -> Optional[dict]:
        if not self.path or not self._stats:
            return None
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {"reward_thresholds": self.summarise()}
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return payload

    def format_summary(self) -> str:
        summary = self.summarise()
        if not summary:
            return ""
        lines = ["Reward success probabilities:"]
        for entry in summary:
            descriptor = entry["name"]
            component = entry.get("component")
            if component:
                descriptor += f" ({component})"
            lines.append(
                f"  - {descriptor}: ≥{entry['threshold']:.1f} "
                f"mean={entry['posterior_mean']:.3f} "
                f"(95% CI [{entry['ci_lower']:.3f}, {entry['ci_upper']:.3f}]) "
                f"| successes {int(entry['successes'])}/{int(entry['trials'])}"
            )
        return "\n".join(lines)


class BestEpisodeRecorder:
    def __init__(
        self,
        num_envs: int,
        trace_path: Optional[str],
        replay_path: Optional[str],
        frame_interval: int = 8,
        frame_limit: int = 12000,
    ) -> None:
        self.num_envs = num_envs
        self.trace_path = trace_path
        self.replay_path = replay_path
        self.frame_interval = max(1, frame_interval)
        self.frame_limit = max(1, frame_limit)
        self.best_reward = -float("inf")
        self._trace_buffers: list[list[dict]] = [[] for _ in range(num_envs)]
        self._frame_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._episode_ids = [0 for _ in range(num_envs)]

    def reset_episode(self, env_idx: int, episode_number: int) -> None:
        if not (0 <= env_idx < self.num_envs):
            return
        self._trace_buffers[env_idx] = []
        self._frame_buffers[env_idx] = []
        self._episode_ids[env_idx] = episode_number

    def record_step(
        self,
        env_idx: int,
        step_count: int,
        info: dict,
        reward: float,
        frame: Optional[np.ndarray],
    ) -> None:
        if not (0 <= env_idx < self.num_envs):
            return
        entry = {
            "step": int(step_count),
            "reward": float(reward),
            "map_name": info.get("map_name"),
            "map_id": info.get("map_id"),
            "coords": list(info.get("agent_coords") or []),
        }
        self._trace_buffers[env_idx].append(entry)
        if (
            frame is not None
            and (step_count % self.frame_interval) == 0
            and len(self._frame_buffers[env_idx]) < self.frame_limit
        ):
            array = np.asarray(frame)
            self._frame_buffers[env_idx].append(array.copy())

    def finalize_episode(self, env_idx: int, total_reward: float) -> None:
        if not (0 <= env_idx < self.num_envs):
            return
        if total_reward <= self.best_reward:
            return
        trace = self._trace_buffers[env_idx]
        if not trace:
            return
        self.best_reward = total_reward
        episode_number = self._episode_ids[env_idx]
        self._persist(env_idx, episode_number, total_reward)

    def _persist(self, env_idx: int, episode_number: int, total_reward: float) -> None:
        trace = self._trace_buffers[env_idx]
        frames = self._frame_buffers[env_idx]
        payload = {
            "episode": episode_number,
            "env_index": env_idx,
            "total_reward": total_reward,
            "frame_interval": self.frame_interval,
            "steps": trace,
        }
        if self.trace_path:
            directory = os.path.dirname(self.trace_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.trace_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        if self.replay_path and frames:
            directory = os.path.dirname(self.replay_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            try:
                np.savez_compressed(
                    self.replay_path,
                    frames=np.stack(frames, axis=0),
                    episode=episode_number,
                    env_index=env_idx,
                    reward=total_reward,
                    frame_interval=self.frame_interval,
                )
            except Exception as exc:
                print(f"[recording] Failed to save best-episode frames: {exc}")
        print(
            f"[recording] Updated best episode: env {env_idx + 1} episode {episode_number} "
            f"reward={total_reward:.2f}"
        )

@dataclass
class TrainConfig:
    rom_path: str
    episodes: int
    max_steps_per_episode: int
    gamma: float
    learning_rate: float
    buffer_size: int
    batch_size: int
    train_frequency: int
    learning_starts: int
    target_sync_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    device: torch.device
    headless: bool
    save_dir: str
    save_every: int
    auto_save_minutes: float
    seed: int
    frame_skip: int
    boot_steps: int
    max_no_input_frames: int
    input_spacing_frames: int
    state_path: str | None
    curriculum_states: list[dict]
    render_map: bool
    map_refresh: int
    map_viz_fps: float
    # MapExplorationReward hyperparameters
    mapexplore_base: float
    mapexplore_neighbor_radius: int
    mapexplore_neighbor_weight: float
    mapexplore_distance_weight: float
    mapexplore_min_reward: float
    mapexplore_persist: bool
    persist_map: bool
    save_map_images: bool
    map_image_every: int
    perf_logging_enabled: bool
    perf_log_path: str
    perf_log_interval_steps: int
    reward_likelihood_enabled: bool
    reward_likelihood_threshold: float
    reward_likelihood_interval_steps: int
    reward_likelihood_log_path: str
    novelty_base: float
    novelty_decay: float
    novelty_min_reward: float
    novelty_stride: int
    novelty_quantisation: int
    novelty_persist: bool
    embedding_base: float
    embedding_decay: float
    embedding_min_reward: float
    embedding_include_map: bool
    embedding_persist: bool
    badge_reward: float
    story_flag_default_reward: float
    story_flags: dict | None
    champion_reward: float
    num_envs: int
    aggregate_map_refresh: int
    vectorized: bool
    render_gameplay_grid: bool
    gameplay_viz_fps: float
    use_ssm_encoder: bool
    ssm_state_dim: int
    ssm_head_dim: int
    ssm_heads: int
    ssm_layers: int
    gru_hidden_size: int
    lstm_hidden_size: int
    record_best_episode: bool
    best_trace_path: str | None
    best_replay_path: str | None
    best_frame_interval: int
    best_frame_limit: int
    visit_count_enabled: bool
    visit_count_scale: float
    visit_count_alpha: float
    visit_count_beta: float
    visit_count_epsilon: float
    visit_count_bin_size: int
    visit_count_include_story: bool
    map_transition_scale: float
    map_transition_max_visits: int
    rnd_enabled: bool
    rnd_scale: float
    posterior_rnd_enabled: bool
    posterior_rnd_event: str
    posterior_rnd_bounds: tuple[float, float]
    rnd_hidden_dim: int
    rnd_learning_rate: float
    rnd_anneal_steps: int
    episodic_bonus_scale: float
    episodic_memory_size: int
    episodic_distance_threshold: float
    state_archive_enabled: bool
    state_archive_dir: str
    state_archive_max_cells: int
    state_archive_capture_min_visits: int
    state_archive_reset_prob: float
    battle_win_reward: float
    battle_loss_penalty: float
    n_step: int
    show_env_maps: bool
    log_interval: int
    verbose_logs: bool
    auxiliary_loss_coef: float
    auxiliary_loss_target: float | None
    auxiliary_loss_schedule_steps: int
    novelty_loss_coef: float
    novelty_loss_target: float | None
    novelty_loss_schedule_steps: int
    progress_interval: int
    display_envs: int
    expose_visit_features: bool
    delete_sav_on_reset: bool
    battle_damage_scale: float
    battle_escape_penalty: float
    revisit_penalty_base: float
    revisit_penalty_excess: float
    revisit_penalty_ratio: float
    latent_event_reward: float
    latent_event_revisit_decay: float
    item_reward: float
    key_item_reward: float
    key_item_ids: list[int] | None
    pokedex_new_species_reward: float
    pokedex_milestones: list[tuple[int, float]] | None
    trainer_wild_reward: float
    trainer_trainer_reward: float
    trainer_gym_reward: float
    trainer_elite_reward: float
    gym_map_ids: list[int] | None
    elite_map_ids: list[int] | None
    quest_definitions: list[dict] | None
    pallet_penalty_map_id: int
    pallet_penalty_interval: int
    pallet_penalty: float
    pallet_penalty_escalate_mode: str | None
    pallet_penalty_escalate_rate: float
    pallet_penalty_max_penalty: float | None
    pallet_penalty_map_ids: list[int] | None
    map_stay_penalties: list[dict] | None
    exploration_progress_reward: float
    stagnation_timeout: int
    stagnation_penalty: float
    stagnation_penalty_interval: int
    frontier_reward: float
    frontier_min_gain: int
    map_visit_reward: float
    step_penalty: float
    idle_penalty: float
    idle_threshold: int
    loss_penalty: float
    blackout_penalty: float
    low_hp_threshold: float
    low_hp_penalty: float
    resource_map_keywords: list[str] | None
    resource_map_reward: float
    resource_item_ids: list[int] | None
    resource_item_reward: float
    curriculum_goals: list[dict] | None
    progress_events: list[ProgressEvent] | None
    progress_metrics_path: str
    reward_thresholds: list[dict]
    reward_metrics_path: str
    auto_demo_after_training: bool
    demo_episodes: int
    demo_max_steps: int
    demo_save_map: str | None
    demo_show_progress: bool
    summary_log_path: str | None
    auto_curriculum_capture: bool
    auto_curriculum_capture_dir: str
    auto_curriculum_capture_episodes: int
    auto_curriculum_story_flags: list[str]
    curriculum_events_log_path: str | None


class CurriculumManager:
    def __init__(
        self,
        entries: list[dict],
        default_state: str | None,
        num_envs: int,
    ) -> None:
        self.default_state = default_state
        self.sequence: list[dict] = []
        for entry in entries:
            repeat = max(1, int(entry.get("episodes", 1)))
            for _ in range(repeat):
                self.sequence.append(
                    {
                        "path": entry.get("path"),
                        "label": entry.get("label"),
                        "requires_badges": set(entry.get("requires_badges") or []),
                        "requires_story_flags": set(entry.get("requires_story_flags") or []),
                        "min_step": int(entry.get("min_step", 0)),
                    }
                )
        self.indices = [0 for _ in range(num_envs)]
        self._missing_warned: set[str] = set()

    def _requirements_met(
        self,
        entry: dict,
        unlocked_badges: set[str],
        unlocked_flags: set[str],
        global_step: int,
    ) -> bool:
        if entry["requires_badges"] and not entry["requires_badges"].issubset(unlocked_badges):
            return False
        if entry["requires_story_flags"] and not entry["requires_story_flags"].issubset(
            unlocked_flags
        ):
            return False
        if global_step < entry["min_step"]:
            return False
        return True

    def assign(
        self,
        env_idx: int,
        unlocked_badges: set[str],
        unlocked_flags: set[str],
        global_step: int,
        verbose: bool = False,
    ) -> tuple[str | None, str | None]:
        if not self.sequence:
            return (self.default_state, None)
        for _ in range(len(self.sequence)):
            idx = self.indices[env_idx]
            entry = self.sequence[idx]
            self.indices[env_idx] = (idx + 1) % len(self.sequence)
            if not self._requirements_met(entry, unlocked_badges, unlocked_flags, global_step):
                continue
            path = entry.get("path")
            label = entry.get("label")
            if path and not os.path.exists(path):
                if path not in self._missing_warned:
                    print(
                        f"[curriculum] Warning: savestate '{path}' missing during training; using ROM boot instead."
                    )
                    self._missing_warned.add(path)
                path = None
            return (path if path else None, label)
        if verbose:
            print(
                "[curriculum] Requirements not met for any savestate; falling back to ROM boot."
            )
        return (self.default_state, None)

    def add_state(
        self,
        path: str,
        label: str | None,
        episodes: int = 1,
        requires_badges: Iterable[str] | None = None,
        requires_story_flags: Iterable[str] | None = None,
        min_step: int = 0,
    ) -> None:
        if not path:
            return
        entry = {
            "path": path,
            "label": label,
            "requires_badges": set(req.lower() for req in (requires_badges or [])),
            "requires_story_flags": set(req.lower() for req in (requires_story_flags or [])),
            "min_step": max(0, int(min_step)),
        }
        count = max(1, int(episodes))
        for _ in range(count):
            self.sequence.append(entry.copy())
        if not self.sequence:
            return
        for idx in range(len(self.indices)):
            self.indices[idx] %= len(self.sequence)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        self.capacity = capacity
        self.storage = []
        self.position = 0
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / max(1, beta_frames)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(
        self,
        obs: np.ndarray,
        map_feat: np.ndarray,
        goal_feat: np.ndarray,
        action: int,
        reward: float,
        discount: float,
        next_obs: np.ndarray,
        next_map_feat: np.ndarray,
        next_goal_feat: np.ndarray,
        done: bool,
    ) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(None)
        self.storage[self.position] = Transition(
            np.array(obs, copy=True),
            np.array(map_feat, copy=True),
            np.array(goal_feat, copy=True),
            int(action),
            float(reward),
            float(discount),
            np.array(next_obs, copy=True),
            np.array(next_map_feat, copy=True),
            np.array(next_goal_feat, copy=True),
            bool(done),
        )
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        size = len(self.storage)
        if size == 0:
            raise ValueError("Cannot sample from empty buffer")
        priorities = self.priorities[:size]
        if not np.any(priorities):
            priorities = np.ones(size, dtype=np.float32)
        scaled = priorities ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(size, batch_size, p=probs)
        samples = [self.storage[i] for i in indices]

        weights = (size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = Transition(*zip(*samples))
        return batch, weights.astype(np.float32), indices

    def __len__(self) -> int:
        return len(self.storage)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            value = float(abs(priority) + 1e-6)
            self.priorities[idx] = value
            self.max_priority = max(self.max_priority, value)


class CatchPokemonReward:
    """Encourages catching a wild encounter on Route 1."""

    def __init__(
        self,
        catch_bonus: float = 150.0,
        encounter_bonus: float = 1.0,
        step_penalty: float = -0.02,
        off_route_penalty: float = -0.2,
        escape_penalty: float = -5.0,
        target_map: str | int = "route 1",
    ):
        self.catch_bonus = catch_bonus
        self.encounter_bonus = encounter_bonus
        self.step_penalty = step_penalty
        self.off_route_penalty = off_route_penalty
        self.escape_penalty = escape_penalty
        self.target_map = (
            target_map.lower() if isinstance(target_map, str) else target_map
        )
        self._catch_awarded = False

    def compute(self, obs, info) -> float:
        reward = self.step_penalty
        # Removed off-route penalty per user preference.
        if info.get("in_battle"):
            reward += self.encounter_bonus
        if info.get("battle_result") == "escaped":
            reward += self.escape_penalty
        if info.get("caught_pokemon") and not self._catch_awarded:
            self._catch_awarded = True
            reward += self.catch_bonus
        return reward

    def reset(self) -> None:
        self._catch_awarded = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    obs = obs.astype(np.float32) / 255.0
    if obs.ndim == 3:
        obs = np.transpose(obs, (2, 0, 1))  # (C, H, W)
    return obs


def _compute_goal_features(info: dict) -> np.ndarray:
    badge_count = float(info.get("badge_count") or 0.0)
    badge_progress = min(max(badge_count / 8.0, 0.0), 1.0)
    story_flags = info.get("story_flags") or {}
    if isinstance(story_flags, dict) and story_flags:
        on_flags = sum(1 for value in story_flags.values() if value)
        story_progress = on_flags / float(len(story_flags))
    else:
        story_progress = 0.0
    champion_flag = 1.0 if info.get("champion_defeated") else 0.0
    elite_progress = info.get("elite_progress")
    if elite_progress is None:
        elite_progress = champion_flag
    elite_progress = float(elite_progress)
    elite_progress = min(max(elite_progress / 4.0, 0.0), 1.0)
    return np.array(
        [
            1.0 - badge_progress,
            story_progress,
            elite_progress,
            champion_flag,
        ],
        dtype=np.float32,
    )


def build_map_and_goal_features(info: dict) -> tuple[np.ndarray, np.ndarray]:
    base = extract_map_features(info)
    goal = _compute_goal_features(info)
    return base, goal


def epsilon_by_step(step: int, cfg: TrainConfig) -> float:
    if cfg.epsilon_decay_steps <= 0:
        return cfg.epsilon_end
    fraction = min(1.0, step / cfg.epsilon_decay_steps)
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def batch_to_tensors(batch: Transition, device: torch.device):
    obs = torch.tensor(np.stack(batch.obs), dtype=torch.float32, device=device)
    map_feat = torch.tensor(
        np.stack(batch.map_feat), dtype=torch.float32, device=device
    )
    goal_feat = torch.tensor(
        np.stack(batch.goal_feat), dtype=torch.float32, device=device
    )
    actions = torch.tensor(batch.action, dtype=torch.long, device=device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    discounts = torch.tensor(batch.discount, dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.stack(batch.next_obs), dtype=torch.float32, device=device)
    next_map_feat = torch.tensor(
        np.stack(batch.next_map_feat), dtype=torch.float32, device=device
    )
    next_goal_feat = torch.tensor(
        np.stack(batch.next_goal_feat), dtype=torch.float32, device=device
    )
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device)
    return (
        obs,
        map_feat,
        goal_feat,
        actions,
        rewards,
        discounts,
        next_obs,
        next_map_feat,
        next_goal_feat,
        dones,
    )


def build_reward_modules(cfg: TrainConfig):
    modules: list = []

    if cfg.mapexplore_base > 0.0:
        modules.append(
            MapExplorationReward(
                base_reward=cfg.mapexplore_base,
                neighbor_radius=cfg.mapexplore_neighbor_radius,
                neighbor_weight=cfg.mapexplore_neighbor_weight,
                distance_weight=cfg.mapexplore_distance_weight,
                min_reward=cfg.mapexplore_min_reward,
                persist_across_episodes=cfg.mapexplore_persist,
            )
        )

    if cfg.novelty_base > 0.0:
        modules.append(
            NoveltyReward(
                base_reward=cfg.novelty_base,
                decay=cfg.novelty_decay,
                min_reward=cfg.novelty_min_reward,
                sample_stride=cfg.novelty_stride,
                quantisation=cfg.novelty_quantisation,
                persist_across_episodes=cfg.novelty_persist,
            )
        )

    if cfg.embedding_base > 0.0:
        modules.append(
            LearnedMapEmbeddingReward(
                base_reward=cfg.embedding_base,
                decay=cfg.embedding_decay,
                min_reward=cfg.embedding_min_reward,
                include_map_id=cfg.embedding_include_map,
                persist_across_episodes=cfg.embedding_persist,
            )
        )

    modules.append(CatchPokemonReward())

    modules.append(
        BattleOutcomeReward(
            win_reward=cfg.battle_win_reward,
            loss_penalty=cfg.battle_loss_penalty,
        )
    )
    if cfg.battle_damage_scale != 0.0 or cfg.battle_escape_penalty != 0.0:
        modules.append(
            BattleDamageReward(
                damage_scale=cfg.battle_damage_scale,
                escape_penalty=cfg.battle_escape_penalty,
            )
        )

    modules.append(BadgeReward(reward_per_badge=cfg.badge_reward))
    modules.append(
        StoryFlagReward(cfg.story_flags, default_reward=cfg.story_flag_default_reward)
    )
    modules.append(ChampionReward(reward=cfg.champion_reward))

    if cfg.map_visit_reward > 0.0:
        modules.append(MapVisitReward(map_reward=cfg.map_visit_reward))

    if cfg.exploration_progress_reward != 0.0 or cfg.stagnation_penalty != 0.0:
        modules.append(
            ExplorationProgressReward(
                reward_scale=cfg.exploration_progress_reward,
                stagnation_timeout=cfg.stagnation_timeout,
                stagnation_penalty=cfg.stagnation_penalty,
                penalty_interval=cfg.stagnation_penalty_interval,
            )
        )

    if cfg.frontier_reward > 0.0:
        modules.append(
            ExplorationFrontierReward(
                distance_reward=cfg.frontier_reward,
                min_gain=cfg.frontier_min_gain,
            )
        )

    if cfg.quest_definitions:
        modules.append(QuestReward(cfg.quest_definitions))

    if cfg.pallet_penalty != 0.0:
        modules.append(
            MapStayPenalty(
                map_id=None if cfg.pallet_penalty_map_ids else cfg.pallet_penalty_map_id,
                map_ids=cfg.pallet_penalty_map_ids,
                interval=cfg.pallet_penalty_interval,
                penalty=cfg.pallet_penalty,
                name="pallet_penalty",
                escalate_mode=getattr(cfg, "pallet_penalty_escalate_mode", None),
                escalate_rate=getattr(cfg, "pallet_penalty_escalate_rate", 1.0),
                max_penalty=getattr(cfg, "pallet_penalty_max_penalty", None),
            )
        )

    for penalty_cfg in getattr(cfg, "map_stay_penalties", []) or []:
        modules.append(
            MapStayPenalty(
                map_id=penalty_cfg.get("map_id"),
                map_ids=penalty_cfg.get("map_ids"),
                interval=penalty_cfg.get("interval", cfg.pallet_penalty_interval),
                penalty=penalty_cfg.get("penalty", cfg.pallet_penalty),
                name=penalty_cfg.get("name"),
                escalate_mode=penalty_cfg.get("escalate_mode"),
                escalate_rate=penalty_cfg.get("escalate_rate", 1.0),
                max_penalty=penalty_cfg.get("max_penalty"),
            )
        )

    modules.append(
        PokedexReward(
            new_species_reward=cfg.pokedex_new_species_reward,
            milestone_rewards=cfg.pokedex_milestones,
        )
    )
    modules.append(
        ItemCollectionReward(
            item_reward=cfg.item_reward,
            key_item_reward=cfg.key_item_reward,
            key_item_ids=cfg.key_item_ids,
        )
    )

    modules.append(
        EfficiencyPenalty(
            step_penalty=cfg.step_penalty,
            idle_penalty=cfg.idle_penalty,
            idle_threshold=cfg.idle_threshold,
        )
    )
    modules.append(
        SafetyPenalty(
            loss_penalty=cfg.loss_penalty,
            blackout_penalty=cfg.blackout_penalty,
            low_hp_threshold=cfg.low_hp_threshold,
            low_hp_penalty=cfg.low_hp_penalty,
        )
    )

    if cfg.resource_map_reward or cfg.resource_item_reward:
        modules.append(
            ResourceManagementReward(
                map_keywords=cfg.resource_map_keywords,
                map_reward=cfg.resource_map_reward,
                utility_item_ids=cfg.resource_item_ids,
                item_reward=cfg.resource_item_reward,
            )
        )

    if cfg.latent_event_reward:
        modules.append(
            LatentEventReward(
                base_reward=cfg.latent_event_reward,
                revisit_decay=cfg.latent_event_revisit_decay,
            )
        )

    modules.append(
        TrainerBattleReward(
            wild_reward=cfg.trainer_wild_reward,
            trainer_reward=cfg.trainer_trainer_reward,
            gym_reward=cfg.trainer_gym_reward,
            elite_reward=cfg.trainer_elite_reward,
            gym_map_ids=cfg.gym_map_ids,
            elite_map_ids=cfg.elite_map_ids,
        )
    )

    return modules


def make_env(cfg: TrainConfig) -> PokemonRedEnv:
    return PokemonRedEnv(
        rom_path=cfg.rom_path,
        show_display=not cfg.headless,
        frame_skip=cfg.frame_skip,
        boot_steps=cfg.boot_steps,
        max_no_input_frames=cfg.max_no_input_frames,
        state_path=cfg.state_path,
        story_flag_defs=cfg.story_flags,
        track_visit_stats=cfg.expose_visit_features,
        delete_sav_on_reset=cfg.delete_sav_on_reset,
        input_spacing_frames=cfg.input_spacing_frames,
    )

def make_env_instance(cfg: TrainConfig, env_index: int) -> PokemonRedEnv:
    # Show a PyBoy window for a limited number of environments unless headless is requested.
    show_display = not cfg.headless and env_index < max(0, cfg.display_envs)
    return PokemonRedEnv(
        rom_path=cfg.rom_path,
        show_display=show_display,
        frame_skip=cfg.frame_skip,
        boot_steps=cfg.boot_steps,
        max_no_input_frames=cfg.max_no_input_frames,
        state_path=cfg.state_path,
        story_flag_defs=cfg.story_flags,
        track_visit_stats=cfg.expose_visit_features,
        delete_sav_on_reset=cfg.delete_sav_on_reset,
        input_spacing_frames=cfg.input_spacing_frames,
    )


def compute_td_loss(
    online_net: LowLevelDQNPolicy,
    target_net: LowLevelDQNPolicy,
    optimizer: torch.optim.Optimizer,
    batch_tensors,
    weights: torch.Tensor,
    auxiliary_coef: float,
    novelty_coef: float,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    online_net.reset_noise()
    target_net.reset_noise()
    (
        obs,
        map_feat,
        goal_feat,
        actions,
        rewards,
        discounts,
        next_obs,
        next_map_feat,
        next_goal_feat,
        dones,
    ) = batch_tensors
    quantiles, _, aux_pred, novelty_pred, latents = online_net(obs, map_feat, goal_feat)
    batch_size = quantiles.size(0)
    num_quantiles = quantiles.size(-1)
    action_index = actions.unsqueeze(-1).unsqueeze(-1).expand(
        batch_size, 1, num_quantiles
    )
    chosen_quantiles = quantiles.gather(1, action_index).squeeze(1)

    with torch.no_grad():
        next_quantiles_online, _, _, _, _ = online_net(next_obs, next_map_feat, next_goal_feat)
        next_actions = next_quantiles_online.mean(dim=2).argmax(dim=1)
        next_action_index = next_actions.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, 1, num_quantiles
        )
        next_quantiles_target, _, _, _, _ = target_net(next_obs, next_map_feat, next_goal_feat)
        next_chosen_quantiles = next_quantiles_target.gather(1, next_action_index).squeeze(1)
        target_quantiles = rewards.unsqueeze(1) + discounts.unsqueeze(1) * (1.0 - dones.unsqueeze(1)) * next_chosen_quantiles

    tau = (torch.arange(num_quantiles, device=obs.device, dtype=torch.float32) + 0.5) / num_quantiles
    tau = tau.view(1, num_quantiles, 1)

    diff = target_quantiles.unsqueeze(1) - chosen_quantiles.unsqueeze(2)
    huber = torch.where(diff.abs() <= 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
    quantile_loss = torch.abs(tau - (diff < 0).float()) * huber
    quantile_loss = quantile_loss.mean(dim=2).sum(dim=1)

    td_loss = (weights * quantile_loss).mean()
    aux_loss = F.mse_loss(aux_pred, map_feat, reduction="none").mean(dim=1)
    aux_loss = (weights * aux_loss).mean()
    novelty_loss = F.mse_loss(novelty_pred, goal_feat, reduction="none").mean(dim=1)
    novelty_loss = (weights * novelty_loss).mean()
    loss = td_loss + auxiliary_coef * aux_loss + novelty_coef * novelty_loss
    td_errors = (target_quantiles.mean(dim=1) - chosen_quantiles.mean(dim=1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
    optimizer.step()
    return float(loss.item()), td_errors.detach(), latents.detach()


def _move_hidden_to_device(hidden, device):
    if hidden is None:
        return None
    gru = hidden.get("gru")
    lstm = hidden.get("lstm")
    if gru is None or lstm is None:
        return None
    lstm_h, lstm_c = lstm
    return {
        "gru": gru.to(device),
        "lstm": (lstm_h.to(device), lstm_c.to(device)),
    }


def _detach_hidden(hidden):
    if hidden is None:
        return None
    gru = hidden.get("gru")
    lstm = hidden.get("lstm")
    if gru is None or lstm is None:
        return None
    lstm_h, lstm_c = lstm
    return {
        "gru": gru.detach(),
        "lstm": (lstm_h.detach(), lstm_c.detach()),
    }


def select_action_eps(
    model: LowLevelDQNPolicy,
    obs: np.ndarray,
    map_feat: np.ndarray,
    goal_feat: np.ndarray,
    epsilon: float,
    action_space,
    device: torch.device,
    hidden_state=None,
) -> tuple[int, dict | None, torch.Tensor | None]:
    model.reset_noise()
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    map_tensor = torch.from_numpy(map_feat).float().unsqueeze(0).to(device)
    goal_tensor = torch.from_numpy(goal_feat).float().unsqueeze(0).to(device)
    hidden = _move_hidden_to_device(hidden_state, device)
    with torch.no_grad():
        quantiles, next_hidden, _, _, latent = model(obs_tensor, map_tensor, goal_tensor, hidden)
    q_mean = quantiles.mean(dim=2)
    greedy_action = int(torch.argmax(q_mean, dim=1).item())
    if np.random.rand() < epsilon:
        action = int(action_space.sample())
    else:
        action = greedy_action
    latent_vec = latent.squeeze(0).detach()
    return action, _detach_hidden(next_hidden), latent_vec


def resolve_device(device_spec: str) -> torch.device:
    if device_spec != "auto":
        return torch.device(device_spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_save(model: torch.nn.Module, path: str) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN with epsilon-greedy policy to catch a Pokemon on Route 1."
    )
    parser.set_defaults(story_flags=None)
    default_rom = os.path.join(os.path.dirname(__file__), "pokemon_red.gb")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file (defaults to training_config.json if present).",
    )
    parser.add_argument("--rom", default=default_rom, help="Path to Pokemon Red ROM.")
    parser.add_argument(
        "--rom-fallback",
        dest="rom_fallbacks",
        action="append",
        default=None,
        help="Additional ROM paths to try if the primary path is unavailable. "
        "May be supplied multiple times.",
    )
    parser.add_argument("--episodes", type=int, default=400, help="Training episodes.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Max environment steps per episode.",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100000, help="Replay buffer capacity."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for updates."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=5000,
        help="Number of steps before training begins.",
    )
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=4,
        help="How often (in steps) to apply a gradient update.",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=3,
        help="Number of steps to accumulate for n-step returns.",
    )
    parser.add_argument(
        "--target-sync",
        type=int,
        default=4000,
        help="Sync interval (in steps) for the target network.",
    )
    parser.add_argument(
        "--epsilon-start", type=float, default=1.0, help="Initial epsilon value."
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.05, help="Final epsilon value."
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=200000,
        help="Number of steps over which epsilon decays.",
    )
    parser.add_argument("--seed", type=int, default=7, help="PRNG seed.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device spec (e.g. cpu, cuda, mps) or 'auto'.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run PyBoy in headless mode to avoid rendering overhead.",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(os.path.dirname(__file__), "checkpoints"),
        help="Directory to store checkpoints.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Episode interval for periodic checkpointing.",
    )
    parser.add_argument(
        "--auto-save-minutes",
        type=float,
        default=10.0,
        help="Wall-clock minutes between forced checkpoint saves (latest.pt).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="How often to print per-step logs during training (in environment steps).",
    )
    parser.add_argument(
        "--perf-logging-enabled",
        action="store_true",
        default=False,
        help="Enable periodic logging of steps/sec to a CSV file.",
    )
    parser.add_argument(
        "--perf-log-path",
        default="checkpoints/perf_log.csv",
        help="CSV file to append performance samples (created if missing).",
    )
    parser.add_argument(
        "--perf-log-interval-steps",
        type=int,
        default=1000,
        help="Step interval between performance (steps/sec) samples.",
    )
    parser.add_argument(
        "--reward-likelihood-enabled",
        action="store_true",
        default=False,
        help="Track likelihood of rewards above a threshold over rolling windows.",
    )
    parser.add_argument(
        "--reward-likelihood-threshold",
        type=float,
        default=200.0,
        help="Reward value considered a significant event.",
    )
    parser.add_argument(
        "--reward-likelihood-interval-steps",
        type=int,
        default=1000,
        help="Number of steps per likelihood window.",
    )
    parser.add_argument(
        "--reward-likelihood-log-path",
        default="checkpoints/reward_likelihood.csv",
        help="CSV file to append reward-likelihood samples (created if missing).",
    )
    parser.add_argument(
        "--headless-mode",
        action="store_true",
        default=False,
        help="Convenience flag: disable rendering/logging and force headless PyBoy for overnight runs.",
    )
    parser.add_argument(
        "--no-logs",
        action="store_false",
        dest="verbose_logs",
        help="Disable verbose episode/step logging (useful for headless overnight runs).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Episode interval for always printing a summary even when verbose logs are disabled.",
    )
    parser.add_argument(
        "--display-envs",
        type=int,
        default=1,
        help="Number of PyBoy windows to display when rendering gameplay.",
    )
    parser.add_argument(
        "--pyboy-window",
        action="store_true",
        dest="pyboy_window",
        help="Force a PyBoy gameplay window to be shown (overrides display-envs if needed).",
    )
    parser.add_argument(
        "--no-pyboy-window",
        action="store_false",
        dest="pyboy_window",
        help="Disable the PyBoy gameplay window regardless of display-envs.",
    )
    parser.set_defaults(pyboy_window=None)
    parser.add_argument(
        "--auxiliary-loss-coef",
        type=float,
        default=0.05,
        help="Weight applied to the auxiliary representation reconstruction loss.",
    )
    parser.add_argument(
        "--auxiliary-loss-target",
        type=float,
        default=None,
        help="Target auxiliary-loss coefficient to reach over time (None disables scheduling).",
    )
    parser.add_argument(
        "--auxiliary-loss-schedule-steps",
        type=int,
        default=0,
        help="Number of global steps over which to interpolate the auxiliary-loss coefficient toward the target.",
    )
    parser.add_argument(
        "--novelty-loss-coef",
        type=float,
        default=0.1,
        help="Weight applied to the novelty critic conditioning loss.",
    )
    parser.add_argument(
        "--novelty-loss-target",
        type=float,
        default=None,
        help="Target novelty-loss coefficient (None disables scheduling).",
    )
    parser.add_argument(
        "--novelty-loss-schedule-steps",
        type=int,
        default=0,
        help="Number of global steps used to reach the novelty-loss target.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Number of emulator frames to advance per environment step.",
    )
    parser.add_argument(
        "--input-spacing-frames",
        type=int,
        default=0,
        help="Number of blank frames to insert after each action to avoid rapid button mashing.",
    )
    parser.add_argument(
        "--boot-steps",
        type=int,
        default=120,
        help="Steps to fast-forward after reset before control begins.",
    )
    parser.add_argument(
        "--no-input-timeout",
        type=int,
        default=600,
        help="Terminate an episode if no inputs were pressed for this many frames.",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional PyBoy state file to load on reset (starts near Route 1).",
    )
    parser.add_argument(
        "--auto-curriculum-capture",
        action="store_true",
        help="Automatically save savestates when new badges or story flags are earned.",
    )
    parser.add_argument(
        "--auto-curriculum-capture-dir",
        default="checkpoints/curriculum_states",
        help="Directory to write auto-captured savestates.",
    )
    parser.add_argument(
        "--auto-curriculum-capture-episodes",
        type=int,
        default=1,
        help="How many times to inject each captured savestate into the curriculum cycle.",
    )
    parser.add_argument(
        "--auto-curriculum-story-flags",
        default="",
        help="Comma-separated list of story flag names eligible for auto-capture (empty = all tracked flags).",
    )
    parser.add_argument(
        "--curriculum-events-log-path",
        default="logs/curriculum_events.csv",
        help="CSV file to log curriculum savestate capture events.",
    )
    parser.add_argument(
        "--render-map",
        action="store_true",
        help="Render a live Route 1 occupancy map alongside gameplay.",
    )
    parser.add_argument(
        "--no-render-map",
        action="store_false",
        dest="render_map",
        help="Disable occupancy-map rendering even if enabled in the config.",
    )
    parser.add_argument(
        "--map-refresh",
        type=int,
        default=4,
        help="Update the map visual every N environment steps.",
    )
    parser.add_argument(
        "--map-fps",
        type=float,
        default=30.0,
        help="Target draw FPS for Matplotlib map windows (0 disables throttling).",
    )
    parser.add_argument(
        "--persist-map",
        action="store_true",
        help="Do not clear the occupancy map between episodes.",
    )
    parser.add_argument(
        "--save-map-images",
        action="store_true",
        help="Save map PNGs periodically to the checkpoint directory.",
    )
    parser.add_argument(
        "--map-image-every",
        type=int,
        default=10,
        help="Episode interval for saving map images when enabled.",
    )
    parser.add_argument(
        "--record-best-episode",
        action="store_true",
        dest="record_best_episode",
        help="Capture traces/frames for the highest-reward episode.",
    )
    parser.add_argument(
        "--no-record-best-episode",
        action="store_false",
        dest="record_best_episode",
        help="Disable best-episode capture even if enabled in config.",
    )
    parser.set_defaults(record_best_episode=None)
    parser.add_argument(
        "--best-trace-path",
        default="checkpoints/best_episode_trace.json",
        help="JSON file to store the best-episode coordinate trace.",
    )
    parser.add_argument(
        "--best-replay-path",
        default="checkpoints/best_episode_frames.npz",
        help="Compressed NPZ containing sampled frames from the best episode.",
    )
    parser.add_argument(
        "--best-frame-interval",
        type=int,
        default=8,
        help="Sample every Nth frame when recording best-episode gameplay.",
    )
    parser.add_argument(
        "--best-frame-limit",
        type=int,
        default=12000,
        help="Maximum number of sampled frames to retain for the best episode.",
    )
    parser.add_argument(
        "--gru-hidden-size",
        type=int,
        default=144,
        help="Hidden size of the GRU block inside the policy.",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=144,
        help="Hidden size of the LSTM block inside the policy.",
    )
    parser.add_argument(
        "--visit-count-enabled",
        action="store_true",
        help="Enable global+episodic visit-count intrinsic rewards.",
    )
    parser.add_argument(
        "--visit-count-scale",
        type=float,
        default=0.0,
        help="Scaling factor for visit-count intrinsic bonuses.",
    )
    parser.add_argument(
        "--visit-count-alpha",
        type=float,
        default=1.0,
        help="Coefficient for the global 1/N term in the visit-count reward.",
    )
    parser.add_argument(
        "--visit-count-beta",
        type=float,
        default=0.1,
        help="Bonus for the first visit to a cell within an episode.",
    )
    parser.add_argument(
        "--visit-count-epsilon",
        type=float,
        default=1.0,
        help="Stability constant added to the denominator of the global count bonus.",
    )
    parser.add_argument(
        "--visit-count-bin-size",
        type=int,
        default=4,
        help="Coordinate bin size when hashing map tiles into cells.",
    )
    parser.add_argument(
        "--visit-count-include-story",
        action="store_true",
        help="Include active story flags in the visit-count hash.",
    )
    parser.add_argument(
        "--map-transition-scale",
        type=float,
        default=0.0,
        help="Bonus scale for first-time map transitions.",
    )
    parser.add_argument(
        "--map-transition-max-visits",
        type=int,
        default=8,
        help="Number of times a map transition can yield a bonus (0=unlimited).",
    )
    parser.add_argument(
        "--rnd-enabled",
        action="store_true",
        help="Enable Random Network Distillation intrinsic rewards.",
    )
    parser.add_argument(
        "--rnd-scale",
        type=float,
        default=0.0,
        help="Scaling factor for the RND intrinsic reward.",
    )
    parser.add_argument(
        "--posterior-rnd-enabled",
        action="store_true",
        help="When set, adjust RND scale each episode based on a milestone posterior.",
    )
    parser.add_argument(
        "--posterior-rnd-event",
        default="Oak Parcel Assigned",
        help="Name of the progress event used to drive posterior-based RND scaling.",
    )
    parser.add_argument(
        "--posterior-rnd-bounds",
        default=None,
        help="Optional min,max bounds (e.g., '0.3,0.8') for posterior-based RND scale.",
    )
    parser.add_argument(
        "--rnd-hidden-dim",
        type=int,
        default=256,
        help="Hidden size for the RND predictor/target networks.",
    )
    parser.add_argument(
        "--rnd-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the RND predictor network.",
    )
    parser.add_argument(
        "--rnd-anneal-steps",
        type=int,
        default=5000000,
        help="Steps over which to anneal the RND reward scale toward zero.",
    )
    parser.add_argument(
        "--episodic-bonus-scale",
        type=float,
        default=0.0,
        help="Scale for episodic latent-novelty bonuses.",
    )
    parser.add_argument(
        "--episodic-memory-size",
        type=int,
        default=256,
        help="Maximum latent vectors to retain for episodic novelty memory per env.",
    )
    parser.add_argument(
        "--episodic-distance-threshold",
        type=float,
        default=0.8,
        help="Minimum L2 distance in latent space to earn episodic novelty bonus.",
    )
    parser.add_argument(
        "--state-archive-enabled",
        action="store_true",
        help="Enable Go-Explore style state archive captures/resets.",
    )
    parser.add_argument(
        "--state-archive-dir",
        default="state_archive",
        help="Directory relative to save_dir for storing archive savestates.",
    )
    parser.add_argument(
        "--state-archive-max-cells",
        type=int,
        default=10000,
        help="Maximum savestates to retain in the archive.",
    )
    parser.add_argument(
        "--state-archive-capture-min-visits",
        type=int,
        default=2,
        help="Only capture a cell if its global visit count is at or below this value.",
    )
    parser.add_argument(
        "--state-archive-reset-prob",
        type=float,
        default=0.0,
        help="Probability of resetting an env from a frontier archive cell instead of power-on.",
    )
    parser.add_argument(
        "--reward-thresholds",
        default="",
        help="Comma-separated reward thresholds for Bayesian progress tracking (e.g., '50,100,200').",
    )
    parser.add_argument(
        "--reward-metrics-path",
        default="checkpoints/reward_metrics.json",
        help="Path to write reward-threshold posterior statistics.",
    )
    parser.add_argument(
        "--summary-log-path",
        default=None,
        help="Optional CSV file to append episode summaries (writes header if missing).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments to run for training.",
    )
    parser.add_argument(
        "--aggregate-map-refresh",
        type=int,
        nargs="?",
        const=8,
        default=8,
        help="Update frequency (steps) for the aggregate map visual in parallel mode (defaults to 8).",
    )
    parser.add_argument(
        "--show-env-maps",
        action="store_true",
        default=True,
        help="Display per-environment map panels alongside the aggregate view.",
    )
    parser.add_argument(
        "--no-show-env-maps",
        action="store_false",
        dest="show_env_maps",
        help="Hide individual environment map panels and show only the aggregate view.",
    )
    parser.add_argument(
        "--gameplay-grid",
        action="store_true",
        dest="render_gameplay_grid",
        default=True,
        help="Render the gameplay grid window alongside the aggregate map.",
    )
    parser.add_argument(
        "--gameplay-fps",
        type=float,
        default=24.0,
        help="Target draw FPS for the gameplay grid (0 disables throttling).",
    )
    parser.add_argument(
        "--no-gameplay-grid",
        action="store_false",
        dest="render_gameplay_grid",
        help="Disable the gameplay grid while keeping the aggregate map.",
    )
    parser.add_argument(
        "--watch-only",
        action="store_true",
        help="Skip training and only run replay episodes using a checkpoint.",
    )
    parser.add_argument(
        "--watch-after-training",
        action="store_true",
        help="After training completes, automatically run replay episodes.",
    )
    parser.add_argument(
        "--auto-demo-after-training",
        action="store_true",
        dest="auto_demo_after_training",
        help="Alias for --watch-after-training that also honours config defaults.",
    )
    parser.add_argument(
        "--no-auto-demo-after-training",
        action="store_false",
        dest="auto_demo_after_training",
        help="Disable automatic demo after training even if enabled in the config.",
    )
    parser.set_defaults(auto_demo_after_training=None)
    parser.add_argument(
        "--replay-checkpoint",
        default=None,
        help="Checkpoint path to load during replay (defaults to the best checkpoint in save_dir).",
    )
    parser.add_argument(
        "--replay-episodes",
        type=int,
        default=1,
        help="Number of episodes to play back during replay mode.",
    )
    parser.add_argument(
        "--replay-max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode during replay (defaults to training max_steps).",
    )
    parser.add_argument(
        "--save-replay-map",
        default=None,
        help="Optional file path to save the aggregate map after replay.",
    )
    parser.add_argument(
        "--demo-episodes",
        type=int,
        default=None,
        help="Override the number of episodes to play during the automatic demo.",
    )
    parser.add_argument(
        "--demo-max-steps",
        type=int,
        default=None,
        help="Override the max steps per demo episode.",
    )
    parser.add_argument(
        "--demo-save-map",
        default=None,
        help="Override the automatic demo map save path.",
    )
    parser.add_argument(
        "--show-progress-summary",
        action="store_true",
        dest="show_progress_summary",
        default=True,
        help="Print Bayesian progress metrics before replaying a checkpoint.",
    )
    parser.add_argument(
        "--no-progress-summary",
        action="store_false",
        dest="show_progress_summary",
        help="Suppress the Bayesian progress summary before replay.",
    )
    parser.add_argument(
        "--vectorized",
        action="store_true",
        help="Enable multiprocessing vectorized environment workers.",
    )
    parser.add_argument(
        "--use-ssm-encoder",
        action="store_true",
        help="Enable the lightweight SSM side-encoder to fuse with the CNN features.",
    )
    parser.add_argument(
        "--ssm-state-dim",
        type=int,
        default=32,
        help="State size per SSM head (controls the recurrent memory size).",
    )
    parser.add_argument(
        "--ssm-head-dim",
        type=int,
        default=32,
        help="Output dimension per SSM head (controls how many features are appended).",
    )
    parser.add_argument(
        "--ssm-heads",
        type=int,
        default=2,
        help="Number of SSM heads (multi-value style; shared B/C projections).",
    )
    parser.add_argument(
        "--ssm-layers",
        type=int,
        default=1,
        help="How many stacked lightweight SSM blocks to apply.",
    )
    parser.add_argument(
        "--story-flags-json",
        default=None,
        help="Optional JSON file describing story flag memory locations and rewards.",
    )
    parser.add_argument(
        "--badge-reward",
        type=float,
        default=200.0,
        help="Reward applied when a new gym badge is obtained.",
    )
    parser.add_argument(
        "--story-flag-default-reward",
        type=float,
        default=150.0,
        help="Default reward applied when a configured story flag becomes true.",
    )
    parser.add_argument(
        "--champion-reward",
        type=float,
        default=1000.0,
        help="Reward applied when the Champion is defeated.",
    )
    # Novelty reward controls
    parser.add_argument(
        "--novelty-base",
        type=float,
        default=1.0,
        help="Base reward for novel screens before decay.",
    )
    parser.add_argument(
        "--novelty-decay",
        type=float,
        default=0.9,
        help="Multiplicative decay applied on each revisit of the same token.",
    )
    parser.add_argument(
        "--novelty-min-reward",
        type=float,
        default=0.0,
        help="Minimum novelty reward after decay.",
    )
    parser.add_argument(
        "--novelty-stride",
        type=int,
        default=4,
        help="Stride used when downsampling observations for novelty hashing.",
    )
    parser.add_argument(
        "--novelty-quantisation",
        type=int,
        default=32,
        help="Quantisation bucket size for novelty hashing (higher = more tolerant).",
    )
    # Learned map embedding controls
    parser.add_argument(
        "--embedding-base",
        type=float,
        default=1.0,
        help="Base reward for new coordinate embeddings.",
    )
    parser.add_argument(
        "--embedding-decay",
        type=float,
        default=0.9,
        help="Decay applied each time an embedding is revisited.",
    )
    parser.add_argument(
        "--embedding-min-reward",
        type=float,
        default=0.0,
        help="Minimum embedding reward after decay.",
    )
    parser.add_argument(
        "--embedding-include-map",
        action="store_true",
        default=True,
        help="Include map_id in the embedding key (can disable with --no-embedding-include-map).",
    )
    parser.add_argument(
        "--no-embedding-include-map",
        action="store_false",
        dest="embedding_include_map",
        help="Do not include map_id in the embedding key.",
    )
    # MapExplorationReward knobs
    parser.add_argument(
        "--mapexplore-base", type=float, default=1.0,
        help="Base reward for a brand new coordinate."
    )
    parser.add_argument(
        "--mapexplore-neighbor-radius", type=int, default=1,
        help="Chebyshev radius for neighbor density penalty."
    )
    parser.add_argument(
        "--mapexplore-neighbor-weight", type=float, default=0.15,
        help="Penalty per visited neighbor within radius."
    )
    parser.add_argument(
        "--mapexplore-distance-weight", type=float, default=0.5,
        help="Bonus scale for distance from episode start."
    )
    parser.add_argument(
        "--mapexplore-min-reward", type=float, default=0.05,
        help="Minimum reward floor for a new coordinate."
    )
    parser.add_argument(
        "--battle-win-reward",
        type=float,
        default=20.0,
        help="Bonus reward when the agent wins a trainer battle.",
    )
    parser.add_argument(
        "--battle-loss-penalty",
        type=float,
        default=-15.0,
        help="Penalty applied when a trainer battle is lost or results in blackout.",
    )
    args = parser.parse_args()

    # Track which CLI options were explicitly provided so config files do not override them
    cli_specified: set[str] = set()
    option_actions = getattr(parser, "_option_string_actions", {})
    for raw_arg in sys.argv[1:]:
        if raw_arg == "--":
            break
        if not raw_arg.startswith("-") or raw_arg == "-":
            continue
        option = raw_arg.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None and action.dest:
            cli_specified.add(action.dest)

    defaults: dict[str, object] = {}
    for name in vars(args):
        try:
            defaults[name] = parser.get_default(name)
        except (AttributeError, KeyError):
            continue

    additional_defaults = {
        "item_reward": 1.0,
        "key_item_reward": 5.0,
        "pokedex_new_species_reward": 10.0,
        "trainer_wild_reward": 5.0,
        "trainer_trainer_reward": 20.0,
        "trainer_gym_reward": 100.0,
        "trainer_elite_reward": 250.0,
        "frontier_reward": 1.0,
        "frontier_min_gain": 1,
        "map_visit_reward": 5.0,
        "step_penalty": -0.001,
        "idle_penalty": -0.1,
        "idle_threshold": 20,
        "loss_penalty": -25.0,
        "blackout_penalty": -50.0,
        "low_hp_threshold": 0.1,
        "low_hp_penalty": -2.0,
        "resource_map_reward": 5.0,
        "resource_item_reward": 2.0,
        "auxiliary_loss_coef": 0.05,
        "auxiliary_loss_target": None,
        "auxiliary_loss_schedule_steps": 0,
        "novelty_loss_coef": 0.1,
        "novelty_loss_target": None,
        "novelty_loss_schedule_steps": 0,
        "latent_event_reward": 25.0,
        "latent_event_revisit_decay": 0.5,
        "verbose_logs": True,
        "headless_mode": False,
        "progress_interval": 10,
        "perf_logging_enabled": False,
        "perf_log_path": "checkpoints/perf_log.csv",
        "perf_log_interval_steps": 1000,
        "reward_likelihood_enabled": False,
        "reward_likelihood_threshold": 200.0,
        "reward_likelihood_interval_steps": 1000,
        "reward_likelihood_log_path": "checkpoints/reward_likelihood.csv",
        "auto_curriculum_capture": False,
        "auto_curriculum_capture_dir": "checkpoints/curriculum_states",
        "auto_curriculum_capture_episodes": 1,
        "use_ssm_encoder": False,
        "ssm_state_dim": 32,
        "ssm_head_dim": 32,
        "ssm_heads": 2,
        "ssm_layers": 1,
        "display_envs": 1,
        "mapexplore_persist": True,
        "novelty_persist": True,
        "embedding_persist": True,
        "expose_visit_features": True,
        "delete_sav_on_reset": True,
        "battle_damage_scale": 6.0,
        "battle_escape_penalty": -12.0,
        "revisit_penalty_base": 0.02,
        "revisit_penalty_excess": 0.01,
        "revisit_penalty_ratio": 0.015,
        "pallet_penalty_map_id": 0,
        "pallet_penalty_interval": 200,
        "pallet_penalty": 0.0,
        "pallet_penalty_escalate_mode": None,
        "pallet_penalty_escalate_rate": 1.0,
        "pallet_penalty_max_penalty": None,
        "exploration_progress_reward": 0.0,
        "stagnation_timeout": 800,
        "stagnation_penalty": -8.0,
        "stagnation_penalty_interval": 120,
        "progress_metrics_path": "checkpoints/progress_metrics.json",
        "reward_metrics_path": "checkpoints/reward_metrics.json",
        "auto_demo_after_training": True,
        "demo_episodes": 1,
        "demo_max_steps": 5000000,
        "demo_save_map": "checkpoints/demo_map.png",
        "demo_show_progress": True,
        "record_best_episode": False,
        "best_trace_path": "checkpoints/best_episode_trace.json",
        "best_replay_path": "checkpoints/best_episode_frames.npz",
        "best_frame_interval": 8,
        "best_frame_limit": 12000,
        "curriculum_events_log_path": "logs/curriculum_events.csv",
    }
    list_defaults = {
        "key_item_ids": [],
        "pokedex_milestones": [],
        "gym_map_ids": [],
        "elite_map_ids": [],
        "quest_definitions": [],
        "resource_map_keywords": ["pokemon center"],
        "resource_item_ids": [],
        "curriculum_goals": [],
        "curriculum_states": [],
        "pallet_penalty_map_ids": [],
        "map_stay_penalties": [],
        "progress_events": [],
        "reward_thresholds": [],
        "rom_fallbacks": [],
        "auto_curriculum_story_flags": [],
    }

    for key, value in additional_defaults.items():
        if key not in defaults:
            defaults[key] = value
        if key == "auto_demo_after_training":
            continue
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    for key, value in list_defaults.items():
        if key not in defaults:
            defaults[key] = value
        if not hasattr(args, key):
            setattr(args, key, list(value) if isinstance(value, (list, tuple)) else value)
    config_path = args.config
    config_dir = None
    if not config_path:
        candidate = os.path.join(os.path.dirname(__file__), "training_config.json")
        if os.path.exists(candidate):
            config_path = candidate

    if config_path:
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config_data = json.load(fh)
        except Exception as exc:
            print(f"[config] Failed to load {config_path}: {exc}")
        else:
            config_dir = os.path.dirname(os.path.abspath(config_path))
            path_keys = {
                "rom",
                "save_dir",
                "state_path",
                "story_flags_json",
                "progress_metrics_path",
                "reward_metrics_path",
                "demo_save_map",
                "best_trace_path",
                "best_replay_path",
                "curriculum_events_log_path",
                "auto_curriculum_capture_dir",
            }
            path_list_keys = {
                "rom_fallbacks",
            }
            for key, value in config_data.items():
                if key not in defaults:
                    continue
                if key in path_keys and isinstance(value, str) and value:
                    if not os.path.isabs(value):
                        value = os.path.normpath(os.path.join(config_dir, value))
                if key in path_list_keys:
                    entries = value if isinstance(value, list) else [value]
                    normalised_list: list[str] = []
                    for entry in entries:
                        if not isinstance(entry, str):
                            continue
                        entry_path = entry
                        if config_dir and entry_path and not os.path.isabs(entry_path):
                            entry_path = os.path.normpath(os.path.join(config_dir, entry_path))
                        normalised_list.append(entry_path)
                    value = normalised_list
                if key in cli_specified:
                    continue
                if getattr(args, key) == defaults[key]:
                    setattr(args, key, value)
            args.config = config_path
            print(f"[config] Loaded parameters from {config_path}")
    if getattr(args, "auto_demo_after_training", None) is None:
        setattr(args, "auto_demo_after_training", False)
    setattr(args, "_config_dir", config_dir)

    # Apply headless-mode presets if requested (via CLI or config).
    headless_mode = getattr(args, "headless_mode", False)
    if headless_mode:
        setattr(args, "headless", True)
        setattr(args, "render_map", False)
        setattr(args, "save_map_images", False)
        # Only force quiet logging if the user did not override it.
        verbose_default = defaults.get("verbose_logs", True)
        if getattr(args, "verbose_logs", verbose_default) == verbose_default:
            setattr(args, "verbose_logs", False)

    # Load story flag definitions from JSON file if supplied.
    story_flags = getattr(args, "story_flags", None)
    story_flags_json = getattr(args, "story_flags_json", None)
    if story_flags_json:
        json_path = story_flags_json
        if isinstance(json_path, str) and not os.path.isabs(json_path) and config_dir:
            json_path = os.path.normpath(os.path.join(config_dir, json_path))
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                story_flags = json.load(fh)
        except Exception as exc:
            print(f"[config] Failed to load story flags from {story_flags_json}: {exc}")
        else:
            setattr(args, "story_flags", story_flags)

    if story_flags is None:
        setattr(args, "story_flags", None)

    # Finalize ROM path by checking the primary path along with fallbacks and env overrides.
    config_dir = os.path.dirname(os.path.abspath(args.config)) if getattr(args, "config", None) else None
    rom_fallbacks = getattr(args, "rom_fallbacks", []) or []
    try:
        resolved_rom = _resolve_rom_path(getattr(args, "rom", None), rom_fallbacks, config_dir)
    except FileNotFoundError as exc:
        print(f"[config] {exc}")
        raise SystemExit(1)
    setattr(args, "rom", resolved_rom)
    try:
        rom_size = os.path.getsize(resolved_rom)
    except OSError:
        rom_size = None
    size_str = f"{rom_size / 1024:.1f} KiB" if rom_size is not None else "unknown size"
    print(f"[config] Using Pokemon ROM: {resolved_rom} ({size_str})")

    return args


def build_config(args) -> TrainConfig:
    key_item_ids = _coerce_int_list(getattr(args, "key_item_ids", []))
    pokedex_milestones = _coerce_float_tuple_list(getattr(args, "pokedex_milestones", []))
    gym_map_ids = _coerce_int_list(getattr(args, "gym_map_ids", []))
    elite_map_ids = _coerce_int_list(getattr(args, "elite_map_ids", []))
    resource_map_keywords = _coerce_str_list(getattr(args, "resource_map_keywords", []))
    resource_item_ids = _coerce_int_list(getattr(args, "resource_item_ids", []))
    quest_definitions = getattr(args, "quest_definitions", []) or []
    curriculum_goals = getattr(args, "curriculum_goals", []) or []
    config_dir = getattr(args, "_config_dir", None)
    curriculum_states = _sanitize_curriculum_states(
        getattr(args, "curriculum_states", []) or [], config_dir
    )
    auto_curriculum_story_flags = _coerce_str_list(
        getattr(args, "auto_curriculum_story_flags", []) or []
    )
    progress_events = parse_progress_events(getattr(args, "progress_events", []) or [])
    progress_metrics_path = getattr(
        args, "progress_metrics_path", os.path.join(args.save_dir, "progress_metrics.json")
    )
    reward_metrics_path = getattr(
        args, "reward_metrics_path", os.path.join(args.save_dir, "reward_metrics.json")
    )
    summary_log_path = getattr(args, "summary_log_path", None)
    if summary_log_path:
        summary_log_path = os.path.normpath(summary_log_path)
    curriculum_events_log_path = getattr(args, "curriculum_events_log_path", None)
    if curriculum_events_log_path:
        curriculum_events_log_path = os.path.normpath(curriculum_events_log_path)
    reward_thresholds = _normalise_reward_thresholds(getattr(args, "reward_thresholds", []) or [])
    record_best_episode = getattr(args, "record_best_episode", None)
    if record_best_episode is None:
        record_best_episode = False
    demo_max_steps = getattr(args, "demo_max_steps", None)
    if demo_max_steps is None:
        demo_max_steps = args.max_steps
    pallet_penalty_map_ids = _coerce_int_list(getattr(args, "pallet_penalty_map_ids", []))
    pallet_penalty_max_penalty = getattr(args, "pallet_penalty_max_penalty", None)
    if pallet_penalty_max_penalty is not None:
        try:
            pallet_penalty_max_penalty = float(pallet_penalty_max_penalty)
        except (TypeError, ValueError):
            pallet_penalty_max_penalty = None
    aux_loss_target = getattr(args, "auxiliary_loss_target", None)
    if aux_loss_target is not None:
        try:
            aux_loss_target = float(aux_loss_target)
        except (TypeError, ValueError):
            aux_loss_target = None
    novelty_loss_target = getattr(args, "novelty_loss_target", None)
    if novelty_loss_target is not None:
        try:
            novelty_loss_target = float(novelty_loss_target)
        except (TypeError, ValueError):
            novelty_loss_target = None
    map_stay_penalties = _sanitize_map_stay_penalties(getattr(args, "map_stay_penalties", []))
    requested_display_envs = max(0, int(getattr(args, "display_envs", 1)))
    pyboy_window = getattr(args, "pyboy_window", None)
    if pyboy_window is False:
        requested_display_envs = 0
    elif pyboy_window is True and requested_display_envs == 0:
        requested_display_envs = 1
    visit_count_enabled = bool(getattr(args, "visit_count_enabled", False))
    visit_count_scale = float(getattr(args, "visit_count_scale", 0.0))
    visit_count_alpha = float(getattr(args, "visit_count_alpha", 1.0))
    visit_count_beta = float(getattr(args, "visit_count_beta", 0.1))
    visit_count_epsilon = float(getattr(args, "visit_count_epsilon", 1.0))
    visit_count_bin_size = max(1, int(getattr(args, "visit_count_bin_size", 4)))
    visit_count_include_story = bool(getattr(args, "visit_count_include_story", False))
    map_transition_scale = float(getattr(args, "map_transition_scale", 0.0))
    map_transition_max_visits = int(getattr(args, "map_transition_max_visits", 8))
    rnd_enabled = bool(getattr(args, "rnd_enabled", False))
    rnd_scale = float(getattr(args, "rnd_scale", 0.0))
    posterior_rnd_enabled = bool(getattr(args, "posterior_rnd_enabled", False))
    posterior_rnd_event = str(getattr(args, "posterior_rnd_event", "") or "").strip()
    posterior_rnd_bounds = _coerce_float_bounds(
        getattr(args, "posterior_rnd_bounds", (0.4, 0.9)),
        (0.4, 0.9),
    )
    rnd_hidden_dim = int(getattr(args, "rnd_hidden_dim", 256))
    rnd_learning_rate = float(getattr(args, "rnd_learning_rate", 1e-4))
    rnd_anneal_steps = int(getattr(args, "rnd_anneal_steps", 5000000))
    episodic_bonus_scale = float(getattr(args, "episodic_bonus_scale", 0.0))
    episodic_memory_size = int(getattr(args, "episodic_memory_size", 256))
    episodic_distance_threshold = float(getattr(args, "episodic_distance_threshold", 0.8))
    state_archive_enabled = bool(getattr(args, "state_archive_enabled", False))
    archive_dir = getattr(args, "state_archive_dir", os.path.join(args.save_dir, "state_archive"))
    if archive_dir and not os.path.isabs(archive_dir):
        archive_dir = os.path.normpath(os.path.join(args.save_dir, archive_dir))
    state_archive_max_cells = int(getattr(args, "state_archive_max_cells", 10000))
    state_archive_capture_min_visits = int(getattr(args, "state_archive_capture_min_visits", 2))
    state_archive_reset_prob = float(getattr(args, "state_archive_reset_prob", 0.0))
    return TrainConfig(
        rom_path=args.rom,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        gamma=args.gamma,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_frequency=args.train_frequency,
        learning_starts=args.learning_starts,
        target_sync_interval=args.target_sync,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        device=resolve_device(args.device),
        headless=args.headless,
        save_dir=args.save_dir,
        save_every=args.save_every,
        auto_save_minutes=max(0.1, float(getattr(args, "auto_save_minutes", 10.0))),
        seed=args.seed,
        frame_skip=args.frame_skip,
        input_spacing_frames=max(0, int(getattr(args, "input_spacing_frames", 0))),
        boot_steps=args.boot_steps,
        max_no_input_frames=args.no_input_timeout,
        state_path=args.state_path if args.state_path else None,
        curriculum_states=curriculum_states,
        render_map=args.render_map,
        map_refresh=max(1, args.map_refresh),
        map_viz_fps=max(0.0, float(getattr(args, "map_fps", 30.0))),
        mapexplore_base=args.mapexplore_base,
        mapexplore_neighbor_radius=args.mapexplore_neighbor_radius,
        mapexplore_neighbor_weight=args.mapexplore_neighbor_weight,
        mapexplore_distance_weight=args.mapexplore_distance_weight,
        mapexplore_min_reward=args.mapexplore_min_reward,
        mapexplore_persist=bool(getattr(args, "mapexplore_persist", True)),
        persist_map=args.persist_map,
        save_map_images=args.save_map_images,
        map_image_every=max(1, args.map_image_every),
        perf_logging_enabled=bool(getattr(args, "perf_logging_enabled", False)),
        perf_log_path=getattr(args, "perf_log_path", "checkpoints/perf_log.csv"),
        perf_log_interval_steps=max(1, int(getattr(args, "perf_log_interval_steps", 1000))),
        reward_likelihood_enabled=bool(getattr(args, "reward_likelihood_enabled", False)),
        reward_likelihood_threshold=float(
            getattr(args, "reward_likelihood_threshold", 200.0)
        ),
        reward_likelihood_interval_steps=max(
            1, int(getattr(args, "reward_likelihood_interval_steps", 1000))
        ),
        reward_likelihood_log_path=getattr(
            args, "reward_likelihood_log_path", "checkpoints/reward_likelihood.csv"
        ),
        novelty_base=args.novelty_base,
        novelty_decay=args.novelty_decay,
        novelty_min_reward=args.novelty_min_reward,
        novelty_stride=args.novelty_stride,
        novelty_quantisation=args.novelty_quantisation,
        novelty_persist=bool(getattr(args, "novelty_persist", True)),
        embedding_base=args.embedding_base,
        embedding_decay=args.embedding_decay,
        embedding_min_reward=args.embedding_min_reward,
        embedding_include_map=args.embedding_include_map,
        embedding_persist=bool(getattr(args, "embedding_persist", True)),
        battle_win_reward=args.battle_win_reward,
        battle_loss_penalty=args.battle_loss_penalty,
        badge_reward=args.badge_reward,
        story_flag_default_reward=args.story_flag_default_reward,
        story_flags=getattr(args, "story_flags", None),
        champion_reward=args.champion_reward,
        num_envs=max(1, args.num_envs),
        aggregate_map_refresh=max(1, args.aggregate_map_refresh),
        vectorized=args.vectorized,
        render_gameplay_grid=bool(getattr(args, "render_gameplay_grid", True)),
        gameplay_viz_fps=max(0.0, float(getattr(args, "gameplay_fps", 24.0))),
        use_ssm_encoder=bool(getattr(args, "use_ssm_encoder", False)),
        ssm_state_dim=int(getattr(args, "ssm_state_dim", 32)),
        ssm_head_dim=int(getattr(args, "ssm_head_dim", 32)),
        ssm_heads=int(getattr(args, "ssm_heads", 2)),
        ssm_layers=int(getattr(args, "ssm_layers", 1)),
        gru_hidden_size=int(getattr(args, "gru_hidden_size", 144)),
        lstm_hidden_size=int(getattr(args, "lstm_hidden_size", 144)),
        record_best_episode=bool(record_best_episode),
        best_trace_path=getattr(args, "best_trace_path", None),
        best_replay_path=getattr(args, "best_replay_path", None),
        best_frame_interval=max(1, int(getattr(args, "best_frame_interval", 8))),
        best_frame_limit=max(1, int(getattr(args, "best_frame_limit", 12000))),
        visit_count_enabled=visit_count_enabled,
        visit_count_scale=visit_count_scale,
        visit_count_alpha=visit_count_alpha,
        visit_count_beta=visit_count_beta,
        visit_count_epsilon=max(1e-6, visit_count_epsilon),
        visit_count_bin_size=visit_count_bin_size,
        visit_count_include_story=visit_count_include_story,
        map_transition_scale=map_transition_scale,
        map_transition_max_visits=max(0, map_transition_max_visits),
        rnd_enabled=rnd_enabled,
        rnd_scale=rnd_scale,
        posterior_rnd_enabled=posterior_rnd_enabled,
        posterior_rnd_event=posterior_rnd_event,
        posterior_rnd_bounds=posterior_rnd_bounds,
        rnd_hidden_dim=rnd_hidden_dim,
        rnd_learning_rate=rnd_learning_rate,
        rnd_anneal_steps=max(1, rnd_anneal_steps),
        episodic_bonus_scale=episodic_bonus_scale,
        episodic_memory_size=max(1, episodic_memory_size),
        episodic_distance_threshold=max(1e-6, episodic_distance_threshold),
        state_archive_enabled=state_archive_enabled,
        state_archive_dir=archive_dir,
        state_archive_max_cells=max(1, state_archive_max_cells),
        state_archive_capture_min_visits=max(1, state_archive_capture_min_visits),
        state_archive_reset_prob=max(0.0, min(1.0, state_archive_reset_prob)),
        n_step=max(1, args.n_step),
        show_env_maps=bool(args.show_env_maps),
        log_interval=max(1, args.log_interval),
        verbose_logs=bool(getattr(args, "verbose_logs", True)),
        auxiliary_loss_coef=float(getattr(args, "auxiliary_loss_coef", 0.05)),
        auxiliary_loss_target=aux_loss_target,
        auxiliary_loss_schedule_steps=max(
            0, int(getattr(args, "auxiliary_loss_schedule_steps", 0))
        ),
        novelty_loss_coef=float(getattr(args, "novelty_loss_coef", 0.1)),
        novelty_loss_target=novelty_loss_target,
        novelty_loss_schedule_steps=max(
            0, int(getattr(args, "novelty_loss_schedule_steps", 0))
        ),
        progress_interval=max(1, int(getattr(args, "progress_interval", 10))),
        display_envs=requested_display_envs,
        expose_visit_features=bool(getattr(args, "expose_visit_features", True)),
        delete_sav_on_reset=bool(getattr(args, "delete_sav_on_reset", True)),
        battle_damage_scale=float(getattr(args, "battle_damage_scale", 6.0)),
        battle_escape_penalty=float(getattr(args, "battle_escape_penalty", -12.0)),
        revisit_penalty_base=float(getattr(args, "revisit_penalty_base", 0.02)),
        revisit_penalty_excess=float(getattr(args, "revisit_penalty_excess", 0.01)),
        revisit_penalty_ratio=float(getattr(args, "revisit_penalty_ratio", 0.015)),
        latent_event_reward=float(getattr(args, "latent_event_reward", 25.0)),
        latent_event_revisit_decay=float(getattr(args, "latent_event_revisit_decay", 0.5)),
        item_reward=float(getattr(args, "item_reward", 1.0)),
        key_item_reward=float(getattr(args, "key_item_reward", 5.0)),
        key_item_ids=key_item_ids,
        pokedex_new_species_reward=float(getattr(args, "pokedex_new_species_reward", 10.0)),
        pokedex_milestones=pokedex_milestones,
        trainer_wild_reward=float(getattr(args, "trainer_wild_reward", 5.0)),
        trainer_trainer_reward=float(getattr(args, "trainer_trainer_reward", 20.0)),
        trainer_gym_reward=float(getattr(args, "trainer_gym_reward", 100.0)),
        trainer_elite_reward=float(getattr(args, "trainer_elite_reward", 250.0)),
        gym_map_ids=gym_map_ids,
        elite_map_ids=elite_map_ids,
        quest_definitions=quest_definitions,
        pallet_penalty_map_id=int(getattr(args, "pallet_penalty_map_id", 0)),
        pallet_penalty_interval=max(1, int(getattr(args, "pallet_penalty_interval", 200))),
        pallet_penalty=float(getattr(args, "pallet_penalty", 0.0)),
        pallet_penalty_escalate_mode=getattr(args, "pallet_penalty_escalate_mode", None),
        pallet_penalty_escalate_rate=float(
            getattr(args, "pallet_penalty_escalate_rate", 1.0)
        ),
        pallet_penalty_max_penalty=pallet_penalty_max_penalty,
        pallet_penalty_map_ids=pallet_penalty_map_ids or None,
        map_stay_penalties=map_stay_penalties or [],
        exploration_progress_reward=float(getattr(args, "exploration_progress_reward", 0.0)),
        stagnation_timeout=max(1, int(getattr(args, "stagnation_timeout", 800))),
        stagnation_penalty=float(getattr(args, "stagnation_penalty", -8.0)),
        stagnation_penalty_interval=max(1, int(getattr(args, "stagnation_penalty_interval", 120))),
        frontier_reward=float(getattr(args, "frontier_reward", 1.0)),
        frontier_min_gain=max(1, int(getattr(args, "frontier_min_gain", 1))),
        map_visit_reward=float(getattr(args, "map_visit_reward", 5.0)),
        step_penalty=float(getattr(args, "step_penalty", -0.001)),
        idle_penalty=float(getattr(args, "idle_penalty", -0.1)),
        idle_threshold=max(1, int(getattr(args, "idle_threshold", 20))),
        loss_penalty=float(getattr(args, "loss_penalty", -25.0)),
        blackout_penalty=float(getattr(args, "blackout_penalty", -50.0)),
        low_hp_threshold=float(getattr(args, "low_hp_threshold", 0.1)),
        low_hp_penalty=float(getattr(args, "low_hp_penalty", -2.0)),
        resource_map_keywords=resource_map_keywords,
        resource_map_reward=float(getattr(args, "resource_map_reward", 5.0)),
        resource_item_ids=resource_item_ids,
        resource_item_reward=float(getattr(args, "resource_item_reward", 2.0)),
        curriculum_goals=curriculum_goals,
        progress_events=progress_events,
        progress_metrics_path=progress_metrics_path,
        reward_thresholds=reward_thresholds,
        reward_metrics_path=reward_metrics_path,
        summary_log_path=summary_log_path,
        auto_demo_after_training=bool(getattr(args, "auto_demo_after_training", False)),
        demo_episodes=max(1, int(getattr(args, "demo_episodes", 1))),
        demo_max_steps=max(1, int(demo_max_steps)),
        demo_save_map=getattr(args, "demo_save_map", None),
        demo_show_progress=bool(getattr(args, "demo_show_progress", True)),
        auto_curriculum_capture=bool(getattr(args, "auto_curriculum_capture", False)),
        auto_curriculum_capture_dir=getattr(
            args, "auto_curriculum_capture_dir", "checkpoints/curriculum_states"
        ),
        auto_curriculum_capture_episodes=max(
            1, int(getattr(args, "auto_curriculum_capture_episodes", 1))
        ),
        auto_curriculum_story_flags=auto_curriculum_story_flags,
        curriculum_events_log_path=curriculum_events_log_path,
    )


def _ssm_cfg_from_train_cfg(cfg: TrainConfig) -> dict:
    return {
        "use_ssm_encoder": getattr(cfg, "use_ssm_encoder", False),
        "ssm_state_dim": getattr(cfg, "ssm_state_dim", 32),
        "ssm_head_dim": getattr(cfg, "ssm_head_dim", 32),
        "ssm_heads": getattr(cfg, "ssm_heads", 2),
        "ssm_layers": getattr(cfg, "ssm_layers", 1),
    }


def _resolve_checkpoint_path(cfg: TrainConfig, explicit: str | None) -> str | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join(cfg.save_dir, "dqn_route1_best.pt"))
    candidates.append(os.path.join(cfg.save_dir, "dqn_route1_latest.pt"))
    try:
        ckpt_dir = Path(cfg.save_dir)
        if ckpt_dir.is_dir():
            sorted_pts = sorted(
                ckpt_dir.glob("*.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            candidates.extend([str(p) for p in sorted_pts])
    except Exception:
        pass
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None



def run_replay(
    cfg: TrainConfig,
    checkpoint_path: str | None,
    episodes: int,
    max_steps: int | None = None,
    save_map_path: str | None = None,
    show_progress: bool = True,
) -> None:
    """Load a trained policy and play back episodes with visualization."""

    checkpoint = _resolve_checkpoint_path(cfg, checkpoint_path)
    if not checkpoint:
        print(
            "[replay] Unable to locate a checkpoint. "
            "Provide one via --replay-checkpoint or ensure training has saved dqn_route1_best.pt."
        )
        return

    if show_progress:
        metrics = BayesProgressTracker.load_metrics(cfg.progress_metrics_path)
        if metrics:
            text = BayesProgressTracker.format_metrics(metrics)
            if text:
                print(text)
        else:
            print("[progress] No stored progress metrics yet; run training to generate them.")

    max_steps = max_steps or cfg.max_steps_per_episode
    num_envs = max(1, cfg.num_envs)
    render_map = cfg.render_map or bool(save_map_path)

    print(
        f"[replay] Loading checkpoint {checkpoint} "
        f"for {episodes} episode(s) with max {max_steps} steps."
    )
    device = cfg.device
    set_seed(cfg.seed)
    envs = [
        EpsilonEnv(make_env_instance(cfg, idx), build_reward_modules(cfg))
        for idx in range(num_envs)
    ]
    use_curriculum_manager = bool(cfg.curriculum_states or cfg.auto_curriculum_capture)
    curriculum_manager = (
        CurriculumManager(cfg.curriculum_states, cfg.state_path, num_envs)
        if use_curriculum_manager
        else None
    )
    try:
        probe_obs, probe_info = envs[0].reset(seed=cfg.seed)
    except Exception as exc:  # pragma: no cover - sanity check
        for env in envs:
            env.close()
        print(f"[replay] Failed to initialize environment: {exc}")
        return

    obs_shape = (
        (probe_obs.shape[2], probe_obs.shape[0], probe_obs.shape[1])
        if probe_obs.ndim == 3
        else probe_obs.shape
    )
    probe_map_feat, probe_goal_feat = build_map_and_goal_features(probe_info)
    map_feat_dim = probe_map_feat.shape[0]
    goal_feat_dim = probe_goal_feat.shape[0]
    n_actions = envs[0].action_space.n
    policy = LowLevelDQNPolicy(
        SimpleDQN(
            obs_shape,
            map_feat_dim,
            goal_feat_dim,
            n_actions,
            ssm_cfg=_ssm_cfg_from_train_cfg(cfg),
            gru_hidden_size=cfg.gru_hidden_size,
            lstm_hidden_size=cfg.lstm_hidden_size,
        )
    ).to(device)
    try:
        state_dict = torch.load(checkpoint, map_location="cpu")
        policy.load_state_dict(state_dict)
        policy = policy.to(device)
    except Exception as exc:
        print(f"[replay] Failed to load checkpoint {checkpoint}: {exc}")
        return

    curriculum_assignments = ["" for _ in range(num_envs)]
    replay_badges_unlocked: set[str] = set()
    replay_story_flags_unlocked: set[str] = set()
    replay_global_step = 0
    prev_story_flags = [set() for _ in range(num_envs)]

    def _update_replay_unlocks(info: dict) -> None:
        badges = info.get("badges") or {}
        for name, unlocked in badges.items():
            if unlocked:
                replay_badges_unlocked.add(str(name).lower())
        story_flags = info.get("story_flags") or {}
        for name, active in story_flags.items():
            if active:
                replay_story_flags_unlocked.add(str(name).lower())
    policy.eval()

    map_viz = None
    gameplay_viz = None
    frame_shape = probe_info.get("raw_frame")
    if isinstance(frame_shape, np.ndarray):
        frame_shape = frame_shape.shape
    else:
        frame_shape = probe_obs.shape

    if render_map:
        try:
            map_viz = MultiRouteMapVisualizer(
                num_envs,
                show_env_panels=cfg.show_env_maps,
                target_fps=cfg.map_viz_fps,
            )
        except RuntimeError as exc:
            print(f"[replay] Map visualization unavailable: {exc}")
            map_viz = None
    if render_map and cfg.render_gameplay_grid:
        try:
            gameplay_viz = GameplayGridVisualizer(
                num_envs,
                frame_shape=frame_shape,
                target_fps=cfg.gameplay_viz_fps,
            )
        except RuntimeError as exc:
            print(f"[replay] Gameplay grid unavailable: {exc}")
            gameplay_viz = None
        except Exception as exc:  # pragma: no cover
            print(f"[replay] Failed to initialise gameplay grid: {exc}")
            gameplay_viz = None

    map_refresh = max(1, cfg.map_refresh)
    aggregate_refresh = max(1, cfg.aggregate_map_refresh)
    perf_log_path = _resolve_path_like(getattr(cfg, "perf_log_path", None), cfg.save_dir)
    reward_like_log_path = _resolve_path_like(
        getattr(cfg, "reward_likelihood_log_path", None), cfg.save_dir
    )
    summary_log_path = (
        _resolve_path_like(cfg.summary_log_path, cfg.save_dir) if getattr(cfg, "summary_log_path", None) else None
    )
    curriculum_events_log_path = (
        _resolve_path_like(cfg.curriculum_events_log_path, cfg.save_dir)
        if getattr(cfg, "curriculum_events_log_path", None)
        else None
    )
    perf_last_step = 0
    perf_last_time = time.monotonic()
    perf_console_last_time = perf_last_time
    reward_like_interval = max(1, cfg.reward_likelihood_interval_steps)
    reward_like_hits = 0
    reward_like_steps = 0
    force_progress_log = False

    try:
        for episode in range(episodes):
            obs_list: list[np.ndarray] = []
            info_list: list[dict] = []
            map_feat_list: list[np.ndarray] = []
            goal_feat_list: list[np.ndarray] = []
            done = [False] * num_envs
            actor_hidden = [policy.init_hidden(1, device) for _ in range(num_envs)]
            episode_rewards = [0.0] * num_envs
            env_steps = [0] * num_envs

            for idx, env in enumerate(envs):
                seed = cfg.seed + episode * num_envs + idx
                base_state = cfg.state_path
                state_label = None
                if curriculum_manager:
                    override, state_label = curriculum_manager.assign(
                        idx,
                        replay_badges_unlocked,
                        replay_story_flags_unlocked,
                        replay_global_step,
                        cfg.verbose_logs,
                    )
                    base_state = override
                underlying = getattr(env, "env", env)
                if hasattr(underlying, "state_path"):
                    underlying.state_path = base_state
                obs, info = env.reset(seed=seed)
                desc = ""
                if curriculum_manager:
                    base_desc = os.path.basename(base_state) if base_state else "power_on"
                    desc = f"{state_label} ({base_desc})" if state_label else base_desc
                    if cfg.verbose_logs:
                        print(
                            f"[curriculum] Episode {episode + 1:04d} env {idx + 1}: loading {desc}"
                        )
                elif base_state:
                    desc = os.path.basename(base_state)
                curriculum_assignments[idx] = desc or "power_on"
                info["curriculum_state"] = curriculum_assignments[idx]
                _update_replay_unlocks(info)
                prev_story_flags[idx] = {
                    name for name, active in (info.get("story_flags") or {}).items() if active
                }
                raw_frame = info.get("raw_frame")
                obs_list.append(preprocess_obs(obs))
                info_list.append(info)
                prev_caught_totals[idx] = int(info.get("pokemon_caught_total") or 0)
                prev_defeated_totals[idx] = int(info.get("pokemon_defeated_total") or 0)
                map_feat, goal_feat = build_map_and_goal_features(info)
                map_feat_list.append(map_feat)
                goal_feat_list.append(goal_feat)
                prev_story_flags[idx] = {
                    name for name, active in (info.get("story_flags") or {}).items() if active
                }
                if map_viz:
                    map_viz.new_episode(idx, episode + 1)
                    if not cfg.persist_map:
                        map_viz.reset(idx)
                    map_viz.update(idx, info, reward=0.0, terminal=False, update_aggregate=True)
                if gameplay_viz:
                    gameplay_viz.new_episode(idx, episode + 1)
                    frame = raw_frame if isinstance(raw_frame, np.ndarray) else obs
                    gameplay_viz.update(idx, frame, info=info, reward=0.0, terminal=False)

            total_steps = 0
            while total_steps < max_steps and not all(done):
                for idx, env in enumerate(envs):
                    if done[idx]:
                        continue
                    map_feat = map_feat_list[idx]
                    goal_feat = goal_feat_list[idx]
                    action, new_hidden, _ = select_action_eps(
                        policy,
                        obs_list[idx],
                        map_feat,
                        goal_feat,
                        epsilon=0.0,
                        action_space=env.action_space,
                        device=device,
                        hidden_state=actor_hidden[idx],
                    )
                    actor_hidden[idx] = new_hidden
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    next_info["curriculum_state"] = curriculum_assignments[idx]
                    replay_global_step += 1
                    obs_list[idx] = preprocess_obs(next_obs)
                    info_list[idx] = next_info
                    map_feat, goal_feat = build_map_and_goal_features(next_info)
                    map_feat_list[idx] = map_feat
                    goal_feat_list[idx] = goal_feat
                    _update_replay_unlocks(next_info)
                    current_story_flags: set[str] = set()
                    if cfg.auto_curriculum_capture:
                        current_story_flags = {
                            name
                            for name, active in (next_info.get("story_flags") or {}).items()
                            if active
                        }
                    done_flag = bool(terminated or truncated)
                    done[idx] = done_flag
                    env_steps[idx] += 1
                    total_steps += 1
                    episode_rewards[idx] += reward

                    if map_viz:
                        update_map = (env_steps[idx] % map_refresh == 0) or done_flag
                        update_agg = (env_steps[idx] % aggregate_refresh == 0) or done_flag
                        if update_map:
                            map_viz.update(
                                idx,
                                next_info,
                                reward=reward,
                                terminal=done_flag,
                                update_aggregate=update_agg,
                            )
                    if gameplay_viz:
                        raw_frame = next_info.get("raw_frame")
                        frame = raw_frame if isinstance(raw_frame, np.ndarray) else next_obs
                        gameplay_viz.update(
                            idx,
                            frame,
                            info=next_info,
                            reward=reward,
                            terminal=done_flag,
                        )
                    if total_steps >= max_steps:
                        break

            mean_reward = sum(episode_rewards) / len(episode_rewards)
            print(
                f"[replay] Episode {episode + 1}/{episodes} "
                f"| mean reward {mean_reward:+.2f} | steps {sum(env_steps)}"
            )
            if map_viz and cfg.save_map_images:
                os.makedirs(cfg.save_dir, exist_ok=True)
                map_path = os.path.join(
                    cfg.save_dir, f"replay_map_ep{episode + 1:04d}.png"
                )
                try:
                    map_viz.save(map_path)
                    print(f"[replay] Saved map snapshot to {map_path}")
                except Exception as exc:
                    print(f"[replay] Failed to save map snapshot: {exc}")

        if save_map_path and map_viz:
            try:
                os.makedirs(os.path.dirname(save_map_path) or ".", exist_ok=True)
                map_viz.save(save_map_path)
                print(f"[replay] Aggregate map saved to {save_map_path}")
            except Exception as exc:
                print(f"[replay] Failed to save replay map {save_map_path}: {exc}")
    finally:
        if map_viz:
            map_viz.close()
        if cfg.render_gameplay_grid and gameplay_viz:
            gameplay_viz.close()
        for env in envs:
            env.close()


def train(cfg: TrainConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)
    if cfg.delete_sav_on_reset and cfg.rom_path:
        purge_rom_save(cfg.rom_path)
    num_envs = max(1, cfg.num_envs)
    envs = [
        EpsilonEnv(make_env_instance(cfg, idx), build_reward_modules(cfg))
        for idx in range(num_envs)
    ]
    use_curriculum_manager = bool(cfg.curriculum_states or cfg.auto_curriculum_capture)
    curriculum_manager = (
        CurriculumManager(cfg.curriculum_states, cfg.state_path, num_envs)
        if use_curriculum_manager
        else None
    )
    prev_badge_bits = [0 for _ in range(num_envs)]
    prev_badge_sets = [set() for _ in range(num_envs)]
    prev_champion_flag = [False for _ in range(num_envs)]
    prev_story_flags = [set() for _ in range(num_envs)]
    captured_badge_events = [set() for _ in range(num_envs)]
    captured_story_events = [set() for _ in range(num_envs)]
    captured_quest_events = [set() for _ in range(num_envs)]
    curriculum_assignments = ["" for _ in range(num_envs)]
    curriculum_event_header = "wall_time,episode,env,global_step,event,label,path"
    unlocked_badges_global: set[str] = set()
    unlocked_story_flags_global: set[str] = set()
    progress_logging_ready = [False for _ in range(num_envs)]
    prev_caught_totals = [0 for _ in range(num_envs)]
    prev_defeated_totals = [0 for _ in range(num_envs)]
    progress_payload = None

    probe_obs, probe_info = envs[0].reset(seed=cfg.seed)
    obs_shape = (probe_obs.shape[2], probe_obs.shape[0], probe_obs.shape[1]) if probe_obs.ndim == 3 else probe_obs.shape
    probe_map_feat, probe_goal_feat = build_map_and_goal_features(probe_info)
    map_feat_dim = probe_map_feat.shape[0]
    goal_feat_dim = probe_goal_feat.shape[0]
    n_actions = envs[0].action_space.n
    map_viz = None
    gameplay_viz = None
    if cfg.render_map:
        try:
            map_viz = MultiRouteMapVisualizer(
                num_envs,
                show_env_panels=cfg.show_env_maps,
                target_fps=cfg.map_viz_fps,
            )
        except RuntimeError as exc:
            print(f"[visualization] Disabling map view: {exc}")
            map_viz = None

    # Performance/reward logging setup (keep defined once near the top of train)
    perf_log_path = _resolve_path_like(getattr(cfg, "perf_log_path", None), cfg.save_dir)
    reward_like_log_path = _resolve_path_like(
        getattr(cfg, "reward_likelihood_log_path", None), cfg.save_dir
    )
    summary_log_path = (
        _resolve_path_like(cfg.summary_log_path, cfg.save_dir) if getattr(cfg, "summary_log_path", None) else None
    )
    curriculum_events_log_path = (
        _resolve_path_like(cfg.curriculum_events_log_path, cfg.save_dir)
        if getattr(cfg, "curriculum_events_log_path", None)
        else None
    )
    perf_last_step = 0
    perf_last_time = time.monotonic()
    perf_console_last_time = perf_last_time
    reward_like_interval = max(1, cfg.reward_likelihood_interval_steps)
    reward_like_hits = 0
    reward_like_steps = 0

    curriculum_capture_dir = None
    allowed_story_flags = None
    capture_label_prefixes = ("story_", "badge_", "event_", "quest_", "new_town", "champion_flag")
    capture_map_whitelist = {
        "viridian",
        "route 2",
        "route 3",
        "viridian forest",
        "pewter",
        "cerulean",
    }
    capture_label_limits: dict[str, int] = {
        "event_pokemon_caught": 2,
        "event_battle_win": 2,
        "new_town": 4,
    }
    capture_default_limit = 3
    capture_counts: Counter[str] = Counter()
    if cfg.auto_curriculum_capture:
        curriculum_capture_dir = cfg.auto_curriculum_capture_dir or ""
        if not os.path.isabs(curriculum_capture_dir):
            curriculum_capture_dir = os.path.normpath(os.path.join(cfg.save_dir, curriculum_capture_dir))
        else:
            curriculum_capture_dir = os.path.normpath(curriculum_capture_dir)
        _prune_curriculum_dir(curriculum_capture_dir, keep_per_label=5)
        allowed_story_flags = (
            {name.lower() for name in cfg.auto_curriculum_story_flags}
            if cfg.auto_curriculum_story_flags
            else None
        )

    def _capture_event_state(
        env_idx: int,
        label: str,
        reason: str,
        episode_num: int,
        step_num: int,
        map_name: str | None = None,
    ) -> None:
        nonlocal capture_counts
        if not cfg.auto_curriculum_capture or not curriculum_capture_dir:
            return
        canonical_label = label.split(".", 1)[0]
        allowed_label = any(canonical_label.startswith(prefix) for prefix in capture_label_prefixes)
        allowed_map = False
        if map_name:
            lower = map_name.lower()
            if any(lower.startswith(keyword) for keyword in capture_map_whitelist):
                allowed_map = True
        if not (allowed_label or allowed_map):
            return
        label_limit = capture_label_limits.get(canonical_label, capture_default_limit)
        if capture_counts[canonical_label] >= label_limit:
            return
        slug = _slugify(label)
        filename = f"{slug}_ep{episode_num:04d}_env{env_idx + 1}_step{step_num:07d}.state"
        destination = os.path.join(curriculum_capture_dir, filename)
        if _capture_savestate(envs[env_idx], destination):
            print(f"[curriculum] Captured savestate for {reason} -> {destination}")
            capture_counts[canonical_label] += 1
            if curriculum_manager:
                req_badges = None
                req_flags = None
                normalized = label or ""
                if normalized.startswith("badge_"):
                    req_badges = [normalized.split("badge_", 1)[1]]
                elif normalized.startswith("story_"):
                    req_flags = [normalized.split("story_", 1)[1]]
                curriculum_manager.add_state(
                    destination,
                    label,
                    cfg.auto_curriculum_capture_episodes,
                    requires_badges=req_badges,
                    requires_story_flags=req_flags,
                    min_step=step_num,
                )
            if curriculum_events_log_path:
                _append_csv(
                    curriculum_events_log_path,
                    curriculum_event_header,
                    f"{time.time():.3f},{episode_num},{env_idx + 1},{step_num},{reason},{label},{destination}",
                )


    def _schedule_loss_value(
        base: float,
        target: Optional[float],
        schedule_steps: int,
        global_step: int,
    ) -> float:
        if target is None:
            return base
        if schedule_steps <= 0:
            return target
        frac = min(1.0, max(0.0, global_step / schedule_steps))
        return base + (target - base) * frac

    progress_tracker = None
    if cfg.progress_events:
        if cfg.progress_metrics_path:
            try:
                os.remove(cfg.progress_metrics_path)
                print(f"[progress] Cleared {cfg.progress_metrics_path} before training.")
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"[progress] Warning: could not remove {cfg.progress_metrics_path}: {exc}")
        progress_tracker = BayesProgressTracker(cfg.progress_events, num_envs, cfg.progress_metrics_path)
    reward_tracker = RewardLikelihoodTracker(cfg.reward_thresholds, cfg.reward_metrics_path)
    best_recorder = None
    if cfg.record_best_episode:
        best_recorder = BestEpisodeRecorder(
            num_envs,
            cfg.best_trace_path,
            cfg.best_replay_path,
            cfg.best_frame_interval,
            cfg.best_frame_limit,
        )

    device = cfg.device
    ssm_cfg = _ssm_cfg_from_train_cfg(cfg)
    online_net = LowLevelDQNPolicy(
        SimpleDQN(
            obs_shape,
            map_feat_dim,
            goal_feat_dim,
            n_actions,
            ssm_cfg=ssm_cfg,
            gru_hidden_size=cfg.gru_hidden_size,
            lstm_hidden_size=cfg.lstm_hidden_size,
        )
    ).to(device)
    target_net = LowLevelDQNPolicy(
        SimpleDQN(
            obs_shape,
            map_feat_dim,
            goal_feat_dim,
            n_actions,
            ssm_cfg=ssm_cfg,
            gru_hidden_size=cfg.gru_hidden_size,
            lstm_hidden_size=cfg.lstm_hidden_size,
        )
    ).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = torch.optim.Adam(online_net.parameters(), lr=cfg.learning_rate)
    replay_buffer = ReplayBuffer(cfg.buffer_size)
    n_step = max(1, cfg.n_step)
    gamma = cfg.gamma
    nstep_buffers = [deque() for _ in range(num_envs)]
    actor_hidden = [online_net.init_hidden(1, device) for _ in range(num_envs)]
    policy_latents: list[torch.Tensor | None] = [None for _ in range(num_envs)]
    visit_counter = VisitCounter(
        num_envs,
        bin_size=cfg.visit_count_bin_size,
        include_story=cfg.visit_count_include_story,
    )
    episodic_memory = (
        EpisodicLatentMemory(
            num_envs,
            max_items=cfg.episodic_memory_size,
            distance_threshold=cfg.episodic_distance_threshold,
        )
        if cfg.episodic_bonus_scale > 0.0
        else None
    )
    rnd_module = None
    rnd_optimizer = None
    rnd_normalizer = None
    dynamic_rnd_scale = cfg.rnd_scale
    if cfg.rnd_enabled and cfg.rnd_scale > 0.0:
        latent_dim = online_net.backbone.hidden_size  # type: ignore[attr-defined]
        rnd_module = RNDModule(latent_dim, hidden_dim=cfg.rnd_hidden_dim).to(device)
        rnd_optimizer = torch.optim.Adam(rnd_module.predictor.parameters(), lr=cfg.rnd_learning_rate)
        rnd_normalizer = RunningRewardNormalizer()
    state_archive = None
    if cfg.state_archive_enabled:
        state_archive = StateArchive(
            cfg.state_archive_dir,
            max_cells=cfg.state_archive_max_cells,
            capture_min_visits=cfg.state_archive_capture_min_visits,
        )

    def rnd_scale_for_step(step: int) -> float:
        base_scale = dynamic_rnd_scale if cfg.posterior_rnd_enabled else cfg.rnd_scale
        if not (rnd_module and base_scale > 0.0):
            return 0.0
        frac = min(1.0, max(0.0, step / max(1, cfg.rnd_anneal_steps)))
        return base_scale * (1.0 - frac)

    def _update_dynamic_rnd_scale_from_payload(payload: dict | None) -> None:
        nonlocal dynamic_rnd_scale
        if not (cfg.posterior_rnd_enabled and payload):
            return
        events = payload.get("events") or payload.get("progress_events") or []
        if not events:
            return
        target = None
        target_name = (cfg.posterior_rnd_event or "").strip().lower()
        if target_name:
            for event in events:
                if str(event.get("name") or "").strip().lower() == target_name:
                    target = event
                    break
        if target is None:
            target = events[0]
        try:
            mean = float(target.get("posterior_mean") or 0.0)
        except (TypeError, ValueError):
            mean = 0.0
        mean = max(0.0, min(1.0, mean))
        lo, hi = cfg.posterior_rnd_bounds
        if hi < lo:
            lo, hi = hi, lo
        dynamic_rnd_scale = lo + (hi - lo) * (1.0 - mean)

    def flush_nstep(idx: int, force: bool = False) -> None:
        queue = nstep_buffers[idx]
        while queue and (force or len(queue) >= n_step):
            reward_sum = 0.0
            steps = 0
            done_final = False
            next_obs_final = queue[0][5]
            next_map_final = queue[0][6]
            next_goal_final = queue[0][7]
            for obs_tuple in list(queue)[:n_step]:
                _, _, _, _, r, nxt_obs, nxt_map, nxt_goal, done_flag = obs_tuple
                reward_sum += (gamma ** steps) * r
                steps += 1
                next_obs_final = nxt_obs
                next_map_final = nxt_map
                next_goal_final = nxt_goal
                if done_flag:
                    done_final = True
                    break
            discount = 0.0 if done_final else gamma ** steps
            first = queue[0]
            replay_buffer.push(
                first[0],
                first[1],
                first[2],
                first[3],
                reward_sum,
                discount,
                next_obs_final,
                next_map_final,
                next_goal_final,
                done_final,
            )
            queue.popleft()
            if not force and len(queue) < n_step:
                break

    global_step = 0
    reward_window = deque(maxlen=25)
    best_reward = -float("inf")
    last_loss = None
    best_model_path = os.path.join(cfg.save_dir, "dqn_route1_best.pt")
    latest_path = os.path.join(cfg.save_dir, "dqn_route1_latest.pt")
    auto_save_seconds = max(60.0, cfg.auto_save_minutes * 60.0)
    last_auto_save = time.monotonic()
    map_refresh = max(1, cfg.map_refresh)
    aggregate_refresh = max(1, cfg.aggregate_map_refresh)
    log_interval = max(1, cfg.log_interval)
    # Treat nonpositive save_every as "do not save on episode cadence".
    episode_save_every = cfg.save_every if cfg.save_every and cfg.save_every > 0 else None

    print(f"Starting training on device {device}. ROM: {cfg.rom_path}")
    progress_payload = None
    try:
        prev_in_battle_flags = [False for _ in range(num_envs)]
        for episode in range(cfg.episodes):
            obs_list = []
            info_list = []
            map_feat_list = []
            goal_feat_list = []
            done = [False] * num_envs
            episode_rewards = [0.0] * num_envs
            env_steps = [0] * num_envs
            episode_reward_components = [defaultdict(float) for _ in range(num_envs)]
            episode_intrinsic_components = [defaultdict(float) for _ in range(num_envs)]
            for idx, env in enumerate(envs):
                seed = cfg.seed + episode * num_envs + idx
                base_state = cfg.state_path
                state_label = None
                archive_entry = None
                archive_desc = None
                if state_archive and cfg.state_archive_reset_prob > 0.0:
                    if random.random() < cfg.state_archive_reset_prob:
                        candidate = state_archive.sample_frontier()
                        if candidate:
                            base_state = candidate.path
                            archive_entry = candidate
                            archive_desc = f"archive:{candidate.label}"
                            state_archive.touch(candidate, global_step=max(1, global_step))
                if archive_entry is None and curriculum_manager:
                    override, state_label = curriculum_manager.assign(
                        idx,
                        unlocked_badges_global,
                        unlocked_story_flags_global,
                        global_step,
                        cfg.verbose_logs,
                    )
                    base_state = override
                underlying = getattr(env, "env", env)
                if hasattr(underlying, "state_path"):
                    underlying.state_path = base_state
                obs, info = env.reset(seed=seed)
                visit_counter.start_episode(idx)
                if episodic_memory:
                    episodic_memory.reset(idx)
                policy_latents[idx] = None
                desc = ""
                if archive_entry is not None:
                    desc = archive_desc or os.path.basename(base_state) if base_state else "archive"
                    if cfg.verbose_logs:
                        print(
                            f"[archive] Episode {episode + 1:04d} env {idx + 1}: loading {desc}"
                        )
                elif curriculum_manager:
                    base_desc = os.path.basename(base_state) if base_state else "power_on"
                    desc = f"{state_label} ({base_desc})" if state_label else base_desc
                    if cfg.verbose_logs:
                        print(
                            f"[curriculum] Episode {episode + 1:04d} env {idx + 1}: loading {desc}"
                        )
                elif base_state:
                    desc = os.path.basename(base_state)
                curriculum_assignments[idx] = desc or "power_on"
                info["curriculum_state"] = curriculum_assignments[idx]
                raw_frame = info.get("raw_frame")
                obs_list.append(preprocess_obs(obs))
                info_list.append(info)
                prev_caught_totals[idx] = int(info.get("pokemon_caught_total") or 0)
                prev_defeated_totals[idx] = int(info.get("pokemon_defeated_total") or 0)
                map_feat, goal_feat = build_map_and_goal_features(info)
                map_feat_list.append(map_feat)
                goal_feat_list.append(goal_feat)
                prev_badge_bits[idx] = int(info.get("badge_bits") or 0)
                prev_badge_sets[idx] = {
                    name for name, unlocked in (info.get("badges") or {}).items() if unlocked
                }
                for badge_name in prev_badge_sets[idx]:
                    unlocked_badges_global.add(badge_name.lower())
                prev_champion_flag[idx] = bool(
                    info.get("champion_flag_raw") or info.get("champion_defeated")
                )
                prev_story_flags[idx] = {
                    name for name, active in (info.get("story_flags") or {}).items() if active
                }
                unlocked_story_flags_global.update(flag.lower() for flag in prev_story_flags[idx])
                prev_in_battle_flags[idx] = bool(info.get("in_battle"))
                progress_logging_ready[idx] = False
                actor_hidden[idx] = online_net.init_hidden(1, device)
                if progress_tracker:
                    progress_tracker.begin_episode(idx, info)
                if best_recorder:
                    best_recorder.reset_episode(idx, episode + 1)
                if cfg.verbose_logs:
                    print(
                        f"[episode {episode + 1:04d}] env {idx + 1}: map={info.get('map_name')} "
                        f"coords={info.get('agent_coords')}",
                        flush=True,
                    )
                if cfg.render_map and cfg.render_gameplay_grid and gameplay_viz is None:
                    frame_shape = raw_frame.shape if isinstance(raw_frame, np.ndarray) else obs.shape
                    try:
                        gameplay_viz = GameplayGridVisualizer(
                            num_envs,
                            frame_shape=frame_shape,
                            target_fps=cfg.gameplay_viz_fps,
                        )
                    except RuntimeError as exc:
                        print(f"[visualization] Disabling gameplay grid: {exc}")
                        gameplay_viz = None
                    except Exception as exc:
                        print(f"[visualization] Failed to initialise gameplay grid: {exc}")
                        gameplay_viz = None
                if map_viz:
                    map_viz.new_episode(idx, episode + 1)
                    if not cfg.persist_map:
                        map_viz.reset(idx)
                    map_viz.update(idx, info, reward=0.0, terminal=False, update_aggregate=True)
                if cfg.render_gameplay_grid and gameplay_viz:
                    gameplay_viz.new_episode(idx, episode + 1)
                    frame = raw_frame if isinstance(raw_frame, np.ndarray) else obs
                    gameplay_viz.update(idx, frame, info=info, reward=0.0, terminal=False)

            for step in range(cfg.max_steps_per_episode):
                active = False
                for idx, env in enumerate(envs):
                    if done[idx]:
                        continue
                    active = True
                    map_feat = map_feat_list[idx]
                    goal_feat = goal_feat_list[idx]
                    epsilon = epsilon_by_step(global_step, cfg)
                    action, new_hidden, latent_vec = select_action_eps(
                        online_net,
                        obs_list[idx],
                        map_feat,
                        goal_feat,
                        epsilon,
                        env.action_space,
                        device,
                        actor_hidden[idx],
                    )
                    actor_hidden[idx] = new_hidden
                    policy_latents[idx] = latent_vec
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    raw_frame = next_info.get("raw_frame")
                    next_obs_proc = preprocess_obs(next_obs)
                    next_map_feat, next_goal_feat = build_map_and_goal_features(next_info)
                    done_flag = bool(terminated or truncated)
                    nstep_buffers[idx].append(
                        (
                            obs_list[idx],
                            map_feat,
                            goal_feat,
                            action,
                            reward,
                            next_obs_proc,
                            next_map_feat,
                            next_goal_feat,
                            done_flag,
                        )
                    )
                    flush_nstep(idx)
                    intrinsic_components = {}
                    if cfg.visit_count_enabled or state_archive:
                        stats = visit_counter.observe(idx, next_info)
                        if cfg.visit_count_enabled:
                            visit_bonus = VisitCounter.intrinsic_bonus(
                                stats.n_global,
                                stats.n_episode,
                                cfg.visit_count_alpha,
                                cfg.visit_count_beta,
                                cfg.visit_count_epsilon,
                            )
                            visit_bonus *= cfg.visit_count_scale
                            if visit_bonus:
                                reward += visit_bonus
                                intrinsic_components["visit"] = visit_bonus
                            if cfg.map_transition_scale > 0.0 and stats.transition_key:
                                transition_bonus = VisitCounter.transition_bonus(
                                    stats.transition_visits,
                                    cfg.map_transition_scale,
                                    cfg.map_transition_max_visits,
                                )
                                if transition_bonus:
                                    reward += transition_bonus
                                    intrinsic_components["transition"] = transition_bonus
                        if (
                            state_archive
                            and stats.n_global <= cfg.state_archive_capture_min_visits
                            and not truncated
                        ):
                            state_archive.maybe_add(
                                stats.cell_id,
                                env,
                                reward_estimate=episode_rewards[idx],
                                global_step=global_step,
                                label=curriculum_assignments[idx],
                                global_visits=stats.n_global,
                            )
                    if episodic_memory and policy_latents[idx] is not None and cfg.episodic_bonus_scale > 0.0:
                        latent_np = policy_latents[idx].detach().cpu().numpy()
                        episodic_bonus = episodic_memory.bonus(idx, latent_np) * cfg.episodic_bonus_scale
                        if episodic_bonus:
                            reward += episodic_bonus
                            intrinsic_components["episodic"] = episodic_bonus
                    if rnd_module and rnd_normalizer and policy_latents[idx] is not None:
                        latent_tensor = policy_latents[idx].unsqueeze(0).to(device)
                        rnd_error = rnd_module(latent_tensor).detach()
                        rnd_normalizer.update(rnd_error)
                        scaled_error = rnd_normalizer.normalize(rnd_error).clamp(min=0.0)
                        rnd_bonus = float((scaled_error * rnd_scale_for_step(global_step)).item())
                        if rnd_bonus:
                            reward += rnd_bonus
                            intrinsic_components["rnd"] = rnd_bonus
                    policy_latents[idx] = None
                    if intrinsic_components:
                        next_info = dict(next_info)
                        next_info["intrinsic_components"] = intrinsic_components
                    obs_list[idx] = next_obs_proc
                    info_list[idx] = next_info
                    map_feat_list[idx] = next_map_feat
                    goal_feat_list[idx] = next_goal_feat
                    comp_list = next_info.get("reward_components") or []
                    if comp_list:
                        comp_totals = episode_reward_components[idx]
                        for comp_name, comp_value in comp_list:
                            try:
                                comp_totals[str(comp_name)] += float(comp_value)
                            except (TypeError, ValueError):
                                continue
                    intrinsic_total_map = next_info.get("intrinsic_components") or {}
                    if intrinsic_total_map:
                        intrinsic_totals = episode_intrinsic_components[idx]
                        for comp_name, comp_value in intrinsic_total_map.items():
                            try:
                                intrinsic_totals[str(comp_name)] += float(comp_value)
                            except (TypeError, ValueError):
                                continue
                    badge_bits = int(next_info.get("badge_bits") or 0)
                    badge_dict = next_info.get("badges") or {}
                    champion_flag_raw = bool(
                        next_info.get("champion_flag_raw") or next_info.get("champion_defeated")
                    )
                    was_in_battle = prev_in_battle_flags[idx]
                    now_in_battle = bool(next_info.get("in_battle"))
                    if cfg.verbose_logs and was_in_battle and not now_in_battle:
                        battle_result = next_info.get("battle_result")
                        battle_code = next_info.get("battle_result_code")
                        enemy_hp_info = next_info.get("enemy_hp") or {}
                        enemy_hp_snapshot = next_info.get("battle_enemy_hp_snapshot") or {}
                        hp_display = enemy_hp_snapshot or enemy_hp_info
                        enemy_hp_current = hp_display.get("current")
                        enemy_hp_max = hp_display.get("max")
                        caught_total = next_info.get("pokemon_caught_total")
                        defeated_total = next_info.get("pokemon_defeated_total")
                        last_outcome = next_info.get("last_battle_outcome")
                        last_won = next_info.get("last_battle_won")
                        last_lost = next_info.get("last_battle_lost")
                        last_blackout = next_info.get("last_battle_blackout")
                        last_fled = next_info.get("last_battle_fled")
                        print(
                            f"[battle] episode {episode + 1:04d} env {idx + 1}: "
                            f"result={battle_result} code=0x{(battle_code or 0):02X} "
                            f"enemy_hp={enemy_hp_current}/{enemy_hp_max} "
                            f"snapshot={enemy_hp_snapshot} "
                            f"last_outcome={last_outcome} won={last_won} "
                            f"lost={last_lost} blackout={last_blackout} fled={last_fled} "
                            f"caught_total={caught_total} defeated_total={defeated_total} "
                            f"step={global_step}",
                            flush=True,
                        )
                    prev_in_battle_flags[idx] = now_in_battle
                    current_story_flags: set[str] = set()
                    if cfg.auto_curriculum_capture:
                        story_flag_map = next_info.get("story_flags") or {}
                        current_story_flags = {
                            name for name, active in story_flag_map.items() if active
                        }
                    episode_rewards[idx] += reward
                    global_step += 1
                    env_steps[idx] += 1
                    if cfg.perf_logging_enabled and perf_log_path:
                        if global_step - perf_last_step >= cfg.perf_log_interval_steps:
                            now = time.monotonic()
                            dt = max(now - perf_last_time, 1e-6)
                            steps_delta = global_step - perf_last_step
                            steps_per_sec = steps_delta / dt
                            _append_csv(
                                perf_log_path,
                                "wall_time,global_step,steps,seconds,steps_per_sec",
                                f"{time.time():.3f},{global_step},{steps_delta},{dt:.6f},{steps_per_sec:.6f}",
                            )
                            if now - perf_console_last_time >= 60.0:
                                print(
                                    f"[perf] global_step={global_step} steps/sec={steps_per_sec:.2f} "
                                    f"(window {steps_delta} steps over {dt:.1f}s)",
                                    flush=True,
                                )
                                perf_console_last_time = now
                            perf_last_step = global_step
                            perf_last_time = now
                    if cfg.reward_likelihood_enabled and reward_like_log_path:
                        reward_like_steps += 1
                        if reward >= cfg.reward_likelihood_threshold:
                            reward_like_hits += 1
                        if global_step % reward_like_interval == 0:
                            prob = reward_like_hits / max(reward_like_steps, 1)
                            _append_csv(
                                reward_like_log_path,
                                "wall_time,global_step,window_steps,threshold_hits,prob",
                                f"{time.time():.3f},{global_step},{reward_like_steps},{reward_like_hits},{prob:.6f}",
                            )
                            reward_like_hits = 0
                            reward_like_steps = 0
                    tracking_ready = progress_logging_ready[idx]
                    if not tracking_ready and env_steps[idx] >= cfg.boot_steps:
                        progress_logging_ready[idx] = True
                        tracking_ready = True
                        prev_badge_bits[idx] = badge_bits
                        prev_badge_sets[idx] = {
                            name for name, unlocked in badge_dict.items() if unlocked
                        }
                        prev_champion_flag[idx] = champion_flag_raw
                        if progress_tracker:
                            progress_tracker.sync_baseline(idx, next_info)
                    if progress_tracker and tracking_ready:
                        progress_tracker.observe(idx, next_info, env_steps[idx])
                    if tracking_ready:
                        if badge_bits != prev_badge_bits[idx]:
                            gained_bits = badge_bits & (~prev_badge_bits[idx])
                            if gained_bits:
                                for bit in range(min(len(BADGE_NAMES), 8)):
                                    if gained_bits & (1 << bit):
                                        badge_name = BADGE_NAMES[bit]
                                        if badge_dict.get(badge_name) and badge_name not in prev_badge_sets[idx]:
                                            print(
                                                f"[progress] step {global_step} env {idx + 1}: acquired {badge_name} badge",
                                                flush=True,
                                            )
                                            prev_badge_sets[idx].add(badge_name)
                                            unlocked_badges_global.add(badge_name.lower())
                                            if badge_name not in captured_badge_events[idx]:
                                                captured_badge_events[idx].add(badge_name)
                                                _capture_event_state(
                                                    idx,
                                                    f"badge_{badge_name}",
                                                    f"{badge_name} badge",
                                                    episode + 1,
                                                    global_step,
                                                    next_info.get("map_name"),
                                                )
                            prev_badge_bits[idx] = badge_bits
                        if champion_flag_raw and not prev_champion_flag[idx]:
                            print(
                                f"[progress] step {global_step} env {idx + 1}: champion flag set",
                                flush=True,
                            )
                            _capture_event_state(
                                idx,
                                "champion_flag",
                                "champion flag",
                                episode + 1,
                                global_step,
                                next_info.get("map_name"),
                            )
                            prev_champion_flag[idx] = champion_flag_raw

                        if cfg.auto_curriculum_capture:
                            new_flags = current_story_flags - prev_story_flags[idx]
                            for flag_name in new_flags:
                                if allowed_story_flags and flag_name.lower() not in allowed_story_flags:
                                    continue
                                if flag_name in captured_story_events[idx]:
                                    continue
                                captured_story_events[idx].add(flag_name)
                                _capture_event_state(
                                    idx,
                                    f"story_{flag_name}",
                                    f"story flag {flag_name}",
                                    episode + 1,
                                    global_step,
                                    next_info.get("map_name"),
                                )
                                unlocked_story_flags_global.add(flag_name.lower())
                            prev_story_flags[idx] = current_story_flags
                    else:
                        if cfg.auto_curriculum_capture:
                            prev_story_flags[idx] = current_story_flags

                    if cfg.auto_curriculum_capture:
                        caught_total = int(next_info.get("pokemon_caught_total") or 0)
                        if caught_total > prev_caught_totals[idx]:
                            _capture_event_state(
                                idx,
                                "event_pokemon_caught",
                                "pokemon caught",
                                episode + 1,
                                global_step,
                                next_info.get("map_name"),
                            )
                        prev_caught_totals[idx] = caught_total

                        defeated_total = int(next_info.get("pokemon_defeated_total") or 0)
                        if defeated_total > prev_defeated_totals[idx]:
                            _capture_event_state(
                                idx,
                                "event_battle_win",
                                "battle victory",
                                episode + 1,
                                global_step,
                                next_info.get("map_name"),
                            )
                        prev_defeated_totals[idx] = defeated_total

                        new_town = next_info.get("new_town_visited")
                        if new_town:
                            label = f"new_town_{_slugify(str(new_town))}"
                            _capture_event_state(
                                idx,
                                label,
                                f"new town {new_town}",
                                episode + 1,
                                global_step,
                                next_info.get("map_name"),
                            )

                        quest_events = next_info.get("quest_events") or []
                        for quest_event in quest_events:
                            quest_label = f"quest_{quest_event}"
                            if quest_label in captured_quest_events[idx]:
                                continue
                            captured_quest_events[idx].add(quest_label)
                            _capture_event_state(
                                idx,
                                quest_label,
                                f"quest {quest_event}",
                                episode + 1,
                                global_step,
                                next_info.get("map_name"),
                            )

                    coords = next_info.get("agent_coords")
                    map_name = next_info.get("map_name")
                    if cfg.verbose_logs and ((env_steps[idx] % log_interval == 0) or done_flag):
                        print(
                            f"[episode {episode + 1:04d}] step {step:05d} "
                            f"env {idx + 1}: reward={reward:+7.3f} "
                            f"map={map_name} coords={coords}",
                            flush=True,
                        )

                    if map_viz:
                        update_map = (env_steps[idx] % map_refresh == 0) or done_flag
                        update_agg = (env_steps[idx] % aggregate_refresh == 0) or done_flag
                        if update_map:
                            map_viz.update(
                                idx,
                                next_info,
                                reward=reward,
                                terminal=done_flag,
                                update_aggregate=update_agg,
                            )

                    if cfg.render_gameplay_grid and gameplay_viz:
                        update_frame = (env_steps[idx] % map_refresh == 0) or done_flag
                        if update_frame:
                            gameplay_viz.update(
                                idx,
                                raw_frame if isinstance(raw_frame, np.ndarray) else next_obs,
                                info=next_info,
                                reward=reward,
                                terminal=done_flag,
                            )
                    if best_recorder:
                        frame_to_record = raw_frame if isinstance(raw_frame, np.ndarray) else None
                        best_recorder.record_step(idx, env_steps[idx], next_info, reward, frame_to_record)
                    if done_flag and progress_tracker:
                        progress_tracker.finish_episode(idx)

                    if (
                        len(replay_buffer) >= cfg.batch_size
                        and global_step >= cfg.learning_starts
                        and global_step % cfg.train_frequency == 0
                    ):
                        batch, weights, indices = replay_buffer.sample(cfg.batch_size)
                        batch_tensors = batch_to_tensors(batch, device)
                        weights_tensor = torch.tensor(
                            weights, dtype=torch.float32, device=device
                        )
                        loss_val, td_errors, latent_batch = compute_td_loss(
                            online_net,
                            target_net,
                            optimizer,
                            batch_tensors,
                            weights_tensor,
                            _schedule_loss_value(
                                cfg.auxiliary_loss_coef,
                                cfg.auxiliary_loss_target,
                                cfg.auxiliary_loss_schedule_steps,
                                global_step,
                            ),
                            _schedule_loss_value(
                                cfg.novelty_loss_coef,
                                cfg.novelty_loss_target,
                                cfg.novelty_loss_schedule_steps,
                                global_step,
                            ),
                        )
                        if rnd_module and rnd_optimizer and latent_batch is not None:
                            rnd_optimizer.zero_grad(set_to_none=True)
                            rnd_loss = rnd_module(latent_batch.to(device)).mean()
                            rnd_loss.backward()
                            torch.nn.utils.clip_grad_norm_(rnd_module.predictor.parameters(), 1.0)
                            rnd_optimizer.step()
                        replay_buffer.update_priorities(
                            indices, td_errors.abs().cpu().numpy()
                        )
                        last_loss = loss_val
                        if cfg.verbose_logs and (global_step % log_interval == 0):
                            print(
                                f"[train] step {global_step:7d} "
                                f"loss={loss_val:8.4f} "
                                f"buffer={len(replay_buffer)} "
                                f"eps={epsilon_by_step(global_step, cfg):.3f}",
                                flush=True,
                            )

                    if (
                        global_step >= cfg.learning_starts
                        and global_step % cfg.target_sync_interval == 0
                    ):
                        target_net.load_state_dict(online_net.state_dict())

                    if done_flag:
                        flush_nstep(idx, force=True)
                        actor_hidden[idx] = online_net.init_hidden(1, device)
                        done[idx] = True
                        if best_recorder:
                            best_recorder.finalize_episode(idx, episode_rewards[idx])

                if all(done) or not active:
                    break
            for idx in range(num_envs):
                flush_nstep(idx, force=True)
            if progress_tracker:
                for idx in range(num_envs):
                    progress_tracker.finish_episode(idx)
                force_progress_log = True

            mean_reward = float(np.mean(episode_rewards))
            reward_window.append(mean_reward)
            if reward_tracker.is_active:
                for idx in range(num_envs):
                    reward_tracker.update(
                        episode_rewards[idx],
                        episode_reward_components[idx],
                        episode_intrinsic_components[idx],
                    )
            if reward_tracker.is_active and ((episode + 1) % cfg.progress_interval == 0):
                payload = reward_tracker.save()
                if payload:
                    summary = reward_tracker.format_summary()
                    if summary:
                        print(summary)
            running_avg = float(np.mean(reward_window))
            epsilon = epsilon_by_step(global_step, cfg)
            loss_val = last_loss if last_loss is not None else float("nan")
            if summary_log_path:
                reward_list_serial = ";".join(f"{r:.6f}" for r in episode_rewards)
                env_steps_serial = ";".join(str(s) for s in env_steps)
                map_names_serial = ";".join(
                    str(info_list[idx].get("map_name", "")) if idx < len(info_list) else ""
                    for idx in range(num_envs)
                )
                curriculum_serial = ";".join(curriculum_assignments)
                _append_csv(
                    summary_log_path,
                    "wall_time,episode,mean_reward,rewards,total_env_steps,global_step,epsilon,loss,num_envs,map_names,env_steps,curriculum_states",
                    f"{time.time():.3f},{episode + 1},{mean_reward:.6f},{reward_list_serial},"
                    f"{sum(env_steps)},{global_step},{epsilon:.6f},{loss_val:.6f},{num_envs},"
                    f"{map_names_serial},{env_steps_serial},{curriculum_serial}",
                )
            rewards_str = ", ".join(f"{r:8.2f}" for r in episode_rewards)
            summary_due = cfg.verbose_logs or ((episode + 1) % cfg.progress_interval == 0) or (episode == 0) or (episode + 1 == cfg.episodes)
            if summary_due:
                print(
                    f"Episode {episode + 1:04d} | Rewards [{rewards_str}] | Mean {mean_reward:8.2f} | "
                    f"Avg(25) {running_avg:8.2f} | Steps {global_step:7d} | "
                    f"Epsilon {epsilon:6.3f} | Loss {loss_val:8.4f}",
                    flush=True,
                )
            if map_viz and cfg.save_map_images and ((episode + 1) % cfg.map_image_every == 0):
                os.makedirs(cfg.save_dir, exist_ok=True)
                map_path = os.path.join(cfg.save_dir, f"dqn_route1_ep{episode + 1:04d}.pt.map.png")
                try:
                    map_viz.save(map_path)
                    print(f"[map] Saved {map_path}")
                except Exception as exc:
                    print(f"[map] Failed to save map image: {exc}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                maybe_save(online_net, best_model_path)

            if episode_save_every and (episode + 1) % episode_save_every == 0:
                checkpoint_path = os.path.join(cfg.save_dir, f"dqn_route1_ep{episode + 1:04d}.pt")
                maybe_save(online_net, checkpoint_path)

            now = time.monotonic()
            if now - last_auto_save >= auto_save_seconds:
                maybe_save(online_net, latest_path)
                last_auto_save = now
                print(
                    f"[checkpoint] Autosaved latest model at episode {episode + 1:04d}"
                )
            if progress_tracker and force_progress_log:
                payload = progress_tracker.save()
                if payload:
                    summary_text = BayesProgressTracker.format_metrics(payload)
                    if summary_text:
                        print(summary_text)
                    _update_dynamic_rnd_scale_from_payload(payload)
                force_progress_log = False
        if progress_tracker:
            progress_payload = progress_tracker.save()
            if progress_payload:
                summary_text = BayesProgressTracker.format_metrics(progress_payload)
                if summary_text:
                    print(summary_text)
                _update_dynamic_rnd_scale_from_payload(progress_payload)
    except KeyboardInterrupt:
        print("\n[training] Interrupted by user. Saving latest checkpoint and shutting down...")
        maybe_save(online_net, latest_path)
        try:
            if 'map_viz' in locals() and map_viz and cfg.save_map_images:
                os.makedirs(cfg.save_dir, exist_ok=True)
                map_path = os.path.join(cfg.save_dir, "route_map_interrupt.png")
                map_viz.save(map_path)
                print(f"[map] Saved interrupt map to {map_path}")
        except Exception:
            pass
        if reward_tracker.is_active:
            reward_tracker.save()
    finally:
        if progress_tracker and progress_payload is None:
            payload = progress_tracker.save()
            if payload:
                summary_text = BayesProgressTracker.format_metrics(payload)
                if summary_text:
                    print(summary_text)
                _update_dynamic_rnd_scale_from_payload(payload)
        if reward_tracker.is_active:
            reward_tracker.save()
        if map_viz:
            map_viz.close()
        if cfg.render_gameplay_grid and gameplay_viz:
            gameplay_viz.close()
        for env in envs:
            env.close()


if __name__ == "__main__":
    arguments = parse_args()
    config = build_config(arguments)
    replay_steps = arguments.replay_max_steps or config.max_steps_per_episode

    if arguments.watch_only:
        run_replay(
            config,
            arguments.replay_checkpoint,
            arguments.replay_episodes,
            replay_steps,
            arguments.save_replay_map,
            show_progress=arguments.show_progress_summary,
        )
    else:
        train(config)
        should_demo = arguments.watch_after_training
        if arguments.auto_demo_after_training is not None:
            should_demo = should_demo or arguments.auto_demo_after_training
        else:
            should_demo = should_demo or config.auto_demo_after_training
        if should_demo:
            demo_episodes = (
                arguments.demo_episodes if arguments.demo_episodes is not None else config.demo_episodes
            )
            demo_steps = (
                arguments.demo_max_steps if arguments.demo_max_steps is not None else config.demo_max_steps
            )
            demo_map_path = (
                arguments.demo_save_map if arguments.demo_save_map is not None else config.demo_save_map
            )
            run_replay(
                config,
                arguments.replay_checkpoint,
                demo_episodes,
                demo_steps,
                demo_map_path,
                show_progress=config.demo_show_progress,
            )
