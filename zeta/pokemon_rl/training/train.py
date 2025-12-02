import os
import json
import argparse
from types import SimpleNamespace

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

from ..envs.epsilon_env import EpsilonEnv
from ..agents.simple_dqn import SimpleDQN
from ..envs.map_features import extract_map_features
from ..agents.select_action import select_action
from .visualization import (
    RouteMapVisualizer,
    MultiRouteMapVisualizer,
    GameplayGridVisualizer,
    KantoAggregateProjector,
)
from .minimal_epsilon_setup import (
    CatchPokemonReward,
    build_reward_modules,
    make_env_instance,
    build_map_and_goal_features,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Simple visible runner with config support")
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "configs", "training_config.json"
        ),
        help="Path to JSON config file (defaults to configs/training_config.json if present)",
    )
    return parser.parse_args()


def _load_cfg(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            print(f"[config] Loaded {path}")
            return cfg
        except Exception as exc:
            print(f"[config] Failed to load {path}: {exc}")
    return {}


def preprocess(obs):
    obs = obs.astype(np.float32) / 255.0
    if obs.ndim == 3:
        obs = np.transpose(obs, (2, 0, 1))
    return obs


def _cfg_namespace(cfg: dict) -> SimpleNamespace:
    return SimpleNamespace(
        rom_path=os.path.join(BASE_DIR, cfg.get("rom", "pokemon_red.gb")),
        headless=bool(cfg.get("headless", False)),
        frame_skip=int(cfg.get("frame_skip", 4)),
        input_spacing_frames=int(cfg.get("input_spacing_frames", 0)),
        use_ssm_encoder=bool(cfg.get("use_ssm_encoder", False)),
        ssm_state_dim=int(cfg.get("ssm_state_dim", 32)),
        ssm_head_dim=int(cfg.get("ssm_head_dim", 32)),
        ssm_heads=int(cfg.get("ssm_heads", 2)),
        ssm_layers=int(cfg.get("ssm_layers", 1)),
        boot_steps=int(cfg.get("boot_steps", 120)),
        max_no_input_frames=int(cfg.get("no_input_timeout", 600)),
        state_path=cfg.get("state_path"),
        story_flags=cfg.get("story_flags"),
        novelty_base=float(cfg.get("novelty_base", 1.0)),
        novelty_decay=float(cfg.get("novelty_decay", 0.9)),
        novelty_min_reward=float(cfg.get("novelty_min_reward", 0.05)),
        novelty_stride=int(cfg.get("novelty_stride", 4)),
        novelty_quantisation=int(cfg.get("novelty_quantisation", 32)),
        novelty_persist=bool(cfg.get("novelty_persist", True)),
        mapexplore_base=float(cfg.get("mapexplore_base", 1.0)),
        mapexplore_neighbor_radius=int(cfg.get("mapexplore_neighbor_radius", 1)),
        mapexplore_neighbor_weight=float(cfg.get("mapexplore_neighbor_weight", 0.15)),
        mapexplore_distance_weight=float(cfg.get("mapexplore_distance_weight", 0.5)),
        mapexplore_min_reward=float(cfg.get("mapexplore_min_reward", 0.05)),
        mapexplore_persist=bool(cfg.get("mapexplore_persist", True)),
        embedding_base=float(cfg.get("embedding_base", 0.6)),
        embedding_decay=float(cfg.get("embedding_decay", 0.92)),
        embedding_min_reward=float(cfg.get("embedding_min_reward", 0.05)),
        embedding_include_map=bool(cfg.get("embedding_include_map", True)),
        embedding_persist=bool(cfg.get("embedding_persist", True)),
        badge_reward=float(cfg.get("badge_reward", 200.0)),
        story_flag_default_reward=float(cfg.get("story_flag_default_reward", 150.0)),
        champion_reward=float(cfg.get("champion_reward", 1000.0)),
        battle_win_reward=float(cfg.get("battle_win_reward", 20.0)),
        battle_loss_penalty=float(cfg.get("battle_loss_penalty", -15.0)),
        battle_damage_scale=float(cfg.get("battle_damage_scale", 6.0)),
        battle_escape_penalty=float(cfg.get("battle_escape_penalty", -12.0)),
        auxiliary_loss_coef=float(cfg.get("auxiliary_loss_coef", 0.05)),
        latent_event_reward=float(cfg.get("latent_event_reward", 25.0)),
        latent_event_revisit_decay=float(cfg.get("latent_event_revisit_decay", 0.5)),
        item_reward=float(cfg.get("item_reward", 2.0)),
        key_item_reward=float(cfg.get("key_item_reward", 15.0)),
        key_item_ids=list(cfg.get("key_item_ids", [])),
        pokedex_new_species_reward=float(cfg.get("pokedex_new_species_reward", 25.0)),
        pokedex_milestones=list(cfg.get("pokedex_milestones", [])),
        trainer_wild_reward=float(cfg.get("trainer_wild_reward", 5.0)),
        trainer_trainer_reward=float(cfg.get("trainer_trainer_reward", 25.0)),
        trainer_gym_reward=float(cfg.get("trainer_gym_reward", 150.0)),
        trainer_elite_reward=float(cfg.get("trainer_elite_reward", 400.0)),
        gym_map_ids=list(cfg.get("gym_map_ids", [])),
        elite_map_ids=list(cfg.get("elite_map_ids", [])),
        quest_definitions=list(cfg.get("quest_definitions", [])),
        frontier_reward=float(cfg.get("frontier_reward", 1.5)),
        frontier_min_gain=int(cfg.get("frontier_min_gain", 2)),
        map_visit_reward=float(cfg.get("map_visit_reward", 8.0)),
        step_penalty=float(cfg.get("step_penalty", -0.0005)),
        idle_penalty=float(cfg.get("idle_penalty", -0.2)),
        idle_threshold=int(cfg.get("idle_threshold", 30)),
        loss_penalty=float(cfg.get("loss_penalty", -30.0)),
        blackout_penalty=float(cfg.get("blackout_penalty", -60.0)),
        low_hp_threshold=float(cfg.get("low_hp_threshold", 0.15)),
        low_hp_penalty=float(cfg.get("low_hp_penalty", -3.0)),
        resource_map_keywords=list(cfg.get("resource_map_keywords", [])),
        resource_map_reward=float(cfg.get("resource_map_reward", 6.0)),
        resource_item_ids=list(cfg.get("resource_item_ids", [])),
        resource_item_reward=float(cfg.get("resource_item_reward", 3.0)),
        curriculum_goals=list(cfg.get("curriculum_goals", [])),
        show_env_maps=bool(cfg.get("show_env_maps", True)),
        progress_interval=int(cfg.get("progress_interval", 10)),
        display_envs=int(cfg.get("display_envs", 1)),
        expose_visit_features=bool(cfg.get("expose_visit_features", True)),
        delete_sav_on_reset=bool(cfg.get("delete_sav_on_reset", True)),
        revisit_penalty_base=float(cfg.get("revisit_penalty_base", 0.02)),
        revisit_penalty_excess=float(cfg.get("revisit_penalty_excess", 0.01)),
        revisit_penalty_ratio=float(cfg.get("revisit_penalty_ratio", 0.015)),
        pallet_penalty_map_id=int(cfg.get("pallet_penalty_map_id", 0)),
        pallet_penalty_interval=int(cfg.get("pallet_penalty_interval", 200)),
        pallet_penalty=float(cfg.get("pallet_penalty", 0.0)),
        exploration_progress_reward=float(cfg.get("exploration_progress_reward", 0.0)),
        stagnation_timeout=int(cfg.get("stagnation_timeout", 800)),
        stagnation_penalty=float(cfg.get("stagnation_penalty", -8.0)),
        stagnation_penalty_interval=int(cfg.get("stagnation_penalty_interval", 120)),
        pallet_penalty_map_ids=list(cfg.get("pallet_penalty_map_ids", [])),
        map_stay_penalties=list(cfg.get("map_stay_penalties", [])),
        render_gameplay_grid=bool(cfg.get("render_gameplay_grid", True)),
    )


def main():
    args = _parse_args()
    cfg = _load_cfg(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config)) if args.config else BASE_DIR

    num_envs = int(cfg.get("num_envs", 1))
    episodes = int(cfg.get("episodes", 500))
    max_steps = int(cfg.get("max_steps", 1000))
    epsilon = float(cfg.get("epsilon_start", 1.0))
    epsilon_min = float(cfg.get("epsilon_end", 0.05))
    epsilon_decay = float(cfg.get("epsilon_decay", 0.995))
    render_map = bool(cfg.get("render_map", False))
    render_gameplay_grid = bool(cfg.get("render_gameplay_grid", True))
    persist_map = bool(cfg.get("persist_map", True))
    map_refresh = int(cfg.get("map_refresh", 4))
    aggregate_refresh = int(cfg.get("aggregate_map_refresh", 8))
    show_env_maps = bool(cfg.get("show_env_maps", True))
    map_fps = float(cfg.get("map_fps", 30.0))
    gameplay_fps = float(cfg.get("gameplay_fps", 24.0))

    cfg_ns = _cfg_namespace(cfg)
    envs = [
        EpsilonEnv(make_env_instance(cfg_ns, idx), build_reward_modules(cfg_ns))
        for idx in range(num_envs)
    ]
    probe_obs, probe_info = envs[0].reset()
    obs_shape = (probe_obs.shape[2], probe_obs.shape[0], probe_obs.shape[1]) if probe_obs.ndim == 3 else probe_obs.shape
    probe_map_feat, probe_goal_feat = build_map_and_goal_features(probe_info)
    map_feat_dim = probe_map_feat.shape[0]
    goal_feat_dim = probe_goal_feat.shape[0]
    n_actions = envs[0].action_space.n
    ssm_cfg = {
        "use_ssm_encoder": cfg_ns.use_ssm_encoder,
        "ssm_state_dim": cfg_ns.ssm_state_dim,
        "ssm_head_dim": cfg_ns.ssm_head_dim,
        "ssm_heads": cfg_ns.ssm_heads,
        "ssm_layers": cfg_ns.ssm_layers,
    }
    model = SimpleDQN(obs_shape, map_feat_dim, goal_feat_dim, n_actions, ssm_cfg=ssm_cfg)
    device = next(model.parameters()).device
    actor_hidden = [model.init_hidden(1, device) for _ in range(num_envs)]

    map_viz = None
    gameplay_viz = None
    layout_path = cfg.get("aggregate_layout_path")
    if layout_path and not os.path.isabs(layout_path):
        layout_path = os.path.normpath(os.path.join(config_dir, layout_path))
    background_path = cfg.get("aggregate_background_path")
    if background_path and not os.path.isabs(background_path):
        background_path = os.path.normpath(os.path.join(config_dir, background_path))
    aggregate_projector = None
    if render_map and layout_path:
        try:
            aggregate_projector = KantoAggregateProjector(
                layout_path,
                background_path=background_path,
                tile_scale=int(cfg.get("aggregate_tile_scale", 16)),
                pad_tiles=int(cfg.get("aggregate_pad_tiles", 12)),
                overlay_alpha=float(cfg.get("aggregate_overlay_alpha", 0.72)),
            )
        except Exception as exc:
            print(f"[visualization] Aggregate overlay disabled: {exc}")
            aggregate_projector = None
    if render_map:
        try:
            map_viz = MultiRouteMapVisualizer(
                num_envs,
                show_env_panels=show_env_maps,
                target_fps=map_fps,
                aggregate_projector=aggregate_projector,
            )
        except RuntimeError as exc:
            print(f"[visualization] Map view unavailable: {exc}")
            map_viz = None

    try:
        for ep in range(episodes):
            obs = []
            info = []
            map_feats = []
            goal_feats = []
            for idx, env in enumerate(envs):
                o, i = env.reset()
                raw_frame = i.get("raw_frame")
                obs.append(preprocess(o))
                info.append(i)
                map_feat, goal_feat = build_map_and_goal_features(i)
                map_feats.append(map_feat)
                goal_feats.append(goal_feat)
                actor_hidden[idx] = model.init_hidden(1, device)
                if render_map and render_gameplay_grid and gameplay_viz is None:
                    frame_shape = raw_frame.shape if isinstance(raw_frame, np.ndarray) else o.shape
                    try:
                        gameplay_viz = GameplayGridVisualizer(
                            num_envs,
                            frame_shape=frame_shape,
                            target_fps=gameplay_fps,
                        )
                    except RuntimeError as exc:
                        print(f"[visualization] Gameplay grid unavailable: {exc}")
                        gameplay_viz = None
                    except Exception as exc:
                        print(f"[visualization] Failed to initialise gameplay grid: {exc}")
                        gameplay_viz = None
                if render_map and map_viz:
                    map_viz.new_episode(idx, ep + 1)
                    if not persist_map:
                        map_viz.reset(idx)
                    map_viz.update(idx, i, reward=0.0, terminal=False, update_aggregate=True)
                if render_map and render_gameplay_grid and gameplay_viz:
                    gameplay_viz.new_episode(idx, ep + 1)
                    frame = raw_frame if isinstance(raw_frame, np.ndarray) else o
                    gameplay_viz.update(idx, frame, info=i, reward=0.0, terminal=False)
            done = [False] * num_envs
            ep_rewards = [0.0] * num_envs
            env_steps = [0] * num_envs
            for t in range(max_steps):
                active = False
                for idx, env in enumerate(envs):
                    if done[idx]:
                        continue
                    active = True
                    action, next_hidden = select_action(
                        model,
                        obs[idx],
                        map_feats[idx],
                        goal_feats[idx],
                        epsilon,
                        env.action_space,
                        device=device,
                        hidden_state=actor_hidden[idx],
                    )
                    actor_hidden[idx] = next_hidden
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    raw_frame = next_info.get("raw_frame")
                    obs[idx] = preprocess(next_obs)
                    info[idx] = next_info
                    map_feat, goal_feat = build_map_and_goal_features(next_info)
                    map_feats[idx] = map_feat
                    goal_feats[idx] = goal_feat
                    ep_rewards[idx] += reward
                    env_steps[idx] += 1
                    if render_map and map_viz:
                        update_map = (env_steps[idx] % max(1, map_refresh) == 0) or terminated or truncated
                        update_agg = (env_steps[idx] % max(1, aggregate_refresh) == 0) or terminated or truncated
                        if update_map:
                            map_viz.update(
                                idx,
                                next_info,
                                reward=reward,
                                terminal=terminated or truncated,
                                update_aggregate=update_agg,
                            )
                    if render_map and render_gameplay_grid and gameplay_viz:
                        update_frame = (env_steps[idx] % max(1, map_refresh) == 0) or terminated or truncated
                        if update_frame:
                            gameplay_viz.update(
                                idx,
                                raw_frame if isinstance(raw_frame, np.ndarray) else next_obs,
                                info=next_info,
                                reward=reward,
                                terminal=terminated or truncated,
                            )
                    if terminated or truncated:
                        actor_hidden[idx] = model.init_hidden(1, device)
                        done[idx] = True
                if all(done) or not active:
                    break
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            rewards_str = ", ".join(f"{r:.1f}" for r in ep_rewards)
            print(f"Episode {ep} | Rewards [{rewards_str}] | Mean {np.mean(ep_rewards):.2f} | Epsilon: {epsilon:.3f}")
            if render_map and map_viz and bool(cfg.get("save_map_images", True)) and ((ep + 1) % int(cfg.get("map_image_every", 10)) == 0):
                save_dir = cfg.get("save_dir", "checkpoints")
                os.makedirs(save_dir, exist_ok=True)
                map_path = os.path.join(save_dir, f"route_map_ep{ep + 1:04d}.png")
                try:
                    map_viz.save(map_path)
                    print(f"[map] Saved {map_path}")
                except Exception as exc:
                    print(f"[map] Failed to save map image: {exc}")
    except KeyboardInterrupt:
        print("\n[train] Interrupted by user. Cleaning up...")
        try:
            if render_map and map_viz and bool(cfg.get("save_map_images", True)):
                save_dir = cfg.get("save_dir", "checkpoints")
                os.makedirs(save_dir, exist_ok=True)
                map_path = os.path.join(save_dir, "route_map_interrupt.png")
                map_viz.save(map_path)
                print(f"[map] Saved interrupt map to {map_path}")
        except Exception:
            pass
    finally:
        for env in envs:
            env.close()
        if render_map and map_viz:
            map_viz.close()
        if render_map and render_gameplay_grid and gameplay_viz:
            gameplay_viz.close()


if __name__ == "__main__":
    main()
