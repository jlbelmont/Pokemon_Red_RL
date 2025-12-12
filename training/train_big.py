from __future__ import annotations

import argparse
import atexit
import csv
import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import torch

from env.interface import load_env_config, make_env, project_root
from env.wrappers import SimpleVectorEnv
from mega_model.agent import AgentConfig, SlimHierarchicalDQN
from mega_model.curriculum import CurriculumManager
from mega_model.flags import FlagEncoder
from mega_model.logging_utils import PerformanceTracker, RewardLogger
from mega_model.video_utils import GAME_HEIGHT, GAME_WIDTH, maybe_save_video
from training.utils import (
    apply_config,
    cleanup_shared_memory,
    obs_to_dict,
    resolve_run_dir,
    set_seed,
    setup_logging,
    stack_frames,
    write_config,
)

# Model input resolution (downsampled for training; videos remain native 160x144)
MODEL_WIDTH = GAME_WIDTH // 2  # 80
MODEL_HEIGHT = GAME_HEIGHT // 2  # 72

atexit.register(cleanup_shared_memory)


MODEL_PRESETS = {
    "slim": {"cnn_channels": (16, 32, 64), "gru": 128, "lstm": 128, "ssm": 128, "structured": 128},
    "large": {"cnn_channels": (64, 128, 256), "gru": 512, "lstm": 512, "ssm": 512, "structured": 512},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaled training for hierarchical DQN.")
    parser.add_argument("--config", type=str, default=None, help="YAML file with argument overrides.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=4_000)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="big_run")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rnd-weight", type=float, default=0.1)
    parser.add_argument("--novelty-weight", type=float, default=0.05)
    parser.add_argument("--bayes-weight", type=float, default=0.05)
    parser.add_argument("--save-state-dir", type=str, default="savestates")
    parser.add_argument("--save-state-every", type=int, default=0)
    parser.add_argument("--save-state-on-bayes", action="store_true")
    parser.add_argument("--record-video-every", type=int, default=0)
    parser.add_argument("--video-interval", type=int, default=0)
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--prune-interval", type=int, default=0)
    parser.add_argument("--map-save-interval", type=int, default=0)
    parser.add_argument("--model-size", choices=list(MODEL_PRESETS.keys()), default="large")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-min", type=float, default=None)
    parser.set_defaults(headless=True)
    return parser.parse_args()


def _list_checkpoints(run_dir: Path) -> List[Path]:
    ckpt_dir = run_dir / "checkpoints"
    candidates = list(ckpt_dir.glob("checkpoint_step*.pth"))
    candidates += list(run_dir.glob("checkpoint_step*.pth"))

    def _step_num(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.split("checkpoint_step")[-1])
        except Exception:
            return -1

    return sorted(candidates, key=_step_num, reverse=True)


def _load_checkpoint(agent, checkpoint_path: Path, device: torch.device) -> int:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    with torch.no_grad():
        dummy_frames = torch.zeros(1, agent.config.frame_stack, MODEL_HEIGHT, MODEL_WIDTH, device=device)
        dummy_structured = None
        if agent.config.structured_dim > 0:
            dummy_structured = torch.zeros(1, agent.config.structured_dim, device=device)
        _ = agent.network(
            dummy_frames,
            structured=dummy_structured,
            state=agent.network.initial_state(batch_size=1, device=device),
            done=None,
        )
        _ = agent.target_network(
            dummy_frames,
            structured=dummy_structured,
            state=agent.target_network.initial_state(batch_size=1, device=device),
            done=None,
        )

    def _filtered_load(module, saved_state):
        current = module.state_dict()
        filtered = {}
        skipped = []
        for k, v in saved_state.items():
            if k in current and current[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        if skipped:
            logging.warning("[resume] Skipped %d keys with shape mismatch: %s", len(skipped), skipped[:4])
        module.load_state_dict(filtered, strict=False)

    ckpt_gru = ckpt.get("network_state_dict", {}).get("gru.weight_ih", None)
    curr_gru = agent.network.gru.weight_ih if agent.network.gru is not None else None  # type: ignore[attr-defined]
    if ckpt_gru is not None and curr_gru is not None and ckpt_gru.shape[1] != curr_gru.shape[1]:
        logging.warning(
            "[resume] GRU input mismatch (ckpt %s vs current %s). Skipping checkpoint restore.",
            ckpt_gru.shape,
            curr_gru.shape,
        )
        return 0

    _filtered_load(agent.network, ckpt["network_state_dict"])
    _filtered_load(agent.target_network, ckpt["target_network_state_dict"])
    agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    agent.rnd.load_state_dict(ckpt["rnd_state_dict"])
    agent.rnd_opt.load_state_dict(ckpt["rnd_opt_state_dict"])
    agent.epsilon = ckpt.get("epsilon", agent.epsilon)
    agent.global_step = ckpt.get("global_step", 0)
    if "replay_buffer" in ckpt:
        agent.replay.buffer = ckpt["replay_buffer"].get("buffer", [])
        agent.replay.pos = ckpt["replay_buffer"].get("pos", 0)
    return ckpt.get("step", 0)


def main() -> None:
    args = parse_args()
    args = apply_config(args, args.config)
    setup_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    preset = MODEL_PRESETS[args.model_size]

    run_dir = resolve_run_dir(args.log_dir, args.run_name)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir) if args.video_dir else run_dir.parent / "videos"
    if not video_dir.is_absolute():
        video_dir = project_root() / video_dir

    write_config(run_dir, vars(args))
    reward_logger = RewardLogger(log_dir=run_dir)
    perf = PerformanceTracker()

    positions_path = run_dir / "positions.csv"
    positions_file = open(positions_path, "w", newline="")
    atexit.register(positions_file.close)
    pos_writer = csv.DictWriter(
        positions_file,
        fieldnames=[
            "step_global",
            "env",
            "episode",
            "map_id",
            "coord",
            "badge",
            "deaths",
            "reward",
            "entropy",
            "map_steps",
        ],
    )
    pos_writer.writeheader()

    env_cfg = load_env_config(headless=args.headless, reduce_res=True, two_bit=False, use_rubinstein_reward=True)
    save_state_root = Path(args.save_state_dir)
    if not save_state_root.is_absolute():
        save_state_root = project_root() / save_state_root
    save_state_root.mkdir(parents=True, exist_ok=True)
    curriculum = CurriculumManager(base_dir=save_state_root, run_name=args.run_name)

    def _bootstrap_init_states() -> None:
        if curriculum.list_states():
            return
        repo_root = project_root()
        candidates = [
            repo_root / "env" / "assets" / "pyboy_states" / "Bulbasaur.state",
            repo_root / "env" / "pokemonred_puffer" / "pyboy_states" / "Bulbasaur.state",
        ]
        for c in candidates:
            if c.exists():
                dest = curriculum.run_dir / c.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(c.read_bytes())
                curriculum.register_state("pallet_start", dest, tier="early")
                break

    _bootstrap_init_states()
    env_cfg["state_dir"] = str(curriculum.run_dir)

    def env_fn(idx: int):
        return make_env(env_cfg, seed=args.seed + idx)

    envs = SimpleVectorEnv([lambda idx=i: env_fn(idx) for i in range(args.num_envs)])
    current_state_tags: List[Optional[str]] = [None for _ in range(args.num_envs)]

    def _sample_reset_options() -> list:
        opts: list = []
        for env_idx in range(args.num_envs):
            sample = curriculum.sample_state()
            if sample:
                tag, path = sample
                current_state_tags[env_idx] = tag
                try:
                    opts.append({"state": path.read_bytes()})
                    continue
                except Exception:
                    pass
            current_state_tags[env_idx] = None
            opts.append(None)
        return opts

    obs, _ = envs.reset(seed=args.seed, options=_sample_reset_options())
    obs = obs_to_dict(obs)

    num_actions = envs.single_action_space.n
    frame_stacks: List[Deque[np.ndarray]] = [deque(maxlen=args.frame_stack) for _ in range(args.num_envs)]
    for i in range(args.num_envs):
        for _ in range(args.frame_stack):
            frame_stacks[i].append(obs["screen"][i])

    flag_encoder = FlagEncoder()
    structured_dim = flag_encoder.output_dim

    agent_cfg = AgentConfig(
        num_actions=num_actions,
        frame_stack=args.frame_stack,
        structured_dim=structured_dim,
        replay_capacity=args.buffer_size,
        target_update_interval=args.target_update,
        rnd_weight=args.rnd_weight,
        novelty_weight=args.novelty_weight,
        bayes_weight=args.bayes_weight,
    )
    agent = SlimHierarchicalDQN(
        agent_cfg,
        device=device,
        cnn_channels=preset["cnn_channels"],
        gru_size=preset["gru"],
        lstm_size=preset["lstm"],
        ssm_size=preset["ssm"],
        structured_hidden=preset["structured"],
    )
    agent.flag_encoder = flag_encoder
    if args.epsilon_min is not None:
        agent.epsilon_min = args.epsilon_min
    agent.epsilon = 1.0
    agent.epsilon_decay = max(0.0, (agent.epsilon - agent.epsilon_min) / max(1, args.total_steps))

    start_step = 0
    if args.resume:
        loaded = False
        for ckpt_path in _list_checkpoints(run_dir):
            try:
                start_step = _load_checkpoint(agent, ckpt_path, device)
                logging.info("[resume] Loaded %s at step %d", ckpt_path, start_step)
                loaded = True
                break
            except Exception as e:
                logging.warning("[resume] Failed to load %s (%s)", ckpt_path, e)
                continue
        if not loaded:
            logging.warning("[resume] No checkpoint could be loaded in %s", run_dir / "checkpoints")
    else:
        agent.global_step = 0

    if args.epsilon_start is not None:
        agent.epsilon = args.epsilon_start
        agent.epsilon_decay = max(0.0, (agent.epsilon - agent.epsilon_min) / max(1, args.total_steps))

    states = agent.reset_hidden(args.num_envs)
    dones = np.zeros(args.num_envs, dtype=bool)
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_rnd = np.zeros(args.num_envs, dtype=np.float32)
    episode_novel = np.zeros(args.num_envs, dtype=np.float32)
    episode_bayes = np.zeros(args.num_envs, dtype=np.float32)
    episode_total = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    episode_ids = np.zeros(args.num_envs, dtype=np.int64)
    last_badges = np.zeros(args.num_envs, dtype=np.int32)
    last_map = np.full(args.num_envs, -1, dtype=np.int32)
    map_steps = np.zeros(args.num_envs, dtype=np.int64)

    recent_lengths: Deque[float] = deque(maxlen=200)
    recent_env: Deque[float] = deque(maxlen=200)
    recent_total: Deque[float] = deque(maxlen=200)
    recent_r_env: Deque[float] = deque(maxlen=200)
    recent_r_rnd: Deque[float] = deque(maxlen=200)
    recent_r_novel: Deque[float] = deque(maxlen=200)
    recent_r_bayes: Deque[float] = deque(maxlen=200)

    start_time = time.time()
    video_every = args.video_interval if args.video_interval else args.record_video_every
    frames_for_video: List[np.ndarray] = []
    video_saved = False

    for step in range(start_step, args.total_steps):
        frame_batch = torch.as_tensor(
            stack_frames(frame_stacks, target_hw=(MODEL_HEIGHT, MODEL_WIDTH)),
            device=device,
        )
        obs_tensors = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
        structured, milestone_flags = agent.flag_encoder(obs_tensors)
        done_mask = torch.as_tensor(dones, device=device, dtype=torch.float32)

        if video_every:
            frames_for_video.append(obs["screen"][0])

        actions, new_states, aux = agent.act(frame_batch, structured, states, done_mask)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
        next_obs = obs_to_dict(next_obs)
        dones = np.logical_or(terminated, truncated)
        done_flags = dones.copy()
        episode_returns += rewards
        episode_lengths += 1
        if args.max_episode_steps and args.max_episode_steps > 0:
            timed_out = episode_lengths >= args.max_episode_steps
            done_flags = np.logical_or(done_flags, timed_out)
            dones = np.logical_or(dones, timed_out)

        intrinsic = agent.intrinsic_rewards(aux, milestone_flags, env_ids=range(args.num_envs))
        r_env = torch.as_tensor(rewards, device=device, dtype=torch.float32)
        total_reward = (
            r_env
            + args.rnd_weight * intrinsic["rnd"]
            + args.novelty_weight * intrinsic["novel"]
            + args.bayes_weight * intrinsic["bayes"]
        )
        rnd_np = intrinsic["rnd"].detach().cpu().numpy()
        rnd_raw_np = intrinsic["rnd_raw"].detach().cpu().numpy() if "rnd_raw" in intrinsic else rnd_np
        novel_np = intrinsic["novel"].detach().cpu().numpy()
        bayes_np = intrinsic["bayes"].detach().cpu().numpy()
        total_np = total_reward.detach().cpu().numpy()
        episode_rnd += rnd_np
        episode_novel += novel_np
        episode_bayes += bayes_np
        episode_total += total_np

        perf.step(args.num_envs)

        for env_idx, done in enumerate(done_flags):
            info = infos[env_idx] if isinstance(infos, (list, tuple)) else {}
            stats = info.get("stats", {}) if isinstance(info, dict) else {}
            hist = stats.get("action_hist", None)
            entropy = None
            if hist is not None:
                hist = np.array(hist, dtype=np.float32)
                total = hist.sum()
                if total > 0:
                    p = hist / total
                    entropy = float(-(p * np.log(p + 1e-8)).sum())

            map_id_for_log = stats.get("map_id", -1)
            reward_logger.log_step(
                step_global=step,
                env_id=env_idx,
                episode_id=int(episode_ids[env_idx]),
                rewards={
                    "env": rewards[env_idx],
                    "rnd": rnd_np[env_idx],
                    "novel": novel_np[env_idx],
                    "bayes": bayes_np[env_idx],
                    "total": total_np[env_idx],
                },
                weights={"rnd": args.rnd_weight, "novel": args.novelty_weight, "bayes": args.bayes_weight},
                milestone_flags=milestone_flags[env_idx].detach().cpu().numpy().tolist(),
                map_id=map_id_for_log,
                posterior_mean=float(intrinsic["posterior_mean"][env_idx].detach().cpu().item()),
                rnd_scale=float(intrinsic["bayes_scale"][env_idx].detach().cpu().item()),
                rnd_raw=float(rnd_raw_np[env_idx]),
            )

            pos_writer.writerow(
                {
                    "step_global": step,
                    "env": env_idx,
                    "episode": int(episode_ids[env_idx]),
                    "map_id": stats.get("map_id", 0),
                    "coord": stats.get("coord", 0),
                    "badge": stats.get("badge", 0),
                    "deaths": stats.get("deaths", 0),
                    "reward": float(rewards[env_idx]),
                    "entropy": entropy if entropy is not None else "",
                    "map_steps": int(map_steps[env_idx]),
                }
            )

            raw_map = stats.get("map_id", -1)
            try:
                map_id = int(raw_map)
            except Exception:
                map_id = -1
            if map_id >= 0 and map_id != last_map[env_idx]:
                tag = f"map_{map_id:03d}"
                path = curriculum.maybe_save_milestone(envs.envs[env_idx], tag, step + 1)
                if path:
                    current_state_tags[env_idx] = tag
                last_map[env_idx] = map_id
                map_steps[env_idx] = 0
            else:
                map_steps[env_idx] += 1
                if args.map_save_interval and args.map_save_interval > 0:
                    if map_steps[env_idx] % args.map_save_interval == 0 and map_id >= 0:
                        tag = f"map_{map_id:03d}_step{map_steps[env_idx]:06d}_g{step+1:07d}"
                        save_path = curriculum.maybe_save_milestone(envs.envs[env_idx], tag, step + 1)
                        if save_path:
                            current_state_tags[env_idx] = tag

            if done:
                perf.episode()
                curriculum.record_outcome(current_state_tags[env_idx], float(episode_total[env_idx]))
                bayes_updates = agent.update_bayes_from_milestones(
                    milestone_flags[env_idx].detach().cpu()
                )
                reward_logger.log_episode(
                    {
                        "episode_id": int(episode_ids[env_idx]),
                        "env_id": env_idx,
                        "length_steps": int(episode_lengths[env_idx]),
                        "return_env": float(episode_returns[env_idx]),
                        "return_rnd": float(episode_rnd[env_idx]),
                        "return_novel": float(episode_novel[env_idx]),
                        "return_bayes": float(episode_bayes[env_idx]),
                        "return_total_with_intrinsic": float(episode_total[env_idx]),
                        "milestones_reached": ";".join(
                            [k for k, v in bayes_updates.items() if v.get("updated", False)]
                        ),
                    }
                )
                for k, v in bayes_updates.items():
                    if {"alpha", "beta"} <= set(v.keys()):
                        reward_logger.log_progress(k, v["alpha"], v["beta"], v.get("alarm_score", 0.0))

                recent_lengths.append(float(episode_lengths[env_idx]))
                recent_env.append(float(episode_returns[env_idx]))
                recent_total.append(float(episode_total[env_idx]))
                recent_r_env.append(float(np.mean(r_env.cpu().numpy())))
                recent_r_rnd.append(float(np.mean(rnd_np)))
                recent_r_novel.append(float(np.mean(novel_np)))
                recent_r_bayes.append(float(np.mean(bayes_np)))

                badge_count = stats.get("badge", 0)
                if badge_count > last_badges[env_idx]:
                    tag = f"badge_{badge_count}"
                    save_path = curriculum.maybe_save_milestone(envs.envs[env_idx], tag, step + 1)
                    if save_path:
                        current_state_tags[env_idx] = tag
                    last_badges[env_idx] = badge_count

                logging.info(
                    "[EP %d env=%d] len=%d R_env=%.2f R_total=%.2f badges=%s map=%s",
                    episode_ids[env_idx],
                    env_idx,
                    episode_lengths[env_idx],
                    episode_returns[env_idx],
                    episode_total[env_idx],
                    badge_count,
                    stats.get("map_id", -1),
                )
                episode_ids[env_idx] += 1
                episode_returns[env_idx] = 0.0
                episode_rnd[env_idx] = 0.0
                episode_novel[env_idx] = 0.0
                episode_bayes[env_idx] = 0.0
                episode_total[env_idx] = 0.0
                episode_lengths[env_idx] = 0
                agent.reset_episode(env_idx)
                map_steps[env_idx] = 0
                sample = curriculum.sample_state()
                reset_opts = None
                if sample:
                    tag, path = sample
                    current_state_tags[env_idx] = tag
                    try:
                        reset_opts = {"state": path.read_bytes()}
                    except Exception:
                        reset_opts = None
                else:
                    current_state_tags[env_idx] = None
                if args.save_state_every and (episode_ids[env_idx] % args.save_state_every == 0):
                    save_path = curriculum.run_dir / f"ep{episode_ids[env_idx]:05d}_env{env_idx}.state"
                    envs.envs[env_idx]._save_emulator_state(save_path)
                    curriculum.register_state(f"ep{episode_ids[env_idx]:05d}", save_path, tier="mid")
                reset_obs, _ = envs.envs[env_idx].reset(seed=args.seed + env_idx, options=reset_opts)
                reset_obs = obs_to_dict(reset_obs)
                for key in next_obs:
                    next_obs[key][env_idx] = reset_obs[key]
                frame_stacks[env_idx].clear()
                for _ in range(args.frame_stack):
                    frame_stacks[env_idx].append(reset_obs["screen"])
                new_states[env_idx] = agent.network.initial_state(batch_size=1, device=device)
                dones[env_idx] = False
                if video_every and env_idx == 0 and episode_ids[env_idx] % video_every == 0:
                    video_dir.mkdir(parents=True, exist_ok=True)
                    out_path = video_dir / f"train_ep{episode_ids[env_idx]:05d}.mp4"
                    if frames_for_video:
                        artifacts = maybe_save_video(frames_for_video, "mp4", out_path, fps=15)
                        for art in artifacts:
                            logging.info("[video] saved %s (%d frames)", art["path"], art["frames"])
                            video_saved = True
                    frames_for_video.clear()
                elif env_idx == 0:
                    frames_for_video.clear()
            else:
                frame_stacks[env_idx].append(next_obs["screen"][env_idx])

        obs = next_obs
        states = new_states

        next_frame_batch = torch.as_tensor(
            stack_frames(frame_stacks, target_hw=(MODEL_HEIGHT, MODEL_WIDTH)),
            device=device,
        )
        next_obs_tensors = {k: torch.as_tensor(v, device=device) for k, v in next_obs.items()}
        next_structured, _ = agent.flag_encoder(next_obs_tensors)

        agent.add_transition(
            frame_batch,
            structured,
            actions,
            total_reward,
            {"env": r_env, "rnd": intrinsic["rnd"], "novel": intrinsic["novel"], "bayes": intrinsic["bayes"]},
            torch.as_tensor(done_flags, device=device, dtype=torch.float32),
            next_frame_batch,
            next_structured,
            states,
        )
        if step >= args.learning_starts and step % args.train_every == 0:
            agent.learn(args.batch_size)

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = (step + 1) * args.num_envs / max(elapsed, 1e-6)
            avg_len = np.mean(recent_lengths) if recent_lengths else 0.0
            avg_env = np.mean(recent_env) if recent_env else 0.0
            avg_total = np.mean(recent_total) if recent_total else 0.0
            mean_r_env = np.mean(recent_r_env) if recent_r_env else 0.0
            mean_r_rnd = np.mean(recent_r_rnd) if recent_r_rnd else 0.0
            mean_r_novel = np.mean(recent_r_novel) if recent_r_novel else 0.0
            mean_r_bayes = np.mean(recent_r_bayes) if recent_r_bayes else 0.0
            logging.info(
                "[step %d/%d] SPS=%.1f eps=%.3f avg_len=%.1f avg_env=%.2f avg_total=%.2f | r_env=%.3f r_rnd=%.3f r_novel=%.3f r_bayes=%.3f | episodes=%d envs=%d | %s",
                step + 1,
                args.total_steps,
                sps,
                agent.epsilon,
                avg_len,
                avg_env,
                avg_total,
                mean_r_env,
                mean_r_rnd,
                mean_r_novel,
                mean_r_bayes,
                int(np.sum(episode_ids)),
                args.num_envs,
                curriculum.summary(),
            )

        if args.eval_interval and (step + 1) % args.eval_interval == 0:
            logging.info("[eval] eval loop not implemented; interval=%d", args.eval_interval)

        if args.save_interval and (step + 1) % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"checkpoint_step{step+1}.pth"
            torch.save(
                {
                    "network_state_dict": agent.network.state_dict(),
                    "target_network_state_dict": agent.target_network.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "rnd_state_dict": agent.rnd.state_dict(),
                    "rnd_opt_state_dict": agent.rnd_opt.state_dict(),
                    "epsilon": agent.epsilon,
                    "global_step": agent.global_step,
                    "episode_ids": episode_ids.tolist(),
                    "step": step + 1,
                },
                ckpt_path,
            )
            logging.info("[ckpt] saved %s", ckpt_path)

        if args.prune_interval and (step + 1) % args.prune_interval == 0:
            before = curriculum.summary()
            curriculum.promote_or_demote()
            after = curriculum.summary()
            logging.info("[prune] %s -> %s", before, after)

    if video_every and frames_for_video and not video_saved:
        video_dir.mkdir(parents=True, exist_ok=True)
        base = video_dir / f"{args.run_name}_final"
        out_path = base.with_suffix(".mp4")
        suffix = 1
        while out_path.exists():
            out_path = base.with_name(f"{base.stem}_{suffix}").with_suffix(".mp4")
            suffix += 1
        artifacts = maybe_save_video(frames_for_video, "mp4", out_path, fps=15)
        for art in artifacts:
            logging.info("[video] saved %s (%d frames)", art["path"], art["frames"])

    final_state_path = curriculum.run_dir / f"final_step{args.total_steps:07d}.state"
    try:
        envs.envs[0]._save_emulator_state(final_state_path)
        curriculum.register_state(f"final_step{args.total_steps}", final_state_path, tier="mid")
        logging.info("[savestate] saved final state to %s", final_state_path)
    except Exception as e:
        logging.warning("[savestate] failed to save final state: %s", e)

    final_ckpt_path = ckpt_dir / f"checkpoint_step{args.total_steps}_full.pth"
    try:
        torch.save(
            {
                "network_state_dict": agent.network.state_dict(),
                "target_network_state_dict": agent.target_network.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "rnd_state_dict": agent.rnd.state_dict(),
                "rnd_opt_state_dict": agent.rnd_opt.state_dict(),
                "epsilon": agent.epsilon,
                "global_step": agent.global_step,
                "episode_ids": episode_ids.tolist(),
                "replay_buffer": {"buffer": agent.replay.buffer, "pos": agent.replay.pos},
                "step": args.total_steps,
            },
            final_ckpt_path,
        )
        logging.info("[ckpt] saved final checkpoint with buffer %s", final_ckpt_path)
    except Exception as e:
        logging.warning("[ckpt] failed to save final checkpoint with buffer: %s", e)

    reward_logger.flush_steps()
    reward_logger.flush_episodes()
    positions_file.flush()


if __name__ == "__main__":
    main()
