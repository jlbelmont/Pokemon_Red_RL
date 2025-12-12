from __future__ import annotations

import argparse
import atexit
import csv
import json
import logging
import signal
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
from mega_model.streaming import BestAgentSelector, StreamClient, progress_score
from mega_model.video_utils import maybe_save_video
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

_shutdown_requested = False
atexit.register(cleanup_shared_memory)


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[train] Shutdown signal received, will save checkpoint and exit...")


def _save_checkpoint(
    agent: SlimHierarchicalDQN,
    step: int,
    episode_ids: np.ndarray,
    checkpoint_dir: Path,
    progress_path: Path,
) -> str:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{step:08d}.pt"
    torch.save(
        {
            "step": step,
            "episode_ids": episode_ids.tolist(),
            "network_state_dict": agent.network.state_dict(),
            "target_network_state_dict": agent.target_network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "rnd_state_dict": agent.rnd.state_dict(),
            "rnd_opt_state_dict": agent.rnd_opt.state_dict(),
            "epsilon": agent.epsilon,
            "global_step": agent.global_step,
        },
        checkpoint_path,
    )
    progress_data = {
        "env_steps": step,
        "episodes": int(episode_ids.sum()),
        "last_checkpoint": str(checkpoint_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    progress_path.write_text(json.dumps(progress_data, indent=2))
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(torch.load(checkpoint_path), latest_path)
    return str(checkpoint_path)


def _load_checkpoint(agent: SlimHierarchicalDQN, checkpoint_path: Path, device: torch.device) -> tuple[int, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.network.load_state_dict(checkpoint["network_state_dict"])
    agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.rnd.load_state_dict(checkpoint["rnd_state_dict"])
    agent.rnd_opt.load_state_dict(checkpoint["rnd_opt_state_dict"])
    agent.epsilon = checkpoint["epsilon"]
    agent.global_step = checkpoint["global_step"]
    step = checkpoint["step"]
    episode_ids = np.array(checkpoint["episode_ids"], dtype=np.int64)
    return step, episode_ids


def _find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _cleanup_old_checkpoints(checkpoint_dir: Path, max_keep: int = 6) -> None:
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if len(checkpoints) > max_keep:
        for old_ckpt in checkpoints[:-max_keep]:
            try:
                old_ckpt.unlink()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the slim hierarchical DQN agent.")
    parser.add_argument("--config", type=str, default=None, help="YAML file with argument overrides.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=2_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--run-name", type=str, default="slim_run")
    parser.add_argument("--log-dir", type=str, default="runs/slim")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rnd-weight", type=float, default=0.1)
    parser.add_argument("--novelty-weight", type=float, default=0.05)
    parser.add_argument("--bayes-weight", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save-state-dir", type=str, default="savestates")
    parser.add_argument("--save-state-every", type=int, default=0)
    parser.add_argument("--save-state-on-bayes", action="store_true")
    parser.add_argument("--use-puffer", action="store_true")
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--record-video-every", dest="record_video_every", type=int, default=0)
    parser.add_argument("--video-dir", dest="video_dir", type=str, default=None)
    parser.add_argument("--cnn-channels", type=str, default="")
    parser.add_argument("--gru-size", type=int, default=128)
    parser.add_argument("--lstm-size", type=int, default=128)
    parser.add_argument("--ssm-size", type=int, default=128)
    parser.add_argument("--structured-hidden", type=int, default=128)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--max-runtime-seconds", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=6)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stream-port", type=int, default=9999)
    parser.add_argument("--stream-interval", type=int, default=60)
    parser.add_argument("--stream-top-k", type=int, default=2)
    parser.add_argument("--stream-candidates-path", type=str, default=None)
    parser.add_argument("--job-rank", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    global _shutdown_requested

    args = parse_args()
    args = apply_config(args, args.config)
    setup_logging()
    set_seed(args.seed)
    device = torch.device(args.device)

    run_dir = resolve_run_dir(args.log_dir, args.run_name)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root() / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    video_dir = Path(args.video_dir) if args.video_dir else run_dir.parent / "videos"
    if not video_dir.is_absolute():
        video_dir = project_root() / video_dir

    write_config(run_dir, vars(args))
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    start_time = time.time()

    env_cfg = load_env_config(headless=args.headless, reduce_res=False, two_bit=False, use_rubinstein_reward=True)
    save_state_root = Path(args.save_state_dir)
    if not save_state_root.is_absolute():
        save_state_root = project_root() / save_state_root
    save_state_root = save_state_root.expanduser().resolve()
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
    cnn_channels = tuple(int(x) for x in args.cnn_channels.split(",")) if args.cnn_channels else None
    agent = SlimHierarchicalDQN(
        agent_cfg,
        device=device,
        cnn_channels=cnn_channels,
        gru_size=args.gru_size,
        lstm_size=args.lstm_size,
        ssm_size=args.ssm_size,
        structured_hidden=args.structured_hidden,
    )
    agent.flag_encoder = flag_encoder
    reward_logger = RewardLogger(log_dir=run_dir)
    perf = PerformanceTracker()

    positions_path = run_dir / "positions.csv"
    positions_file = open(positions_path, "w", newline="")
    atexit.register(positions_file.close)
    pos_writer = csv.DictWriter(
        positions_file,
        fieldnames=["step_global", "env", "episode", "map_id", "coord", "badge", "deaths", "reward", "entropy"],
    )
    pos_writer.writeheader()

    start_step = 0
    episode_ids = np.zeros(args.num_envs, dtype=np.int64)
    if args.mode == "eval" and args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        agent.epsilon = 0.0
        agent.epsilon_min = 0.0
    elif args.resume or (args.mode == "train" and args.checkpoint):
        resume_path = Path(args.checkpoint) if args.checkpoint else _find_latest_checkpoint(checkpoint_dir)
        if resume_path and resume_path.exists():
            print(f"[train] Resuming from checkpoint: {resume_path}")
            start_step, episode_ids = _load_checkpoint(agent, resume_path, device)
            print(f"[train] Resumed at step {start_step}, episodes: {episode_ids.tolist()}")
        else:
            print("[train] No checkpoint found, starting fresh.")

    frames_for_video: List[np.ndarray] = []
    video_saved = False
    last_badge = np.zeros(args.num_envs, dtype=np.int32)
    last_map = np.zeros(args.num_envs, dtype=np.int32)

    stream_client: Optional[StreamClient] = None
    best_selector: Optional[BestAgentSelector] = None
    should_stream: Dict[int, bool] = {i: False for i in range(args.num_envs)}

    if args.stream:
        print(f"[train] Streaming enabled on port {args.stream_port}")
        stream_client = StreamClient(
            port=args.stream_port,
            interval_steps=args.stream_interval,
            metadata={"job_rank": args.job_rank},
        )
        candidates_path = args.stream_candidates_path or str(run_dir / "stream_candidates.json")
        best_selector = BestAgentSelector(candidates_path=candidates_path, top_k=args.stream_top_k)
        print(f"[train] Stream candidates: {candidates_path}")

    states = agent.reset_hidden(args.num_envs)
    dones = np.zeros(args.num_envs, dtype=bool)
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_rnd = np.zeros(args.num_envs, dtype=np.float32)
    episode_novel = np.zeros(args.num_envs, dtype=np.float32)
    episode_bayes = np.zeros(args.num_envs, dtype=np.float32)
    episode_total = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    agent.epsilon_decay = (1.0 - agent.epsilon_min) / max(1, args.total_steps)

    last_checkpoint_step = start_step

    for step in range(start_step, args.total_steps):
        if _shutdown_requested:
            print(f"[train] Shutdown requested at step {step}, saving checkpoint...")
            ckpt_path = _save_checkpoint(agent, step, episode_ids, checkpoint_dir, progress_path)
            print(f"[train] Saved checkpoint: {ckpt_path}")
            break

        if args.max_runtime_seconds > 0:
            elapsed = time.time() - start_time
            if elapsed >= args.max_runtime_seconds:
                print(f"[train] Max runtime ({args.max_runtime_seconds}s) reached at step {step}, saving checkpoint...")
                ckpt_path = _save_checkpoint(agent, step, episode_ids, checkpoint_dir, progress_path)
                print(f"[train] Saved checkpoint: {ckpt_path}")
                break

        frame_batch = torch.as_tensor(stack_frames(frame_stacks), device=device)
        obs_tensors = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
        structured, milestone_flags = agent.flag_encoder(obs_tensors)
        done_mask = torch.as_tensor(dones, device=device, dtype=torch.float32)

        actions, new_states, aux = agent.act(frame_batch, structured, states, done_mask)
        new_states = list(new_states)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
        next_obs = obs_to_dict(next_obs)
        dones = np.logical_or(terminated, truncated)
        done_flags = dones.copy()
        episode_returns += rewards
        episode_lengths += 1
        perf.step(args.num_envs)

        intrinsic = agent.intrinsic_rewards(aux, milestone_flags, env_ids=range(args.num_envs))
        r_env = torch.as_tensor(rewards, device=device, dtype=torch.float32)
        total_reward = (
            r_env
            + args.rnd_weight * intrinsic["rnd"]
            + args.novelty_weight * intrinsic["novel"]
            + args.bayes_weight * intrinsic["bayes"]
        )
        rnd_np = intrinsic["rnd"].detach().cpu().numpy()
        novel_np = intrinsic["novel"].detach().cpu().numpy()
        bayes_np = intrinsic["bayes"].detach().cpu().numpy()
        total_np = total_reward.detach().cpu().numpy()
        episode_rnd += rnd_np
        episode_novel += novel_np
        episode_bayes += bayes_np
        episode_total += total_np

        if args.record_video_every:
            frames_for_video.append(obs["screen"][0])

        for env_idx, done in enumerate(done_flags):
            info = infos[env_idx] if isinstance(infos, (list, tuple)) else {}
            stats = info.get("stats", {}) if isinstance(info, dict) else {}

            if done:
                perf.episode()
                agent.update_bayes_from_milestones(milestone_flags[env_idx].detach().cpu())
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
                    }
                )
                episode_ids[env_idx] += 1
                episode_returns[env_idx] = 0.0
                episode_rnd[env_idx] = 0.0
                episode_novel[env_idx] = 0.0
                episode_bayes[env_idx] = 0.0
                episode_total[env_idx] = 0.0
                episode_lengths[env_idx] = 0
                agent.reset_episode(env_idx)
                if args.save_state_every and (episode_ids[env_idx] % args.save_state_every == 0):
                    save_path = curriculum.run_dir / f"ep{episode_ids[env_idx]:05d}_env{env_idx}.state"
                    envs.envs[env_idx]._save_emulator_state(save_path)
                    curriculum.register_state(f"ep{episode_ids[env_idx]:05d}", save_path, tier="mid")
                if args.record_video_every and env_idx == 0 and episode_ids[env_idx] % args.record_video_every == 0:
                    video_dir.mkdir(parents=True, exist_ok=True)
                    out_path = video_dir / f"train_ep{episode_ids[env_idx]:05d}.mp4"
                    if frames_for_video:
                        artifacts = maybe_save_video(frames_for_video, "mp4", out_path, fps=15)
                        for art in artifacts:
                            print(f"[video] saved {art['path']} ({art['frames']} frames)")
                            video_saved = True
                    frames_for_video.clear()
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
                reset_obs, _ = envs.envs[env_idx].reset(seed=args.seed + env_idx, options=reset_opts)
                reset_obs = obs_to_dict(reset_obs)
                for key in next_obs:
                    next_obs[key][env_idx] = reset_obs[key]
                frame_stacks[env_idx].clear()
                for _ in range(args.frame_stack):
                    frame_stacks[env_idx].append(reset_obs["screen"])
                new_states = list(new_states)
                new_states[env_idx] = agent.network.initial_state(batch_size=1, device=device)
                last_badge[env_idx] = stats.get("badge", 0)
                last_map[env_idx] = stats.get("map_id", 0)
                dones[env_idx] = False
            else:
                frame_stacks[env_idx].append(next_obs["screen"][env_idx])

            hist = stats.get("action_hist", None)
            entropy = None
            if hist is not None:
                hist = np.array(hist, dtype=np.float32)
                total = hist.sum()
                if total > 0:
                    p = hist / total
                    entropy = float(-(p * np.log(p + 1e-8)).sum())

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
                }
            )

            if args.stream and stream_client and best_selector:
                info_summary = {
                    "badge_count": stats.get("badge", 0),
                    "story_flags": stats.get("events", {}),
                }
                score, tie = progress_score(info_summary, episode_total[env_idx])
                should_stream[env_idx] = best_selector.update(
                    agent_id=f"job{args.job_rank}_env{env_idx}",
                    info=info_summary,
                    episode_reward=float(episode_total[env_idx]),
                    env_idx=env_idx,
                    job_rank=args.job_rank,
                )
                if should_stream[env_idx] and stream_client:
                    stream_client.maybe_send(obs["screen"][env_idx], step, f"job{args.job_rank}_env{env_idx}")
                    stream_client.send_status(
                        agent_id=f"job{args.job_rank}_env{env_idx}",
                        score=score,
                        episode_reward=float(episode_total[env_idx]),
                        step=step,
                        extra={"tie": tie},
                    )

        obs = next_obs
        states = new_states

        next_frame_batch = torch.as_tensor(stack_frames(frame_stacks), device=device)
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

        if step % max(1, args.log_interval) == 0 and step > 0:
            stats = perf.stats()
            print(f"step {step} | SPS {stats['sps']:.1f} | episodes/hr {stats['episodes_per_hour']:.1f}")

        if args.mode == "train" and args.checkpoint_interval > 0:
            if (step - last_checkpoint_step) >= args.checkpoint_interval:
                ckpt_path = _save_checkpoint(agent, step, episode_ids, checkpoint_dir, progress_path)
                print(f"[train] Checkpoint saved: {ckpt_path}")
                _cleanup_old_checkpoints(checkpoint_dir, args.max_checkpoints)
                last_checkpoint_step = step

    if args.mode == "train" and not _shutdown_requested:
        final_step = min(step + 1, args.total_steps) if "step" in locals() else args.total_steps
        ckpt_path = _save_checkpoint(agent, final_step, episode_ids, checkpoint_dir, progress_path)
        print(f"[train] Final checkpoint saved: {ckpt_path}")

    if args.record_video_every and frames_for_video and not video_saved:
        video_dir.mkdir(parents=True, exist_ok=True)
        base = video_dir / "train_final"
        out_path = base.with_suffix(".mp4")
        suffix = 1
        while out_path.exists():
            out_path = base.with_name(f"{base.stem}_{suffix}").with_suffix(".mp4")
            suffix += 1
        artifacts = maybe_save_video(frames_for_video, "mp4", out_path, fps=15)
        for art in artifacts:
            print(f"[video] saved {art['path']} ({art['frames']} frames)")

    if stream_client is not None:
        stream_client.close()
    reward_logger.close()
    envs.close()
    print("[train] Training complete.")


if __name__ == "__main__":
    main()
