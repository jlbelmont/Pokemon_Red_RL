# Pokémon Red RL Project – Goal Brief

## High-Level Objective
- Train a high-performing reinforcement-learning agent that can complete the full Pokémon Red storyline inside a deterministic PyBoy emulator.
- Favor long-horizon progress (badges, Elite Four, champion) over short-term farming by combining a dueling distributional DQN (`epsilon/pokemon_rl/simple_dqn.py`) with heavy reward shaping (exploration, quests, novelty, efficiency, safety).
- Run reproducibly on macOS (M1 Max target hardware) with 8 parallel environments so the policy always boots from a clean ROM state and scales to multi-million step episodes.

## Environment & Training Goals
- `epsilon/pokemon_rl/minimal_epsilon_setup.py` orchestrates everything: resolves ROMs, purges `.sav` files before every launch (`utils.purge_rom_save`), boots envs, tracks replay, and manages checkpoints.
- `PokemonRedEnv` + `EpsilonEnv` expose raw frames, map features, and goal-conditioning channels so `SimpleDQN` always sees spatial context (see `epsilon/pokemon_rl/map_features.py`).
- Default config (`epsilon/pokemon_rl/training_config.json`) runs 1 M steps/episode across 8 envs, `learning_starts=20 k`, `epsilon_decay_steps=4 M`, and auto-saves to `checkpoints/`.
- Support both long training runs and lightweight debugging: `run_local_train.sh` enforces a 1-env headless run with `caffeinate`, while `watch_local.sh` replays the best checkpoint without the PyBoy window.

## Reward, Progress, and Success Criteria
- Modular rewards in `epsilon/pokemon_rl/rewards/` include badges, story flags, exploration, resource management, etc., each contributing to the total signal that drives the agent toward game completion.
- The Bayes progress tracker (`progress_tracking.py`) must log **true** badge/champion events only when bits flip from a clean baseline—hence the `.sav` purge tests (`tests/test_rom_purge.py`) and badge/champion regression tests (`tests/test_progress_tracking.py`).
- `BADGE_NAMES = ["boulder", "cascade", "thunder", "rainbow", "soul", "marsh", "volcano", "earth"]` define the canonical order. Champion credit demands all eight badges and a freshly observed champion flag (event bit 1 at `0xD867`).
- Every 10 k steps per env, `minimal_epsilon_setup.py` logs badge/champion flips, Bayesian summaries, and writes per-episode CSVs via `--summary-log-path` (default `logs/train_summary.csv`). False positives are unacceptable; downstream analytics assume absolute accuracy.

## Observability, Ops, and Automation Goals
- Headless jobs must keep the Mac display awake: run long trainings through `caffeinate -d -i` (see handoff command block) and pipe stdout/stderr into timestamped logs under `logs/`.
- Auto-purge ROM saves before **every** reset/run to prevent inherited badges or champion flags. If the ROM ships as a `.zip`, extend the purge logic so the real `.sav` path is still removed.
- Track performance via `perf_logging_enabled` plus `perf_log_path`, and keep Bayesian summaries in `checkpoints/progress_metrics.json` for quick health checks before demos.
- Preserve clean debugging hooks: expose a single-env launch that boots from zero, dumps early badge masks, and proves that cascade/volcano/earth bits stay unset until legitimately earned.
- Curriculum hooks: `curriculum_states` lets you cycle per-env savestates (or fresh boots) so later quest segments can be rehearsed without waiting for a lucky rollout.

## Current Focus / Next Agent Tasks
1. Verify that the `.sav` purge hits the correct file even when the ROM path differs (symlinks, zipped assets, alternative directories). Extend logging if needed.
2. Instrument boot-time badge mask logging to capture the first few hundred steps; identify why cascade/volcano/earth badges still appear ~100–300 steps in and eliminate that source.
3. Provide a simple CLI flag (e.g., `--debug-single-env` or documented `run_local_train.sh`) so future debugging reproduces the clean boot scenario quickly.
4. Keep `logs/train_summary.csv` and Bayesian metrics trustworthy—no badge/champion entry should be recorded without a verifiable bit flip relative to that episode’s baseline.

## Key References
- Architecture snapshot: `ARCHITECTURE_OVERVIEW.txt`
- Training driver: `epsilon/pokemon_rl/minimal_epsilon_setup.py`
- Progress logic + tests: `epsilon/pokemon_rl/progress_tracking.py`, `tests/test_progress_tracking.py`
- ROM hygiene: `epsilon/pokemon_rl/utils.py`, `tests/test_rom_purge.py`
- Run scripts: `run_local_train.sh`, `watch_local.sh`, plus the handoff headless `caffeinate` command (see previous summary message).
