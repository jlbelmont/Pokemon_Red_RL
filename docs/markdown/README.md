# Pokemon Red RL – Parcel Quest Pipeline

This repo hosts the reorganised `epsilon/pokemon_rl` package that powers the parcel-focused
Pokemon Red reinforcement-learning experiments on an Apple Silicon laptop. The current
stack emphasises three outcomes:

1. **Parcel quest reliability** – Oak's parcel pickup and drop-off must happen early and
often so downstream badges are even reachable.
2. **Long-horizon planning** – quest definitions, badge/quest rewards, and Bayesian
progress tracking cover the entire campaign up through the Champion fight.
3. **Training continuity** – checkpoints, replay buffers, posterior logs, and run
metadata survive interruptions so you can resume training or evaluation later without
guesswork.

## Repository layout

```
epsilon/
  pokemon_rl/
    agents/            # DQN + recurrent policy definitions
    bayesian_proj/     # Progress-event parsing & posterior logging
    configs/           # Training configs (default: training_config.json)
    envs/              # Gymnasium-compatible Pokemon Red wrapper
    intrinsic/         # Visit counts, episodic novelty, RND
    rewards/           # Reward/penalty modules and quest logic
    training/          # Main training loops, scripts, visualisation helpers
    assets/            # Aggregate map layout + background overlays
  archive/             # Legacy repo snapshots & deprecated scripts
checkpoints/           # Latest/best checkpoints, replay buffer dumps, run traces
logs/                  # Stdout logs, curriculum summaries, Bayesian metrics
artifacts/             # Posterior snapshots, analysis-ready CSVs/images
docs/                  # Progress reports and write-ups
RUNS.md                # Running log of long jobs & important settings
```

## Environment & setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Apple M1 Pro/M1 Max hardware can run four or eight parallel envs with PyBoy headless.
The ROM path defaults to `epsilon/pokemon_rl/pokemon_red.gb`, but you can override it via
`POKEMON_ROM_PATH` or the `rom`/`rom_fallbacks` config keys.

## Training & replay entry points

Headless training with the production config (4 envs, 6k steps, posterior-driven RND):

```bash
caffeinate -dimsu bash -lc '
  cd /Users/jbelmont/Downloads/College/MS/DRL/Final && \
  SAVE_DIR=checkpoints/headless_4env_parcel && \
  mkdir -p "$SAVE_DIR" logs checkpoints/curriculum_states && \
  PYTHONUNBUFFERED=1 .venv/bin/python -m epsilon.pokemon_rl.training.minimal_epsilon_setup \
    --config epsilon/pokemon_rl/configs/training_config.json \
    --save-dir "$SAVE_DIR" \
    --summary-log-path logs/train_summary_4env.csv \
    --curriculum-events-log-path logs/curriculum_events_4env.csv
'
```

Replay/watch an existing checkpoint with matplotlib overlays and PyBoy disabled:

```bash
.venv/bin/python -m epsilon.pokemon_rl.training.minimal_epsilon_setup \
  --config epsilon/pokemon_rl/configs/training_config.json \
  --watch-only \
  --replay-checkpoint checkpoints/headless_4env_parcel/dqn_route1_best.pt \
  --render-map --gameplay-grid
```

The config exposes `resume_checkpoint`, `fresh_start`, `replay_buffer_path`, and
`auto_demo_after_training` knobs so long jobs can resume automatically after an abort.
`RUNS.md` tracks the command, git SHA, and checkpoints used for each long run.

## Logging & monitoring

* **Checkpoints** – `dqn_route1_latest.pt` is updated atomically, plus a "best episode"
  snapshot when the rolling reward improves. Optimiser, RNG state, visit counters, state
  archive, and replay buffers (up to `replay_buffer_max_save`) are persisted via
  `save_training_state`.
* **Summary CSV** – `summary_log_path` now contains per-episode quest hit counts
  (`quest_hits`), exploration coverage (`coverage_tiles`), and the new-town list
  (`coverage_towns`) in addition to rewards and epsilon/loss entries.
* **Posterior tracking** – `bayesian_proj/progress_tracking.py` logs Beta posteriors for
  parcel flags, each badge, counter thresholds, and Champion completion at
  `progress_metrics_path`. Those payloads also drive posterior-derived RND scaling.
* **Reward metrics** – `reward_metrics_path` captures threshold-success posteriors per
  reward component (episode return, quest chain, visit bonus, etc.).
* **Maps & gameplay** – Matplotlib viewers can be toggled via `render_map` and
  `render_gameplay_grid`. Aggregate overlays now sit on top of the stitched Kanto map via
  `epsilon/pokemon_rl/assets/kanto_map_layout.json` and `kanto_background.png` (layout &
  art derived from [drubinstein/pokemonred_puffer](https://github.com/drubinstein/pokemonred_puffer)).
* **Replay video** – when `post_training_video_enabled` is true (default), every training
  run spawns a fresh 5-minute replay using the best checkpoint and writes a compressed
  MP4 to `post_training_video_path`. You can run the same exporter manually with
  `--replay-video-output some_path.mp4 --watch-only ...`.

## Quest & reward shaping

The default `training_config.json` encodes the entire story arc:

* `quest_definitions` chain parcel pickup/delivery, Viridian-to-Pewter traversal, Nugget
  Bridge, Mt. Moon, S.S. Anne, Rocket Hideout, Pokemon Tower, Safari, Silph, Cinnabar,
  Victory Road, Elite Four, and Champion fights. Completion bonuses escalate so the
  agent is pulled toward longer horizons once the parcel posterior improves.
* `story_flags` (see table below) directly reward Oak's parcel assignment/delivery, each
  badge, and key quest items. Setting `story_flag_default_reward` keeps rare events
  positive even when not explicitly enumerated.
* `map_stay_penalties`, `pallet_penalty_*`, `visit_count_*`, and frontier rewards
  penalise staying in Pallet/Route 1 too long and encourage unique map entries.
* Posterior-driven RND toggles (`posterior_rnd_enabled` and `posterior_rnd_event`) shrink
  intrinsic exploration bonuses once the parcel quest becomes reliable, freeing capacity
  for badge pursuits.

## Aggregate map visualisation

`KantoAggregateProjector` reads `assets/kanto_map_layout.json` (a lightly cleaned version
of the layout shipped in `pokemonred_puffer`) and paints visit counts onto the
`kanto_background.png` atlas. Config knobs:

```json
"aggregate_layout_path": "../assets/kanto_map_layout.json",
"aggregate_background_path": "../assets/kanto_background.png",
"aggregate_tile_scale": 16,
"aggregate_overlay_alpha": 0.72
```

You can replace the background with your own art – just update the config paths. Setting
`render_map=false` or omitting the layout path falls back to the legacy 256×256 tile map.

_Attribution_: the aggregate layout & background art are derived from
[drubinstein/pokemonred_puffer](https://github.com/drubinstein/pokemonred_puffer).

## Key RAM flags & quest bits

These are the primary addresses/bits used by `story_flags` and `progress_events`. Values
are in hex and refer to the Pokemon Red RAM map.

| Flag / Event              | Address | Bit | Reward | Description |
|---------------------------|---------|-----|--------|-------------|
| `oak_parcel_assigned`     | 0xD74E  | 0   | 320    | Parcel quest accepted in Viridian Mart |
| `oak_parcel_received`     | 0xD74E  | 1   | 1500   | Parcel delivered to Oak (pivotal milestone) |
| `oak_pokeballs_received`  | 0xD74B  | 4   | 1200   | Tutorial Poké Balls obtained |
| `oak_pokedex_received`    | 0xD74B  | 5   | 600    | Pokédex delivered; unlocks Pokédex rewards |
| `boulder_badge_flag`      | 0xD356  | 0   | 1800   | Defeat Brock |
| `cascade_badge_flag`      | 0xD356  | 1   | 2100   | Defeat Misty |
| `thunder_badge_flag`      | 0xD356  | 2   | 2400   | Defeat Lt. Surge |
| `rainbow_badge_flag`      | 0xD356  | 3   | 2600   | Defeat Erika |
| `soul_badge_flag`         | 0xD356  | 4   | 2800   | Defeat Koga |
| `marsh_badge_flag`        | 0xD356  | 5   | 3000   | Defeat Sabrina |
| `volcano_badge_flag`      | 0xD356  | 6   | 3200   | Defeat Blaine |
| `earth_badge_flag`        | 0xD356  | 7   | 3400   | Defeat Giovanni |
| `champion_flag`           | 0xD747  | 0   | 4200   | Champion (Rival) defeated |
| `mt_moon_fossil`          | 0xD8B3  | 0   | 240    | Mt. Moon fossil acquired |
| `ss_ticket`               | 0xD744  | 1   | 400    | S.S. Anne access |
| `bike_voucher`            | 0xD744  | 2   | 350    | Bike voucher obtained |
| `lift_key`                | 0xD744  | 4   | 450    | Celadon Rocket HQ lift key |
| `card_key`                | 0xD744  | 5   | 450    | Silph Co card key |
| `poke_flute`              | 0xD744  | 6   | 500    | Snorlax unblock progression |
| `secret_key`              | 0xD744  | 7   | 450    | Cinnabar Gym access |
| `exp_all`                 | 0xD8B3  | 1   | 300    | EXP All key item |
| `legendary_bird_*`        | 0xD8BC  | 5–7| 300 ea | Moltres/Zapdos/Articuno captures |

Use this table when extending rewards or adding new progress events – each entry can be
referenced directly in `training_config.json`.

## Maintaining best runs & curriculum state

* Latest checkpoint: `checkpoints/<run>/dqn_route1_latest.pt`
* Best episode: `checkpoints/<run>/dqn_route1_best.pt` plus
  `best_episode_trace.json` & `best_episode_frames.npz`
* Replay buffer: `replay_buffer.pt` (size capped by `replay_buffer_max_save`)
* Curriculum archive: `checkpoints/<run>/state_archive/` retains frontier savestates.
* Run metadata: append to `RUNS.md` after each significant run (start/end timestamps,
  command, notable outcomes).

A clean cycle looks like:

1. Start training (command above) and let it run ≥6 hours.
2. If interrupted, re-run the same command. The loader checks `resume_checkpoint` and the
   replay buffer snapshot automatically.
3. After each run, copy `logs/run.log`, `logs/train_summary_*.csv`, `progress_metrics.json`,
   `reward_metrics.json`, and aggregate maps into a dated folder under `archives/` so you
   can regenerate posterior plots later.
4. Use `analysis/run_pipeline.py` and `analysis/reward_success.py` to rebuild posterior
   figures; update the LaTeX report if needed.

This flow preserves the best-performing models, keeps Bayesian metrics up to date, and
makes debugging (e.g., checking whether parcel posteriors budged) far faster.
