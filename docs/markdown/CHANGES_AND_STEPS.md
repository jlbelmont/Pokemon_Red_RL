# Recent Enhancements

1. **Configurable Exploration & Rewards** – Adjusted `training_config.json` to enlarge the replay buffer (now 10 M), slow epsilon decay (100 M steps, floor 0.25), strengthen exploration bonuses, and add explicit story-flag rewards (parcel, Pokédex, etc.) plus harsher Pallet/Route stay penalties.
2. **Curriculum Savestates** – Added `curriculum_states` support so each env can boot from a scheduled savestate (or clean ROM) with clear `[curriculum]` logs and a `curriculum_states` column in `logs/train_summary.csv`.
3. **Auto-Capture Pipeline** – When `auto_curriculum_capture` is enabled, new badges/story flags automatically trigger savestate captures, which are logged to `logs/curriculum_events.csv` and immediately fed back into the curriculum rotation.
4. **Story Flag Accuracy** – Champion flag now reads the correct event bit (`0xD867` bit 1) to avoid false positives; badge logging already relied on proper bit flips plus baseline masking.
5. **Observability** – Reward summaries, perf logs, curriculum logs, and progress trackers now provide a holistic view: per-env map names/steps, curriculum states, Bayesian badge metrics, and recorded savestate paths.
6. **Quest & Penalty Generalisation** – QuestReward supports staged chains with prerequisites, MapStayPenalty can escalate over dwell time, curriculum states can declare unlock requirements (badges/flags/min step), and auxiliary/novelty loss weights can smoothly schedule to new targets without touching the network.

# Suggested Next Steps (No Major Rewrite)

1. **Adaptive Stay Penalties** – Scale Pallet/Route dwell penalties with dwell time (e.g., quadratic ramp) so idling quickly becomes prohibitive without touching model architecture.
2. **Quest Chains** – Extend `QuestReward` with multi-stage chains (e.g., “reach Viridian”, “enter mart”, “deliver parcel”) so you get dense reward per quest step; it reuses the existing modular reward stack.
3. **Macro Action Library** – Introduce reusable macros (short scripted action sequences) for tedious traversals (leave Pallet, enter mart). Store them alongside savestates and let the low-level policy call them sparingly—no network changes needed.
4. **Curriculum Scheduling Rules** – Add unlock conditions so new savestates aren’t injected until earlier goals are met (using existing progress tracker metrics). This keeps later states from appearing too early.
5. **Auxiliary Loss Tuning** – Slightly raise the auxiliary/novelty loss weights to force the latent to encode map/story features more strongly, improving long-horizon planning without altering the network shape.

## Exploration & Long-Horizon Progression Plan

Goals:
- Drive the agent out of Route 1 by making globally familiar tiles uninteresting while keeping new maps exciting.
- Layer scalable intrinsic motivation (Random Network Distillation + episodic novelty memory) onto the CNN+SSM latent so sparsely rewarded quests still have a training signal.
- Maintain a lightweight Go-Explore–style archive of savestates keyed by coarse cells and reset from frontier cells to rehearse later parts of the game without hand-authored scripts.
- Keep reward shaping generic (cells, map transitions, story-flag hashes) so new quest lines automatically benefit.
- Prepare the policy stack for HIRO-style hierarchy by wrapping the existing DQN in a low-level controller that already accepts an optional latent goal.

Implementation highlights:
- `epsilon/pokemon_rl/intrinsic/visit_counter.py` adds a `VisitCounter` (global + per-episode counts) plus map-transition bonuses and an episodic latent-memory helper.
- `epsilon/pokemon_rl/intrinsic/rnd.py` implements RND with running reward normalisation; the predictor trains alongside the Q-network and feeds intrinsic bonuses.
- `epsilon/pokemon_rl/state_archive.py` maintains disk-backed frontier cells; the trainer can capture new cells opportunistically and occasionally reset an env from an archive snapshot.
- `simple_dqn.SimpleDQN` now surfaces its latent output, and `LowLevelDQNPolicy` wraps it so future high-level controllers can inject goals.
- `minimal_epsilon_setup.py` wires everything into the training loop (reward composition, optimizer updates, archive resets, config plumbing, logging).

Config knobs (see `training_config.json`):
- `visit_count_enabled`, `visit_count_scale`, `visit_count_alpha`, `visit_count_beta`, `visit_count_bin_size`, `map_transition_scale`.
- `rnd_enabled`, `rnd_scale`, `rnd_hidden_dim`, `rnd_learning_rate`, `rnd_anneal_steps`, `episodic_bonus_scale`, `episodic_memory_size`, `episodic_distance_threshold`.
- `state_archive_enabled`, `state_archive_dir`, `state_archive_max_cells`, `state_archive_capture_min_visits`, `state_archive_reset_prob`.

Example run (headless, visit-count + RND + archive resets enabled):

```bash
caffeinate -d -i bash -lc '
  cd /Users/jbelmont/Downloads/College/MS/DRL/Final
  SAVE_DIR=checkpoints/headless_8env_1m
  PYTHONUNBUFFERED=1 .venv/bin/python -u epsilon/pokemon_rl/minimal_epsilon_setup.py \
    --config epsilon/pokemon_rl/training_config.json \
    --episodes 5 \
    --max-steps 1000000 \
    --save-dir "$SAVE_DIR" \
    --num-envs 8 \
    --headless --render-map --no-show-env-maps --no-gameplay-grid --display-envs 0 \
    --visit-count-enabled \
    --rnd-enabled \
    --state-archive-enabled \
    --summary-log-path logs/train_summary.csv \
    --curriculum-events-log-path logs/curriculum_events.csv
'
```
