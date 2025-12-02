# Puffer-Style Optimisation Roadmap

This doc outlines what it would take to pull the core throughput ideas from
[drubinstein/pokemonred_puffer](https://github.com/drubinstein/pokemonred_puffer) into the
current `epsilon/pokemon_rl` stack without rewriting the entire repo.

## Milestone 1: Vectorised Environment Runner

Puffer’s biggest gain is batching emulator steps. Our trainer iterates over envs one at a
time, which means:

- ~80% of wall time is PyBoy API overhead + Python dispatch.
- Every observation/action/reward is handled via Python lists.

To match Puffer, we need a vector wrapper:

1. Build a `VectorPokemonEnv` (based on `gym.vector.AsyncVectorEnv`) that wraps multiple
   `PokemonRedEnv` instances and exposes stacked observations of shape
   `(num_envs, C, H, W)`.
2. Refactor `train()` / `run_replay()` to consume the stacked tensors. That means:
   - `select_action_eps` runs once on a batch rather than looping per-env.
   - `VisitCounter`, `QuestReward`, `BayesProgressTracker`, etc., take batched info
     dictionaries.
   - Curriculum/state archive pathways work with vector indices instead of per-env loops.
3. Make batching optional: keep the current per-env path so we can flip between legacy
   behaviour and the vector runner while debugging.

## Milestone 2: Batched Policy & Torch Compile

Once the env is vectorised, we can apply Puffer’s policy tricks:

- Modify `SimpleDQN.forward()` to accept `[batch, C, H, W]` inputs.
- Update `select_action_eps()` to call the policy once per batch, return batched Q-values
  and hidden states.
- Use `torch.compile(policy)` (PyTorch 2.2+) to fuse the CNN/SSM/LSTM operations for
  Apple Silicon.

This reduces Python overhead per step and keeps the GPU/ANE fully utilised.

## Milestone 3: Tensor Replay Buffer

Puffer keeps replay transitions as device tensors and samples them via tensor ops. To
borrow that:

1. Replace the list-based `ReplayBuffer` with a structure that pre-allocates tensors for
   obs, actions, rewards, etc., on the target device.
2. Switch sampling to `torch.randint` plus tensor indexing, yielding batches that are
   already on-device (no per-sample numpy→torch conversion).
3. Integrate a tensorised priority mechanism if PER is required (optional; Puffer uses a
   uniform buffer plus large batches).

## Milestone 4: DataLoader-style N-step Writer

Puffer’s runner writes multi-step trajectories straight into the replay buffer in large
chunks. To emulate that:

- Gather `n_step` transitions per env and push them into the tensor buffer in one go.
- Use background threads/processes to offload PyBoy frame decoding or preprocessing.

## Milestone 5: Profiling & Validation

- Compare steps/sec before/after each milestone using the built-in perf logger.
- Ensure the parcel quest logic, Bayesian tracking, and checkpoint/resume flow still work
  under the batched model.

## Risks & Open Questions

- Every milestone touches most modules: rewards, intrinsic bonuses, progress tracking,
  curriculum, etc. Expect multi-day integration/testing for each stage.
- PyBoy might have quirks when driven from multiple threads; `AsyncVectorEnv` may require
  `forkserver` or per-process emulator instances.
- Torch compile/`torch.compile` support on MPS is new; we’ll need to pin PyTorch 2.2+ and
  verify there are no precision regressions.

## Suggested Execution Order

1. Land Milestone 1 on a feature branch; run short smoke tests.
2. Layer Milestone 2 on top once vector env is stable.
3. Iterate on Milestones 3–4 as separate PRs to keep diffs reviewable.
4. After each milestone, run the standard training command and record metrics in
   `RUNS.md` so we can measure speed improvements.

