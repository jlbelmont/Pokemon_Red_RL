# Model Goal

- Ensure the Pokémon Red RL agent steadily progresses through long quests (like the parcel delivery) by combining curriculum savestates with transparent logging so humans can monitor and tune the process in real time.

## Implementation Goals

1. ✅ Surface the curriculum state for every environment reset (and per-episode summary) so training logs clearly show which savestate or boot mode each env used.
2. ✅ Write a dedicated CSV log for every auto-captured savestate event (badge/story flag) with timestamp, env index, and file path, making it easy to audit curriculum growth.

## Exploration & Progression Upgrades

We extended the plan beyond curriculum savestates so the agent can escape Route 1 without hand-written quest scripts:

1. ✅ Added global + episodic visit counting (binned `(map_id, x, y)` cells plus optional story-flag hashes) so previously farmed areas automatically become low value, while per-episode first-visit bonuses keep local exploration worthwhile.
2. ✅ Introduced map-transition bonuses keyed on `(map_from → map_to)` pairs. These pay out only the first few times a gate/door is crossed, nudging the policy toward Viridian, Route 2, etc.
3. ✅ Plugged Random Network Distillation and episodic latent-memory bonuses into the SimpleDQN latent so remote states stay intrinsically interesting and micro-loops are discouraged.
4. ✅ Implemented a lightweight Go-Explore style state archive that captures savestates for new coarse cells and occasionally resets envs from frontier entries. This lets the replay buffer practise later quest stages without a perfect single-episode trajectory.
5. ✅ Wrapped the backbone in `LowLevelDQNPolicy`, exposing latent vectors and optional goal slots so a future HIRO-style controller can set high-level navigation goals while the existing low-level policy handles movement.
