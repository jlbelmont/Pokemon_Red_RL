# Run Log

Document each experiment so we can trace which configuration produced which results.

## How to record a run

1. Choose a unique identifier, e.g. `20251201_parcel_quota`.
2. Launch training via `scripts/run_experiment.sh <RUN_ID>`.
3. After the run finishes, append a new entry below summarising:
   - Run ID / date
   - Key config changes (e.g. posterior RND bounds, curriculum tweaks)
   - Notable outcomes (parcel posterior, badges, auto-captured states)

## Entries

- *(pending)* `YYYYMMDD_description` – add notes after your next run.
