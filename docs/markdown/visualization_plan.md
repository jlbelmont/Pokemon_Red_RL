# Visualization Plan

1. **Aggregate Progress Logs**
   - Copy each run's `progress_metrics.json` to `artifacts/runs/<run_id>/`.
   - Ensure `logs/train_summary_8env.csv` exists for intrinsic reward stats.

2. **Generate Posterior Traces**
   ```bash
   PYTHONPATH=. .venv/bin/python analysis/run_pipeline.py
   ```
   - Modify `analysis/plots.py` to save figures:
     ```python
     plot_posterior_traces(posterior_df, milestone="oak_parcel_assigned")
     plt.savefig("figures/parcel_posterior.pdf")
     ```

3. **Lead-time Table**
   - Annotate stagnation episodes in `analysis/labels.csv` (run_id, milestone, stagnation_episode).
   - Re-run pipeline to fill `analysis/evaluation.compute_lead_time`.

4. **Reward Calibration**
   - Use `analysis/rewards.fit_gamma_hyperparams` output to document intrinsic scaling.
   - Export as JSON for inclusion in the report.

5. **Final Figures**
   - `figures/parcel_posterior.pdf`: posterior trajectories for parcel milestones.
   - `figures/new_town_success.pdf`: success curve for new-town milestone.
   - `figures/lead_time_bar.pdf`: bar chart of lead episodes before stagnation.
