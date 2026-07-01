# Maze2D Seed 0 SRTD Report

Generated on 2026-06-30 from the fallback Ambient-style 2D maze dataset.

## Training

All seven seed-0 policies completed 200,000 training steps using the cached fallback dataset:

`data/generated/maze2d_fallback_seed0_p50_q5000_h100.npz`

Run directories:

- `gcs_only`: `runs/maze2d_gcs_only_seed0_20260630_192251`
- `rrt_only`: `runs/maze2d_rrt_only_seed0_20260630_192251`
- `cotrain`: `runs/maze2d_cotrain_seed0_20260630_192251`
- `ambient_scalar`: `runs/maze2d_ambient_scalar_seed0_20260630_192251`
- `sr_tmin`: `runs/maze2d_sr_tmin_seed0_20260630_192350`
- `sr_freqmask`: `runs/maze2d_sr_freqmask_seed0_20260630_192251`
- `sr_full`: `runs/maze2d_sr_full_seed0_20260630_190733`

## Evaluation

Command:

```bash
.venv/bin/python -m srtd.eval.report \
  --runs \
  runs/maze2d_gcs_only_seed0_20260630_192251 \
  runs/maze2d_rrt_only_seed0_20260630_192251 \
  runs/maze2d_cotrain_seed0_20260630_192251 \
  runs/maze2d_ambient_scalar_seed0_20260630_192251 \
  runs/maze2d_sr_tmin_seed0_20260630_192350 \
  runs/maze2d_sr_freqmask_seed0_20260630_192251 \
  runs/maze2d_sr_full_seed0_20260630_190733 \
  --num-trials 1000 \
  --out runs/maze2d_seed0_report \
  --save-rollouts 5
```

Artifacts:

- Tracked metrics copy: `reports/maze2d_seed0_metrics.csv`
- Runtime metrics: `runs/maze2d_seed0_report/metrics.csv`
- Shared trials: `runs/maze2d_seed0_report/shared_trials.npz`
- Saved rollout paths: `runs/maze2d_seed0_report/rollout_paths.npz`
- Pareto figure: `runs/maze2d_seed0_report/success_vs_smoothness_pareto.png`
- Residual energy figure: `runs/maze2d_seed0_report/generated_residual_energy.png`
- Success/collision figure: `runs/maze2d_seed0_report/success_collision_rates.png`
- Rollout grid: `runs/maze2d_seed0_report/rollout_grid_same_start_goal.png`

## Results

| policy | success_rate | smoothness_mean | collision_rate | endpoint_error | hf_residual_energy_mean |
|---|---:|---:|---:|---:|---:|
| gcs_only | 0.245 | 59.434 | 0.539 | 1.238 | 0.03655 |
| rrt_only | 0.635 | 12.120 | 0.071 | 0.247 | 0.01963 |
| cotrain | 0.662 | 11.785 | 0.074 | 0.238 | 0.01950 |
| ambient_scalar | 0.759 | 14.056 | 0.120 | 0.252 | 0.02142 |
| sr_tmin | 0.545 | 25.950 | 0.143 | 0.362 | 0.02274 |
| sr_freqmask | 0.744 | 22.813 | 0.049 | 0.223 | 0.02356 |
| sr_full | 0.688 | 20.096 | 0.051 | 0.246 | 0.02318 |

## Interpretation

This seed-0 fallback experiment completed successfully, but it does not validate the main SRTD policy hypothesis yet.

- `ambient_scalar` achieved the best success rate: `0.759`.
- `sr_freqmask` was close in success: `0.744`, and had the lowest collision rate: `0.049`.
- `sr_full` improved over `cotrain` on success (`0.688` vs `0.662`) and collision rate (`0.051` vs `0.074`), but was less smooth than `cotrain`.
- `sr_tmin` underperformed, which suggests the current per-chunk `tmin` criterion is too restrictive or poorly calibrated on the fallback generator.
- The SRTD variants did not meet the target policy gate against `ambient_scalar` in this first fallback run.

Immediate next technical work should focus on diagnosing the spectral gates before more expensive sweeps:

- inspect the `sr_tmin` histogram and usable RRT fraction by diffusion timestep,
- add diagnostic AUROC/high-frequency residual reports,
- tune `clean_bad_visible_quantile`, `global_band_cutoff_norm`, and compatibility temperature,
- compare with a less aggressive `sr_tmin` schedule before expanding to multiple seeds.

