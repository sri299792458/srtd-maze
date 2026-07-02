# Maze2D Seed 0 SRTD Report

Generated on 2026-07-01 from the fallback Ambient-style 2D maze dataset after
the review fixes to DDIM sampling, smoothness evaluation, generated residual
energy, and fixed sample-proportional SRTD source loss weighting.

## Training

All seven seed-0 policies completed 200,000 training steps using the cached
fallback dataset:

`data/generated/maze2d_fallback_seed0_p50_q5000_h100.npz`

Run directories:

- `gcs_only`: `runs/maze2d_gcs_only_seed0_20260701_072457`
- `rrt_only`: `runs/maze2d_rrt_only_seed0_20260701_073116`
- `cotrain`: `runs/maze2d_cotrain_seed0_20260701_073116`
- `ambient_scalar`: `runs/maze2d_ambient_scalar_seed0_20260701_073116`
- `sr_tmin`: `runs/maze2d_sr_tmin_seed0_20260701_073116`
- `sr_freqmask`: `runs/maze2d_sr_freqmask_seed0_20260701_072457`
- `sr_full`: `runs/maze2d_sr_full_seed0_20260701_073116`

## Evaluation

Command:

```bash
.venv/bin/python -m srtd.eval.report \
  --runs \
  runs/maze2d_gcs_only_seed0_20260701_072457 \
  runs/maze2d_rrt_only_seed0_20260701_073116 \
  runs/maze2d_cotrain_seed0_20260701_073116 \
  runs/maze2d_ambient_scalar_seed0_20260701_073116 \
  runs/maze2d_sr_tmin_seed0_20260701_073116 \
  runs/maze2d_sr_freqmask_seed0_20260701_072457 \
  runs/maze2d_sr_full_seed0_20260701_073116 \
  --num-trials 1000 \
  --out runs/maze2d_seed0_retrain_20260701_report \
  --seed 0 \
  --save-rollouts 5
```

Artifacts:

- Tracked metrics copy: `reports/maze2d_seed0_metrics.csv`
- Runtime metrics: `runs/maze2d_seed0_retrain_20260701_report/metrics.csv`
- Shared trials: `runs/maze2d_seed0_retrain_20260701_report/shared_trials.npz`
- Saved rollout paths: `runs/maze2d_seed0_retrain_20260701_report/rollout_paths.npz`
- Pareto figure: `runs/maze2d_seed0_retrain_20260701_report/success_vs_smoothness_pareto.png`
- Residual energy figure: `runs/maze2d_seed0_retrain_20260701_report/generated_residual_energy.png`
- Success/collision figure: `runs/maze2d_seed0_retrain_20260701_report/success_collision_rates.png`
- Rollout grid: `runs/maze2d_seed0_retrain_20260701_report/rollout_grid_same_start_goal.png`

The heavy runtime artifacts are local and ignored by git. No new release asset
has been published for this rerun.

## Results

| policy | success_rate | smoothness_mean | collision_rate | endpoint_error | hf_residual_energy_mean |
|---|---:|---:|---:|---:|---:|
| gcs_only | 0.244 | 140.004 | 0.540 | 1.238 | 0.43420 |
| rrt_only | 0.756 | 32.395 | 0.084 | 0.255 | 0.32379 |
| cotrain | 0.770 | 31.236 | 0.069 | 0.239 | 0.31982 |
| ambient_scalar | 0.795 | 37.310 | 0.117 | 0.272 | 0.31980 |
| sr_tmin | 0.640 | 54.834 | 0.180 | 0.402 | 0.32207 |
| sr_freqmask | 0.873 | 63.019 | 0.075 | 0.221 | 0.29376 |
| sr_full | 0.836 | 69.766 | 0.027 | 0.224 | 0.32925 |

## Interpretation

This post-fix seed-0 fallback experiment supports the spectral-mask policy
hypothesis on success rate.

Important caveat: this rerun predates the audit patch that added Diffusion
Policy's `squaredcos_cap_v2` VP schedule, a true VP Ambient x0-loss baseline,
filtered execution, padded primary collision reporting, and reproducibility
bundling. The result is useful as a fallback signal, but it is not yet a
faithful Ambient baseline comparison.

- `sr_freqmask` achieved the best success rate: `0.873`, with 95% CI
  `[0.852, 0.894]`.
- `ambient_scalar` remained a strong baseline at `0.795`, with 95% CI
  `[0.770, 0.820]`.
- `sr_full` was second on success at `0.836` and had the lowest collision rate:
  `0.027`.
- `sr_freqmask` also had the lowest generated high-frequency residual energy:
  `0.29376`.
- `sr_tmin` still underperformed at `0.640`, suggesting the current per-chunk
  `tmin` criterion remains too restrictive or poorly calibrated.
- The SRTD variants trade off smoothness under the cubic-spline acceleration
  metric; both high-performing SRTD policies are less smooth than
  `ambient_scalar`.

Current `sr_tmin` usable RRT fraction under the old `sine_sigma` schedule:

| t | usable RRT fraction |
|---:|---:|
| 0 | 0.0211 |
| 5 | 0.1059 |
| 10 | 0.2240 |
| 18 | 0.2434 |
| 25 | 0.2472 |
| 50 | 0.2482 |
| 75 | 0.2482 |
| 99 | 0.2482 |

Next technical work should focus on stress-testing this result:

- repeat over multiple seeds,
- inspect `sr_tmin` usable RRT fraction by diffusion timestep,
- tune `global_band_cutoff_norm`, `compat_temperature`, and residual margins,
- compare `sr_freqmask` against a less aggressive `sr_tmin` schedule.

## Historical Note

The earlier 2026-06-30 seed-0 report was generated before the review fixes and
is treated as historical diagnostic output. The metrics in this file and
`reports/maze2d_seed0_metrics.csv` are from the 2026-07-01 rerun.
