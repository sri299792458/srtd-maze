# Maze2D Fixed-Cadence Audited 3-Seed Report

Generated on 2026-07-02 by re-evaluating the 15 trained 2026-07-01 audited checkpoints after fixing rollout observation cadence. The checkpoints are unchanged; only evaluation changed.

The cadence fix keeps interpolation substeps for collision and smoothness measurement, but updates policy `obs_prev` and `obs_curr` only at the 10 Hz command cadence used during training.

## Artifacts

Tracked metrics:

- Filtered/padded per-run metrics: `reports/maze2d_audit3seed_fixedcadence_filtered_padded_metrics.csv`
- Raw/padded per-run metrics: `reports/maze2d_audit3seed_fixedcadence_raw_padded_metrics.csv`
- Aggregate metrics: `reports/maze2d_audit3seed_fixedcadence_aggregate.csv`
- Filtered/padded paired stats: `reports/maze2d_audit3seed_fixedcadence_filtered_padded_paired_stats.csv`
- Raw/padded paired stats: `reports/maze2d_audit3seed_fixedcadence_raw_padded_paired_stats.csv`

Runtime report directories:

- `runs/audit3seed_filtered_padded_fixedcadence_20260702_report`
- `runs/audit3seed_raw_padded_fixedcadence_20260702_report`

## Filtered/Padded Fixed-Cadence Results

| policy | n | success_rate | collision_rate | smoothness_mean | hf_residual_energy_delta_mean | out_of_bounds_rate |
|---|---:|---:|---:|---:|---:|---:|
| sr_freqmask | 3 | 0.488 +/- 0.025 | 0.443 +/- 0.037 | 33.341 +/- 0.246 | 0.718 +/- 0.008 | 0.005 +/- 0.004 |
| sr_freqmask_shuffled_clean_stats | 3 | 0.457 +/- 0.009 | 0.471 +/- 0.003 | 33.268 +/- 1.955 | 0.716 +/- 0.006 | 0.004 +/- 0.003 |
| cotrain | 3 | 0.419 +/- 0.015 | 0.467 +/- 0.020 | 23.182 +/- 2.169 | 0.689 +/- 0.005 | 0.009 +/- 0.008 |
| ambient_sampler_x0_mse | 3 | 0.309 +/- 0.020 | 0.598 +/- 0.007 | 22.092 +/- 1.637 | 0.694 +/- 0.001 | 0.016 +/- 0.009 |
| ambient_scalar_ambient_loss | 3 | 0.106 +/- 0.021 | 0.831 +/- 0.011 | 22.752 +/- 1.778 | 0.715 +/- 0.004 | 0.012 +/- 0.007 |


Paired trial statistics:

- `sr_freqmask - cotrain` success diff: `0.069` paired bootstrap 95% CI `[0.029, 0.103]`; McNemar p = `2.186e-10`
- `sr_freqmask_shuffled_clean_stats - cotrain` success diff: `0.038` paired bootstrap 95% CI `[0.017, 0.059]`; McNemar p = `0.0003413`
- `sr_freqmask - sr_freqmask_shuffled_clean_stats` success diff: `0.031` paired bootstrap 95% CI `[-0.005, 0.064]`; McNemar p = `0.003824`

## Raw/Padded Fixed-Cadence Results

| policy | n | success_rate | collision_rate | smoothness_mean | hf_residual_energy_delta_mean | out_of_bounds_rate |
|---|---:|---:|---:|---:|---:|---:|
| sr_freqmask | 3 | 0.544 +/- 0.031 | 0.407 +/- 0.031 | 322.878 +/- 11.580 | 0.789 +/- 0.004 | 0.011 +/- 0.008 |
| sr_freqmask_shuffled_clean_stats | 3 | 0.535 +/- 0.008 | 0.422 +/- 0.018 | 349.904 +/- 31.634 | 0.785 +/- 0.009 | 0.010 +/- 0.005 |
| cotrain | 3 | 0.531 +/- 0.014 | 0.383 +/- 0.036 | 111.883 +/- 12.155 | 0.742 +/- 0.008 | 0.016 +/- 0.007 |
| ambient_sampler_x0_mse | 3 | 0.408 +/- 0.013 | 0.507 +/- 0.016 | 121.309 +/- 4.122 | 0.751 +/- 0.002 | 0.020 +/- 0.006 |
| ambient_scalar_ambient_loss | 3 | 0.129 +/- 0.012 | 0.808 +/- 0.018 | 198.681 +/- 18.878 | 0.766 +/- 0.007 | 0.015 +/- 0.012 |


Paired trial statistics:

- `sr_freqmask - cotrain` success diff: `0.013` paired bootstrap 95% CI `[-0.015, 0.042]`; McNemar p = `0.261`
- `sr_freqmask_shuffled_clean_stats - cotrain` success diff: `0.004` paired bootstrap 95% CI `[-0.019, 0.027]`; McNemar p = `0.7231`
- `sr_freqmask - sr_freqmask_shuffled_clean_stats` success diff: `0.009` paired bootstrap 95% CI `[-0.023, 0.038]`; McNemar p = `0.436`

## Interpretation

The cadence fix modestly improves most policies but does not change the scientific conclusion.

- In filtered/padded evaluation, `sr_freqmask` improves over `cotrain`: `0.488 +/- 0.025` versus `0.419 +/- 0.015`, with paired success difference `+0.069` and paired bootstrap CI `[+0.029, +0.103]`.
- In raw/padded evaluation, `sr_freqmask` is only slightly above `cotrain`: `0.544 +/- 0.031` versus `0.531 +/- 0.014`, with paired success difference `+0.013` and CI `[-0.015, +0.042]`.
- The mechanism claim remains weak. `sr_freqmask_shuffled_clean_stats` is close to `sr_freqmask`: filtered paired difference `+0.031` with CI `[-0.005, +0.064]`; raw paired difference `+0.009` with CI `[-0.023, +0.038]`.
- `cotrain` remains smoother and lower in delta-motion high-frequency residual energy than the frequency-mask variants.
- The faithful VP Ambient-loss baseline remains poor, so it is still a tuning/implementation warning rather than a conclusion about Ambient Diffusion Policy.

Current conclusion: the fixed-cadence audit preserves a modest success advantage for `sr_freqmask` in the primary filtered/padded setting, but it still does not establish clean spectral residual compatibility as the mechanism.
