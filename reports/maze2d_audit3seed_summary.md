# Maze2D Audited 3-Seed Report

Generated on 2026-07-01 from the audited fallback Maze2D sweep after adding the Diffusion Policy `squaredcos_cap_v2` VP schedule, true VP Ambient x0-loss baseline, filtered execution, padded collision reporting, and the frequency-mask ablations requested by review.

## Training

The sweep trained five policy families over seeds 0, 1, and 2, for 15 total runs:

- `cotrain`
- `ambient_sampler_x0_mse` - the earlier scalar sampler gate / x0-MSE proxy baseline
- `ambient_scalar_ambient_loss` - the faithful VP Ambient x0-loss baseline implemented for the audit
- `sr_freqmask`
- `sr_freqmask_shuffled_clean_stats` - clean-stat ablation that keeps the mask structure but shuffles clean spectral statistics

Manifest: `logs/audit3seed_20260701_213340_manifest.txt`

All 15 runs completed and wrote final checkpoints under `runs/*_20260701_213343`.

## Evaluation

Primary audited report:

```bash
python -m srtd.eval.report \
  --runs $(awk '{print $4}' logs/audit3seed_20260701_213340_manifest.txt) \
  --num-trials 1000 \
  --out runs/audit3seed_filtered_padded_20260701_report \
  --seed 0 \
  --save-rollouts 5 \
  --execution-mode filtered \
  --lowpass-alpha 0.35 \
  --interpolation-steps 4 \
  --primary-collision-padding padded
```

Companion raw report:

```bash
python -m srtd.eval.report \
  --runs $(awk '{print $4}' logs/audit3seed_20260701_213340_manifest.txt) \
  --num-trials 1000 \
  --out runs/audit3seed_raw_padded_20260701_report \
  --seed 0 \
  --save-rollouts 5 \
  --execution-mode raw \
  --interpolation-steps 4 \
  --primary-collision-padding padded
```

Tracked metrics:

- Per-run filtered metrics: `reports/maze2d_audit3seed_filtered_padded_metrics.csv`
- Per-run raw metrics: `reports/maze2d_audit3seed_raw_padded_metrics.csv`
- Aggregated metrics: `reports/maze2d_audit3seed_aggregate.csv`

Runtime artifacts:

- `runs/audit3seed_filtered_padded_20260701_report/metrics.csv`
- `runs/audit3seed_filtered_padded_20260701_report/shared_trials.npz`
- `runs/audit3seed_filtered_padded_20260701_report/rollout_paths.npz`
- `runs/audit3seed_raw_padded_20260701_report/metrics.csv`
- `runs/audit3seed_raw_padded_20260701_report/shared_trials.npz`
- `runs/audit3seed_raw_padded_20260701_report/rollout_paths.npz`

## Filtered/Padded Results

| policy | n | success_rate | collision_rate | hf_residual_energy_mean | action_target_jump_mean |
|---|---:|---:|---:|---:|---:|
| sr_freqmask | 3 | 0.470 +/- 0.018 | 0.472 +/- 0.035 | 0.714 +/- 0.006 | 0.071 +/- 0.002 |
| sr_freqmask_shuffled_clean_stats | 3 | 0.463 +/- 0.016 | 0.467 +/- 0.029 | 0.713 +/- 0.006 | 0.071 +/- 0.001 |
| cotrain | 3 | 0.405 +/- 0.026 | 0.489 +/- 0.025 | 0.686 +/- 0.005 | 0.065 +/- 0.001 |
| ambient_sampler_x0_mse | 3 | 0.294 +/- 0.022 | 0.621 +/- 0.023 | 0.691 +/- 0.001 | 0.068 +/- 0.000 |
| ambient_scalar_ambient_loss | 3 | 0.107 +/- 0.015 | 0.831 +/- 0.011 | 0.712 +/- 0.004 | 0.073 +/- 0.001 |


## Raw/Padded Results

| policy | n | success_rate | collision_rate | hf_residual_energy_mean | action_target_jump_mean |
|---|---:|---:|---:|---:|---:|
| sr_freqmask | 3 | 0.522 +/- 0.010 | 0.439 +/- 0.019 | 0.789 +/- 0.001 | 0.073 +/- 0.001 |
| sr_freqmask_shuffled_clean_stats | 3 | 0.505 +/- 0.012 | 0.451 +/- 0.007 | 0.788 +/- 0.009 | 0.074 +/- 0.002 |
| cotrain | 3 | 0.498 +/- 0.019 | 0.422 +/- 0.042 | 0.740 +/- 0.007 | 0.064 +/- 0.001 |
| ambient_sampler_x0_mse | 3 | 0.398 +/- 0.003 | 0.519 +/- 0.031 | 0.749 +/- 0.005 | 0.067 +/- 0.002 |
| ambient_scalar_ambient_loss | 3 | 0.129 +/- 0.018 | 0.809 +/- 0.021 | 0.765 +/- 0.008 | 0.074 +/- 0.001 |


## Interpretation

The audited result is diagnostic rather than a clean confirmation of the original seed-0 claim.

- `sr_freqmask` has the best mean success in both reports: `0.470 +/- 0.018` filtered/padded and `0.522 +/- 0.010` raw/padded.
- `sr_freqmask_shuffled_clean_stats` is close: `0.463 +/- 0.016` filtered/padded and `0.505 +/- 0.012` raw/padded. This weakens the claim that the clean spectral-stat compatibility term is the decisive mechanism in this setup.
- `cotrain` is behind `sr_freqmask` under filtered/padded evaluation, but is close under raw/padded evaluation: `0.498 +/- 0.019` versus `0.522 +/- 0.010`.
- `ambient_sampler_x0_mse` is lower than cotrain and the frequency-mask variants, so the earlier scalar sampler gate is not competitive in the audited sweep.
- The faithful VP `ambient_scalar_ambient_loss` baseline performs very poorly here: `0.107 +/- 0.015` filtered/padded and `0.129 +/- 0.018` raw/padded, with very high collision. That should be treated as a baseline implementation/tuning warning, not as evidence that Ambient Diffusion Policy is intrinsically weak.
- Absolute success rates are far lower than the older seed-0 `0.873` result because this audit changed the schedule, baseline, seeds, collision accounting, and execution protocol.

Current conclusion: frequency masking still helps relative to cotrain in the main filtered/padded report, but the shuffled-stat ablation is too close for a strong mechanistic claim. The next research step should isolate visibility, compatibility, and low-frequency components under the same 3-seed protocol before writing this up as a positive result.
