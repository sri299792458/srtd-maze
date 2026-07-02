# Running Notes

## 2026-06-30

- Created a new project workspace at `/home/srinivas/Desktop/g1pilot-workspace/srtd-maze`.
- Initialized a standalone git repository on branch `main`.
- Created a local Python virtual environment at `.venv`.
- Read both pasted reference/spec files before implementation.
- Copied the pasted files into `references/`:
  - `trajectory_diffusion_spectral_skill_autoregression.txt`
  - `ambient_maze_srtd_implementation_spec.txt`
- First implementation target: the Ambient-style 2D Maze noisy-trajectory smoke test.
- Public Ambient project materials do not appear to expose a direct code/data repo, so the first pass uses the fallback generator specified in the prompt.
- Downloaded local reference copies:
  - Ambient Diffusion Policy arXiv API query and PDF: `references/downloads/ambient_diffusion_policy_2606.12365v1.pdf`.
  - Sander Dieleman blog HTML: `references/downloads/sander_diffusion_is_spectral_autoregression.html`.
  - Hou-Zhang PDF retrieval is still a TODO; the old Caltech and author URLs returned HTTP 503 from this environment, and a guessed CVF path returned 404. See `references/downloads/SOURCES.md`.
- Installed the venv package editable with GPU PyTorch:
  - Torch: `2.12.1+cu130`.
  - CUDA visible: yes.
  - Devices visible: 2.
  - First device: `NVIDIA RTX A5500`.
- Disk note: the first GPU install failed with `No space left on device`; purging pip cache recovered about 17 GB and the GPU install then succeeded. Current free space after install was about 10 GB.
- Verification:
  - `pytest -q`: 6 tests passed.
  - Smoke train: `python -m srtd.training.train_maze2d --config srtd/configs/maze2d_smoke_sr_tmin.yaml --max-steps 3`.
  - Smoke run directory: `runs/maze2d_sr_tmin_seed0_20260630_185621`.
  - Smoke report: `python -m srtd.eval.report --runs runs/maze2d_sr_tmin_seed0_20260630_185621 --num-trials 2 --out runs/smoke_report`.
- Implementation order:
  1. core spectral residual operators,
  2. clean spectral statistics and visibility masks,
  3. SR `tmin` annotation,
  4. samplers and losses,
  5. fallback maze generator and training/evaluation harness,
  6. tests and an initial commit.

## 2026-06-30 Training Launch

- User clarified that the next step should be actual training, not diagnostics.
- Launched full `sr_full` training for seed 0 with:
  `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m srtd.training.train_maze2d --config srtd/configs/maze2d_sr_full.yaml --seed 0`
- Persistent process:
  - PID: `3631838`
  - Log: `logs/train_sr_full_seed0.log`
  - Active run directory: `runs/maze2d_sr_full_seed0_20260630_190733`
  - GPU: `CUDA_VISIBLE_DEVICES=1`, visible in-process as CUDA device 0.
- Before relaunching, fixed fallback generation startup cost:
  - cached generated fallback datasets under `data/generated/`,
  - added progress logging during fallback generation,
  - cached padded obstacle rectangles in `MazeEnv` instead of rebuilding them at every collision sample.
- The first full fallback dataset was cached at:
  `data/generated/maze2d_fallback_seed0_p50_q5000_h100.npz`
- Training confirmed started on CUDA:
  - reached `step=1600`,
  - loss decreased from `0.221029` at step 100 to `0.029435` at step 1600.

## 2026-06-30 Final Seed-0 Deliverables

- Completed 200,000 training steps for all seven seed-0 policies:
  - `gcs_only`
  - `rrt_only`
  - `cotrain`
  - `ambient_scalar`
  - `sr_tmin`
  - `sr_freqmask`
  - `sr_full`
- Final checkpoint directories:
  - `runs/maze2d_gcs_only_seed0_20260630_192251`
  - `runs/maze2d_rrt_only_seed0_20260630_192251`
  - `runs/maze2d_cotrain_seed0_20260630_192251`
  - `runs/maze2d_ambient_scalar_seed0_20260630_192251`
  - `runs/maze2d_sr_tmin_seed0_20260630_192350`
  - `runs/maze2d_sr_freqmask_seed0_20260630_192251`
  - `runs/maze2d_sr_full_seed0_20260630_190733`
- Ran the final 1,000 shared-trial report:
  `python -m srtd.eval.report --runs <seven run dirs> --num-trials 1000 --out runs/maze2d_seed0_report --save-rollouts 5`
- Final report artifacts:
  - `runs/maze2d_seed0_report/metrics.csv`
  - `runs/maze2d_seed0_report/success_vs_smoothness_pareto.png`
  - `runs/maze2d_seed0_report/generated_residual_energy.png`
  - `runs/maze2d_seed0_report/success_collision_rates.png`
  - `runs/maze2d_seed0_report/rollout_grid_same_start_goal.png`
  - `reports/maze2d_seed0_metrics.csv`
  - `reports/maze2d_seed0_summary.md`
- Key result:
  - `ambient_scalar` had the best success rate at `0.759`.
  - `sr_freqmask` was close at `0.744` and had the lowest collision rate at `0.049`.
  - `sr_full` beat `cotrain` on success (`0.688` vs `0.662`) and collision rate (`0.051` vs `0.074`) but was less smooth.
  - `sr_tmin` underperformed at `0.545`, suggesting the current per-chunk spectral `tmin` gate is too restrictive or poorly calibrated.
- Conclusion:
  - The final deliverables are complete for seed 0 on the fallback maze setup.
  - The first run is a useful negative/diagnostic result, not a confirmed win over `ambient_scalar`.
  - Superseded as current evidence by the 2026-07-01 post-fix rerun below.

## 2026-07-01 Review Fixes

- Read the external review in `/home/srinivas/.codex/attachments/53e559aa-e4b9-4f6d-9374-4559a254fc7a/pasted-text.txt`.
- Fixed the VP DDIM sampler:
  - added `ddim_x0_step`,
  - deterministic updates now preserve the estimated noise direction,
  - added a formula test for an oracle `x0` prediction.
- Replaced finite-difference smoothness with cubic-spline integrated squared acceleration.
- Changed generated high-frequency residual energy to use fixed windows of delta motion instead of variable-length absolute rollout paths.
- Made SRTD source loss weighting fixed and sample-proportional:
  - removed the old equal-source compatibility path,
  - removed the `source_loss_weighting` config knob.
- Added diagnostics for:
  - fallback smoothness by source,
  - fallback residual energy by source,
  - `sr_tmin` usable RRT fraction by timestep.
- Ran diagnostics on the cached fallback dataset:
  - clean smoothness: `6.597253882838576`,
  - RRT smoothness: `574.6017420248655`,
  - clean residual energy: `0.06841986887156963`,
  - RRT residual energy: `0.16783318338394165`,
  - both basic data-direction gates passed.
- Marked the original seed-0 report as historical diagnostic output because it was generated before these fixes.
- Verification:
  - `pytest -q`: 13 tests passed.
  - `compileall`: passed.
  - one-trial report smoke after fixes completed.

## 2026-07-01 Post-Fix Seed-0 Rerun

- Retrained all seven seed-0 policies for 200,000 steps after the review fixes
  and after removing the equal-source compatibility path:
  - `runs/maze2d_gcs_only_seed0_20260701_072457`
  - `runs/maze2d_rrt_only_seed0_20260701_073116`
  - `runs/maze2d_cotrain_seed0_20260701_073116`
  - `runs/maze2d_ambient_scalar_seed0_20260701_073116`
  - `runs/maze2d_sr_tmin_seed0_20260701_073116`
  - `runs/maze2d_sr_freqmask_seed0_20260701_072457`
  - `runs/maze2d_sr_full_seed0_20260701_073116`
- Ran the 1,000 shared-trial evaluation:
  `python -m srtd.eval.report --runs <seven post-fix run dirs> --num-trials 1000 --out runs/maze2d_seed0_retrain_20260701_report --seed 0 --save-rollouts 5`
- Updated tracked report files:
  - `reports/maze2d_seed0_metrics.csv`
  - `reports/maze2d_seed0_summary.md`
- Runtime report artifacts:
  - `runs/maze2d_seed0_retrain_20260701_report/metrics.csv`
  - `runs/maze2d_seed0_retrain_20260701_report/success_vs_smoothness_pareto.png`
  - `runs/maze2d_seed0_retrain_20260701_report/generated_residual_energy.png`
  - `runs/maze2d_seed0_retrain_20260701_report/success_collision_rates.png`
  - `runs/maze2d_seed0_retrain_20260701_report/rollout_grid_same_start_goal.png`
- Key result:
  - `sr_freqmask` had the best success rate at `0.873`.
  - `sr_full` was second on success at `0.836` and had the lowest collision rate at `0.027`.
  - `ambient_scalar` remained strong at `0.795`, but was below `sr_freqmask`.
  - `sr_tmin` still underperformed at `0.640`.
  - `sr_freqmask` also had the lowest generated high-frequency residual energy at `0.2937557602347806`.
- Conclusion:
  - The post-fix seed-0 fallback rerun supports the spectral-mask objective on success.
  - This is still a single-seed fallback result; next step is multi-seed stress testing and `sr_tmin` gate tuning.

## 2026-07-01 Audit/Faithfulness Patch

- Read the external review in `/home/srinivas/.codex/attachments/d5aaf068-81d7-4ad7-832b-7adf0cfaa871/pasted-text.txt`.
- Added an explicit Diffusion Policy VP schedule:
  - `diffusion_policy_cosine` implements diffusers `squaredcos_cap_v2`,
  - legacy `cosine` remains the old sine-sigma schedule for existing checkpoint interpretation,
  - `python -m srtd.diffusion.schedule_report` reports sigma/index mappings.
- Schedule diagnostic result for 100 train steps:
  - old `sine_sigma`: `sigma=0.074 -> t_idx=5`,
  - Diffusion Policy `squaredcos_cap_v2`: `sigma=0.074 -> t_idx=4`,
  - `squaredcos_cap_v2` has `sigma(t=18) ~= 0.3034`, so the Ambient appendix `tmin=18` crossing is not equivalent to the scalar `sigma=0.074` under this local mapping.
- Added true VP Ambient x0-loss support:
  - new config: `srtd/configs/maze2d_ambient_scalar_ambient_loss.yaml`,
  - renamed current scalar-gated x0-MSE baseline config: `srtd/configs/maze2d_ambient_sampler_x0_mse.yaml`,
  - smoke training for the Ambient-loss path completed for 2 steps on CUDA.
- Added Diffusion Policy-style evaluation controls:
  - raw or first-order low-pass filtered target execution,
  - optional linear interpolation substeps,
  - padded and unpadded collision metrics,
  - primary collision defaults to padded in the report CLI.
- Added rollout-quality metrics:
  - cubic spline acceleration,
  - finite-difference acceleration,
  - jerk,
  - turn rate,
  - padded/unpadded obstacle clearance,
  - raw action-target jump size.
- Added reproducibility bundling:
  - `python -m srtd.eval.bundle`,
  - report CLI also accepts `--bundle-out`.
- Added `sr_tmin` audit table support at `t = 0, 5, 10, 18, 25, 50, 75, 99`.
  - Existing seed-0 `sr_tmin` annotation table under the old `sine_sigma` schedule:
    - `t=0`: `0.021084394305944443`
    - `t=5`: `0.10587459057569504`
    - `t=10`: `0.2240358293056488`
    - `t=18`: `0.24343234300613403`
    - `t=25`: `0.24720415472984314`
    - `t=50`: `0.24821311235427856`
    - `t=75`: `0.24821311235427856`
    - `t=99`: `0.24821311235427856`
- Added frequency-mask mechanism switches:
  - `visibility_only`,
  - `compatibility_only`,
  - `lowfreq_only`,
  - `shuffled_clean_stats`,
  - `rrt_only_freqmask`.
- Caveat:
  - the tracked 0.873 `sr_freqmask` result predates this audit patch and should be rerun under the new faithful baseline/evaluation settings before making stronger claims.

## 2026-07-01 Audited 3-Seed Training and Evaluation

- Added minimal audited sweep configs using the `diffusion_policy_cosine`
  schedule:
  - `srtd/configs/maze2d_audit_cotrain.yaml`
  - `srtd/configs/maze2d_audit_sr_freqmask.yaml`
  - `srtd/configs/maze2d_audit_sr_freqmask_shuffled_clean_stats.yaml`
- Generated cached fallback datasets for seeds 1 and 2:
  - `data/generated/maze2d_fallback_seed1_p50_q5000_h100.npz`
  - `data/generated/maze2d_fallback_seed2_p50_q5000_h100.npz`
- Trained 15 total policies over seeds 0, 1, and 2:
  - `cotrain`
  - `ambient_sampler_x0_mse`
  - `ambient_scalar_ambient_loss`
  - `sr_freqmask`
  - `sr_freqmask_shuffled_clean_stats`
- Training manifest:
  - `logs/audit3seed_20260701_213340_manifest.txt`
- All 15 final checkpoints completed under `runs/*_20260701_213343`.
- Ran the primary audited filtered/padded report:
  `python -m srtd.eval.report --runs <15 manifest runs> --num-trials 1000 --out runs/audit3seed_filtered_padded_20260701_report --seed 0 --save-rollouts 5 --execution-mode filtered --lowpass-alpha 0.35 --interpolation-steps 4 --primary-collision-padding padded`
- Ran the raw/padded companion report:
  `python -m srtd.eval.report --runs <15 manifest runs> --num-trials 1000 --out runs/audit3seed_raw_padded_20260701_report --seed 0 --save-rollouts 5 --execution-mode raw --interpolation-steps 4 --primary-collision-padding padded`
- Tracked audit outputs:
  - `reports/maze2d_audit3seed_summary.md`
  - `reports/maze2d_audit3seed_aggregate.csv`
  - `reports/maze2d_audit3seed_filtered_padded_metrics.csv`
  - `reports/maze2d_audit3seed_raw_padded_metrics.csv`
- Local reproducibility bundle:
  - `runs/audit3seed_repro_20260701.tar.gz`
  - size: about 290 MB
  - includes all three generated datasets, all 15 final checkpoints/configs,
    spectral annotations/figures, and both report directories
  - excludes intermediate step checkpoints by default
- Main filtered/padded aggregate result:
  - `sr_freqmask`: `0.470 +/- 0.018` success, `0.472 +/- 0.035` collision
  - `sr_freqmask_shuffled_clean_stats`: `0.463 +/- 0.016` success,
    `0.467 +/- 0.029` collision
  - `cotrain`: `0.405 +/- 0.026` success, `0.489 +/- 0.025` collision
  - `ambient_sampler_x0_mse`: `0.294 +/- 0.022` success
  - `ambient_scalar_ambient_loss`: `0.107 +/- 0.015` success
- Raw/padded companion result:
  - `sr_freqmask`: `0.522 +/- 0.010` success
  - `sr_freqmask_shuffled_clean_stats`: `0.505 +/- 0.012` success
  - `cotrain`: `0.498 +/- 0.019` success
  - `ambient_sampler_x0_mse`: `0.398 +/- 0.003` success
  - `ambient_scalar_ambient_loss`: `0.129 +/- 0.018` success
- Interpretation:
  - `sr_freqmask` is best on mean success in both audited reports.
  - The shuffled-clean-stats ablation is too close to claim that clean spectral
    compatibility is the decisive mechanism.
  - The faithful VP Ambient x0-loss baseline performs poorly enough that it
    should be treated as an implementation/tuning warning rather than a
    conclusion about Ambient Diffusion Policy.
  - The current result is a useful diagnostic research deliverable, not a
    clean positive paper result.

## 2026-07-02 Audit Review Patch

- Read the external review in
  `/home/srinivas/.codex/attachments/843e8ebd-452b-4ac1-9b4e-7bd2d9cc8a43/pasted-text.txt`.
- Fixed the rollout interpolation bug called out in review:
  - dense interpolation points are still used for collision and smoothness,
  - policy observations now keep separate `obs_prev`/`obs_curr` state at the
    10 Hz command cadence used during training,
  - interpolation substeps no longer alter the model observation distribution.
- Report changes:
  - added per-trial `trial_metrics.csv`,
  - added pairwise `paired_stats.csv` with McNemar exact p-values and paired
    bootstrap confidence intervals for success differences,
  - changed rollout path keys to include seed labels so multi-seed grids do
    not overwrite earlier seeds,
  - aggregate summary plots now use policy means instead of duplicated labels,
  - split high-frequency residual reporting into absolute-position and
    delta-motion metrics,
  - replaced `-inf` clearance means with:
    - `out_of_bounds_rate`,
    - success-only finite clearance,
    - collision-free finite clearance,
    - all-run finite clipped clearance.
- Added compact config inheritance through `extends`.
- Added next-audit configs for:
  - `gcs_only` and `rrt_only`,
  - Ambient sampler x0-MSE at `tmin_idx = 4, 18, 30`,
  - Ambient x0-loss at `tmin_idx = 4, 18, 30` and buffer
    `10, 100, 1000, inf`,
  - frequency-mask ablations:
    `visibility_only`, `compatibility_only`, `lowfreq_only`,
    `constant_lowpass_mask`, `random_mask_same_density`,
    `shuffled_clean_stats`, and `shuffled_target_residuals`.
- Added `run_label` support so sweep variants can share a training method
  while producing distinct run directories and report labels.
- Important caveat:
  - the 2026-07-01 audited 3-seed metrics predate this cadence fix and should
    be treated as historical diagnostic artifacts until rerun.
