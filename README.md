# Spectral Residual Trajectory Diffusion: 2D Maze Prototype

This workspace implements the first smoke test from the pasted SRTD spec:
use a cheap 2D maze trajectory setting to test whether spectral residual
gating can keep low-frequency route information from jittery RRT-like data
while suppressing high-frequency residual artifacts.

The initial scope is intentionally narrow:

- fallback 5 m x 5 m maze data generation,
- trajectory spectral residual diagnostics,
- clean spectral statistics fitted only on clean training chunks,
- SNR-based visibility masks,
- per-chunk spectral-residual `tmin` annotations,
- Ambient scalar and spectral-residual samplers,
- standard, spectral-masked, and saliency-weighted losses,
- unit tests for the core mathematical gates.

## Quick Start

```bash
source .venv/bin/activate
pip install -e .
pytest
python -m srtd.training.train_maze2d --config srtd/configs/maze2d_sr_tmin.yaml --max-steps 10
```

Long training is not run by default. The config defaults mirror the requested
experiment, but the CLI accepts `--max-steps` for fast smoke tests.

## Seed-0 Results

The lightweight git history tracks source code, configs, tests, and the final
seed-0 summary metrics in `reports/`.

Current post-fix seed-0 rerun:

- Summary: `reports/maze2d_seed0_summary.md`
- Metrics: `reports/maze2d_seed0_metrics.csv`
- Runtime report directory: `runs/maze2d_seed0_retrain_20260701_report`

Key post-fix result: `sr_freqmask` reached `0.873` success on 1000 shared
trials, ahead of `ambient_scalar` at `0.795`; `sr_full` reached `0.836` and had
the lowest collision rate at `0.027`.

Important caveat: those tracked metrics predate the audit patch that added the
Diffusion Policy `squaredcos_cap_v2` VP schedule, VP Ambient x0-loss,
filtered execution, padded primary collision reporting, and reproducibility
bundling. Treat them as the strongest fallback result so far, not as a faithful
Ambient baseline comparison.

Audit commands:

```bash
python -m srtd.diffusion.schedule_report --train-steps 100 --sigma 0.074 --t 0 5 10 18 25 50 75 99
python -m srtd.eval.report --runs <run dirs> --execution-mode filtered --primary-collision-padding padded
python -m srtd.eval.bundle --out runs/repro_bundle.tar.gz --dataset data/generated/maze2d_fallback_seed0_p50_q5000_h100.npz --runs <run dirs> --report-dir <report dir>
```

Heavy runtime artifacts are ignored locally and are not kept in the main
branch. The existing GitHub Release asset
`srtd-maze-seed0-results.tar.gz` is the pre-fix historical run archive:
https://github.com/sri299792458/srtd-maze/releases/tag/seed0-results-v1
