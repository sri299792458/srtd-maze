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

