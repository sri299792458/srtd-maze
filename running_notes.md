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
