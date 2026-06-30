from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from srtd.spectral.residual import spectral_residual


def plot_psd_gcs_vs_rrt(clean_actions: torch.Tensor, rrt_actions: torch.Tensor, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean_psd = torch.fft.rfft(clean_actions, dim=1, norm="ortho").abs().square().mean(dim=0)
    rrt_psd = torch.fft.rfft(rrt_actions, dim=1, norm="ortho").abs().square().mean(dim=0)
    freq = np.arange(clean_psd.shape[0])
    fig, axes = plt.subplots(1, clean_psd.shape[1], figsize=(9, 3), squeeze=False)
    for d, ax in enumerate(axes[0]):
        ax.loglog(freq[1:] + 1, clean_psd[1:, d].cpu().numpy() + 1e-12, label="GCS-like")
        ax.loglog(freq[1:] + 1, rrt_psd[1:, d].cpu().numpy() + 1e-12, label="RRT-like")
        ax.set_title(f"dim {d}")
        ax.set_xlabel("frequency bin")
        ax.set_ylabel("PSD")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_visibility(visible: torch.Tensor, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    heat = visible.mean(dim=-1).detach().cpu().numpy().T
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(heat, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xlabel("diffusion timestep")
    ax.set_ylabel("frequency bin")
    fig.colorbar(im, ax=ax, label="visibility")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_sr_tmin_hist(tmin: np.ndarray, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(tmin, bins=40)
    ax.set_xlabel("SR tmin index")
    ax.set_ylabel("chunks")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_residual_saliency_examples(
    actions: torch.Tensor,
    out: str | Path,
    max_examples: int = 8,
) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sal = spectral_residual(actions[:max_examples])["saliency_scalar"].detach().cpu().numpy()
    xy = actions[:max_examples].detach().cpu().numpy()
    n = min(max_examples, len(xy))
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.4), squeeze=False)
    for i, ax in enumerate(axes[0]):
        sizes = 8.0 + 35.0 * sal[i] / (sal[i].max() + 1e-8)
        ax.plot(xy[i, :, 0], xy[i, :, 1], color="0.35", linewidth=1.0)
        ax.scatter(xy[i, :, 0], xy[i, :, 1], s=sizes, c=sal[i], cmap="magma")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

