import torch
import torch.nn.functional as F

from srtd.diffusion.schedules import VPSchedule
from srtd.spectral.envelope import fit_clean_spectral_stats
from srtd.training.losses import (
    add_vp_ambient_noise,
    combine_source_losses,
    q_frequency_weights,
    spectral_mse,
    vp_ambient_x0_loss,
)


def test_spectral_mse_parseval():
    pred = torch.randn(4, 16, 2)
    target = torch.randn(4, 16, 2)
    weights = torch.ones_like(pred)
    assert torch.allclose(spectral_mse(pred, target, weights), F.mse_loss(pred, target), atol=1e-6)


def test_combine_source_losses_sample_weighted():
    p_loss = torch.tensor(10.0)
    q_loss = torch.tensor(0.0)
    loss = combine_source_losses(p_loss, q_loss, n_p=1, n_q=9)
    assert torch.allclose(loss, torch.tensor(1.0))


def test_combine_source_losses_single_source():
    p_loss = torch.tensor(10.0)
    loss = combine_source_losses(p_loss, None, n_p=3, n_q=0)
    assert torch.allclose(loss, p_loss)


def test_vp_ambient_x0_loss_oracle_is_smaller_than_zero_prediction():
    schedule = VPSchedule.diffusion_policy_cosine(train_steps=100)
    x0 = torch.randn(8, 16, 2)
    t_idx = torch.full((8,), 18, dtype=torch.long)
    tmin_idx = 4
    x_t, x_tmin = add_vp_ambient_noise(x0, t_idx, tmin_idx, schedule)

    oracle = vp_ambient_x0_loss(x0, x_t, x_tmin, t_idx, tmin_idx, schedule)
    zero = vp_ambient_x0_loss(torch.zeros_like(x0), x_t, x_tmin, t_idx, tmin_idx, schedule)

    assert oracle < zero


def test_q_frequency_weight_ablation_modes_are_valid():
    schedule = VPSchedule.sine_sigma(train_steps=10)
    clean = torch.randn(6, 16, 2)
    target = torch.randn(3, 16, 2)
    stats = fit_clean_spectral_stats(clean, schedule=schedule)
    t_idx = torch.tensor([2, 5, 8])

    modes = [
        "full",
        "visibility_only",
        "compatibility_only",
        "lowfreq_only",
        "constant_lowpass_mask",
        "random_mask_same_density",
        "shuffled_clean_stats",
        "shuffled_target_residuals",
    ]
    weights = [
        q_frequency_weights(target, t_idx, stats, schedule, mask_mode=mode)
        for mode in modes
    ]

    assert all(weight.shape == weights[0].shape for weight in weights)
    assert all(torch.isfinite(weight).all() for weight in weights)
