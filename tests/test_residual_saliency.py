import torch

from srtd.spectral.residual import high_frequency_residual_energy, spectral_residual


def test_residual_saliency_spike():
    t = torch.arange(16) * 2 * torch.pi / 16
    smooth = torch.stack([torch.sin(t), torch.cos(t)], dim=-1).unsqueeze(0)
    actions = smooth.clone()
    actions[0, 8, 0] += 4.0
    saliency = spectral_residual(actions)["saliency_scalar"][0]
    assert abs(int(saliency.argmax().item()) - 8) <= 1


def test_rrt_jitter_has_higher_residual():
    x = torch.linspace(-1.0, 1.0, 16)
    smooth = torch.stack([x, 0.25 * torch.sin(torch.pi * x)], dim=-1).unsqueeze(0)
    jitter = smooth.clone()
    jitter[0, 1::2, 1] += 0.25
    jitter[0, 2::2, 1] -= 0.25
    smooth_score = high_frequency_residual_energy(smooth)
    jitter_score = high_frequency_residual_energy(jitter)
    assert jitter_score.item() > smooth_score.item()
