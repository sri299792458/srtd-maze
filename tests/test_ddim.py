import torch

from srtd.diffusion.ddim import ddim_x0_step


def test_ddim_x0_step_preserves_oracle_noise_direction():
    x0 = torch.tensor([[[1.0, -0.5], [0.25, 0.75]]])
    eps = torch.tensor([[[0.2, -1.0], [0.5, 0.3]]])
    sigma_t = torch.tensor(0.8)
    sigma_prev = torch.tensor(0.25)
    alpha_t = torch.sqrt(1.0 - sigma_t.square())
    alpha_prev = torch.sqrt(1.0 - sigma_prev.square())
    x_t = alpha_t * x0 + sigma_t * eps

    x_prev = ddim_x0_step(x_t, x0, sigma_t, sigma_prev)
    expected = alpha_prev * x0 + sigma_prev * eps

    assert torch.allclose(x_prev, expected, atol=1e-6)

