import torch
import torch.nn.functional as F

from srtd.training.losses import combine_source_losses, spectral_mse


def test_spectral_mse_parseval():
    pred = torch.randn(4, 16, 2)
    target = torch.randn(4, 16, 2)
    weights = torch.ones_like(pred)
    assert torch.allclose(spectral_mse(pred, target, weights), F.mse_loss(pred, target), atol=1e-6)


def test_combine_source_losses_sample_weighted():
    p_loss = torch.tensor(10.0)
    q_loss = torch.tensor(0.0)
    loss = combine_source_losses(p_loss, q_loss, n_p=1, n_q=9, source_loss_weighting="sample")
    assert torch.allclose(loss, torch.tensor(1.0))


def test_combine_source_losses_source_equal():
    p_loss = torch.tensor(10.0)
    q_loss = torch.tensor(0.0)
    loss = combine_source_losses(p_loss, q_loss, n_p=1, n_q=9, source_loss_weighting="source_equal")
    assert torch.allclose(loss, torch.tensor(5.0))
