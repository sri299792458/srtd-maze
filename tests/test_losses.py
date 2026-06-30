import torch
import torch.nn.functional as F

from srtd.training.losses import spectral_mse


def test_spectral_mse_parseval():
    pred = torch.randn(4, 16, 2)
    target = torch.randn(4, 16, 2)
    weights = torch.ones_like(pred)
    assert torch.allclose(spectral_mse(pred, target, weights), F.mse_loss(pred, target), atol=1e-6)

