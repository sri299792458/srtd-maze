import torch


def test_fft_roundtrip():
    actions = torch.randn(8, 16, 2)
    coeffs = torch.fft.rfft(actions, dim=1, norm="ortho")
    recon = torch.fft.irfft(coeffs, n=16, dim=1, norm="ortho")
    assert torch.allclose(actions, recon, atol=1e-5)

