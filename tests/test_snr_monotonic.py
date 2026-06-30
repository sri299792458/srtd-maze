import torch

from srtd.spectral.snr import max_visible_frequency, visibility_mask


def test_visibility_monotonic():
    freq = torch.arange(9, dtype=torch.float32)
    clean_psd = (1.0 / (freq + 1.0).square()).view(-1, 1).repeat(1, 2)
    sigma = torch.linspace(0.05, 0.95, 20)
    visible = visibility_mask(clean_psd, sigma, tau=0.2, temperature=0.25)
    max_freq = max_visible_frequency(visible, threshold=0.5)
    assert torch.all(max_freq[1:] <= max_freq[:-1])

