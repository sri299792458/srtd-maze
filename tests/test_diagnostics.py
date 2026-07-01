import numpy as np

from srtd.data.maze2d_dataset import MazeChunk, MazeEpisode
from srtd.eval.diagnostics import fallback_data_sanity, sr_tmin_usable_fraction_by_t


def test_sr_tmin_usable_fraction_by_t():
    chunks = [
        MazeChunk(np.zeros(6), np.zeros((16, 2)), 0, 0, 0),
        MazeChunk(np.zeros(6), np.zeros((16, 2)), 1, 1, 0, sr_tmin_idx=2),
        MazeChunk(np.zeros(6), np.zeros((16, 2)), 1, 2, 0, sr_tmin_idx=4),
        MazeChunk(np.zeros(6), np.zeros((16, 2)), 1, 3, 0, sr_tmin_idx=None),
    ]
    frac = sr_tmin_usable_fraction_by_t(chunks, train_steps=5)
    assert np.allclose(frac, [0.0, 0.0, 1 / 3, 1 / 3, 2 / 3])


def test_fallback_data_sanity_flags_jittered_auxiliary():
    t = np.linspace(0.0, 1.0, 40)
    clean_path = np.stack([t, t], axis=-1).astype(np.float32)
    jittered = clean_path.copy()
    jittered[1::2, 1] += 0.1
    jittered[2::2, 1] -= 0.1
    episodes = [
        MazeEpisode(clean_path, clean_path[0], clean_path[-1], "p", 0),
        MazeEpisode(jittered, jittered[0], jittered[-1], "q", 1),
    ]

    report = fallback_data_sanity(episodes, dt=0.1)

    assert report["rrt_less_smooth_than_clean"]
    assert report["rrt_more_residual_than_clean"]

