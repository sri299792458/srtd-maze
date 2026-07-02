import numpy as np
import torch

from srtd.data.normalization import normalize_xy
from srtd.diffusion.schedules import VPSchedule
from srtd.eval.maze_env import MazeEnv
from srtd.eval.rollout import rollout_policy


def test_interpolation_does_not_change_policy_observation_cadence(monkeypatch):
    observed = []
    targets_m = [
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[2.0, 0.0]], dtype=np.float32),
    ]

    def fake_ddim_sample(model, obs, schedule, horizon, action_dim, inference_steps):
        observed.append(obs.detach().cpu().numpy()[0].copy())
        return torch.as_tensor(normalize_xy(targets_m[len(observed) - 1])[None])

    monkeypatch.setattr("srtd.eval.rollout.ddim_sample", fake_ddim_sample)
    env = MazeEnv(bounds=(-1.0, -1.0, 5.0, 1.0), obstacles=[], padding=0.0)

    rollout_policy(
        torch.nn.Identity(),
        env,
        VPSchedule.sine_sigma(train_steps=10),
        start=np.asarray([0.0, 0.0], dtype=np.float32),
        goal=np.asarray([4.0, 0.0], dtype=np.float32),
        policy_horizon=1,
        execute_horizon=1,
        inference_steps=1,
        success_radius_m=0.01,
        timeout_s=0.21,
        dt=0.1,
        interpolation_steps=4,
        primary_collision_padded=False,
    )

    assert len(observed) == 2
    expected_second_obs = np.concatenate(
        [
            normalize_xy(np.asarray([0.0, 0.0], dtype=np.float32)),
            normalize_xy(np.asarray([1.0, 0.0], dtype=np.float32)),
            normalize_xy(np.asarray([4.0, 0.0], dtype=np.float32)),
        ]
    )
    np.testing.assert_allclose(observed[1], expected_second_obs)
