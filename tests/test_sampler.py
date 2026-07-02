import numpy as np

from srtd.data.maze2d_dataset import MazeChunk
from srtd.diffusion.schedules import VPSchedule
from srtd.training.samplers import AmbientScalarSampler, SRTminSampler


def _chunk(source_id: int, tmin: int | None):
    return MazeChunk(
        obs=np.zeros(6, dtype=np.float32),
        actions=np.zeros((16, 2), dtype=np.float32),
        source_id=source_id,
        episode_id=0,
        chunk_start=0,
        sr_tmin_idx=tmin,
    )


def test_sampler_admissibility():
    schedule = VPSchedule.cosine(train_steps=10)
    sampler = SRTminSampler([_chunk(0, None), _chunk(1, 4)], schedule)
    assert sampler.admissible(sampler.chunks[0], 0)
    assert sampler.admissible(sampler.chunks[0], 9)
    assert not sampler.admissible(sampler.chunks[1], 3)
    assert sampler.admissible(sampler.chunks[1], 4)


def test_ambient_sampler_can_use_explicit_tmin_idx_strictly():
    schedule = VPSchedule.sine_sigma(train_steps=10)
    sampler = AmbientScalarSampler(
        [_chunk(0, None), _chunk(1, None)],
        schedule,
        tmin_idx_scalar=4,
        strict_after_tmin=True,
    )
    assert sampler.admissible(sampler.chunks[0], 0)
    assert not sampler.admissible(sampler.chunks[1], 4)
    assert sampler.admissible(sampler.chunks[1], 5)
