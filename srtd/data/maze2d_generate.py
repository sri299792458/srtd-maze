from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import numpy as np

from srtd.data.maze2d_dataset import MazeEpisode, save_episodes
from srtd.eval.maze_env import MazeEnv


def _resample_path(path: np.ndarray, horizon: int) -> np.ndarray:
    if len(path) == 1:
        return np.repeat(path, horizon, axis=0).astype(np.float32)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] < 1e-8:
        return np.repeat(path[:1], horizon, axis=0).astype(np.float32)
    target = np.linspace(0.0, dist[-1], horizon)
    out = np.empty((horizon, 2), dtype=np.float32)
    out[:, 0] = np.interp(target, dist, path[:, 0])
    out[:, 1] = np.interp(target, dist, path[:, 1])
    return out


def _shortcut(path: np.ndarray, env: MazeEnv, rng: np.random.Generator, iters: int = 80) -> np.ndarray:
    points = [p.copy() for p in path]
    for _ in range(iters):
        if len(points) <= 2:
            break
        i, j = sorted(rng.choice(len(points), size=2, replace=False))
        if j <= i + 1:
            continue
        if env.segment_is_free(points[i], points[j]):
            points = points[: i + 1] + points[j:]
    return np.asarray(points, dtype=np.float32)


def _smooth_path(path: np.ndarray, env: MazeEnv, rng: np.random.Generator) -> np.ndarray:
    dense = _resample_path(path, max(64, len(path) * 4))
    smoothed = dense.copy()
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel = kernel / kernel.sum()
    for _ in range(6):
        candidate = smoothed.copy()
        for d in range(2):
            candidate[:, d] = np.convolve(smoothed[:, d], kernel, mode="same")
        candidate[0] = dense[0]
        candidate[-1] = dense[-1]
        if env.path_is_free(candidate):
            smoothed = candidate
    return _shortcut(smoothed, env, rng, iters=20)


def _grid_astar(
    env: MazeEnv,
    start: np.ndarray,
    goal: np.ndarray,
    resolution: float = 0.1,
) -> np.ndarray | None:
    xmin, ymin, xmax, ymax = env.bounds
    width = int(round((xmax - xmin) / resolution)) + 1
    height = int(round((ymax - ymin) / resolution)) + 1

    def to_cell(point: np.ndarray) -> tuple[int, int]:
        ix = int(round((point[0] - xmin) / resolution))
        iy = int(round((point[1] - ymin) / resolution))
        return max(0, min(width - 1, ix)), max(0, min(height - 1, iy))

    def to_point(cell: tuple[int, int]) -> np.ndarray:
        return np.asarray([xmin + cell[0] * resolution, ymin + cell[1] * resolution], dtype=np.float32)

    start_cell = to_cell(start)
    goal_cell = to_cell(goal)
    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start_cell))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start_cell: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_cell:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            points = np.asarray([to_point(c) for c in reversed(cells)], dtype=np.float32)
            points[0] = start
            points[-1] = goal
            return points
        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            p0, p1 = to_point(current), to_point(nxt)
            if not env.segment_is_free(p0, p1):
                continue
            step_cost = float(np.linalg.norm(p1 - p0))
            new_cost = cost[current] + step_cost
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                priority = new_cost + float(np.linalg.norm(p1 - goal))
                heapq.heappush(open_heap, (priority, nxt))
                came_from[nxt] = current
    return None


def _rrt_plan(
    env: MazeEnv,
    start: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    step_size: float = 0.25,
    goal_bias: float = 0.12,
    max_iters: int = 2500,
) -> np.ndarray | None:
    nodes = [start.astype(np.float32)]
    parents = [-1]
    for _ in range(max_iters):
        sample = goal if rng.random() < goal_bias else env.sample_free(rng)
        dists = np.linalg.norm(np.asarray(nodes) - sample, axis=1)
        nearest_idx = int(np.argmin(dists))
        direction = sample - nodes[nearest_idx]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            continue
        new = nodes[nearest_idx] + direction / norm * min(step_size, norm)
        if not env.segment_is_free(nodes[nearest_idx], new):
            continue
        nodes.append(new.astype(np.float32))
        parents.append(nearest_idx)
        if np.linalg.norm(new - goal) <= step_size and env.segment_is_free(new, goal):
            nodes.append(goal.astype(np.float32))
            parents.append(len(nodes) - 2)
            idx = len(nodes) - 1
            path = []
            while idx >= 0:
                path.append(nodes[idx])
                idx = parents[idx]
            return np.asarray(path[::-1], dtype=np.float32)
    return None


def _jitter_path(
    path: np.ndarray,
    env: MazeEnv,
    rng: np.random.Generator,
    scale: float = 0.045,
) -> np.ndarray:
    out = path.copy()
    for i in range(1, len(out) - 1):
        candidate = out[i] + rng.normal(0.0, scale, size=2)
        if env.is_free(candidate) and env.segment_is_free(out[i - 1], candidate) and env.segment_is_free(candidate, out[i + 1]):
            out[i] = candidate
    return out.astype(np.float32)


def _sample_pair(env: MazeEnv, rng: np.random.Generator, min_distance: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(10_000):
        start = env.sample_free(rng)
        goal = env.sample_free(rng)
        if np.linalg.norm(goal - start) >= min_distance:
            return start, goal
    raise RuntimeError("failed to sample start/goal pair")


def generate_fallback_episodes(
    env: MazeEnv,
    num_clean: int = 50,
    num_rrt: int = 5000,
    horizon: int = 100,
    seed: int = 0,
) -> list[MazeEpisode]:
    rng = np.random.default_rng(seed)
    episodes: list[MazeEpisode] = []
    episode_id = 0

    def add_clean(start: np.ndarray, goal: np.ndarray) -> bool:
        nonlocal episode_id
        path = _grid_astar(env, start, goal)
        if path is None:
            return False
        path = _smooth_path(path, env, rng)
        positions = _resample_path(path, horizon)
        episodes.append(MazeEpisode(positions, start, goal, "p", episode_id))
        episode_id += 1
        return True

    def add_rrt(start: np.ndarray, goal: np.ndarray) -> bool:
        nonlocal episode_id
        path = _rrt_plan(env, start, goal, rng)
        if path is None:
            path = _grid_astar(env, start, goal)
        if path is None:
            return False
        path = _shortcut(path, env, rng, iters=15)
        positions = _resample_path(path, horizon)
        positions = _jitter_path(positions, env, rng)
        episodes.append(MazeEpisode(positions, start, goal, "q", episode_id))
        episode_id += 1
        return True

    while sum(e.source == "p" for e in episodes) < num_clean:
        s, g = _sample_pair(env, rng)
        add_clean(s, g)
    while sum(e.source == "q" for e in episodes) < num_rrt:
        s, g = _sample_pair(env, rng)
        add_rrt(s, g)
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze", default="assets/maze2d_default.yaml")
    parser.add_argument("--out", default="data/generated/maze2d_fallback.npz")
    parser.add_argument("--num-clean", type=int, default=50)
    parser.add_argument("--num-rrt", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = MazeEnv.from_yaml(args.maze)
    episodes = generate_fallback_episodes(
        env,
        num_clean=args.num_clean,
        num_rrt=args.num_rrt,
        horizon=args.horizon,
        seed=args.seed,
    )
    save_episodes(Path(args.out), episodes)
    print(f"saved {len(episodes)} episodes to {args.out}")


if __name__ == "__main__":
    main()

