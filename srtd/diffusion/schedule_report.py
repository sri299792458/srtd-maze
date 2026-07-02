from __future__ import annotations

import argparse
import json

from srtd.diffusion.schedules import VPSchedule


def schedule_report(
    train_steps: int = 100,
    sigma_thresholds: list[float] | None = None,
    t_indices: list[int] | None = None,
) -> dict:
    sigma_thresholds = sigma_thresholds or [0.074]
    t_indices = t_indices or [0, 5, 10, 18, 25, 50, 75, 99]
    schedules = [
        VPSchedule.sine_sigma(train_steps=train_steps),
        VPSchedule.diffusion_policy_cosine(train_steps=train_steps),
    ]
    out = {"train_steps": train_steps, "schedules": {}}
    for schedule in schedules:
        out["schedules"][schedule.name] = {
            "sigma_to_t_idx": {
                str(threshold): schedule.sigma_to_t_idx(threshold)
                for threshold in sigma_thresholds
            },
            "sigma_at_t": {
                str(t): schedule.t_idx_to_sigma(t)
                for t in t_indices
                if 0 <= t < schedule.train_steps
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--sigma", type=float, nargs="*", default=[0.074])
    parser.add_argument("--t", type=int, nargs="*", default=[0, 5, 10, 18, 25, 50, 75, 99])
    args = parser.parse_args()
    print(
        json.dumps(
            schedule_report(
                train_steps=args.train_steps,
                sigma_thresholds=args.sigma,
                t_indices=args.t,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
