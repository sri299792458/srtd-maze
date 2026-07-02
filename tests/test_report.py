from srtd.eval.report import paired_trial_stats


def test_paired_trial_stats_uses_shared_seed_trials():
    rows = [
        {"policy": "a", "seed": 0, "trial_idx": 0, "success": 1, "collision": 0, "endpoint_error": 0.1, "smoothness": 1.0},
        {"policy": "b", "seed": 0, "trial_idx": 0, "success": 0, "collision": 1, "endpoint_error": 0.4, "smoothness": 2.0},
        {"policy": "a", "seed": 0, "trial_idx": 1, "success": 0, "collision": 1, "endpoint_error": 0.3, "smoothness": 3.0},
        {"policy": "b", "seed": 0, "trial_idx": 1, "success": 0, "collision": 1, "endpoint_error": 0.5, "smoothness": 4.0},
    ]

    stats = paired_trial_stats(rows)

    assert len(stats) == 1
    assert stats[0]["policy_a"] == "a"
    assert stats[0]["policy_b"] == "b"
    assert stats[0]["paired_trials"] == 2
    assert stats[0]["success_diff_a_minus_b"] == 0.5
    assert stats[0]["success_a_only"] == 1
    assert stats[0]["success_b_only"] == 0
