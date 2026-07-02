from srtd.diffusion.schedule_report import schedule_report
from srtd.diffusion.schedules import VPSchedule


def test_diffusion_policy_cosine_sigma_mapping_is_explicit():
    old_schedule = VPSchedule.sine_sigma(train_steps=100)
    dp_schedule = VPSchedule.diffusion_policy_cosine(train_steps=100)

    assert old_schedule.sigma_to_t_idx(0.074) == 5
    assert dp_schedule.sigma_to_t_idx(0.074) == 4
    assert dp_schedule.t_idx_to_sigma(18) > 0.30


def test_schedule_report_contains_ambient_comparison_indices():
    report = schedule_report(train_steps=100, sigma_thresholds=[0.074], t_indices=[18])
    assert report["schedules"]["sine_sigma"]["sigma_to_t_idx"]["0.074"] == 5
    assert report["schedules"]["diffusion_policy_cosine"]["sigma_to_t_idx"]["0.074"] == 4
    assert "18" in report["schedules"]["diffusion_policy_cosine"]["sigma_at_t"]
