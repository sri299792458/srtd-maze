from __future__ import annotations

import tarfile

from srtd.eval.bundle import create_repro_bundle


def _names(path):
    with tarfile.open(path, "r:gz") as tar:
        return set(tar.getnames())


def test_repro_bundle_defaults_to_final_checkpoints_and_multiple_inputs(tmp_path):
    dataset_a = tmp_path / "dataset_a.npz"
    dataset_b = tmp_path / "dataset_b.npz"
    dataset_a.write_bytes(b"a")
    dataset_b.write_bytes(b"b")

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    for name in ["config.json", "checkpoint_last.pt", "losses.txt", "checkpoint_step_10000.pt"]:
        (run_dir / name).write_text(name, encoding="utf-8")
    (run_dir / "annotations").mkdir()
    (run_dir / "annotations" / "sr_tmin.npy").write_bytes(b"annotation")

    report_a = tmp_path / "report_a"
    report_b = tmp_path / "report_b"
    report_a.mkdir()
    report_b.mkdir()
    (report_a / "metrics.csv").write_text("policy,success_rate\n", encoding="utf-8")
    (report_b / "eval_config.json").write_text("{}", encoding="utf-8")

    bundle = create_repro_bundle(
        tmp_path / "bundle.tar.gz",
        [dataset_a, dataset_b],
        [run_dir],
        [report_a, report_b],
    )

    names = _names(bundle)
    assert "data/dataset_a.npz" in names
    assert "data/dataset_b.npz" in names
    assert "runs/run_a/checkpoint_last.pt" in names
    assert "runs/run_a/checkpoint_step_10000.pt" not in names
    assert "runs/run_a/annotations/sr_tmin.npy" in names
    assert "runs/report_a/metrics.csv" in names
    assert "runs/report_b/eval_config.json" in names
    assert "SHA256SUMS.json" in names


def test_repro_bundle_can_include_step_checkpoints(tmp_path):
    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    for name in ["config.json", "checkpoint_last.pt", "losses.txt", "checkpoint_step_10000.pt"]:
        (run_dir / name).write_text(name, encoding="utf-8")

    bundle = create_repro_bundle(
        tmp_path / "bundle.tar.gz",
        None,
        [run_dir],
        None,
        include_step_checkpoints=True,
    )

    assert "runs/run_a/checkpoint_step_10000.pt" in _names(bundle)
