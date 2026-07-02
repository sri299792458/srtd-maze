from srtd.config import load_config


def test_load_config_extends_parent_relative_to_child(tmp_path):
    (tmp_path / "base.yaml").write_text(
        """
method: cotrain
training:
  total_steps: 10
  lr: 0.001
diffusion:
  schedule: cosine
""",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        """
extends: base.yaml
method: sr_freqmask
training:
  lr: 0.0001
""",
        encoding="utf-8",
    )

    cfg = load_config(tmp_path / "child.yaml")

    assert cfg["method"] == "sr_freqmask"
    assert cfg["training"]["total_steps"] == 10
    assert cfg["training"]["lr"] == 0.0001
    assert cfg["diffusion"]["schedule"] == "cosine"
