from samileides.config import ExperimentConfig
from samileides.data import repo_root


def test_load_smoke_config():
    cfg = ExperimentConfig.load(repo_root() / "configs" / "experiments" / "smoke.yaml")
    assert cfg.name == "smoke"
    assert cfg.data.source == "greek"
    assert cfg.model.d_model == 256
    assert cfg.tokenizer.vocab_size == 4000
    assert cfg.training.per_device_batch_size == 64
    assert cfg.inference.beam == 5


def test_load_pilot_config_is_transformer_big():
    cfg = ExperimentConfig.load(repo_root() / "configs" / "experiments" / "pilot.yaml")
    assert cfg.model.d_model == 1024
    assert cfg.model.encoder_layers == 6
    assert cfg.tokenizer.vocab_size == 32000


def test_load_vref_configs():
    base = repo_root() / "configs" / "experiments" / "vref"
    for encoding in ("struct", "vtok", "text"):
        cfg = ExperimentConfig.load(base / f"vref_ie_{encoding}.yaml")
        assert cfg.data.source == "vref"
        assert cfg.data.vref_encoding == encoding
        assert cfg.data.max_ratio == 0            # ratio filter disabled
        assert cfg.model.d_model == 512           # transformer-base, like ie_base
        assert cfg.training.max_steps == 180000
        assert cfg.probe is not None
        assert cfg.probe.every_steps == 1000
        assert cfg.probe.patience_steps == 20000
        assert cfg.probe.min_gain == 1.0
        assert cfg.data.expected_train_manifest == "experiments/vref-train-manifest.txt"


def test_config_without_probe_section_has_none():
    cfg = ExperimentConfig.load(repo_root() / "configs" / "experiments" / "ie_base.yaml")
    assert cfg.probe is None


def test_unknown_yaml_keys_ignored(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: x\nphase: one-to-many\n"
        "data:\n  selection: a.csv\n  holdouts: b.yaml\n  future_key: 1\n"
        "model:\n  d_model: 128\n",
        encoding="utf-8",
    )
    cfg = ExperimentConfig.load(p)
    assert cfg.data.selection == "a.csv"
    assert cfg.model.d_model == 128
