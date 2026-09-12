"""Tests for sit_fer.core.config.Config"""

import yaml
from sit_fer.core.config import Config


def test_get_with_dot_notation(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "training": {"lr": 0.0001, "batch_size": 16},
        "model": {"num_classes": 7},
    }))

    config = Config(config_path=str(config_path))

    assert config.get("training.lr") == 0.0001
    assert config.get("training.batch_size") == 16
    assert config.get("model.num_classes") == 7


def test_get_missing_key_returns_default():
    config = Config()
    assert config.get("does.not.exist") is None
    assert config.get("does.not.exist", default=42) == 42


def test_set_creates_nested_keys():
    config = Config()
    config.set("a.b.c", 123)
    assert config.get("a.b.c") == 123


def test_kwargs_override_file_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"training": {"lr": 0.1}}))

    config = Config(config_path=str(config_path), training={"lr": 0.5})

    assert config.get("training.lr") == 0.5


def test_save_round_trip(tmp_path):
    config = Config()
    config.set("training.epochs", 80)

    out_path = tmp_path / "nested" / "saved.yaml"
    config.save(str(out_path))

    reloaded = Config(config_path=str(out_path))
    assert reloaded.get("training.epochs") == 80
